//! Install queue: one job per entry id, first come first served, at most `max_concurrent` jobs
//! (default 1, at most 2) running at a time, and never two downloads from one host at once.
//!
//! Jobs are plain tokio tasks spawned on the caller's runtime (the Tauri runtime in the app, the
//! headless runtime for `--headless fetch`, a test runtime in tests), so the queue owns no thread.
//! A job waits for the jobs of its `requires` first, then for a slot in FIFO order.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::sync::{watch, Notify};

/// Cooperative cancellation shared by a job, its download and its extraction.
#[derive(Clone, Default)]
pub(crate) struct CancelToken(Arc<CancelInner>);

#[derive(Default)]
struct CancelInner {
    flag: AtomicBool,
    notify: Notify,
}

impl CancelToken {
    pub fn cancel(&self) {
        self.0.flag.store(true, Ordering::SeqCst);
        self.0.notify.notify_waiters();
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.flag.load(Ordering::SeqCst)
    }

    /// Resolves once [`Self::cancel`] was called (immediately if it already was).
    pub async fn cancelled(&self) {
        loop {
            let notified = self.0.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

/// Final result of a job: the install folder, or the error text.
pub(crate) type JobResult = Result<PathBuf, String>;

/// Error text of a cancelled job (also shown to the user).
pub(crate) const CANCELLED: &str = "cancelled";

/// One queued or running install.
pub(crate) struct Job {
    pub id: String,
    pub seq: u64,
    pub cancel: CancelToken,
    pub deps: Vec<Arc<Job>>,
    done: watch::Sender<Option<JobResult>>,
}

impl Job {
    /// Wait for the job to finish (any number of waiters).
    pub async fn wait(&self) -> JobResult {
        let mut rx = self.done.subscribe();
        loop {
            if let Some(r) = rx.borrow_and_update().clone() {
                return r;
            }
            if rx.changed().await.is_err() {
                return Err("the install task ended unexpectedly".into());
            }
        }
    }

    #[cfg(test)]
    pub fn is_finished(&self) -> bool {
        self.done.borrow().is_some()
    }
}

struct QueueState {
    next_seq: u64,
    waiting: BTreeSet<u64>,
    active: usize,
    limit: usize,
    jobs: HashMap<String, Arc<Job>>,
}

/// The queue of one store.
pub(crate) struct Queue {
    state: Mutex<QueueState>,
    changed: Notify,
    host_locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

/// A held download slot; released on drop.
pub(crate) struct Slot<'a> {
    queue: &'a Queue,
}

/// A place in the line of [`Queue::acquire`]; leaves the line on drop unless it became a [`Slot`].
struct InLine<'a> {
    queue: &'a Queue,
    seq: u64,
    waiting: bool,
}

impl Drop for InLine<'_> {
    fn drop(&mut self) {
        if !self.waiting {
            return;
        }
        let removed = self.queue.lock().waiting.remove(&self.seq);
        if removed {
            // The waiter behind may be first now.
            self.queue.changed.notify_waiters();
        }
    }
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        {
            let mut st = self.queue.lock();
            st.active = st.active.saturating_sub(1);
        }
        self.queue.changed.notify_waiters();
    }
}

impl Queue {
    pub fn new(limit: usize) -> Self {
        Self {
            state: Mutex::new(QueueState {
                next_seq: 0,
                waiting: BTreeSet::new(),
                active: 0,
                limit: limit.clamp(1, 2),
                jobs: HashMap::new(),
            }),
            changed: Notify::new(),
            host_locks: Mutex::new(HashMap::new()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, QueueState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn set_limit(&self, limit: usize) {
        self.lock().limit = limit.clamp(1, 2);
        self.changed.notify_waiters();
    }

    #[cfg(test)]
    pub fn limit(&self) -> usize {
        self.lock().limit
    }

    pub fn get(&self, id: &str) -> Option<Arc<Job>> {
        self.lock().jobs.get(id).cloned()
    }

    pub fn ids(&self) -> Vec<String> {
        self.lock().jobs.keys().cloned().collect()
    }

    pub fn next_seq(&self) -> u64 {
        let mut st = self.lock();
        st.next_seq += 1;
        st.next_seq
    }

    /// The existing job for `id`, or a new one (`true`) registered under `id`.
    pub fn get_or_create(&self, id: &str, deps: Vec<Arc<Job>>) -> (Arc<Job>, bool) {
        let mut st = self.lock();
        if let Some(j) = st.jobs.get(id) {
            return (j.clone(), false);
        }
        st.next_seq += 1;
        let (done, _) = watch::channel(None);
        let job = Arc::new(Job {
            id: id.to_string(),
            seq: st.next_seq,
            cancel: CancelToken::default(),
            deps,
            done,
        });
        st.jobs.insert(id.to_string(), job.clone());
        (job, true)
    }

    /// Publish the job's result and forget it (a later install creates a new job).
    pub fn finish(&self, job: &Job, result: JobResult) {
        {
            let mut st = self.lock();
            if st.jobs.get(&job.id).is_some_and(|j| j.seq == job.seq) {
                st.jobs.remove(&job.id);
            }
        }
        // `send_replace`, not `send`: `send` drops the value when nobody is subscribed yet, and a
        // waiter that subscribes after the job finished would then wait forever.
        job.done.send_replace(Some(result));
    }

    /// Wait for a slot in FIFO order of `seq`. Returns `None` when `cancel` fires first.
    ///
    /// Drop-safe: when the future is dropped while waiting (a caller's `timeout` or `select!`), its
    /// place in line is given up, so it never blocks the waiters behind it.
    pub async fn acquire(&self, seq: u64, cancel: &CancelToken) -> Option<Slot<'_>> {
        self.lock().waiting.insert(seq);
        let mut place = InLine {
            queue: self,
            seq,
            waiting: true,
        };
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let mut st = self.lock();
                if cancel.is_cancelled() {
                    // `place` leaves the line (and wakes the others) when it drops.
                    drop(st);
                    return None;
                }
                if st.waiting.first() == Some(&seq) && st.active < st.limit {
                    st.waiting.remove(&seq);
                    st.active += 1;
                    place.waiting = false;
                    drop(st);
                    // The next waiter may fit too (limit 2).
                    self.changed.notify_waiters();
                    return Some(Slot { queue: self });
                }
            }
            tokio::select! {
                _ = &mut notified => {}
                _ = cancel.cancelled() => {}
            }
        }
    }

    /// The lock serializing downloads from one host.
    pub fn host_lock(&self, host: &str) -> Arc<tokio::sync::Mutex<()>> {
        self.host_locks
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entry(host.to_string())
            .or_default()
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    #[tokio::test]
    async fn slots_are_fifo_and_respect_the_limit() {
        let q = Arc::new(Queue::new(1));
        let running = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut tasks = Vec::new();
        for i in 0..4u64 {
            let seq = q.next_seq();
            let (q, running, peak, order) = (q.clone(), running.clone(), peak.clone(), order.clone());
            tasks.push(tokio::spawn(async move {
                let _slot = q.acquire(seq, &CancelToken::default()).await.unwrap();
                let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                order.lock().unwrap().push(i);
                tokio::time::sleep(Duration::from_millis(20)).await;
                running.fetch_sub(1, Ordering::SeqCst);
            }));
            // Register in order.
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        for t in tasks {
            t.await.unwrap();
        }
        assert_eq!(peak.load(Ordering::SeqCst), 1);
        assert_eq!(*order.lock().unwrap(), [0, 1, 2, 3]);
    }

    #[tokio::test]
    async fn limit_two_runs_two_and_cancel_leaves_the_line() {
        let q = Queue::new(2);
        let a = q.acquire(q.next_seq(), &CancelToken::default()).await.unwrap();
        let b = q.acquire(q.next_seq(), &CancelToken::default()).await.unwrap();
        let cancel = CancelToken::default();
        let seq = q.next_seq();
        let c2 = cancel.clone();
        let waiter = async { q.acquire(seq, &c2).await.is_none() };
        let canceller = async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            cancel.cancel();
        };
        let (was_cancelled, ()) = tokio::join!(waiter, canceller);
        assert!(was_cancelled);
        drop(a);
        drop(b);
        q.set_limit(1);
        let _c = q.acquire(q.next_seq(), &CancelToken::default()).await.unwrap();
        assert_eq!(q.limit(), 1);
    }

    #[tokio::test]
    async fn a_waiter_dropped_by_a_timeout_does_not_block_the_line() {
        let q = Queue::new(1);
        let held = q.acquire(q.next_seq(), &CancelToken::default()).await.unwrap();
        // A caller wraps its wait in a timeout while the only slot is busy (e.g. fetch_unpinned).
        let dropped = tokio::time::timeout(Duration::from_millis(30), q.acquire(q.next_seq(), &CancelToken::default())).await;
        assert!(dropped.is_err(), "timed out while the slot was held");
        let next = q.next_seq();
        let cancel = CancelToken::default();
        let waiter = async {
            tokio::time::timeout(Duration::from_secs(2), q.acquire(next, &cancel))
                .await
                .map(|slot| slot.is_some())
        };
        // The slot is released only after the next waiter is in line.
        let release = async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            drop(held);
        };
        let (got, ()) = tokio::join!(waiter, release);
        assert!(matches!(got, Ok(true)), "the next waiter must get the slot: {got:?}");
        assert!(q.lock().waiting.is_empty());
    }

    #[tokio::test]
    async fn a_result_published_before_anyone_waits_is_not_lost() {
        let q = Queue::new(1);
        let (job, _) = q.get_or_create("early", vec![]);
        q.finish(&job, Err("boom".into()));
        let r = tokio::time::timeout(Duration::from_secs(2), job.wait()).await;
        assert_eq!(r.expect("wait must not hang"), Err("boom".to_string()));
    }

    #[tokio::test]
    async fn jobs_are_shared_and_results_reach_every_waiter() {
        let q = Queue::new(1);
        let (j1, created) = q.get_or_create("a", vec![]);
        assert!(created);
        let (j2, created) = q.get_or_create("a", vec![]);
        assert!(!created);
        assert_eq!(j1.seq, j2.seq);
        let w = tokio::spawn({
            let j = j2.clone();
            async move { j.wait().await }
        });
        q.finish(&j1, Ok(PathBuf::from("x")));
        assert_eq!(w.await.unwrap().unwrap(), PathBuf::from("x"));
        assert!(j1.is_finished());
        assert!(q.get("a").is_none());
    }
}
