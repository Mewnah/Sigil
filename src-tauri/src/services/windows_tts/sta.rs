//! The SAPI thread (`sigil-sapi`): a COM single-threaded apartment that owns every SAPI object of
//! the plugin and pumps its messages.
//!
//! Why a thread of its own: SAPI voices and the engines they load live in the apartment that
//! created them. On the webview's main thread a slow engine would freeze the UI, and an
//! apartment-threaded engine called from SAPI's worker thread needs its apartment to pump messages
//! (calls arrive as window messages). This thread pumps them while it waits for work and while a
//! sentence renders ([`pump_messages`]).
//!
//! Requests run one at a time, in order, as closures over [`StaState`] ([`run`]); their results
//! come back through a oneshot channel. A request that panics is answered by its dropped channel,
//! and the voices are recreated for the next one. A thread that ended is restarted on the next
//! request.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{self, TryRecvError};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use tokio::sync::oneshot;
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::Media::Speech::{ISpeechVoice, SpVoice};
use windows::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_ALL, COINIT_APARTMENTTHREADED, COINIT_DISABLE_OLE1DDE};
use windows::Win32::System::Threading::GetCurrentThreadId;
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, MsgWaitForMultipleObjects, PeekMessageW, PostThreadMessageW, TranslateMessage, MSG, PM_NOREMOVE,
    PM_REMOVE, QS_ALLINPUT, WM_APP, WM_USER,
};

use super::render::Renderer;

/// A request for the SAPI thread.
pub(crate) type Job = Box<dyn FnOnce(&mut StaState) + Send>;

/// Thread message that only wakes the loop (a request was queued).
const WM_SIGIL_SAPI_JOB: u32 = WM_APP + 0x51;
/// Longest idle wait between message pumps (a lost wake-up message costs at most this).
const IDLE_WAIT_MS: u32 = 1_000;
/// How long a new thread may take to initialize COM.
const START_TIMEOUT: Duration = Duration::from_secs(10);

/// The SAPI objects, created on first use (on the SAPI thread only).
#[derive(Default)]
pub(crate) struct StaState {
    /// Automation voice of the legacy speak-to-device command.
    device_voice: Option<ISpeechVoice>,
    /// The voice that renders into the TTS sink.
    renderer: Option<Renderer>,
}

impl StaState {
    /// The automation voice (created on first use).
    pub(crate) fn device_voice(&mut self) -> Result<&ISpeechVoice, String> {
        if self.device_voice.is_none() {
            // SAFETY: COM is initialized on this (SAPI) thread.
            let voice: ISpeechVoice = unsafe { CoCreateInstance(&SpVoice, None, CLSCTX_ALL) }
                .map_err(|e| format!("Windows speech is not available: {}", e.message()))?;
            self.device_voice = Some(voice);
        }
        Ok(self.device_voice.as_ref().expect("created above"))
    }

    /// The sink renderer (created on first use).
    pub(crate) fn renderer(&mut self) -> Result<&mut Renderer, String> {
        if self.renderer.is_none() {
            self.renderer = Some(Renderer::new()?);
        }
        Ok(self.renderer.as_mut().expect("created above"))
    }

    /// Abandon the sink renderer (a wedged voice); the next sentence creates a new one. The old
    /// voice is leaked, not released: releasing a voice whose engine is stuck can wait for that
    /// engine, which would freeze this thread and with it every Windows voice request.
    pub(crate) fn abandon_renderer(&mut self) {
        if let Some(stuck) = self.renderer.take() {
            std::mem::forget(stuck);
        }
    }
}

struct Worker {
    tx: mpsc::Sender<Job>,
    thread_id: u32,
}

static WORKER: Mutex<Option<Worker>> = Mutex::new(None);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Run `f` on the SAPI thread after every request queued before it. The receiver gets the result,
/// or is closed when the thread could not run it (start failure, panic).
pub(crate) fn run<T: Send + 'static>(f: impl FnOnce(&mut StaState) -> T + Send + 'static) -> oneshot::Receiver<T> {
    let (tx, rx) = oneshot::channel();
    let job: Job = Box::new(move |state| {
        let _ = tx.send(f(state));
    });
    if let Err(e) = submit(job) {
        tracing::error!(target: "sigil::tts", "Windows speech: {e}");
    }
    rx
}

fn submit(job: Job) -> Result<(), String> {
    let mut worker = lock(&WORKER);
    let mut job = job;
    // A thread that ended drops its receiver: start a new one and queue there.
    for _ in 0..2 {
        if worker.is_none() {
            *worker = Some(spawn()?);
        }
        let w = worker.as_ref().expect("started above");
        match w.tx.send(job) {
            Ok(()) => {
                wake(w.thread_id);
                return Ok(());
            }
            Err(mpsc::SendError(back)) => {
                tracing::warn!(target: "sigil::tts", "Windows speech: the SAPI thread ended; starting a new one");
                job = back;
                *worker = None;
            }
        }
    }
    Err("the Windows speech thread could not start".into())
}

fn wake(thread_id: u32) {
    // SAFETY: posting to a thread id that created its queue before publishing the id. A failure
    // (the thread just ended) is covered by the idle wait / restart.
    let _ = unsafe { PostThreadMessageW(thread_id, WM_SIGIL_SAPI_JOB, WPARAM(0), LPARAM(0)) };
}

fn spawn() -> Result<Worker, String> {
    let (tx, rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<u32, String>>(1);
    std::thread::Builder::new()
        .name("sigil-sapi".into())
        .spawn(move || thread_main(rx, ready_tx))
        .map_err(|e| format!("could not start the SAPI thread: {e}"))?;
    let thread_id = ready_rx
        .recv_timeout(START_TIMEOUT)
        .map_err(|_| "the SAPI thread did not start".to_string())??;
    Ok(Worker { tx, thread_id })
}

fn thread_main(rx: mpsc::Receiver<Job>, ready: mpsc::SyncSender<Result<u32, String>>) {
    // SAFETY: COM initialization of this new thread (balanced below).
    let hr = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE) };
    if hr.is_err() {
        let _ = ready.send(Err(format!("COM could not start on the SAPI thread (0x{:08X})", hr.0 as u32)));
        return;
    }
    // Create this thread's message queue before its id is published (posts would fail before).
    let mut msg = MSG::default();
    // SAFETY: a valid MSG; PM_NOREMOVE only creates the queue.
    let _ = unsafe { PeekMessageW(&mut msg, None, WM_USER, WM_USER, PM_NOREMOVE) };
    // SAFETY: plain query of this thread's id.
    let _ = ready.send(Ok(unsafe { GetCurrentThreadId() }));
    tracing::debug!(target: "sigil::tts", "Windows speech: SAPI thread started");

    let mut state = StaState::default();
    loop {
        loop {
            match rx.try_recv() {
                Ok(job) => run_job(job, &mut state),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // Replaced: release the SAPI objects on their own thread, then COM.
                    drop(state);
                    // SAFETY: balances the CoInitializeEx above, after every COM object is gone.
                    unsafe { CoUninitialize() };
                    return;
                }
            }
        }
        // SAFETY: no handles; returns on a message (a queued request posts one) or after the wait.
        let _ = unsafe { MsgWaitForMultipleObjects(None, false, IDLE_WAIT_MS, QS_ALLINPUT) };
        pump_messages();
    }
}

fn run_job(job: Job, state: &mut StaState) {
    if let Err(panic) = catch_unwind(AssertUnwindSafe(|| job(state))) {
        tracing::error!(
            target: "sigil::tts",
            "Windows speech: a request panicked ({}); the voices are recreated",
            panic_message(panic.as_ref())
        );
        // A panic may have left the renderer mid-sentence: abandon it like a wedged one.
        state.abandon_renderer();
        state.device_voice = None;
    }
}

/// Dispatch every waiting message of this thread (SAPI and COM use them in this apartment).
pub(crate) fn pump_messages() {
    let mut msg = MSG::default();
    // SAFETY: a valid MSG; standard message loop calls on the calling thread.
    unsafe {
        while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block_on<T>(rx: oneshot::Receiver<T>) -> Option<T> {
        tauri::async_runtime::block_on(async { tokio::time::timeout(Duration::from_secs(20), rx).await.ok()?.ok() })
    }

    #[test]
    fn requests_run_in_order_on_one_sta_thread_and_survive_panics() {
        let a = run(|_| std::thread::current().name().map(str::to_string));
        let b = run(|_| (std::thread::current().id(), apartment_is_sta()));
        assert_eq!(block_on(a).flatten().as_deref(), Some("sigil-sapi"));
        let (thread, sta) = block_on(b).expect("ran");
        assert!(sta, "single-threaded apartment");

        // A panicking request closes its channel; the next one still runs (same thread). (The
        // panic message on stderr is expected.)
        let bad = run(|_| -> u32 { panic!("expected test panic on the SAPI thread") });
        assert_eq!(block_on(bad), None);
        let after = run(|_| std::thread::current().id());
        assert_eq!(block_on(after), Some(thread));

        // Order: results arrive in submission order.
        let order = std::sync::Arc::new(Mutex::new(Vec::new()));
        let rxs: Vec<_> = (0..5)
            .map(|i| {
                let order = order.clone();
                run(move |_| order.lock().unwrap().push(i))
            })
            .collect();
        for rx in rxs {
            block_on(rx).unwrap();
        }
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2, 3, 4]);
    }

    /// Whether the calling thread is in a single-threaded apartment.
    fn apartment_is_sta() -> bool {
        use windows::Win32::System::Com::{CoGetApartmentType, APTTYPE, APTTYPEQUALIFIER, APTTYPE_MAINSTA, APTTYPE_STA};
        let mut kind = APTTYPE::default();
        let mut qualifier = APTTYPEQUALIFIER::default();
        // SAFETY: valid out pointers.
        unsafe { CoGetApartmentType(&mut kind, &mut qualifier) }.is_ok() && (kind == APTTYPE_STA || kind == APTTYPE_MAINSTA)
    }
}
