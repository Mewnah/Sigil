//! Takes waiting to be saved or discarded: processed clips held in memory (a 10 s clip is about
//! 1 MB), never on disk. The newest [`MAX_TAKES`] are kept; older ones are dropped, so an abandoned
//! recorder never grows memory.

use std::collections::VecDeque;
use std::sync::{Mutex, PoisonError};

use super::process::ProcessedTake;
use super::VoiceTake;

/// Takes kept at once.
pub const MAX_TAKES: usize = 6;

/// A held take.
#[derive(Clone, Debug)]
pub(super) struct Take {
    pub info: VoiceTake,
    pub processed: ProcessedTake,
    /// The clip as a 16-bit WAV (what is saved and what the preview plays).
    pub wav: Vec<u8>,
}

/// A bounded set of takes (the plugin uses [`TAKES`]; tests make their own).
pub(super) struct TakeStore {
    takes: Mutex<VecDeque<Take>>,
}

/// The plugin's takes.
pub(super) static TAKES: TakeStore = TakeStore::new();

impl Default for TakeStore {
    fn default() -> Self {
        TakeStore::new()
    }
}

/// A take id: `take-` and 12 lower-case hex digits.
pub fn valid_take_id(id: &str) -> bool {
    id.len() == 17 && id.starts_with("take-") && id[5..].chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
}

pub(super) fn new_take_id() -> String {
    format!("take-{}", super::store::token(12))
}

fn gone(id: &str) -> String {
    format!("take {id:?} is gone (record or import it again)")
}

impl TakeStore {
    pub(super) const fn new() -> TakeStore {
        TakeStore { takes: Mutex::new(VecDeque::new()) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<Take>> {
        self.takes.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Hold a take (dropping the oldest past [`MAX_TAKES`]).
    pub(super) fn insert(&self, take: Take) {
        let mut t = self.lock();
        t.retain(|x| x.info.take_id != take.info.take_id);
        t.push_back(take);
        while t.len() > MAX_TAKES {
            t.pop_front();
        }
    }

    /// A copy of take `id`.
    pub(super) fn get(&self, id: &str) -> Result<Take, String> {
        if !valid_take_id(id) {
            return Err(format!("invalid take id {id:?}"));
        }
        self.lock().iter().find(|t| t.info.take_id == id).cloned().ok_or_else(|| gone(id))
    }

    /// Take `id` out of the store in one step. A save claims its take this way, so two saves of
    /// one take (a double Enter) never both store it: the second finds it gone. A save that fails
    /// puts it back with [`Self::insert`].
    pub(super) fn take(&self, id: &str) -> Result<Take, String> {
        if !valid_take_id(id) {
            return Err(format!("invalid take id {id:?}"));
        }
        let mut t = self.lock();
        let i = t.iter().position(|x| x.info.take_id == id).ok_or_else(|| gone(id))?;
        t.remove(i).ok_or_else(|| gone(id))
    }

    /// The WAV bytes of take `id`.
    pub(super) fn wav(&self, id: &str) -> Result<Vec<u8>, String> {
        self.get(id).map(|t| t.wav)
    }

    /// Drop take `id` (no-op when it is not held).
    pub(super) fn remove(&self, id: &str) {
        self.lock().retain(|t| t.info.take_id != id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn take(id: &str) -> Take {
        Take {
            info: VoiceTake { take_id: id.into(), source: "recorded".into(), ..VoiceTake::default() },
            processed: ProcessedTake { clip: vec![0.0; 4], rate: 16_000, quality: Default::default(), clipped: false },
            wav: vec![1, 2, 3],
        }
    }

    #[test]
    fn takes_are_bounded_and_ids_validated() {
        let ids: Vec<String> = (0..MAX_TAKES + 2).map(|_| new_take_id()).collect();
        assert!(ids.iter().all(|i| valid_take_id(i)), "{ids:?}");
        let store = TakeStore::new();
        for id in &ids {
            store.insert(take(id));
        }
        assert!(store.get(&ids[0]).is_err() && store.get(&ids[1]).is_err(), "the oldest were dropped");
        assert_eq!(store.wav(&ids[MAX_TAKES + 1]).unwrap(), vec![1, 2, 3]);
        store.remove(&ids[MAX_TAKES + 1]);
        assert!(store.get(&ids[MAX_TAKES + 1]).unwrap_err().contains("gone"));
        assert!(store.get("../etc").unwrap_err().contains("invalid"));
        for bad in ["take-", "take-xyz000000000", "TAKE-000000000000", "take-0000000000000"] {
            assert!(!valid_take_id(bad), "{bad}");
        }
    }

    #[test]
    fn a_take_is_claimed_once() {
        let store = TakeStore::new();
        let id = new_take_id();
        store.insert(take(&id));
        let claims: Vec<Result<Take, String>> = std::thread::scope(|s| {
            let a = s.spawn(|| store.take(&id));
            let b = s.spawn(|| store.take(&id));
            vec![a.join().unwrap(), b.join().unwrap()]
        });
        assert_eq!(claims.iter().filter(|c| c.is_ok()).count(), 1, "one claim wins");
        assert!(claims.iter().any(|c| c.as_ref().is_err_and(|e| e.contains("gone"))));
        assert!(store.get(&id).is_err(), "a claimed take is no longer held");
        // A failed save puts it back.
        let claimed = claims.into_iter().find_map(Result::ok).unwrap();
        store.insert(claimed);
        assert_eq!(store.take(&id).unwrap().wav, vec![1, 2, 3]);
        assert!(store.take("../x").unwrap_err().contains("invalid"));
    }
}
