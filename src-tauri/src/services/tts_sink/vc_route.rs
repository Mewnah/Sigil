//! The sink's route into the running voice changer: its aux input (TTS mixed into the microphone
//! before conversion, so speech comes out in the converted voice). Back-pressure, the 60 s queue cap
//! and the mic duck / mute while TTS plays are the voice changer's (`voice_changer::aux_feed`, P0).

use tauri::{AppHandle, Manager, Runtime};

use crate::services::voice_changer::VoiceChangerState;

/// What the sink needs from the voice changer (a fake in tests).
pub(crate) trait VcRoute: Send + Sync {
    /// Capture format `(rate, channels)` while the voice changer runs, else `None`.
    fn format(&self) -> Option<(u32, usize)>;
    /// Queue one block of `tag`'s audio without waiting: `Err(Some(block))` = full, retry later;
    /// `Err(None)` = not running / restarted in another format.
    fn try_push(&self, tag: u64, block: Vec<f32>, format: (u32, usize)) -> Result<(), Option<Vec<f32>>>;
    /// Samples of `tag` still queued (not yet mixed).
    fn pending(&self, tag: u64) -> usize;
    /// Drop `tag`'s queued audio (fading out a block already playing).
    fn cancel(&self, tag: u64);
}

/// The voice changer of the running app.
pub(crate) struct AppVc<R: Runtime>(pub AppHandle<R>);

impl<R: Runtime> VcRoute for AppVc<R> {
    fn format(&self) -> Option<(u32, usize)> {
        self.0.try_state::<VoiceChangerState>()?.tts_aux_format()
    }

    fn try_push(&self, tag: u64, block: Vec<f32>, format: (u32, usize)) -> Result<(), Option<Vec<f32>>> {
        match self.0.try_state::<VoiceChangerState>() {
            Some(vc) => vc.tts_aux_try_push(tag, block, format),
            None => Err(None),
        }
    }

    fn pending(&self, tag: u64) -> usize {
        self.0.try_state::<VoiceChangerState>().map_or(0, |vc| vc.tts_aux_pending(tag))
    }

    fn cancel(&self, tag: u64) {
        if let Some(vc) = self.0.try_state::<VoiceChangerState>() {
            vc.tts_aux_cancel(tag);
        }
    }
}

/// No voice changer (tests): the voice-changer route falls back to the device.
#[cfg(test)]
pub(crate) struct NoVc;

#[cfg(test)]
impl VcRoute for NoVc {
    fn format(&self) -> Option<(u32, usize)> {
        None
    }

    fn try_push(&self, _tag: u64, _block: Vec<f32>, _format: (u32, usize)) -> Result<(), Option<Vec<f32>>> {
        Err(None)
    }

    fn pending(&self, _tag: u64) -> usize {
        0
    }

    fn cancel(&self, _tag: u64) {}
}
