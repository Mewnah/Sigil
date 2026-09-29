//! Real-time voice changer: capture → **classic** phase vocoder (DSP), optional **WebSocket** converter,
//! optional **ONNX** chunk model, or **RVC** neural VC → shared post chain / harness → playback.
//!
//! Module layout:
//! - [`params`]: IPC params, defaults, clamps / normalization and the RVC path, EP and F0 rules.
//! - [`state`]: [`VoiceChangerState`], the Tauri-managed state shared by commands and the worker,
//!   with the params snapshot / apply rules and the persistence wiring.
//! - [`budget`]: the capture-callback push and the worker-side backlog with the latency budget.
//! - [`playback`]: output queue → lock-free ring hand-off, real-time playout, channel adapter.
//! - [`resample`]: the stateful band-limited resampler of the monitor output.
//! - [`aux_feed`]: aux (TTS) clips mixed into the capture stream, with back-pressure and the mic
//!   gate (`aux` is a reserved file name on Windows).
//! - [`worker`]: model load, streams, the processing loop, run generations and
//!   `voice_changer:error` emission.
//! - [`commands`]: params / enable / start / stop / model-slot IPC commands.
//! - [`fallback`]: failed-block concealment (muted with click-free edges, never the dry voice).
//! - [`devices`]: opening the saved input / output / monitor devices (exact name, else the one
//!   renumbered device of the same family, else a clear "not connected" error).
//! - [`metrics`], [`debug`]: metrics and debug-snapshot IPC commands.
//! - [`persist`]: atomic `settings.json` / `slot.json` persistence (Rust is authoritative; the TS
//!   service hydrates from `get_voice_changer_params`).
//!
//! The IPC command list lives in `src-tauri/plugin-commands/voice-changer.txt`, and the frozen P0
//! Rust <-> TS parameter / metrics contract in `params_contract.json` next to this file.

mod aux_feed;
mod budget;
mod commands;
mod debug;
mod devices;
mod fallback;
mod metrics;
mod params;
mod persist;
mod playback;
mod resample;
mod state;
mod worker;

#[cfg(test)]
mod contract_tests;

pub use aux_feed::push_aux_wav_bytes;
pub use state::VoiceChangerState;

use tauri::{
    plugin::{Builder, TauriPlugin},
    Manager, RunEvent, Runtime,
};

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("voice-changer")
        .setup(|app, _api| {
            let state = VoiceChangerState::new();
            // Restore the saved settings (Rust is authoritative) before any command can run.
            match app.path().app_data_dir() {
                Ok(dir) => state.attach_persistence(&dir),
                Err(e) => tracing::warn!("voice changer: no app data folder ({e}); settings will not be saved"),
            }
            app.manage(state);
            Ok(())
        })
        .on_event(|app, event| {
            if let RunEvent::Exit = event {
                if let Some(state) = app.try_state::<VoiceChangerState>() {
                    state.flush_settings();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_voice_changer_params,
            commands::set_voice_changer_params,
            commands::set_voice_changer_enabled,
            commands::start_voice_changer,
            commands::stop_voice_changer,
            commands::is_voice_changer_running,
            metrics::get_voice_changer_metrics,
            aux_feed::voice_changer_push_aux_pcm,
            aux_feed::voice_changer_push_aux_wav,
            debug::capture_voice_changer_debug,
            commands::list_voice_changer_model_slots,
        ])
        .build()
}
