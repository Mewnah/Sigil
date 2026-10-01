//! Real-time voice changer: capture → **classic** phase vocoder (DSP), optional **WebSocket** converter,
//! optional **ONNX** chunk model, or **RVC** neural VC → shared post chain / harness → playback.
//!
//! Module layout:
//! - [`params`]: IPC params, defaults, clamps / normalization and the RVC path, EP and F0 rules.
//! - [`state`]: [`VoiceChangerState`], the Tauri-managed state shared by commands and the worker,
//!   with the params snapshot / apply rules and the persistence wiring.
//! - [`budget`]: the capture-ring push and the worker-side backlog with the latency budget.
//! - [`playback`]: output queue → lock-free ring hand-off (with the latency control that trims a
//!   stall's catch-up back to the steady level), real-time playout, channel adapter.
//! - [`resample`]: the stateful band-limited resampler of the monitor output.
//! - [`aux_feed`]: aux (TTS) clips mixed into the capture stream, with back-pressure and the mic
//!   gate (`aux` is a reserved file name on Windows).
//! - [`worker`]: model load, the processing loop on the capture hub (VC-only conditioning:
//!   gain, denoiser, aux mix), voice switching, the governor, metrics and the timeline, run
//!   generations and `voice_changer:error` emission.
//! - [`commands`]: params / enable / start / stop / model-slot IPC commands.
//! - [`fallback`]: failed-block concealment (muted with click-free edges, never the dry voice).
//! - [`devices`]: finding the saved devices (exact name, else the one renumbered device of the
//!   same family, else a clear "not connected" error); the capture as a device-rate consumer of the
//!   capture hub (`audio::hub`: STT keeps the raw feed of the same stream); output / monitor ports
//!   that follow their device (reopen when it returns, follow the system default).
//! - [`metrics`], [`debug`]: metrics and debug-snapshot IPC commands.
//! - [`persist`]: atomic `settings.json` / `slot.json` persistence (Rust is authoritative; the TS
//!   service hydrates from `get_voice_changer_params`).
//!
//! P3 (scaffolded by P3-S0; owners in each module's docs):
//! - [`capabilities`]: what this build runs (`voice_changer_get_capabilities`) and the lever
//!   resolution (scaffold-owned; features turn on through their own `available()`).
//! - [`factory`]: the engine factory dispatching on `manifest.format`, and the hot voice set
//!   (P3-E).
//! - [`timeline`]: the per-tick timeline for telemetry and live recording (P3-E).
//! - [`cmd_import`]: `.pth` / `.zip` import and slot-store IPC (P3-B).
//! - [`cmd_bench`]: autotune and live-record IPC (P3-F).
//!
//! The IPC command list lives in `src-tauri/plugin-commands/voice-changer.txt`, and the frozen P0
//! Rust <-> TS parameter / metrics contract in `params_contract.json` next to this file.

mod aux_feed;
pub(crate) mod budget;
pub(crate) mod capabilities;
mod cmd_bench;
mod cmd_import;
mod commands;
mod debug;
mod devices;
pub(crate) mod factory;
pub(crate) mod fallback;
mod metrics;
pub(crate) mod params;
mod persist;
pub(crate) mod playback;
mod resample;
mod state;
pub(crate) mod timeline;
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
            // P3 (P3-S0 scaffold; implemented by the owning workstreams)
            capabilities::voice_changer_get_capabilities,
            cmd_import::voice_changer_import,
            cmd_import::voice_changer_cancel_import,
            cmd_import::voice_changer_list_slots,
            cmd_import::voice_changer_rename_slot,
            cmd_import::voice_changer_delete_slot,
            cmd_import::voice_changer_duplicate_slot,
            cmd_import::voice_changer_set_active_slot,
            cmd_import::voice_changer_merge_slots,
            cmd_import::voice_changer_import_default_voice,
            cmd_bench::voice_changer_autotune,
            cmd_bench::voice_changer_cancel_autotune,
            cmd_bench::voice_changer_get_autotune,
            cmd_bench::voice_changer_live_record_start,
            cmd_bench::voice_changer_live_record_stop,
        ])
        .build()
}
