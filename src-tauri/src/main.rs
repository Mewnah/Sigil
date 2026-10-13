#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use clap::Parser;
use serde::{Deserialize, Serialize};
use tauri::{command, Manager, State};
use tauri_plugin_window_state::{AppHandleExt, StateFlags};

use windows::Win32::UI::WindowsAndMessaging::{ShowWindow, SW_SHOWMINNOACTIVE};

use crate::services::AppConfiguration;

mod exit_guard;
mod port_precheck;
mod services;
mod webview_recovery;
mod window_size;

#[derive(Parser, Debug)]
struct InitArguments {
    #[arg(short, long, default_value_t = 3030)]
    port: u16,
}

#[derive(Serialize, Deserialize)]
struct NativeFeatures {
    background_input: bool,
}

#[command]
fn get_native_features() -> NativeFeatures {
    NativeFeatures {
        background_input: cfg!(feature = "background_input"),
    }
}

#[command]
fn get_port(state: State<'_, InitArguments>) -> u16 {
    state.port
}

#[command]
fn app_close(app_handle: tauri::AppHandle) {
    let Some(window) = app_handle.get_webview_window("main") else {
        return app_handle.exit(0);
    };
    app_handle.save_window_state(StateFlags::all()).ok(); // don't really care if it saves it

    if window.close().is_err() {
        app_handle.exit(0);
    }
}

fn main() {
    // Headless entry points (`Sigil.exe --headless <cmd>`: EP probe, later bench / fetch / import
    // worker) run before the port pre-check and never start Tauri, bind the port or show a dialog.
    if let Some(headless) = services::cli::headless_args() {
        std::process::exit(services::cli::run(headless));
    }

    let args = InitArguments::parse();
    let start_minimized = start_minimized();

    // Stop if the port is not available. Checks the same address the web plugin's server binds
    // (persisted `lanAccess.mode`: 0.0.0.0 for LAN, the default, or 127.0.0.1 for "This PC
    // only"), so the check matches what warp will do and never prompts the firewall needlessly.
    // A held port is waited for first (a previous Sigil may still be exiting), and a running Sigil is
    // brought to the front instead of failing; nothing is shown during the wait (see `port_precheck`).
    // The check also logs what it found to the app log file (the logger starts later, inside Tauri).
    match port_precheck::run(services::web::startup_bind_ip(), args.port, !start_minimized) {
        port_precheck::Outcome::Free => {}
        port_precheck::Outcome::Raised => return,
        port_precheck::Outcome::Busy { error, holder } => {
            let text = port_precheck::describe(args.port, &error, &holder);
            if start_minimized {
                // Automated run on a PC someone is using: no box, no focus. The exit code says it, and
                // the log has the reason (a release build has no console to print it to).
                port_precheck::log_line(&format!("Sigil did not start: {text}"));
            } else {
                port_precheck::show_message(&text);
            }
            std::process::exit(1);
        }
    }

    let mut context = tauri::generate_context!();
    if start_minimized {
        // Created hidden and without focus; `setup` shows it minimized (see `show_minimized_without_focus`).
        for window in context.config_mut().app.windows.iter_mut().filter(|w| w.label == "main") {
            window.visible = false;
            window.focus = false;
        }
    }
    let mut window_state = tauri_plugin_window_state::Builder::default()
        .skip_initial_state(services::caption_window::CAPTIONS_WINDOW_LABEL);
    if start_minimized {
        // The plugin's restore ends in show() + set_focus(), which would take the foreground.
        window_state = window_state.skip_initial_state("main");
    }

    tauri::Builder::default()
        // Registered first: plugins initialize in registration order, so the diagnostics setup (file log,
        // EcoQoS opt-out, EP probe) runs before every other plugin.
        .plugin(services::diagnostics::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(move |app| {
            let window = app.get_webview_window("main").expect("Failed to get main window");
            window.set_shadow(true).expect("Unsupported platform!");
            if let Ok(hwnd) = window.hwnd() {
                // For the exit guard's diagnostics line only.
                exit_guard::set_main_window(hwnd.0 as isize);
            }
            if start_minimized {
                show_minimized_without_focus(&window);
            }
            // A renderer failure of the host page reloads it (captions pass through that page).
            webview_recovery::watch(&window);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_port,
            get_native_features,
            app_close,
            services::oauth_loopback::oauth_loopback_start
        ])
        // First launch only (no saved state for `main`): size the window from its monitor instead of the
        // fixed 800 x 600. Registered BEFORE the window-state plugin on purpose: both hook the new
        // window in registration order, so this runs first and a saved state is restored over it. Does
        // nothing for SIGIL_START_MINIMIZED runs (see `window_size`).
        .plugin(window_size::init(!start_minimized))
        // The caption window (P6-B) restores its own position and size without show()/set_focus(),
        // so it never steals the foreground from a game; saving still goes through this plugin.
        .plugin(window_state.build())
        // After a restore: grow a window that is too small for Interface size Auto, cut one that is too big
        // for its monitor. Registered AFTER the window-state plugin on purpose, so it sees the restored
        // size. Does nothing for SIGIL_START_MINIMIZED runs (see `window_size`).
        .plugin(window_size::init_after_restore(!start_minimized))
        .manage(AppConfiguration { port: args.port })
        .plugin(services::osc::init())
        .plugin(services::web::init())
        .plugin(services::audio::init())
        .plugin(services::windows_tts::init())
        .plugin(services::uberduck_tts::init())
        .plugin(services::keyboard::init())
        .plugin(services::uwu::init())
        .plugin(services::whisper::init())
        .plugin(services::translate::init())
        .plugin(services::kokoro_tts::init())
        .plugin(services::moonshine_stt::init())
        .plugin(services::melo_tts::init())
        .plugin(services::chatterbox_tts::init())
        .plugin(services::fish_speech::init())
        .plugin(services::neural_onnx_tts::init())
        .plugin(services::neural_sidecar_tts::init())
        .plugin(services::voice_changer::init())
        .plugin(services::vosk_stt::init())
        // P1 foundations (P1-S0b): model store, native runtime loader, system probe, resource
        // governor, helper-process host and benchmark harness.
        .plugin(services::model_store::init())
        .plugin(services::native_runtime::init())
        .plugin(services::sysinfo::init())
        .plugin(services::resources::init())
        .plugin(services::process_host::init())
        .plugin(services::bench::init())
        // P2 speech engine (P2-S0): STT pipelines, the unified TTS sink, in-process local TTS and
        // the OpenAI-compatible streaming TTS client.
        .plugin(services::speech::init())
        .plugin(services::tts_sink::init())
        .plugin(services::local_tts::init())
        .plugin(services::openai_speech::init())
        // P4 translation (P4-S0): the bundled llama-server helper + LLM proxy, and local MT
        // (NLLB on ORT, the Bergamot helper).
        .plugin(services::llm_server::init())
        .plugin(services::mt_local::init())
        // P5 voice-changer families (P5-S0): the shared reference-voice library (zero-shot voices
        // and TTS cloning).
        .plugin(services::voice_library::init())
        // P6 streamer features (P6-S0): caption window and SteamVR overlay (P6-B), readings
        // (P6-L), Edge / Gemini TTS (P6-H), helper-process TTS engines (P6-E/F/G) and the
        // GPT-SoVITS client (P6-G).
        .plugin(services::caption_window::init())
        .plugin(services::vr_overlay::init())
        .plugin(services::text_readings::init())
        .plugin(services::edge_tts::init())
        .plugin(services::gemini_tts::init())
        .plugin(services::tts_helpers::init())
        .plugin(services::gpt_sovits::init())
        // Secondary windows (the caption window, P6-B) close with the main window, so Sigil never
        // keeps running with only them open (Tauri exits when the last window is gone).
        .on_window_event(|window, event| {
            if window.label() != "main" {
                return;
            }
            match event {
                // Closing the main window must end the process: the exit guard (P7-D) makes sure it does.
                tauri::WindowEvent::CloseRequested { .. } => exit_guard::arm(),
                tauri::WindowEvent::Destroyed => close_secondary_windows(window.app_handle()),
                _ => {}
            }
        })
        .build(context)
        .expect("error while building tauri application")
        .run(|_app, event| match event {
            // The app was asked to exit (last window destroyed, `app.exit`): same guard, started once.
            tauri::RunEvent::ExitRequested { .. } => exit_guard::arm(),
            tauri::RunEvent::Exit => {
                // Helper processes (llama-server, TTS servers, sidecars) die with Sigil. The Job
                // Object also kills them if the process is terminated without this event.
                services::process_host::shutdown_all();
                // Release OpenVR before the process exits (P6-B; a no-op when it never started).
                services::vr_overlay::shutdown();
            }
            _ => {}
        });
}

/// `SIGIL_START_MINIMIZED=1`: the main window starts minimized and never takes the foreground. Meant for automated
/// runs (benchmarks, packaged smoke tests) on a PC someone is using: Tauri's focus path (`set_focus`, which the
/// window-state restore also calls) forces the foreground with a synthetic Alt key press, past Windows' foreground lock.
fn start_minimized() -> bool {
    std::env::var("SIGIL_START_MINIMIZED").is_ok_and(|v| v.trim() == "1")
}

/// Show the hidden main window minimized without activating it (`SIGIL_START_MINIMIZED`). `show()` comes first so the
/// webview counts as visible (created with `focus: false`, it shows without activation); the minimize is queued on the
/// main thread right behind it.
fn show_minimized_without_focus(window: &tauri::WebviewWindow) {
    if let Err(e) = window.show() {
        tracing::warn!("start minimized: show failed: {e}");
    }
    let hwnd = match window.hwnd() {
        Ok(hwnd) => hwnd.0 as isize,
        Err(e) => {
            tracing::warn!("start minimized: no window handle: {e}");
            return;
        }
    };
    let queued = window.run_on_main_thread(move || unsafe {
        let _ = ShowWindow(windows::Win32::Foundation::HWND(hwnd as *mut _), SW_SHOWMINNOACTIVE);
    });
    if let Err(e) = queued {
        tracing::warn!("start minimized: could not queue the minimize: {e}");
    }
}

/// Close every webview window except the main one (P6-S0).
fn close_secondary_windows<R: tauri::Runtime>(app: &tauri::AppHandle<R>) {
    for (label, window) in app.webview_windows() {
        if label != "main" {
            let _ = window.close();
        }
    }
}
