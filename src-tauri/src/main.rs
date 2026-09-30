#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use clap::Parser;
use serde::{Deserialize, Serialize};
use tauri::{command, Manager, State};
use tauri_plugin_window_state::{AppHandleExt, StateFlags};

use windows::{
    core::{s, PCSTR},
    Win32::UI::WindowsAndMessaging::{MessageBoxA, MB_ICONWARNING, MB_OK},
};

use crate::services::AppConfiguration;

mod services;

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

    // crash if port is not available. Checks the same address the web plugin's server binds
    // (0.0.0.0:<port>, all IPv4 interfaces), so the check matches what warp will do.
    let port_availability = std::net::TcpListener::bind(format!("0.0.0.0:{}", args.port));
    match port_availability {
        Ok(l) => l.set_nonblocking(true).expect("Failed to set nonblocking"),
        Err(_err) => {
            // Native pre-webview errors stay English (P3: optional Tauri locale bridge if product needs localized OS dialogs).
            unsafe {
                MessageBoxA(
                    None,
                    PCSTR(format!("Port {} is not available!\0", args.port).as_ptr()),
                    s!("Sigil"),
                    MB_OK | MB_ICONWARNING,
                );
            }
            return;
        }
    };

    tauri::Builder::default()
        // Registered first: plugins initialize in registration order, so the diagnostics setup (file log,
        // EcoQoS opt-out, EP probe) runs before every other plugin.
        .plugin(services::diagnostics::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let window = app.get_webview_window("main").expect("Failed to get main window");
            window.set_shadow(true).expect("Unsupported platform!");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_port,
            get_native_features,
            app_close,
            services::oauth_loopback::oauth_loopback_start
        ])
        .plugin(tauri_plugin_window_state::Builder::default().build())
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
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|_app, event| {
            if let tauri::RunEvent::Exit = event {
                // Helper processes (llama-server, TTS servers, sidecars) die with Sigil. The Job
                // Object also kills them if the process is terminated without this event.
                services::process_host::shutdown_all();
            }
        });
}
