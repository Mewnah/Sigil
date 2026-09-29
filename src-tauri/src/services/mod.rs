pub mod audio;
pub mod http_client;
pub mod chatterbox_tts;
pub mod cli;
pub mod diagnostics;
pub mod fish_speech;
pub mod keyboard;
pub mod kokoro_tts;
pub mod melo_tts;
pub mod moonshine_stt;
pub mod neural_audio;
pub mod neural_onnx_tts;
pub mod neural_sidecar_tts;
pub mod oauth_loopback;
pub mod osc;
pub mod pcm_convert;
pub mod rt_thread;
pub mod speech_core;
pub mod translate;
pub mod uberduck_tts;
pub mod uwu;
pub mod voice_changer;
pub mod vosk_stt;
pub mod web;
pub mod whisper;
pub mod windows_tts;

pub struct AppConfiguration {
    pub port: u16,
}
