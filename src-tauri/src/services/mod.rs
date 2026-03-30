pub mod audio;
pub mod http_client;
pub mod chatterbox_tts;
pub mod fish_speech;
pub mod keyboard;
pub mod kokoro_tts;
pub mod melo_tts;
pub mod moonshine_stt;
pub mod oauth_loopback;
pub mod osc;
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
