use std::io::Cursor;

use rodio::{
    cpal::{self, traits::HostTrait},
    Decoder, DeviceTrait, OutputStream, OutputStreamHandle, Sink,
};
use serde::{Deserialize, Serialize};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime,
};

fn get_output_stream(device_name: &str) -> Result<(OutputStream, OutputStreamHandle), String> {
    if device_name == "default" {
        OutputStream::try_default().map_err(|e| e.to_string())
    } else {
        let host = cpal::default_host();
        let devices = host.output_devices().map_err(|e| e.to_string())?;
        let device = devices
            .into_iter()
            .find(|d| d.name().unwrap_or_default() == device_name)
            .ok_or("Device not found")?;
        OutputStream::try_from_device(&device).map_err(|e| e.to_string())
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RpcAudioPlayAsync {
    pub device_name: String,
    pub data: Vec<u8>,
    pub volume: f32, // 1 - base
    pub rate: f32,   // 1 - base
}

#[command]
pub async fn play_async(data: RpcAudioPlayAsync) -> Result<(), String> {
    let (_stream, stream_handle) = get_output_stream(&data.device_name)?;
    let sink = Sink::try_new(&stream_handle).map_err(|e| e.to_string())?;
    sink.set_volume(data.volume);
    sink.set_speed(data.rate);

    let source = Decoder::new(Cursor::new(data.data)).map_err(|e| e.to_string())?;
    sink.append(source);
    sink.sleep_until_end();
    Ok(())
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("audio")
        .invoke_handler(tauri::generate_handler![play_async])
        .build()
}
