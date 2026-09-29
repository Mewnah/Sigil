//! Float32 PCM in ~[-1, 1] for DSP / ML from cpal device buffers.

use cpal::traits::DeviceTrait;
use cpal::{BuildStreamError, Device, InputCallbackInfo, SampleFormat, SizedSample, Stream, StreamConfig, StreamError, SupportedStreamConfig};

pub fn samples_to_f32_i8(data: &[i8]) -> Vec<f32> {
    data.iter().map(|&s| s as f32 * (1.0 / 128.0)).collect()
}

pub fn samples_to_f32_i16(data: &[i16]) -> Vec<f32> {
    data.iter().map(|&s| s as f32 * (1.0 / 32768.0)).collect()
}

pub fn samples_to_f32_i32(data: &[i32]) -> Vec<f32> {
    const SCALE: f64 = 1.0 / 2_147_483_648.0;
    data.iter().map(|&s| (s as f64 * SCALE) as f32).collect()
}

pub fn samples_to_f32_i64(data: &[i64]) -> Vec<f32> {
    const SCALE: f64 = 1.0 / 9_223_372_036_854_775_808.0;
    data.iter().map(|&s| (s as f64 * SCALE) as f32).collect()
}

pub fn samples_to_f32_u8(data: &[u8]) -> Vec<f32> {
    data.iter()
        .map(|&s| (s as f32 - 128.0) * (1.0 / 128.0))
        .collect()
}

pub fn samples_to_f32_u16(data: &[u16]) -> Vec<f32> {
    data.iter()
        .map(|&s| (s as f32 - 32_768.0) * (1.0 / 32_768.0))
        .collect()
}

pub fn samples_to_f32_u32(data: &[u32]) -> Vec<f32> {
    const MID: f64 = 2_147_483_648.0;
    const SCALE: f64 = 1.0 / 2_147_483_648.0;
    data.iter()
        .map(|&s| ((s as f64 - MID) * SCALE) as f32)
        .collect()
}

pub fn samples_to_f32_u64(data: &[u64]) -> Vec<f32> {
    const MID: f64 = 9_223_372_036_854_775_808.0;
    const SCALE: f64 = 1.0 / 9_223_372_036_854_775_808.0;
    data.iter()
        .map(|&s| ((s as f64 - MID) * SCALE) as f32)
        .collect()
}

pub fn samples_to_f32_f64(data: &[f64]) -> Vec<f32> {
    data.iter().map(|&s| s as f32).collect()
}

fn samples_to_f32_f32(data: &[f32]) -> Vec<f32> {
    data.to_vec()
}

fn build_converting_input_stream<T, D, E>(
    device: &Device,
    config: &StreamConfig,
    convert: fn(&[T]) -> Vec<f32>,
    mut on_data: D,
    on_error: E,
) -> Result<Stream, BuildStreamError>
where
    T: SizedSample + 'static,
    D: FnMut(Vec<f32>) + Send + 'static,
    E: FnMut(StreamError) + Send + 'static,
{
    device.build_input_stream(config, move |data: &[T], _: &InputCallbackInfo| on_data(convert(data)), on_error, None)
}

/// Open an input stream in the device's native sample format (every format cpal defines) and hand each
/// callback buffer to `on_data` as interleaved f32 PCM in ~[-1, 1].
pub fn build_input_stream_f32<D, E>(device: &Device, config: &SupportedStreamConfig, on_data: D, on_error: E) -> Result<Stream, String>
where
    D: FnMut(Vec<f32>) + Send + 'static,
    E: FnMut(StreamError) + Send + 'static,
{
    let stream_config = config.config();
    let cfg = &stream_config;
    let built = match config.sample_format() {
        SampleFormat::F32 => build_converting_input_stream(device, cfg, samples_to_f32_f32, on_data, on_error),
        SampleFormat::F64 => build_converting_input_stream(device, cfg, samples_to_f32_f64, on_data, on_error),
        SampleFormat::I8 => build_converting_input_stream(device, cfg, samples_to_f32_i8, on_data, on_error),
        SampleFormat::I16 => build_converting_input_stream(device, cfg, samples_to_f32_i16, on_data, on_error),
        SampleFormat::I32 => build_converting_input_stream(device, cfg, samples_to_f32_i32, on_data, on_error),
        SampleFormat::I64 => build_converting_input_stream(device, cfg, samples_to_f32_i64, on_data, on_error),
        SampleFormat::U8 => build_converting_input_stream(device, cfg, samples_to_f32_u8, on_data, on_error),
        SampleFormat::U16 => build_converting_input_stream(device, cfg, samples_to_f32_u16, on_data, on_error),
        SampleFormat::U32 => build_converting_input_stream(device, cfg, samples_to_f32_u32, on_data, on_error),
        SampleFormat::U64 => build_converting_input_stream(device, cfg, samples_to_f32_u64, on_data, on_error),
        // `SampleFormat` is non-exhaustive; formats added by a future cpal need a converter above.
        other => return Err(format!("Unsupported input sample format: {other:?}")),
    };
    built.map_err(|e| format!("Failed to open input stream ({:?}): {e}", config.sample_format()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_formats_map_origin_and_extremes() {
        assert_eq!(samples_to_f32_i8(&[0, i8::MIN]), vec![0.0, -1.0]);
        assert_eq!(samples_to_f32_i16(&[0, i16::MIN]), vec![0.0, -1.0]);
        assert_eq!(samples_to_f32_i32(&[0, i32::MIN]), vec![0.0, -1.0]);
        assert_eq!(samples_to_f32_i64(&[0, i64::MIN]), vec![0.0, -1.0]);
        assert_eq!(samples_to_f32_u8(&[128, 0]), vec![0.0, -1.0]);
        assert_eq!(samples_to_f32_u16(&[32_768, 0]), vec![0.0, -1.0]);
        assert_eq!(samples_to_f32_u32(&[1 << 31, 0]), vec![0.0, -1.0]);
        assert_eq!(samples_to_f32_u64(&[1 << 63, 0]), vec![0.0, -1.0]);
        assert!(samples_to_f32_i16(&[i16::MAX])[0] < 1.0);
        assert!(samples_to_f32_u8(&[u8::MAX])[0] < 1.0);
        assert_eq!(samples_to_f32_f64(&[0.5, -0.25]), vec![0.5, -0.25]);
        assert_eq!(samples_to_f32_f32(&[0.5, -0.25]), vec![0.5, -0.25]);
    }
}
