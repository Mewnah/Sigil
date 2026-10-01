//! Float32 PCM in ~[-1, 1] for DSP / ML from cpal device buffers.
//!
//! [`build_input_stream_f32_timed`] opens an input stream in the device's native sample format and
//! hands every callback buffer to `on_data` as an interleaved f32 slice without allocating, with
//! cpal's [`InputCallbackInfo`] (capture and callback timestamps; the capture hub turns them into a
//! QPC capture time per block): f32 devices pass their buffer straight through, other formats are
//! converted into a scratch buffer that is preallocated for [`SCRATCH_SECONDS`] of audio (it grows
//! once if a driver ever delivers more).

use cpal::traits::DeviceTrait;
use cpal::{BuildStreamError, Device, InputCallbackInfo, SampleFormat, SizedSample, Stream, StreamConfig, StreamError, SupportedStreamConfig};

/// Scratch capacity of the converting callbacks, in seconds of interleaved audio.
const SCRATCH_SECONDS: f32 = 0.25;

#[inline]
pub fn i8_to_f32(s: i8) -> f32 {
    s as f32 * (1.0 / 128.0)
}

#[inline]
pub fn i16_to_f32(s: i16) -> f32 {
    s as f32 * (1.0 / 32768.0)
}

#[inline]
pub fn i32_to_f32(s: i32) -> f32 {
    (s as f64 * (1.0 / 2_147_483_648.0)) as f32
}

#[inline]
pub fn i64_to_f32(s: i64) -> f32 {
    (s as f64 * (1.0 / 9_223_372_036_854_775_808.0)) as f32
}

#[inline]
pub fn u8_to_f32(s: u8) -> f32 {
    (s as f32 - 128.0) * (1.0 / 128.0)
}

#[inline]
pub fn u16_to_f32(s: u16) -> f32 {
    (s as f32 - 32_768.0) * (1.0 / 32_768.0)
}

#[inline]
pub fn u32_to_f32(s: u32) -> f32 {
    ((s as f64 - 2_147_483_648.0) * (1.0 / 2_147_483_648.0)) as f32
}

#[inline]
pub fn u64_to_f32(s: u64) -> f32 {
    ((s as f64 - 9_223_372_036_854_775_808.0) * (1.0 / 9_223_372_036_854_775_808.0)) as f32
}

#[inline]
pub fn f64_to_f32(s: f64) -> f32 {
    s as f32
}

fn build_converting_input_stream<T, D, E>(
    device: &Device,
    config: &StreamConfig,
    convert: fn(T) -> f32,
    mut on_data: D,
    on_error: E,
) -> Result<Stream, BuildStreamError>
where
    T: SizedSample + 'static,
    D: FnMut(&[f32], &InputCallbackInfo) + Send + 'static,
    E: FnMut(StreamError) + Send + 'static,
{
    let frames = (config.sample_rate.0 as f32 * SCRATCH_SECONDS) as usize;
    let mut scratch: Vec<f32> = Vec::with_capacity(frames.max(1024) * usize::from(config.channels.max(1)));
    device.build_input_stream(
        config,
        move |data: &[T], info: &InputCallbackInfo| {
            scratch.clear();
            scratch.extend(data.iter().map(|&s| convert(s)));
            on_data(&scratch, info);
        },
        on_error,
        None,
    )
}

/// Open an input stream in the device's native sample format (every format cpal defines) and hand
/// each callback buffer to `on_data` as interleaved f32 PCM in ~[-1, 1], without allocating, with
/// cpal's callback info (capture / callback timestamps). `config` may be an output device's config
/// when `device` is a render endpoint opened for WASAPI loopback. (The untimed variant went away
/// in P3, when the voice changer moved onto the capture hub.)
pub fn build_input_stream_f32_timed<D, E>(device: &Device, config: &SupportedStreamConfig, mut on_data: D, on_error: E) -> Result<Stream, String>
where
    D: FnMut(&[f32], &InputCallbackInfo) + Send + 'static,
    E: FnMut(StreamError) + Send + 'static,
{
    let stream_config = config.config();
    let cfg = &stream_config;
    let built = match config.sample_format() {
        SampleFormat::F32 => device.build_input_stream(cfg, move |data: &[f32], info: &InputCallbackInfo| on_data(data, info), on_error, None),
        SampleFormat::F64 => build_converting_input_stream(device, cfg, f64_to_f32, on_data, on_error),
        SampleFormat::I8 => build_converting_input_stream(device, cfg, i8_to_f32, on_data, on_error),
        SampleFormat::I16 => build_converting_input_stream(device, cfg, i16_to_f32, on_data, on_error),
        SampleFormat::I32 => build_converting_input_stream(device, cfg, i32_to_f32, on_data, on_error),
        SampleFormat::I64 => build_converting_input_stream(device, cfg, i64_to_f32, on_data, on_error),
        SampleFormat::U8 => build_converting_input_stream(device, cfg, u8_to_f32, on_data, on_error),
        SampleFormat::U16 => build_converting_input_stream(device, cfg, u16_to_f32, on_data, on_error),
        SampleFormat::U32 => build_converting_input_stream(device, cfg, u32_to_f32, on_data, on_error),
        SampleFormat::U64 => build_converting_input_stream(device, cfg, u64_to_f32, on_data, on_error),
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
        assert_eq!((i8_to_f32(0), i8_to_f32(i8::MIN)), (0.0, -1.0));
        assert_eq!((i16_to_f32(0), i16_to_f32(i16::MIN)), (0.0, -1.0));
        assert_eq!((i32_to_f32(0), i32_to_f32(i32::MIN)), (0.0, -1.0));
        assert_eq!((i64_to_f32(0), i64_to_f32(i64::MIN)), (0.0, -1.0));
        assert_eq!((u8_to_f32(128), u8_to_f32(0)), (0.0, -1.0));
        assert_eq!((u16_to_f32(32_768), u16_to_f32(0)), (0.0, -1.0));
        assert_eq!((u32_to_f32(1 << 31), u32_to_f32(0)), (0.0, -1.0));
        assert_eq!((u64_to_f32(1 << 63), u64_to_f32(0)), (0.0, -1.0));
        assert!(i16_to_f32(i16::MAX) < 1.0);
        assert!(u8_to_f32(u8::MAX) < 1.0);
        assert_eq!((f64_to_f32(0.5), f64_to_f32(-0.25)), (0.5, -0.25));
    }
}
