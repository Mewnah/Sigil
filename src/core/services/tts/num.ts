/**
 * Numeric helpers for TTS settings.
 *
 * Persisted TTS numbers are strings (`zStringNumber`), and the inspector writes raw `<input>` values
 * straight into state, so at runtime a value can be `""`, `"abc"` or `"NaN"`. Two old patterns were
 * wrong:
 * - `parseFloat(x) ?? 1` never falls back, because `parseFloat` returns `NaN`, not a nullish value.
 *   `NaN` then serializes to JSON `null`, which Rust rejects for an `f32` field, or reaches Web Speech.
 * - `parseFloat(x) || 1` replaces a valid `0` (VOICEVOX intonation or volume, mute) with the fallback.
 *
 * These helpers return a finite number or the fallback, and never `NaN` or `Infinity`.
 */

/** `value` as a finite number (numbers, or numeric strings with surrounding whitespace), else `fallback`. */
export function finiteOr(value: unknown, fallback: number): number {
  if (typeof value === "number") return Number.isFinite(value) ? value : fallback;
  if (typeof value !== "string") return fallback;
  const trimmed = value.trim();
  // Number("") is 0, so a blank field must be caught before converting.
  if (!trimmed) return fallback;
  const n = Number(trimmed);
  return Number.isFinite(n) ? n : fallback;
}

/** Like {@link finiteOr}, then clamped to `[min, max]`. The fallback is clamped too. */
export function clampedOr(value: unknown, fallback: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, finiteOr(value, fallback)));
}

/** A finite number greater than 0 (speeds, rates), else `fallback`. */
export function positiveOr(value: unknown, fallback: number): number {
  const n = finiteOr(value, fallback);
  return n > 0 ? n : fallback;
}
