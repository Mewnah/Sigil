import z from "zod";
import { ElementType } from "./client/elements/schema";
import { Services } from "./services-registry";
import type { RustGpuAdapter } from "./core/services/diagnostics/types";

export type MappedGroupDictionary<Options = any> = Record<string, ([string] | [string, string] | [string, string, Options])[]>

export interface IServiceInterface {
  init(): void;
}

export const enum ServiceNetworkState {
  disconnected = "disconnected",
  connecting = "connecting",
  connected = "connected",
  error = "error"
}

export enum TextEventType {
  final,
  interim,
}

const TextEventTypeSchema = z.nativeEnum(TextEventType);

export type BaseEvent<Data = unknown> = {
  topic: string,
  data?: Data
}

export type PartialWithRequired<T, K extends keyof T> = Pick<T, K> & Partial<T>;

export const TextEventSchema = z.object({
  type: TextEventTypeSchema,
  value: z.string().default(""),
  emotes: z.record(z.number().or(z.string()), z.string()).default({}),
  textFieldType: z.enum(["textField", "twitchChat", "kickChat"]).nullish(),
  id: z.number().optional()
});

export type TextEvent = z.infer<typeof TextEventSchema>;

export enum TextEventSource {
  any = "text",
  textfield = "text.textfield",
  stt = "text.stt",
  translation = "text.translation",
  transform = "text.transform",
  transform_source = "text.transform_source",
  transform_raw = "text.transform_raw",
}

export const TextEventSourceSchema = z.nativeEnum(TextEventSource);

export const ObsCaptionEnvelopeSchema = z.object({
  v: z.literal(1),
  source: TextEventSourceSchema,
  type: z.enum(["final", "interim"]),
  value: z.string(),
  at: z.number(),
});

export type ObsCaptionEnvelope = z.infer<typeof ObsCaptionEnvelopeSchema>;

export type InspectorTabPath = {
  tab: Services | ElementType | "settings" | "integrations" | "scenes" | "files" | "fonts" | "project" | "elements" | "models",
  value?: string
}

/* ------------------------------------------------------------------------------------------------
 * Frozen Rust <-> TS contracts (P0-S0). Change only through an INTEGRATE step, together with the
 * Rust side named in each comment.
 * ---------------------------------------------------------------------------------------------- */

/**
 * VAD segmentation settings sent as `vad` to `plugin:audio|start_utterance_capture`
 * (Rust `speech_core::VadConfig`, snake_case; every field optional, Rust fills defaults).
 */
export interface RustVadConfig {
  /** Speech starts at this frame probability (default 0.5). */
  threshold?: number;
  /** Speech ends after `min_silence_ms` below this (default 0.35). */
  neg_threshold?: number;
  /** Audio kept before the detected start (default 400). */
  pre_roll_ms?: number;
  /** Silence that closes an utterance (default 500). */
  min_silence_ms?: number;
  /** Shorter speech is discarded (default 250), except the voiced remainder of a forced split. */
  min_speech_ms?: number;
  /** Forced close / split length (default 18000). */
  max_utterance_ms?: number;
}

/**
 * Payload of the Tauri event `audio:utterance` (Rust `speech_core::UtteranceEvent`). Metadata only:
 * fetch the PCM of an `utterance` once with
 * `new Float32Array(await invoke<ArrayBuffer>("plugin:audio|take_utterance", { captureId, seq }))`
 * (little-endian f32 mono at `sample_rate`). Ignore events whose `capture_id` is not the id returned
 * by `plugin:audio|start_utterance_capture`.
 */
export type RustUtteranceEvent =
  | { kind: "speech_start"; capture_id: number; seq: number; start_ms: number }
  | {
      kind: "utterance";
      capture_id: number;
      seq: number;
      start_ms: number;
      end_ms: number;
      sample_rate: number;
      samples: number;
      forced: boolean;
    }
  | { kind: "dropped"; capture_id: number; seq: number; reason: string }
  | { kind: "error"; capture_id: number; message: string };

/**
 * Rust `whisper::WhisperDecodeOptions`: the optional `decode` argument of `plugin:whisper|start_recording`
 * (snake_case like `RustVadConfig`). Every field is optional; omitted fields use the Rust defaults.
 */
export interface RustWhisperDecodeOptions {
  /** Decoder threads (default: half the logical CPUs, 1–4). */
  n_threads?: number;
  /** Context prompt, tokenized once per model. */
  initial_prompt?: string;
  /** Words Whisper should favour; passed together with the initial prompt. */
  hotwords?: string[];
  /** Utterances whose no-speech probability is above this are dropped (default 0.6; 1 disables). */
  no_speech_threshold?: number;
  /** Curated hallucination / repeat filter (default true). */
  hallucination_filter?: boolean;
  /** Utterances waiting for the decoder before the oldest is dropped (1..10, default 3). */
  max_backlog?: number;
}

/** Payload of the Tauri event `whisper:speech_start` (the Silero gate confirmed speech). */
export interface RustWhisperSpeechStart {
  seq: number;
  start_ms: number;
}

/** Payload of `whisper:utterance_dropped`: the decoder fell behind and skipped the oldest waiting utterance. */
export interface RustWhisperUtteranceDropped {
  seq: number;
  audio_ms: number;
  reason: "backlog" | (string & {});
}

/** Payload of `whisper:capture_error`: the input device failed and the recording ended. */
export interface RustWhisperCaptureError {
  message: string;
}

/**
 * Execution provider requested vs. actually in use (Rust `diagnostics::EpOutcome`). EP ids are
 * lower-case: "cpu" | "cuda" | "dml" (later "tensorrt", "nvrtx"). `reason` explains a fallback.
 */
export interface RustEpOutcome {
  requested: string;
  active: string;
  reason: string | null;
}

/** Row of `plugin:diagnostics|get_ep_outcomes` / event `diagnostics:ep_outcome`. */
export interface RustEpOutcomeEntry extends RustEpOutcome {
  /** "vc", "vc.content", "vc.pitch", "vc.onnx", "tts.kitten", "stt.vad", ... */
  component: string;
  at_unix_ms: number;
}

/* ------------------------------------------------------------------------------------------------
 * Frozen Rust <-> TS contracts (P1-S0b). Change only through an INTEGRATE step, together with the
 * Rust side named in each comment. Rust structs may gain fields; TS readers must tolerate extras.
 * ---------------------------------------------------------------------------------------------- */

/**
 * Preferred model-store mirror (root key `modelMirror`, Rust `model_store::MirrorPreference`; the runtime
 * list is `MODEL_MIRRORS` in src/core/schema.ts). "auto" keeps catalog order; any other value is tried
 * first, then the rest in catalog order.
 */
export type ModelMirror = "auto" | "huggingface" | "hf-mirror" | "modelscope" | "github-release" | "custom";

/** Rust `model_store::SourceKind` (catalog `sources[].kind`). */
export type RustModelSourceKind =
  | "huggingface"
  | "hf-mirror"
  | "modelscope"
  | "github-release"
  | "nvidia-redist"
  | "direct"
  | "custom";

/** Rust `model_store::CatalogSource`. */
export interface RustCatalogSource {
  kind: RustModelSourceKind;
  url: string;
  revision?: string;
}

/** Rust `model_store::CatalogFile` (see src-tauri/model-catalog/_schema.json). */
export interface RustCatalogFile {
  path: string;
  size: number;
  sha256?: string;
  git_oid?: string;
  sources: RustCatalogSource[];
  archive?: { format: "zip" | "tar.bz2" | "tar.gz"; strip: number; include?: string[] };
}

/** Rust `model_store::CatalogEntry`; row of `plugin:model-store|get_catalog`. */
export interface RustCatalogEntry {
  id: string;
  family: string;
  kind: "model" | "runtime" | "template" | "bench";
  version: string;
  title?: string;
  description?: string;
  /** A fact only; never shown as a warning. */
  license: string;
  status: "published" | "unpublished";
  requires?: string[];
  min_vram_mb?: number;
  variant?: string;
  files: RustCatalogFile[];
}

/** Rust `model_store::EntryState`. */
export type RustModelEntryState =
  | "not_installed"
  | "queued"
  | "downloading"
  | "verifying"
  | "extracting"
  | "installed"
  | "failed"
  | "cancelled";

/**
 * Rust `model_store::EntryStatus`: row of `plugin:model-store|get_entry_states`, result of
 * `remove_entry` / `verify_entry` / `import_local`, and payload of the event `models:state`.
 */
export interface RustModelEntryStatus {
  id: string;
  state: RustModelEntryState;
  installed_version: string | null;
  installed_path: string | null;
  bytes_on_disk: number;
  /** Adopted from whisper/ or vosk-models/ without re-downloading. */
  adopted: boolean;
  error: string | null;
}

/** Payload of the event `models:progress` (Rust `model_store::ProgressEvent`, at most 4 Hz). */
export interface RustModelProgress {
  id: string;
  file: string;
  bytes_done: number;
  /** null when the size is unknown. */
  bytes_total: number | null;
  bytes_per_sec: number;
  mirror: RustModelSourceKind;
  attempt: number;
}

/** Argument `config` of `plugin:model-store|set_store_config` (Rust `model_store::StoreConfig`). */
export interface RustModelStoreConfig {
  mirror: ModelMirror;
  custom_url: string | null;
  /** 1..2 (never two downloads on one host). */
  max_concurrent: number;
}

/** Result of `plugin:model-store|get_disk_usage` (Rust `model_store::DiskUsage`). */
export interface RustModelDiskUsage {
  models_root: string | null;
  total_bytes: number;
  per_family: Record<string, number>;
  free_bytes: number | null;
}

/** Whether an execution provider can run on this PC (Rust `sysinfo::Eligibility`). */
export interface RustEligibility {
  eligible: boolean;
  reason: string | null;
  /**
   * Machine-readable reason (`sysinfo::eps` codes: "ok", "unverified", "no_gpu", "gpu_too_old", ...,
   * "gpu_unknown" when the graphics driver did not answer the probe).
   */
  code?: string;
}

/** Rust `sysinfo::GpuInfo`: a DXGI adapter (`diagnostics::gpu_mem::GpuAdapter`) plus architecture. */
export interface RustGpuInfo extends RustGpuAdapter {
  /** "turing", "ampere", "ada", "blackwell", "rdna3", ... */
  arch: string | null;
  /** NVIDIA compute capability, e.g. "8.9". */
  compute_capability: string | null;
  /** Shared-memory GPU (Intel iGPU, AMD APU, tiny VRAM). */
  integrated?: boolean;
  /** Direct3D 12 at feature level 11_0+ (DirectML); null when not tested. */
  d3d12?: boolean | null;
  /** Highest Direct3D 12 feature level, e.g. "12_2". */
  d3d12_feature_level?: string | null;
  nvml_index?: number | null;
  pci_bus_id?: string | null;
}

/** Result of `plugin:sysinfo|get_system_info` and payload of `sysinfo:changed` (Rust `sysinfo::SystemInfo`). */
export interface RustSystemInfo {
  gpus: RustGpuInfo[];
  primary_gpu: number | null;
  nvidia_driver: string | null;
  cuda_driver_version: string | null;
  eps: { cuda: RustEligibility; tensorrt: RustEligibility; tensorrt_rtx: RustEligibility; directml: RustEligibility };
  cpu: {
    brand: string;
    physical_cores: number;
    logical_cores: number;
    avx2: boolean;
    avx512: boolean;
    vnni: boolean;
    /** P-cores on hybrid CPUs; all cores otherwise. */
    performance_cores?: number;
  };
  total_ram_mb: number;
  probed_at_unix_ms: number;
  nvml_version?: string | null;
  /** Plain-English probe findings (English; for the log). */
  notes?: string[];
  /**
   * Rust `sysinfo::GpuProbe`: GPU facts are current, from an earlier check (the driver did not answer
   * the last one), or missing (the driver never answered). Absent in older payloads (= "complete").
   */
  gpu_probe?: "complete" | "previous" | "missing" | (string & {});
}

/** Rust `resources::Subsystem` (VRAM claim owner). */
export type RustResourceSubsystem = "vc" | "stt_final" | "llm" | "tts" | "other";

/** Rust `resources::VramBudget` (MiB). */
export interface RustVramBudget {
  adapter_index: number | null;
  adapter_name: string | null;
  total_mb: number;
  budget_mb: number;
  process_usage_mb: number;
  claimed_mb: number;
  free_for_claims_mb: number;
  /** Claims held for helper processes (`resources::claim_external_vram`). */
  external_claimed_mb?: number;
  /** NVML device-wide numbers (all apps), when NVML is loaded. */
  device_used_mb?: number | null;
  device_free_mb?: number | null;
}

/** Result of `plugin:resources|get_resource_snapshot` and payload of `resources:changed`. */
export interface RustResourceSnapshot {
  budget: RustVramBudget;
  claims: { id: number; subsystem: RustResourceSubsystem; label: string; mb: number; external?: boolean }[];
  vc_running: boolean;
  /**
   * Whether the running voice changer's models use a GPU EP (its own report, else inferred by Rust
   * from its EP outcomes); null/absent while it is not running or not known yet.
   */
  vc_gpu?: boolean | null;
}

/** Rust `resources::RecommendationTopic` (argument `topic` of `plugin:resources|get_recommendation`). */
export type RustRecommendationTopic = "mt_llm" | "qwen3_tts" | "stt_final";

/** Result of `plugin:resources|get_recommendation` (deterministic). */
export interface RustRecommendation {
  topic: RustRecommendationTopic;
  choice: string;
  reason: string;
  alternatives: string[];
  /**
   * Machine-readable reason (`resources::policy` codes: "fits", "vc_running", "small_gpu", ...,
   * "no_gpu_runtime", "gpu_unknown").
   */
  reason_code?: string;
  device?: "gpu" | "cpu" | (string & {});
  /** Estimated footprint of `choice` (MiB). */
  vram_mb?: number;
  /** Room the policy computed for this topic (MiB). */
  available_mb?: number;
}

/** Row of `plugin:process-host|list_helpers` and payload of `process_host:state` (no API key). */
export interface RustHelperInfo {
  name: string;
  state: "starting" | "running" | "restarting" | "stopped" | "failed";
  pid: number | null;
  port: number | null;
  restarts: number;
  error: string | null;
}

/** Row of `plugin:process-host|get_helper_logs({ name })`: the helper's last output lines (key redacted). */
export interface RustHelperLogLine {
  seq: number;
  at_unix_ms: number;
  stream: "stdout" | "stderr" | "host";
  text: string;
}

/** Rust `native_runtime::RuntimeVariant`. */
export type RustRuntimeVariant = "cuda13" | "vulkan" | "cpu-avx512" | "cpu-avx2" | "cpu";

/**
 * Row of `plugin:runtime|list_runtime_packs` (Rust `native_runtime::RuntimePackStatus`).
 * `plugin:runtime|remove_legacy_runtime({ id })` accepts only `legacy/*` ids (runtime folders left by earlier
 * Sigil versions) and returns the refreshed rows.
 */
export interface RustRuntimePackStatus {
  id: string;
  /** Catalog title, when the pack is a catalog entry. */
  title?: string | null;
  /** A runtime folder from an earlier Sigil version that this version never loads (removable). */
  legacy?: boolean;
  variant: RustRuntimeVariant | null;
  dir: string | null;
  loadable: boolean;
  info: {
    abi_version: number;
    name: string;
    version: string;
    variant: string;
    kind: string;
    build_info: string;
    capabilities: number;
  } | null;
  error: string | null;
}

/**
 * cpal device row of `plugin:audio|list_input_devices` / `list_output_devices` (id == name today).
 * Capture `deviceName`s passed to the audio plugin are capture-hub source ids: `default`, an input name,
 * `loopback:default` or `loopback:<output name>`.
 */
export interface RustAudioDevice {
  id: string;
  name: string;
}

/** Mute groups of the capture hub's gate (Rust `hub::MuteGroup`; argument of `plugin:audio|set_capture_mute`). */
export type RustCaptureMuteGroup = "stt" | "voice_changer" | "others";

/** `plugin:audio|get_capture_status` (Rust `hub::HubStatus`): diagnostics of the capture hub. */
export interface RustCaptureStatus {
  sharing: boolean;
  muted: RustCaptureMuteGroup[];
  sources: {
    source: string;
    epoch: number;
    stream: string;
    consumers: string[];
    recovering: boolean;
    error: string | null;
  }[];
  streams: {
    key: string;
    device: string;
    loopback: boolean;
    state: "idle" | "running" | "failed";
    format: { sample_rate: number; channels: number } | null;
    consumers: number;
    generation: number;
    failures: number;
    frames: number;
    error: string | null;
  }[];
}

/**
 * Capture-hub device events (P1-C, Rust `audio` plugin):
 * - `audio:devices_changed` → {@link RustAudioDevicesChanged} (IMMNotificationClient, polling fallback);
 * - `audio:stream_error` → {@link RustAudioStreamEvent} (a capture stream failed; the watchdog restarts it
 *   under the same epoch);
 * - `audio:stream_recovered` → {@link RustAudioStreamEvent};
 * - `audio:stream_closed` → {@link RustAudioStreamEvent} (message null): a source in an outage lost its
 *   last consumer before it recovered.
 */
export const AUDIO_DEVICE_EVENTS = {
  devicesChanged: "audio:devices_changed",
  streamError: "audio:stream_error",
  streamRecovered: "audio:stream_recovered",
  streamClosed: "audio:stream_closed",
} as const;

/** Payload of `audio:devices_changed`. */
export interface RustAudioDevicesChanged {
  reason: "added" | "removed" | "default_changed" | "state_changed" | "poll" | (string & {});
  inputs: RustAudioDevice[];
  outputs: RustAudioDevice[];
  /** Name of the current default device, when known. */
  default_input: string | null;
  default_output: string | null;
}

/** Payload of `audio:stream_error` / `audio:stream_recovered` / `audio:stream_closed`. */
export interface RustAudioStreamEvent {
  /** Capture source key (device name, or "loopback:<output name>"). */
  source: string;
  /** Hub epoch the stream belongs to (unchanged across a watchdog restart). */
  epoch: number;
  /** Error text (stream_error) or null. */
  message: string | null;
}
