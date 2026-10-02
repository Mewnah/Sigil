/**
 * Capture sources of the other players' pipeline (P6-A): TS mirrors of the Rust source plan
 * (`speech::session::SourcePlan`, returned by `plugin:audio|list_loopback_processes { source, fallback }`) and small
 * pure helpers for the service and the inspector.
 */
import {
  PROCESS_SOURCE_PREFIX,
  type RustLoopbackProcess,
  type RustLoopbackProcessList,
  type RustOthersSourcePlan,
} from "@/types";
import { LOOPBACK_DEFAULT_SOURCE, loopbackOutputName } from "@/utils/rustAudioInput";

/**
 * Rust `speech::session::FallbackReason`, plus `capture_failed`, which only the driver sets: the program runs but its
 * capture could not be opened, so the default output is captured until the program is tried again.
 */
export type OthersFallbackReason = "unsupported" | "not_running" | "capture_failed";

/** Rust `speech::session::SourcePlan`: what the other players' pipeline captures now. */
export interface OthersSourcePlan {
  /** The requested source, normalized (`process:VRChat.exe`, `loopback:default`, ...). */
  requested: string;
  /** Capture source to start the pipeline with; null = nothing for now (waiting for the program). */
  source: string | null;
  /** Why `source` is not the requested program. */
  fallback: OthersFallbackReason | null;
  /** Program to watch: plan again while it may start or exit. */
  watch: string | null;
  /** The requested program is running. */
  running: boolean;
}

/** A program row with the P6-A `audio` flag (holds an audio session). */
export type OthersProcess = RustLoopbackProcess & { audio?: boolean };

/** `plugin:audio|list_loopback_processes` with the P6-A additions. */
export interface OthersProcessList extends RustLoopbackProcessList {
  processes: OthersProcess[];
  /** Windows build number (null off Windows / unknown). */
  windowsBuild?: number | null;
  /** The source plan, when `source` was passed (as Rust sends it: read it with {@link parsePlan}). */
  plan?: RustOthersSourcePlan | null;
}

/** IPC command of the program list and the source plan. */
export const LIST_LOOPBACK_PROCESSES = "plugin:audio|list_loopback_processes";

/** Whether `source` names one program (`process:<exe>`). */
export function isProcessSource(source: string | null | undefined): boolean {
  return (source ?? "").trim().toLowerCase().startsWith(PROCESS_SOURCE_PREFIX);
}

/** The program name of a `process:<exe>` source ("" when it is not one). */
export function processExe(source: string | null | undefined): string {
  return isProcessSource(source) ? (source ?? "").trim().slice(PROCESS_SOURCE_PREFIX.length).trim() : "";
}

/** The capture source id of a program. */
export function processSourceId(exe: string): string {
  return `${PROCESS_SOURCE_PREFIX}${exe.trim()}`;
}

/** A plan shape check for what came over IPC (tolerates older builds without a plan). */
export function parsePlan(value: unknown): OthersSourcePlan | null {
  if (!value || typeof value !== "object") return null;
  const v = value as Record<string, unknown>;
  const str = (x: unknown) => (typeof x === "string" && x.length > 0 ? x : null);
  // Only the reasons Rust sends (`capture_failed` is the driver's own).
  const fallback = v.fallback === "unsupported" || v.fallback === "not_running" ? v.fallback : null;
  return {
    requested: typeof v.requested === "string" ? v.requested : "",
    source: str(v.source),
    fallback,
    watch: str(v.watch),
    running: v.running === true,
  };
}

/** Labels the source helpers need (i18n done by the caller). */
export interface SourceLabels {
  program(exe: string): string;
  loopbackDefault: string;
  loopback(name: string): string;
  device(name: string): string;
}

/** Human label of a capture source id. */
export function sourceLabel(source: string, labels: SourceLabels): string {
  if (isProcessSource(source)) return labels.program(processExe(source));
  const output = loopbackOutputName(source);
  if (output === "") return labels.loopbackDefault;
  if (output !== null) return labels.loopback(output);
  return source.trim() === "" ? labels.loopbackDefault : labels.device(source);
}

export { LOOPBACK_DEFAULT_SOURCE };
