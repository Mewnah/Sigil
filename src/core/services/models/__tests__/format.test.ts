import { describe, expect, it } from "vitest";
import type { RustCatalogEntry, RustModelEntryStatus, RustModelProgress } from "@/types";
import {
  displayFileName,
  etaSeconds,
  formatByteProgress,
  formatDownloadProgress,
  formatEta,
  formatSpeed,
  normalizeDownloadProgress,
  parseDownloadProgress,
  percentOf,
} from "@/core/services/stt/downloadProgress";
import {
  activeEntries,
  buildStoreConfig,
  canDownloadEntry,
  checkCustomMirrorUrl,
  deleteScope,
  diskByFamily,
  emptyStatus,
  entryActions,
  entryProgressView,
  entrySize,
  entryTitle,
  familyFallbackLabel,
  groupCatalog,
  hasInstall,
  hasUpdate,
  importBlockers,
  isBusyState,
  isLocalHost,
  isPinned,
  isSatisfied,
  lowOnDisk,
  matchesQuery,
  mergeProgress,
  mirrorKey,
  needsInstall,
  planRequirement,
  readyTransition,
  requirementClosure,
  requirementFraction,
  summarizeRequirement,
  trackRequirementIds,
  versionLabel,
  visibleOpErrors,
  type EntryProgress,
} from "../format";

const HASH = "a".repeat(64);

function entry(id: string, over: Partial<RustCatalogEntry> = {}): RustCatalogEntry {
  const family = id.split("/")[0];
  return {
    id,
    family,
    kind: "model",
    version: "1",
    license: "MIT",
    status: "published",
    files: [{ path: `${id.split("/")[1] ?? id}.bin`, size: 100, sha256: HASH, sources: [] }],
    ...over,
  };
}

function status(id: string, state: RustModelEntryStatus["state"], over: Partial<RustModelEntryStatus> = {}): RustModelEntryStatus {
  return { ...emptyStatus(id), state, ...over };
}

function progress(id: string, file: string, done: number, total: number | null, over: Partial<RustModelProgress> = {}): RustModelProgress {
  return { id, file, bytes_done: done, bytes_total: total, bytes_per_sec: 1000, mirror: "huggingface", attempt: 1, ...over };
}

describe("download progress (generalized formatter)", () => {
  it("keeps the legacy STT payload behaviour", () => {
    expect(parseDownloadProgress({ file: "ggml-small.bin", progress: 42.4 })).toEqual({ file: "ggml-small.bin", percent: 42 });
    expect(parseDownloadProgress({ file: "vosk small", progress: -1 })).toEqual({ file: "vosk small", percent: null });
    expect(parseDownloadProgress({ file: " ", progress: Number.NaN })).toEqual({ file: "model", percent: null });
    expect(parseDownloadProgress({ file: "x", progress: 250 })).toEqual({ file: "x", percent: 100 });
    expect(parseDownloadProgress(null)).toEqual({ file: "model", percent: null });
    expect(formatDownloadProgress({ file: "ggml-base.bin", progress: 7 })).toBe("Downloading ggml-base.bin: 7%...");
    expect(formatDownloadProgress({ file: "ggml-base.bin", progress: -1 })).toBe("Downloading ggml-base.bin...");
    const legacy = normalizeDownloadProgress({ file: "models/x.bin", progress: 50 });
    expect(legacy).toMatchObject({ file: "x.bin", percent: 50, fraction: 0.5, bytesDone: null, etaSeconds: null });
  });

  it("normalizes model-store byte progress with speed, time left, mirror and attempt", () => {
    const v = normalizeDownloadProgress(progress("stt/x", "onnx/model.onnx", 50e6, 150e6, { bytes_per_sec: 5e6, mirror: "hf-mirror", attempt: 2 }));
    expect(v).toEqual({
      file: "model.onnx",
      percent: 33,
      fraction: 50 / 150,
      bytesDone: 50e6,
      bytesTotal: 150e6,
      bytesPerSec: 5e6,
      etaSeconds: 20,
      mirror: "hf-mirror",
      attempt: 2,
    });
    expect(parseDownloadProgress(progress("stt/x", "a/b.bin", 1, 4))).toEqual({ file: "b.bin", percent: 25 });
    expect(formatDownloadProgress(progress("stt/x", "model.onnx", 50e6, 150e6, { bytes_per_sec: 5e6, attempt: 2 }))).toBe(
      "Downloading model.onnx: 33% (50 MB / 150 MB, 5.0 MB/s, 0:20 left, via huggingface, attempt 2)...",
    );
  });

  it("handles unknown sizes, zero speed and junk values", () => {
    const unknown = normalizeDownloadProgress(progress("x", "f.bin", 3_500_000, null, { bytes_per_sec: 0 }));
    expect(unknown.percent).toBeNull();
    expect(unknown.fraction).toBeNull();
    expect(unknown.etaSeconds).toBeNull();
    expect(unknown.bytesPerSec).toBeNull();
    expect(formatByteProgress(unknown)).toBe("3.5 MB");
    expect(formatDownloadProgress(progress("x", "f.bin", 3_500_000, null, { bytes_per_sec: 0, mirror: undefined as never }))).toBe(
      "Downloading f.bin (3.5 MB)...",
    );
    expect(formatDownloadProgress(progress("x", "f.bin", 3_500_000, null, { bytes_per_sec: 0 }))).toBe(
      "Downloading f.bin (3.5 MB, via huggingface)...",
    );
    const junk = normalizeDownloadProgress({ file: "f", bytes_done: Number.NaN, bytes_total: -5, bytes_per_sec: Infinity, attempt: 0 });
    expect(junk).toMatchObject({ bytesDone: null, bytesTotal: null, bytesPerSec: null, attempt: null, percent: null });
    expect(formatByteProgress(junk)).toBeNull();
  });

  it("never shows 100% before the last byte", () => {
    expect(percentOf(999, 1000)).toBe(99);
    expect(percentOf(1000, 1000)).toBe(100);
    expect(percentOf(2000, 1000)).toBe(100);
    expect(percentOf(0, 1000)).toBe(0);
    expect(percentOf(5, 0)).toBeNull();
  });

  it("formats speed, time left and file names", () => {
    expect(formatSpeed(3.2e6)).toBe("3.2 MB/s");
    expect(formatSpeed(0)).toBeNull();
    expect(formatSpeed(null)).toBeNull();
    expect(formatEta(42)).toBe("0:42");
    expect(formatEta(725)).toBe("12:05");
    expect(formatEta(3723)).toBe("1:02:03");
    expect(formatEta(-1)).toBeNull();
    expect(etaSeconds(100, 10)).toBe(10);
    expect(etaSeconds(0, 0)).toBe(0);
    expect(etaSeconds(100, 0)).toBeNull();
    expect(etaSeconds(1e12, 1)).toBeNull(); // more than a day: not worth showing
    expect(displayFileName("a/b/model.onnx")).toBe("model.onnx");
    expect(displayFileName("a\\b.bin")).toBe("b.bin");
    expect(displayFileName("dir/")).toBe("dir");
    expect(displayFileName("")).toBe("model");
  });
});

describe("catalog entries", () => {
  it("sums sizes and flags unknown ones", () => {
    expect(entrySize(entry("stt/a"))).toEqual({ bytes: 100, exact: true });
    const multi = entry("stt/b", {
      files: [
        { path: "a", size: 10, sha256: HASH, sources: [] },
        { path: "b", size: 0, sources: [] },
      ],
    });
    expect(entrySize(multi)).toEqual({ bytes: 10, exact: false });
    expect(isPinned(multi)).toBe(false);
    expect(isPinned(entry("stt/a"))).toBe(true);
    expect(entryTitle({ id: "stt/a", title: "  " })).toBe("stt/a");
    expect(entryTitle({ id: "stt/a", title: "Silero" })).toBe("Silero");
  });

  it("labels release versions with a v and leaves revisions alone", () => {
    expect(versionLabel("1.13.8")).toBe("v1.13.8");
    expect(versionLabel("0.15")).toBe("v0.15");
    expect(versionLabel("2.0.0-rc.13")).toBe("v2.0.0-rc.13");
    expect(versionLabel("5359861c")).toBe("5359861c");
    expect(versionLabel("2024-05-01")).toBe("2024-05-01");
    expect(versionLabel(" ")).toBeNull();
    expect(versionLabel(null)).toBeNull();
  });

  it("detects updates only for installed entries with another recorded version", () => {
    const e = entry("stt/a", { version: "2" });
    expect(hasUpdate(e, status("stt/a", "installed", { installed_version: "1" }))).toBe(true);
    expect(hasUpdate(e, status("stt/a", "installed", { installed_version: "2" }))).toBe(false);
    expect(hasUpdate(e, status("stt/a", "installed"))).toBe(false);
    expect(hasUpdate(e, status("stt/a", "failed", { installed_version: "1" }))).toBe(false);
    expect(hasUpdate(e, undefined)).toBe(false);
  });

  it("offers the actions each state allows", () => {
    const e = entry("stt/a", { version: "2" });
    expect(entryActions(e, undefined)).toEqual({ install: "install", cancel: false, verify: false, remove: false, import: true });
    expect(entryActions(e, status("stt/a", "installed", { installed_version: "2" }))).toEqual({
      install: null,
      cancel: false,
      verify: true,
      remove: true,
      import: false,
    });
    expect(entryActions(e, status("stt/a", "installed", { installed_version: "1" })).install).toBe("update");
    for (const s of ["queued", "downloading", "extracting", "verifying"] as const) {
      expect(entryActions(e, status("stt/a", s))).toEqual({ install: null, cancel: true, verify: false, remove: false, import: false });
    }
    expect(entryActions(e, status("stt/a", "failed"))).toEqual({ install: "retry", cancel: false, verify: false, remove: true, import: true });
    expect(entryActions(e, status("stt/a", "cancelled"))).toEqual({ install: "resume", cancel: false, verify: false, remove: true, import: true });
  });

  it("lets a newer build be imported over an installed entry", () => {
    // A new catalog version: Update (download) and Import both replace the working install.
    const e = entry("stt/a", { version: "2" });
    expect(entryActions(e, status("stt/a", "installed", { installed_version: "1" }))).toMatchObject({ install: "update", import: true });
    // Import-only entries (local builds) can always be re-imported; the tab hides their Update.
    const local = entry("runtime/pack", { version: "2", status: "unpublished" });
    expect(entryActions(local, status("runtime/pack", "installed", { installed_version: "1" }))).toMatchObject({ install: "update", import: true });
    expect(entryActions(local, status("runtime/pack", "installed", { installed_version: "2" }))).toMatchObject({ install: null, import: true });
  });

  it("knows when a working install exists behind a transient state", () => {
    const failedUpdate = status("whisper/base", "failed", { installed_version: "1", bytes_on_disk: 150e6, error: "network" });
    expect(hasInstall(failedUpdate)).toBe(true);
    expect(hasInstall(status("whisper/base", "cancelled", { installed_version: "1" }))).toBe(true);
    expect(hasInstall(status("whisper/base", "installed"))).toBe(true);
    expect(hasInstall(status("whisper/base", "failed"))).toBe(false);
    expect(hasInstall(undefined)).toBe(false);

    // The delete confirmation must not call a working v1 "what was downloaded".
    expect(deleteScope(status("whisper/base", "installed", { installed_version: "1" }))).toBe("installed");
    expect(deleteScope(failedUpdate)).toBe("installed_and_partial");
    expect(deleteScope(status("whisper/base", "cancelled", { installed_version: "1" }))).toBe("installed_and_partial");
    expect(deleteScope(status("whisper/base", "cancelled"))).toBe("partial");
    expect(deleteScope(status("whisper/base", "failed"))).toBe("partial");
    expect(deleteScope(undefined)).toBe("partial");
  });

  it("treats only installed entries, or a user check of one, as satisfied", () => {
    const checking = status("stt/a", "verifying", { installed_version: "1" });
    expect(isSatisfied(checking, "verify")).toBe(true);
    expect(isSatisfied(checking, "import")).toBe(false);
    expect(isSatisfied(checking)).toBe(false);
    expect(isSatisfied(status("stt/a", "verifying"), "verify")).toBe(false);
    expect(isSatisfied(status("stt/a", "installed"))).toBe(true);
    expect(isSatisfied(undefined)).toBe(false);
    const e = entry("stt/a", { version: "2" });
    expect(needsInstall(e, status("stt/a", "installed", { installed_version: "2" }))).toBe(false);
    expect(needsInstall(e, status("stt/a", "installed", { installed_version: "1" }))).toBe(true);
    expect(needsInstall(e, undefined)).toBe(true);
  });

  it("downloads unpublished entries only from a custom mirror and only when pinned", () => {
    expect(canDownloadEntry(entry("stt/a"), false)).toBe(true);
    const local = entry("runtime/pack", { status: "unpublished" });
    expect(canDownloadEntry(local, false)).toBe(false);
    expect(canDownloadEntry(local, true)).toBe(true);
    const unpinned = entry("runtime/pack", { status: "unpublished", files: [{ path: "pack.zip", size: 0, sources: [] }] });
    expect(canDownloadEntry(unpinned, true)).toBe(false);
  });

  it("disables everything while an operation is pending, except cancelling an install", () => {
    const e = entry("stt/a");
    expect(entryActions(e, status("stt/a", "installed"), "verify")).toEqual({
      install: null,
      cancel: false,
      verify: false,
      remove: false,
      import: false,
    });
    expect(entryActions(e, status("stt/a", "queued"), "install").cancel).toBe(true);
    expect(entryActions(e, status("stt/a", "not_installed"), "install").cancel).toBe(false);
    expect(entryActions(e, status("stt/a", "verifying"), "import").cancel).toBe(false);
  });

  it("knows the busy states", () => {
    expect(["queued", "downloading", "verifying", "extracting"].every((s) => isBusyState(s as never))).toBe(true);
    expect(["installed", "failed", "cancelled", "not_installed"].some((s) => isBusyState(s as never))).toBe(false);
    expect(isBusyState(undefined)).toBe(false);
  });
});

describe("search and grouping", () => {
  const catalog: RustCatalogEntry[] = [
    entry("base/test", { kind: "bench", title: "Self-test" }),
    entry("whisper/tiny", { title: "Whisper tiny" }),
    entry("vosk/en", { title: "Vosk English", description: "Small English model" }),
    entry("whisper/base", { title: "Whisper base" }),
    entry("runtime/sherpa", { kind: "runtime", title: "sherpa-onnx", variant: "cpu" }),
    entry("future/x", { kind: "dataset" as never, title: "Future kind" }),
  ];
  const states = {
    "whisper/tiny": status("whisper/tiny", "installed", { bytes_on_disk: 77 }),
    "whisper/base": status("whisper/base", "downloading"),
    "vosk/en": status("vosk/en", "failed"),
  };

  it("matches every search term anywhere in the entry", () => {
    expect(matchesQuery(catalog[2], "vosk english")).toBe(true);
    expect(matchesQuery(catalog[2], "small  ENGLISH")).toBe(true);
    expect(matchesQuery(catalog[2], "vosk german")).toBe(false);
    expect(matchesQuery(catalog[4], "cpu sherpa")).toBe(true);
    expect(matchesQuery(catalog[4], "")).toBe(true);
  });

  it("groups by kind (display order) and family (catalog order) with counts", () => {
    const groups = groupCatalog(catalog, states);
    expect(groups.map((g) => g.kind)).toEqual(["model", "runtime", "bench", "dataset"]);
    const models = groups[0];
    expect(models.families.map((f) => f.family)).toEqual(["whisper", "vosk"]);
    expect(models.families[0].entries.map((e) => e.id)).toEqual(["whisper/tiny", "whisper/base"]);
    expect(models.families[0]).toMatchObject({ installed: 1, busy: 1, bytesOnDisk: 77 });
    expect(models).toMatchObject({ count: 3, installed: 1 });
  });

  it("filters by query and by 'installed only' (which keeps entries needing attention)", () => {
    expect(groupCatalog(catalog, states, { query: "whisper base" }).flatMap((g) => g.families.flatMap((f) => f.entries.map((e) => e.id)))).toEqual([
      "whisper/base",
    ]);
    const installed = groupCatalog(catalog, states, { installedOnly: true });
    expect(installed.flatMap((g) => g.families.flatMap((f) => f.entries.map((e) => e.id)))).toEqual([
      "whisper/tiny",
      "whisper/base",
      "vosk/en",
    ]);
    expect(groupCatalog(catalog, states, { query: "nothing matches this" })).toEqual([]);
  });

  it("lists the entries in flight", () => {
    expect(activeEntries(catalog, states).map((e) => e.id)).toEqual(["whisper/base"]);
  });

  it("counts an install whose update failed or is running as installed", () => {
    const updating = {
      ...states,
      "whisper/base": status("whisper/base", "downloading", { installed_version: "0", bytes_on_disk: 40 }),
      "vosk/en": status("vosk/en", "failed", { installed_version: "0", bytes_on_disk: 60 }),
    };
    const models = groupCatalog(catalog, updating)[0];
    expect(models.families[0]).toMatchObject({ installed: 2, busy: 1, bytesOnDisk: 117 });
    expect(models.families[1]).toMatchObject({ installed: 1, busy: 0, bytesOnDisk: 60 });
    expect(models.installed).toBe(3);
  });

  it("labels families and mirrors", () => {
    expect(familyFallbackLabel("whisper")).toBe("Whisper");
    expect(familyFallbackLabel("stt-legacy")).toBe("Stt legacy");
    expect(familyFallbackLabel("")).toBe("");
    expect(mirrorKey("hf-mirror")).toBe("hf_mirror");
    expect(mirrorKey("github-release")).toBe("github_release");
  });
});

describe("progress across files", () => {
  const e = entry("stt/multi", {
    files: [
      { path: "a.onnx", size: 100, sha256: HASH, sources: [] },
      { path: "b.onnx", size: 300, sha256: HASH, sources: [] },
    ],
  });

  it("adds finished files to the current one", () => {
    let p = mergeProgress(undefined, progress("stt/multi", "a.onnx", 100, 100));
    p = mergeProgress(p, progress("stt/multi", "b.onnx", 150, 300, { bytes_per_sec: 50 }));
    const v = entryProgressView(e, p);
    expect(v).toMatchObject({ bytesDone: 250, bytesTotal: 400, percent: 62, fileIndex: 2, fileCount: 2, bytesPerSec: 50, etaSeconds: 3 });
    expect(v.fraction).toBeCloseTo(0.625);
  });

  it("counts earlier files fully after a resume (no progress event for complete files)", () => {
    // a.onnx finished before the cancel; after Resume only b.onnx reports progress.
    const p = mergeProgress(undefined, progress("stt/multi", "b.onnx", 150, 300, { bytes_per_sec: 50 }));
    const v = entryProgressView(e, p);
    expect(v).toMatchObject({ bytesDone: 250, bytesTotal: 400, percent: 62, fileIndex: 2, etaSeconds: 3 });
    // The first file: nothing before it.
    expect(entryProgressView(e, mergeProgress(undefined, progress("stt/multi", "a.onnx", 40, 100))).bytesDone).toBe(40);
    // A file path the catalog does not list: no file is assumed complete, only listed files' bytes count.
    expect(entryProgressView(e, mergeProgress(undefined, progress("stt/multi", "other.bin", 7, 10))).bytesDone).toBe(0);
  });

  it("follows a file restarting on another mirror", () => {
    let p = mergeProgress(undefined, progress("stt/multi", "a.onnx", 80, 100));
    p = mergeProgress(p, progress("stt/multi", "a.onnx", 10, 100, { mirror: "modelscope", attempt: 2 }));
    expect(entryProgressView(e, p).bytesDone).toBe(10);
    expect(p).toMatchObject({ mirror: "modelscope", attempt: 2 });
  });

  it("caps a file at its catalog size and falls back to the event total for unknown sizes", () => {
    const p = mergeProgress(undefined, progress("stt/multi", "a.onnx", 999, 100));
    expect(entryProgressView(e, p).bytesDone).toBe(100);
    const unknown = entry("stt/u", { files: [{ path: "u.bin", size: 0, sources: [] }] });
    const v = entryProgressView(unknown, mergeProgress(undefined, progress("stt/u", "u.bin", 25, 100)));
    expect(v).toMatchObject({ bytesDone: 25, bytesTotal: 100, percent: 25 });
    const none = entryProgressView(unknown, mergeProgress(undefined, progress("stt/u", "u.bin", 25, null)));
    expect(none).toMatchObject({ bytesTotal: null, fraction: null, percent: null, etaSeconds: null });
  });

  it("sanitizes junk event values", () => {
    const p = mergeProgress(undefined, { ...progress("x", "f", -5, Number.NaN), bytes_per_sec: Number.NaN, attempt: 0 });
    expect(p).toMatchObject({ bytesDone: 0, bytesTotal: null, bytesPerSec: 0, attempt: 1 });
  });
});

describe("requirements", () => {
  const catalog: RustCatalogEntry[] = [
    entry("runtime/sherpa", { kind: "runtime", status: "unpublished" }),
    entry("stt/encoder", { requires: ["runtime/sherpa"] }),
    entry("stt/model", { requires: ["stt/encoder", "runtime/sherpa"] }),
    entry("loop/a", { requires: ["loop/b"] }),
    entry("loop/b", { requires: ["loop/a"] }),
  ];

  it("resolves requirements first, transitively, once each", () => {
    const { order, unknown } = requirementClosure(catalog, ["stt/model", "stt/encoder", " stt/model "]);
    expect(order.map((e) => e.id)).toEqual(["runtime/sherpa", "stt/encoder", "stt/model"]);
    expect(unknown).toEqual([]);
    expect(requirementClosure(catalog, ["nope", "", "nope"]).unknown).toEqual(["nope"]);
    expect(requirementClosure(catalog, ["loop/a"]).order.map((e) => e.id)).toEqual(["loop/b", "loop/a"]);
  });

  it("summarizes what is missing, busy, failed and import-only", () => {
    const states = {
      "runtime/sherpa": status("runtime/sherpa", "installed"),
      "stt/encoder": status("stt/encoder", "downloading"),
      "stt/model": status("stt/model", "failed", { error: "hash mismatch" }),
    };
    const s = summarizeRequirement(catalog, states, ["stt/model"]);
    expect(s.entries.map((e) => e.id)).toEqual(["runtime/sherpa", "stt/encoder", "stt/model"]);
    expect(s.missing.map((e) => e.id)).toEqual(["stt/encoder", "stt/model"]);
    expect(s.busy.map((e) => e.id)).toEqual(["stt/encoder"]);
    expect(s.failed).toEqual([{ entry: catalog[2], error: "hash mismatch" }]);
    expect(s.unpublished).toEqual([]);
    expect(s).toMatchObject({ downloadBytes: 200, downloadExact: true, ready: false });

    const fresh = summarizeRequirement(catalog, {}, ["stt/encoder"]);
    expect(fresh.unpublished.map((e) => e.id)).toEqual(["runtime/sherpa"]);

    const done = summarizeRequirement(
      catalog,
      { "runtime/sherpa": status("runtime/sherpa", "installed"), "stt/encoder": status("stt/encoder", "installed") },
      ["stt/encoder"],
    );
    expect(done).toMatchObject({ ready: true, missing: [], downloadBytes: 0 });
    expect(summarizeRequirement(catalog, {}, []).ready).toBe(true);
    expect(summarizeRequirement(catalog, {}, ["nope"]).ready).toBe(false);
  });

  it("keeps finished entries in the overall progress so the bar never moves backwards", () => {
    const cat: RustCatalogEntry[] = [entry("runtime/x", { kind: "runtime" }), entry("stt/model", { requires: ["runtime/x"] })];
    const ids = ["stt/model"];
    const frac = (tracked: string[], states: Record<string, RustModelEntryStatus>, prog: Record<string, EntryProgress>) =>
      requirementFraction(
        cat.filter((e) => tracked.includes(e.id)),
        states,
        prog,
      );

    // x downloads first (half done), model waits.
    let states: Record<string, RustModelEntryStatus> = {
      "runtime/x": status("runtime/x", "downloading"),
      "stt/model": status("stt/model", "queued"),
    };
    let s = summarizeRequirement(cat, states, ids);
    let tracked = trackRequirementIds([], s, s.busy.length > 0);
    expect(tracked).toEqual(["runtime/x", "stt/model"]);
    const halfX = { "runtime/x": mergeProgress(undefined, progress("runtime/x", "x.bin", 50, 100)) };
    expect(frac(tracked, states, halfX)).toBeCloseTo(0.25);

    // x finishes: it leaves `missing`, but stays tracked and counts fully (0.25 -> 0.5, not 0).
    states = { "runtime/x": status("runtime/x", "installed"), "stt/model": status("stt/model", "downloading") };
    s = summarizeRequirement(cat, states, ids);
    expect(s.missing.map((e) => e.id)).toEqual(["stt/model"]);
    tracked = trackRequirementIds(tracked, s, s.busy.length > 0);
    expect(tracked).toEqual(["runtime/x", "stt/model"]);
    expect(frac(tracked, states, {})).toBeCloseTo(0.5);
    // Only `missing` (the old input) would have dropped the bar back to 0.
    expect(requirementFraction(s.missing, states, {})).toBe(0);
    const halfModel = { "stt/model": mergeProgress(undefined, progress("stt/model", "model.bin", 50, 100)) };
    expect(frac(tracked, states, halfModel)).toBeCloseTo(0.75);

    // Nothing in flight: tracking resets; ids that left the requirement are dropped.
    expect(trackRequirementIds(tracked, s, false)).toEqual([]);
    expect(trackRequirementIds(["gone/id", "runtime/x"], s, true)).toEqual(["runtime/x", "stt/model"]);

    expect(requirementFraction([cat[0], cat[1]], {}, {})).toBe(0);
    const unknownSize = entry("x/y", { files: [{ path: "y", size: 0, sources: [] }] });
    expect(requirementFraction([unknownSize], {}, {})).toBeNull();
  });

  it("keeps an installed entry satisfied while the user checks its files", () => {
    const states = {
      "runtime/sherpa": status("runtime/sherpa", "installed", { installed_version: "1" }),
      "stt/encoder": status("stt/encoder", "verifying", { installed_version: "1" }),
    };
    // Check files (pending verify): still ready, nothing busy.
    const checking = summarizeRequirement(catalog, states, ["stt/encoder"], { "stt/encoder": "verify" });
    expect(checking).toMatchObject({ ready: true, missing: [], busy: [] });
    // `verifying` during an install or an import is not a user check: the entry is missing and busy.
    const importing = summarizeRequirement(catalog, states, ["stt/encoder"], { "stt/encoder": "import" });
    expect(importing.ready).toBe(false);
    expect(importing.busy.map((e) => e.id)).toEqual(["stt/encoder"]);
    expect(summarizeRequirement(catalog, states, ["stt/encoder"]).ready).toBe(false);
    // A check that fails unregisters the entry: missing again, with the reason.
    const broken = summarizeRequirement(
      catalog,
      { ...states, "stt/encoder": status("stt/encoder", "failed", { error: "encoder.bin is corrupt" }) },
      ["stt/encoder"],
      { "stt/encoder": "verify" },
    );
    expect(broken.failed).toEqual([{ entry: catalog[1], error: "encoder.bin is corrupt" }]);
  });

  it("offers Download only for entries whose whole closure can be downloaded", () => {
    // stt/encoder (published) requires runtime/sherpa (unpublished): Download would fail both.
    const s = summarizeRequirement(catalog, {}, ["stt/encoder"]);
    const plan = planRequirement(catalog, {}, s, false);
    expect(plan.download).toEqual([]);
    expect(plan.importOnly.map((e) => e.id)).toEqual(["runtime/sherpa"]);
    expect(plan.blocked.map((e) => e.id)).toEqual(["stt/encoder"]);
    expect(importBlockers(catalog, {}, catalog[1], false).map((e) => e.id)).toEqual(["runtime/sherpa"]);

    // Once the runtime is imported, the rest downloads.
    const imported = { "runtime/sherpa": status("runtime/sherpa", "installed", { installed_version: "1" }) };
    const after = planRequirement(catalog, imported, summarizeRequirement(catalog, imported, ["stt/model"]), false);
    expect(after.download.map((e) => e.id)).toEqual(["stt/encoder", "stt/model"]);
    expect(after).toMatchObject({ importOnly: [], blocked: [] });

    // An installed import-only requirement at an older version would be re-queued (and fail) too.
    const outdated = { "runtime/sherpa": status("runtime/sherpa", "installed", { installed_version: "0" }) };
    expect(importBlockers(catalog, outdated, catalog[1], false).map((e) => e.id)).toEqual(["runtime/sherpa"]);
    const outdatedPlan = planRequirement(catalog, outdated, summarizeRequirement(catalog, outdated, ["stt/encoder"]), false);
    expect(outdatedPlan).toMatchObject({ download: [], blocked: [catalog[1]], importOnly: [catalog[0]] });

    // Several import-only entries: every one of them is listed (one Import button each).
    const cat2: RustCatalogEntry[] = [
      entry("runtime/vc", { kind: "runtime", status: "unpublished" }),
      entry("vc/template", { kind: "template", status: "unpublished", requires: ["runtime/vc"] }),
    ];
    const both = planRequirement(cat2, {}, summarizeRequirement(cat2, {}, ["vc/template"]), false);
    expect(both).toMatchObject({ download: [], blocked: [] });
    expect(both.importOnly.map((e) => e.id)).toEqual(["runtime/vc", "vc/template"]);

    // A working custom mirror serves pinned unpublished files.
    const viaMirror = planRequirement(cat2, {}, summarizeRequirement(cat2, {}, ["vc/template"]), true);
    expect(viaMirror.download.map((e) => e.id)).toEqual(["runtime/vc", "vc/template"]);
  });

  it("fires onReady once per ready transition and again for another ready id set", () => {
    let r = readyTransition(null, true, "whisper/base");
    expect(r).toEqual({ fire: true, firedFor: "whisper/base" });
    r = readyTransition(r.firedFor, true, "whisper/base");
    expect(r.fire).toBe(false);
    // Switching to another model that is also installed.
    r = readyTransition(r.firedFor, true, "whisper/small");
    expect(r).toEqual({ fire: true, firedFor: "whisper/small" });
    r = readyTransition(r.firedFor, false, "whisper/small");
    expect(r).toEqual({ fire: false, firedFor: null });
    expect(readyTransition(r.firedFor, true, "whisper/small").fire).toBe(true);
  });

  it("shows an operation error once, and not when the status already says it", () => {
    const states = {
      "stt/encoder": status("stt/encoder", "failed", { error: "hash mismatch" }),
      "stt/model": status("stt/model", "not_installed"),
    };
    const errors = { "stt/encoder": "hash mismatch", "stt/model": "C:/x.bin does not exist", "runtime/sherpa": "C:/x.bin does not exist" };
    expect(visibleOpErrors(catalog.slice(0, 3), states, errors)).toEqual(["C:/x.bin does not exist"]);
    expect(visibleOpErrors(catalog.slice(0, 3), states, {})).toEqual([]);
  });
});

describe("disk usage", () => {
  it("sorts families by size and drops empty ones", () => {
    expect(
      diskByFamily({ models_root: "C:/m", total_bytes: 30, per_family: { vosk: 10, whisper: 20, base: 0 }, free_bytes: null }),
    ).toEqual([
      { family: "whisper", bytes: 20 },
      { family: "vosk", bytes: 10 },
    ]);
    expect(diskByFamily(null)).toEqual([]);
  });

  it("warns when a download would leave the drive nearly full", () => {
    const disk = { models_root: null, total_bytes: 0, per_family: {}, free_bytes: 2e9 };
    expect(lowOnDisk(disk, 0.5e9)).toBe(false);
    expect(lowOnDisk(disk, 1.5e9)).toBe(true);
    expect(lowOnDisk({ ...disk, free_bytes: null }, 1)).toBeNull();
  });
});

describe("mirror settings", () => {
  it("recognizes local hosts like the Rust side", () => {
    for (const host of [
      "localhost",
      "127.0.0.1",
      "0.0.0.0",
      "192.168.1.20",
      "10.0.0.5",
      "172.20.3.4",
      "169.254.10.1",
      "100.101.102.103",
      "[::1]",
      "[fd12:3456::1]",
      "[fe80::1]",
      "[::ffff:c0a8:2]",
      "::ffff:192.168.0.2",
      "gpu-box",
      "tts.local",
      "nas.lan",
      "server.home.arpa",
      "tts.internal",
      "kokoro.localhost",
    ]) {
      expect(isLocalHost(host), host).toBe(true);
    }
    for (const host of ["8.8.8.8", "100.128.0.1", "172.32.0.1", "[2001:4860:4860::8888]", "my.mirror.example.com", "local.example.com", ""]) {
      expect(isLocalHost(host), host).toBe(false);
    }
  });

  it("checks custom mirror URLs", () => {
    expect(checkCustomMirrorUrl("https://mirror.example.com/sigil/")).toBeNull();
    expect(checkCustomMirrorUrl("http://192.168.1.2:8000/models")).toBeNull();
    expect(checkCustomMirrorUrl("http://nas.lan")).toBeNull();
    expect(checkCustomMirrorUrl("  ")).toBe("empty");
    expect(checkCustomMirrorUrl(undefined)).toBe("empty");
    expect(checkCustomMirrorUrl("mirror.example.com")).toBe("scheme");
    expect(checkCustomMirrorUrl("ftp://mirror.example.com")).toBe("scheme");
    expect(checkCustomMirrorUrl("https://")).toBe("invalid");
    expect(checkCustomMirrorUrl("https://user:pw@mirror.example.com")).toBe("credentials");
    expect(checkCustomMirrorUrl("https://mirror.example.com/?token=1")).toBe("credentials");
    expect(checkCustomMirrorUrl("https://mirror.example.com/#x")).toBe("credentials");
    expect(checkCustomMirrorUrl("http://mirror.example.com")).toBe("insecure");
  });

  it("builds the store config and holds back an unusable custom mirror", () => {
    expect(buildStoreConfig({ modelMirror: "auto", modelMirrorCustomUrl: "", modelMaxConcurrentDownloads: 1 })).toEqual({
      config: { mirror: "auto", custom_url: null, max_concurrent: 1 },
      problem: null,
    });
    expect(
      buildStoreConfig({ modelMirror: "custom", modelMirrorCustomUrl: " https://m.example.com ", modelMaxConcurrentDownloads: 2 }),
    ).toEqual({ config: { mirror: "custom", custom_url: "https://m.example.com", max_concurrent: 2 }, problem: null });
    expect(buildStoreConfig({ modelMirror: "custom", modelMirrorCustomUrl: "", modelMaxConcurrentDownloads: 1 }).problem).toBe("empty");
    expect(
      buildStoreConfig({ modelMirror: "custom", modelMirrorCustomUrl: "http://m.example.com", modelMaxConcurrentDownloads: 1 }).problem,
    ).toBe("insecure");
    // Another mirror: a valid custom URL is kept for later, an invalid one is dropped without complaint.
    expect(
      buildStoreConfig({ modelMirror: "modelscope", modelMirrorCustomUrl: "https://m.example.com", modelMaxConcurrentDownloads: 1 }),
    ).toEqual({ config: { mirror: "modelscope", custom_url: "https://m.example.com", max_concurrent: 1 }, problem: null });
    expect(
      buildStoreConfig({ modelMirror: "modelscope", modelMirrorCustomUrl: "nope", modelMaxConcurrentDownloads: 1 }),
    ).toEqual({ config: { mirror: "modelscope", custom_url: null, max_concurrent: 1 }, problem: null });
    expect(buildStoreConfig({ modelMirror: "auto", modelMirrorCustomUrl: "", modelMaxConcurrentDownloads: 9 }).config.max_concurrent).toBe(2);
    expect(buildStoreConfig({ modelMirror: "auto", modelMirrorCustomUrl: "", modelMaxConcurrentDownloads: Number.NaN }).config.max_concurrent).toBe(1);
  });
});
