/**
 * W2: the release installers embed Microsoft's WebView2 bootstrapper (`bundle.windows.webviewInstallMode` =
 * `embedBootstrapper` in the release overlay) instead of downloading it at install time, and verify-bundle.ps1 checks that the
 * build really used it: the overlay, the NSIS script and the WiX source the bundler generated, the bootstrapper file itself
 * (it is fetched at build time and nothing else pins it), and both finished installers.
 *
 * What the mode does and does not do (Tauri 2.10; the generated NSIS script and the schema both say it): the bootstrapper still
 * downloads the WebView2 runtime itself ("Requires an internet connection" in the schema, a non-zero exit of the bootstrapper
 * aborts the NSIS setup, `Return='check'` rolls the MSI back). Only `offlineInstaller` installs without internet, at about
 * +130 MB per installer. These tests pin the config and the check, not an offline install.
 *
 * The PowerShell checks run for real, in PowerShell 7 and Windows PowerShell 5.1, but every case of a shell goes through ONE
 * process of it: a process per case took seconds each and, on a busy machine (a build next to the tests), more than the test
 * timeout.
 */
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const RELEASE = path.join(HERE, "..");
const ROOT = path.join(RELEASE, "..", "..");
const read = (...p) => fs.readFileSync(path.join(...p), "utf8").replace(/\r\n/g, "\n");

const OVERLAY_TEXT = read(RELEASE, "tauri.release.conf.json");
const OVERLAY = JSON.parse(OVERLAY_TEXT);
const BASE = JSON.parse(read(ROOT, "src-tauri", "tauri.conf.json"));
const VERIFY = read(RELEASE, "verify-bundle.ps1");
const SCHEMA = JSON.parse(read(ROOT, "node_modules", "@tauri-apps", "cli", "config.schema.json"));

/** RFC 7396 merge patch, which is how `tauri build --config <file>` applies the overlay to tauri.conf.json. */
function mergePatch(target, patch) {
  if (patch === null || typeof patch !== "object" || Array.isArray(patch)) return patch;
  const out = target !== null && typeof target === "object" && !Array.isArray(target) ? { ...target } : {};
  for (const [k, v] of Object.entries(patch)) {
    if (v === null) delete out[k];
    else out[k] = mergePatch(out[k], v);
  }
  return out;
}

/** The branches of the schema's WebviewInstallMode (a oneOf of objects with a `type` enum). */
const MODE_BRANCHES = SCHEMA.definitions.WebviewInstallMode.oneOf;

/** Whether `value` is valid against one branch of WebviewInstallMode (required, enum, known properties only, boolean `silent`). */
function matchesBranch(branch, value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  for (const key of branch.required ?? []) if (!(key in value)) return false;
  for (const [key, v] of Object.entries(value)) {
    const spec = branch.properties?.[key];
    if (!spec) return false; // additionalProperties: false
    if (spec.enum && !spec.enum.includes(v)) return false;
    if (spec.type === "boolean" && typeof v !== "boolean") return false;
    if (spec.type === "string" && typeof v !== "string") return false;
  }
  return true;
}

describe("the release overlay's WebView2 install mode", () => {
  const mode = OVERLAY.bundle?.windows?.webviewInstallMode;

  it("asks for the embedded bootstrapper", () => {
    expect(mode).toEqual({ type: "embedBootstrapper" });
  });

  it("is a value the Tauri 2.10 config schema accepts (the bundler would refuse an unknown mode or property)", () => {
    expect(MODE_BRANCHES.length).toBeGreaterThan(3);
    const accepted = MODE_BRANCHES.filter((b) => matchesBranch(b, mode));
    expect(accepted.length).toBe(1);
    expect(accepted[0].properties.type.enum).toEqual(["embedBootstrapper"]);
    // The checker itself rejects what the bundler would: a typo, an extra property.
    expect(MODE_BRANCHES.some((b) => matchesBranch(b, { type: "embedBootstraper" }))).toBe(false);
    expect(MODE_BRANCHES.some((b) => matchesBranch(b, { type: "embedBootstrapper", path: "x" }))).toBe(false);
    // And the WindowsConfig the overlay's `windows` object lands in knows both of its keys.
    const known = Object.keys(SCHEMA.definitions.WindowsConfig.properties);
    for (const key of Object.keys(OVERLAY.bundle.windows)) expect(known, key).toContain(key);
  });

  it("merges into the base config next to what the base and the installer settings already say", () => {
    const merged = mergePatch(BASE, OVERLAY);
    expect(merged.bundle.windows.webviewInstallMode).toEqual({ type: "embedBootstrapper" });
    expect(merged.bundle.windows.nsis).toEqual({ installMode: "currentUser" });
    expect(merged.bundle.windows.digestAlgorithm).toBe(BASE.bundle.windows.digestAlgorithm);
    expect(merged.bundle.resources["target/native-bundle"]).toBe(".");
    // The base keeps the small default for a plain `tauri build` (no bootstrapper fetched while bundling).
    expect(BASE.bundle.windows.webviewInstallMode).toBeUndefined();
  });

  it("is applied to the shipped installers by every release path", () => {
    const pkg = JSON.parse(read(ROOT, "package.json"));
    expect(pkg.scripts["tauri:release"]).toContain("--config scripts/release/tauri.release.conf.json");
    expect(read(ROOT, ".github", "workflows", "release.yml")).toContain("pnpm tauri build --config scripts/release/tauri.release.conf.json");
  });
});

const REGION = (VERIFY.match(/#region webview2-install-mode\n([\s\S]*?)\n#endregion webview2-install-mode/) ?? [])[1];

describe("verify-bundle.ps1 knows the install mode", () => {
  it("expects the mode the overlay asks for, calls the check, and fails the run on its findings", () => {
    expect(REGION, "#region webview2-install-mode").toBeTruthy();
    expect(VERIFY).toContain(`$expectedWebViewMode = '${OVERLAY.bundle.windows.webviewInstallMode.type}'`);
    expect(VERIFY).toMatch(/Get-WebView2ModeFindings \$expectedWebViewMode .*tauri\.release\.conf\.json.*nsis\\x64\\installer\.nsi.*wix\\x64\\main\.wxs/);
    expect(VERIFY).toMatch(/foreach \(\$f in \$webview\.Failures\) \{ Fail \$f \}/);
    expect(VERIFY).toMatch(/foreach \(\$w in \$webview\.Warnings\) \{ Warn \$w \}/);
  });

  it("makes a missing generated file an error once an installer exists (or with -Strict), not a silent warning", () => {
    expect(VERIFY).toMatch(/Get-WebView2ModeFindings .*\(\[bool\]\(\$Strict -or \$nsis\.Count -gt 0\)\) \(\[bool\]\(\$Strict -or \$msi\.Count -gt 0\)\)/);
  });

  it("checks the bootstrapper file and both finished installers, and fails the run on what they find", () => {
    expect(VERIFY).toMatch(/foreach \(\$bootstrapperPath in @\(\$webview\.BootstrapperPaths\)\)/);
    expect(VERIFY).toMatch(/Get-BootstrapperFileFindings \$bootstrapperPath/);
    expect(VERIFY).toMatch(/Get-MsiBootstrapperFindings \(Get-CompoundStreamSizes \$streams\) \$bootstrapperBytes/);
    expect(VERIFY).toMatch(/Get-NsisBootstrapperFindings \$nsisOut \$bootstrapperSha/);
    expect(VERIFY).toMatch(/foreach \(\$problem in \$msiProblems\) \{ Fail /);
    expect(VERIFY).toMatch(/foreach \(\$problem in \$nsisProblems\) \{ Fail /);
    // 7-Zip lists an MSI's streams as an OLE compound file.
    expect(VERIFY).toContain("l -slt -tCompound");
  });
});

/** The relevant lines of a generated NSIS script (src-tauri/target/release/nsis/x64/installer.nsi). */
const nsisScript = (mode, bootstrapper) =>
  [
    '!define MAINBINARYNAME "Sigil"',
    `!define INSTALLWEBVIEW2MODE "${mode}"`,
    '!define WEBVIEW2INSTALLERARGS "/silent"',
    `!define WEBVIEW2BOOTSTRAPPERPATH "${bootstrapper}"`,
    '!define WEBVIEW2INSTALLERPATH ""',
    '!define MINIMUMWEBVIEW2VERSION ""',
    "",
  ].join("\r\n");

const BOOTSTRAPPER = "C:\\Users\\x\\AppData\\Local\\tauri\\Webview2Bootstrapper\\MicrosoftEdgeWebview2Setup.exe";

/** The relevant part of a generated WiX source (src-tauri/target/release/wix/x64/main.wxs). */
const wixEmbed = (binary = `<Binary Id="MicrosoftEdgeWebview2Setup.exe" SourceFile="${BOOTSTRAPPER}"/>`) =>
  [
    "        <!-- WebView2 -->",
    '        <Property Id="WVRTINSTALLED"><RegistrySearch Id="WVRTInstalledSystem" Root="HKLM" /></Property>',
    "        <!-- Embedded webview bootstrapper mode -->",
    `        ${binary}`,
    "        <CustomAction Id='InvokeBootstrapper' BinaryKey='MicrosoftEdgeWebview2Setup.exe' Execute=\"deferred\" ExeCommand='/silent /install' Return='check' />",
    "        <!-- Embedded offline installer -->",
    "",
  ].join("\r\n");
const WIX_EMBED = wixEmbed();
const WIX_DOWNLOAD = [
  "        <!-- WebView2 -->",
  "        <CustomAction Id='DownloadAndInvokeBootstrapper' Directory=\"INSTALLDIR\" Execute=\"deferred\" ExeCommand='powershell.exe -NoProfile' Return='check'/>",
  "        <!-- Embedded webview bootstrapper mode -->",
  "        <!-- Embedded offline installer -->",
  "",
].join("\r\n");
// The same download action, written the other way round in quotes (a template update could do it).
const WIX_DOWNLOAD_DQ = WIX_DOWNLOAD.replace("Id='DownloadAndInvokeBootstrapper'", 'Id="DownloadAndInvokeBootstrapper"');

const GOOD = { overlay: OVERLAY_TEXT, nsis: nsisScript("embedBootstrapper", BOOTSTRAPPER), wix: WIX_EMBED };

/** A listing as `7z l -slt -tCompound x.msi` writes it (taken from a real WiX MSI with the bootstrapper as a Binary). */
const msiListing = (bootstrapperSize) =>
  [
    "7-Zip 26.00 (x64) : Copyright (c) 1999-2026 Igor Pavlov : 2026-02-12",
    "",
    "Scanning the drive for archives:",
    "1 file, 1835008 bytes (1792 KiB)",
    "",
    "Listing archive: t.msi",
    "",
    "--",
    "Path = t.msi",
    "Type = Compound",
    "Physical Size = 1835008",
    "Cluster Size = 512",
    "",
    "----------",
    "Path = a.cab",
    "Size = 14",
    "Packed Size = 64",
    "",
    "Path = !Binary",
    "Size = 504",
    "Packed Size = 512",
    "",
    ...(bootstrapperSize === null ? [] : ["Path = Binary.MicrosoftEdgeWebview2Setup.exe", `Size = ${bootstrapperSize}`, "Packed Size = 1802240", "Created = ", ""]),
    "Path = Binary.WixUI_Bmp_New",
    "Size = 318",
    "Packed Size = 320",
    "",
  ].join("\r\n");

const q = (s) => `'${String(s).replace(/'/g, "''")}'`;
const SHELLS = ["pwsh", "powershell.exe"]; // PowerShell 7 and Windows PowerShell 5.1
const SYSTEM_SIGNED = "(Join-Path $env:SystemRoot 'System32\\cmd.exe')"; // signed by Microsoft (a catalog signature)
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

/** Test cases for the shells: name -> { files: { relative path: string | Buffer }, ps: (case dir) => PowerShell expression }. */
const CASES = {};
/** Defines a case of Get-WebView2ModeFindings on a scratch folder holding the given files (`null`: absent). */
function modeCase(name, { overlay = OVERLAY_TEXT, nsis = GOOD.nsis, wix = GOOD.wix, mode = "embedBootstrapper", nsisRequired = false, wixRequired = false }) {
  const files = {};
  if (overlay !== null) files["overlay.json"] = overlay;
  if (nsis !== null) files["installer.nsi"] = nsis;
  if (wix !== null) files["main.wxs"] = wix;
  CASES[name] = {
    files,
    ps: (d) => `Get-WebView2ModeFindings ${q(mode)} ${q(path.join(d, "overlay.json"))} ${q(path.join(d, "installer.nsi"))} ${q(path.join(d, "main.wxs"))} $${nsisRequired} $${wixRequired}`,
  };
}
const withoutMode = JSON.parse(OVERLAY_TEXT);
delete withoutMode.bundle.windows.webviewInstallMode;
const otherMode = JSON.parse(OVERLAY_TEXT);
otherMode.bundle.windows.webviewInstallMode = { type: "downloadBootstrapper" };

modeCase("good", {});
modeCase("no-overlay-build", { nsis: nsisScript("downloadBootstrapper", ""), wix: WIX_DOWNLOAD });
modeCase("nsis-embeds-nothing", { nsis: nsisScript("embedBootstrapper", "") });
modeCase("nsis-without-define", { nsis: '!define MAINBINARYNAME "Sigil"\r\n' });
modeCase("wix-without-binary", { wix: "<!-- Embedded webview bootstrapper mode -->\r\n" });
modeCase("wix-embeds-and-downloads", { wix: WIX_EMBED + WIX_DOWNLOAD });
modeCase("wix-downloads-double-quoted", { wix: WIX_EMBED + WIX_DOWNLOAD_DQ });
modeCase("wix-binary-single-quoted", { wix: wixEmbed(`<Binary Id='MicrosoftEdgeWebview2Setup.exe' SourceFile='${BOOTSTRAPPER}'/>`) });
modeCase("wix-binary-attributes-swapped", { wix: wixEmbed(`<Binary SourceFile="${BOOTSTRAPPER}" Id="MicrosoftEdgeWebview2Setup.exe" />`) });
modeCase("wix-binary-without-source", { wix: wixEmbed('<Binary Id="MicrosoftEdgeWebview2Setup.exe"/>') });
modeCase("overlay-missing-mode", { overlay: JSON.stringify(withoutMode) });
modeCase("overlay-other-mode", { overlay: JSON.stringify(otherMode) });
modeCase("overlay-not-json", { overlay: "{ not json" });
modeCase("overlay-absent", { overlay: null });
modeCase("generated-files-absent", { nsis: null, wix: null });
modeCase("generated-files-absent-but-installers-built", { nsis: null, wix: null, nsisRequired: true, wixRequired: true });
modeCase("nsis-absent-but-built", { nsis: null, nsisRequired: true });
modeCase("wix-absent-but-built", { wix: null, wixRequired: true });
modeCase("present-files-ignore-required", { nsisRequired: true, wixRequired: true });

// The bootstrapper file: size, signature, signer.
const UNSIGNED = Buffer.alloc(1_800_000);
CASES["file-missing"] = { files: {}, ps: (d) => `Get-BootstrapperFileFindings ${q(path.join(d, "nothing.exe"))}` };
CASES["file-unsigned"] = { files: { "b.exe": UNSIGNED }, ps: (d) => `Get-BootstrapperFileFindings ${q(path.join(d, "b.exe"))}` };
CASES["file-unsigned-too-small"] = { files: { "b.exe": Buffer.alloc(900) }, ps: (d) => `Get-BootstrapperFileFindings ${q(path.join(d, "b.exe"))}` };
CASES["file-unsigned-too-large"] = { files: { "b.exe": Buffer.alloc(3_100_000) }, ps: (d) => `Get-BootstrapperFileFindings ${q(path.join(d, "b.exe"))}` };
CASES["file-microsoft-signed"] = { files: {}, ps: () => `Get-BootstrapperFileFindings ${SYSTEM_SIGNED} 1 100000000` };
CASES["file-microsoft-signed-wrong-size"] = { files: {}, ps: () => `Get-BootstrapperFileFindings ${SYSTEM_SIGNED} 90000000 100000000` };
const SUBJECTS = {
  "CN=Microsoft Corporation, O=Microsoft Corporation, L=Redmond, S=Washington, C=US": true,
  "CN=Microsoft Windows, O=Microsoft Corporation, L=Redmond, S=Washington, C=US": true,
  "O=Microsoft Corporation": true,
  "CN=Microsoft Corporation, O=Someone Else, C=US": false, // a CN is not the organisation
  "CN=Evil, O=Microsoft Corporation Fake, C=US": false,
  "CN=Evil, O=Not Microsoft Corporation": false,
  "": false,
};
CASES["subjects"] = { files: {}, ps: () => `@(${Object.keys(SUBJECTS).map((s) => `(${q(s)} + '=' + (Test-MicrosoftSubject ${q(s)}))`).join(", ")})` };

// The finished installers.
const BOOTSTRAPPER_SHA = sha256(UNSIGNED);
CASES["nsis-carries-it"] = { files: { "ext/$TEMP/MicrosoftEdgeWebview2Setup.exe": UNSIGNED }, ps: (d) => `Get-NsisBootstrapperFindings ${q(path.join(d, "ext"))} ${q(BOOTSTRAPPER_SHA)}` };
CASES["nsis-carries-it-unchecked"] = { files: { "ext/$TEMP/MicrosoftEdgeWebview2Setup.exe": UNSIGNED }, ps: (d) => `Get-NsisBootstrapperFindings ${q(path.join(d, "ext"))}` };
CASES["nsis-carries-another-file"] = { files: { "ext/$TEMP/MicrosoftEdgeWebview2Setup.exe": Buffer.alloc(1_800_001) }, ps: (d) => `Get-NsisBootstrapperFindings ${q(path.join(d, "ext"))} ${q(BOOTSTRAPPER_SHA)}` };
CASES["nsis-without-it"] = { files: { "ext/Sigil.exe": Buffer.alloc(10), "ext/$PLUGINSDIR/System.dll": Buffer.alloc(10) }, ps: (d) => `Get-NsisBootstrapperFindings ${q(path.join(d, "ext"))} ${q(BOOTSTRAPPER_SHA)}` };
const msiCase = (listing, expected) => ({
  files: { "listing.txt": listing },
  ps: (d) => `Get-MsiBootstrapperFindings (Get-CompoundStreamSizes (Get-Content -LiteralPath ${q(path.join(d, "listing.txt"))})) ${expected}`,
});
CASES["msi-carries-it"] = msiCase(msiListing(1_800_000), 1_800_000);
CASES["msi-carries-it-unchecked"] = msiCase(msiListing(1_800_000), 0);
CASES["msi-carries-another-file"] = msiCase(msiListing(1_700_000), 1_800_000);
CASES["msi-without-it"] = msiCase(msiListing(null), 1_800_000);
CASES["msi-sizes"] = {
  files: { "listing.txt": msiListing(1_800_000) },
  ps: (d) => `@((Get-CompoundStreamSizes (Get-Content -LiteralPath ${q(path.join(d, "listing.txt"))})).GetEnumerator() | Sort-Object Name | ForEach-Object { $_.Name + '=' + $_.Value })`,
};

/**
 * Runs every case in one process of `shell`. Returns { [case]: { failures, warnings, paths, error } }, or null when the shell is
 * missing. A case's output is the Failures/Warnings/BootstrapperPaths of the result object, or the strings it returns.
 */
function runShell(shell, root) {
  const lines = [
    "Set-StrictMode -Version Latest",
    "$ErrorActionPreference = 'Stop'",
    `. ([scriptblock]::Create((Get-Content -Raw -LiteralPath ${q(path.join(root, "region.ps1"))})))`,
    "function Report([string]$name, $r) {",
    "  if ($null -ne $r -and $null -ne $r.PSObject.Properties['Failures']) {",
    "    foreach ($x in @($r.Failures)) { \"F|$name|$x\" }",
    "    foreach ($x in @($r.Warnings)) { \"W|$name|$x\" }",
    "    foreach ($x in @($r.BootstrapperPaths)) { \"P|$name|$x\" }",
    "  } elseif ($null -ne $r) {",
    "    foreach ($x in @($r)) { \"F|$name|$x\" }",
    "  }",
    "  \"D|$name\"",
    "}",
  ];
  for (const [name, c] of Object.entries(CASES)) {
    lines.push(`try { $r = ${c.ps(path.join(root, name))}; Report ${q(name)} $r } catch { "E|${name}|$($_.Exception.Message)" }`);
  }
  const script = path.join(root, `cases-${shell.replace(/\W/g, "")}.ps1`);
  fs.writeFileSync(script, lines.join("\n"));
  const r = spawnSync(shell, ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script], { encoding: "utf8", maxBuffer: 1 << 26 });
  if (r.error && /ENOENT/.test(String(r.error))) return null;
  expect(r.stderr, `${shell} stderr`).toBe("");
  expect(r.status, `${shell} exit code`).toBe(0);
  const out = {};
  for (const name of Object.keys(CASES)) out[name] = { failures: [], warnings: [], paths: [], error: null, done: false };
  for (const line of r.stdout.split(/\r?\n/).filter(Boolean)) {
    const [kind, name, ...rest] = line.split("|");
    const text = rest.join("|");
    const entry = out[name];
    if (!entry) continue;
    if (kind === "F") entry.failures.push(text);
    else if (kind === "W") entry.warnings.push(text);
    else if (kind === "P") entry.paths.push(text);
    else if (kind === "E") entry.error = text;
    else if (kind === "D") entry.done = true;
  }
  return out;
}

const RESULTS = {};

describe.runIf(process.platform === "win32")("verify-bundle.ps1's WebView2 checks (run for real in 7 and 5.1)", () => {
  beforeAll(() => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "webview2-mode-"));
    try {
      fs.writeFileSync(path.join(root, "region.ps1"), REGION);
      for (const [name, c] of Object.entries(CASES)) {
        for (const [rel, content] of Object.entries(c.files)) {
          const file = path.join(root, name, rel);
          fs.mkdirSync(path.dirname(file), { recursive: true });
          fs.writeFileSync(file, content);
        }
        fs.mkdirSync(path.join(root, name), { recursive: true });
      }
      for (const shell of SHELLS) {
        const out = runShell(shell, root);
        if (out) RESULTS[shell] = out;
      }
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  }, 240_000);

  /** Runs `check(result, shell)` on the case in every shell that exists (and requires that one does). */
  const each = (name, check) => {
    expect(Object.keys(RESULTS).length, "at least one PowerShell").toBeGreaterThan(0);
    for (const [shell, results] of Object.entries(RESULTS)) {
      const r = results[name];
      expect(r.error, `${shell} ${name}: error`).toBeNull();
      expect(r.done, `${shell} ${name}: ran`).toBe(true);
      check(r, shell);
    }
  };

  it("passes the repository's own overlay with a build that embedded the bootstrapper, and names the file it embeds", () => {
    each("good", (r, shell) => {
      expect(r.failures, shell).toEqual([]);
      expect(r.warnings, shell).toEqual([]);
      expect(r.paths, shell).toEqual([BOOTSTRAPPER]);
    });
    // Required generated files that are there change nothing.
    each("present-files-ignore-required", (r, shell) => expect({ f: r.failures, w: r.warnings }, shell).toEqual({ f: [], w: [] }));
  });

  it("fails a build that did not get the overlay: the NSIS setup and the MSI still download the bootstrapper", () => {
    each("no-overlay-build", (r, shell) => {
      expect(r.warnings, shell).toEqual([]);
      expect(r.failures.length, shell).toBe(3);
      expect(r.failures.join("\n"), shell).toContain("built with WebView2 install mode 'downloadBootstrapper', not 'embedBootstrapper'");
      expect(r.failures.join("\n"), shell).toContain("MSI embeds no WebView2 bootstrapper");
      expect(r.failures.join("\n"), shell).toContain("MSI still downloads the WebView2 bootstrapper");
    });
  });

  it("fails an NSIS script that says embedBootstrapper but carries no bootstrapper", () => {
    each("nsis-embeds-nothing", (r, shell) => expect(r.failures, shell).toEqual(["the NSIS setup embeds no WebView2 bootstrapper (WEBVIEW2BOOTSTRAPPERPATH is empty)"]));
  });

  it("fails an NSIS script without the define at all (a bundler that changed its template must not pass silently)", () => {
    each("nsis-without-define", (r, shell) => {
      expect(r.failures.length, shell).toBe(1);
      expect(r.failures[0], shell).toContain("defines no INSTALLWEBVIEW2MODE");
    });
  });

  it("fails an MSI source that embeds the bootstrapper without the binary, or that downloads next to it", () => {
    each("wix-without-binary", (r, shell) =>
      expect(r.failures, shell).toEqual(["the MSI embeds no WebView2 bootstrapper (no MicrosoftEdgeWebview2Setup.exe binary in the WiX source)"]),
    );
    each("wix-embeds-and-downloads", (r, shell) =>
      expect(r.failures, shell).toEqual(["the MSI still downloads the WebView2 bootstrapper at install time (DownloadAndInvokeBootstrapper in the WiX source)"]),
    );
  });

  it("does not care which quote style or attribute order the bundler's template uses", () => {
    // A download action written with double quotes is still a download (it passed unseen when only single quotes matched).
    each("wix-downloads-double-quoted", (r, shell) =>
      expect(r.failures, shell).toEqual(["the MSI still downloads the WebView2 bootstrapper at install time (DownloadAndInvokeBootstrapper in the WiX source)"]),
    );
    // And an embedded binary written the other way is still found, with its file.
    for (const name of ["wix-binary-single-quoted", "wix-binary-attributes-swapped"]) {
      each(name, (r, shell) => {
        expect({ f: r.failures, w: r.warnings, p: r.paths }, `${shell} ${name}`).toEqual({ f: [], w: [], p: [BOOTSTRAPPER] });
      });
    }
    each("wix-binary-without-source", (r, shell) => {
      expect(r.failures.length, shell).toBe(1);
      expect(r.failures[0], shell).toContain("has no SourceFile");
    });
  });

  it("fails an overlay that no longer asks for the mode, asks for another one, or is not JSON", () => {
    each("overlay-missing-mode", (r, shell) => {
      expect(r.failures.length, shell).toBe(1);
      expect(r.failures[0], shell).toContain("sets no bundle.windows.webviewInstallMode");
    });
    each("overlay-other-mode", (r, shell) => expect(r.failures, shell).toEqual(["the release overlay sets webviewInstallMode 'downloadBootstrapper', expected 'embedBootstrapper'"]));
    each("overlay-not-json", (r, shell) => {
      expect(r.failures.length, shell).toBe(1);
      expect(r.failures[0], shell).toContain("not valid JSON");
    });
    each("overlay-absent", (r, shell) => expect(r.failures[0], shell).toContain("release overlay not found"));
  });

  it("only warns when the generated files are not where they are expected and no installer was built from them", () => {
    each("generated-files-absent", (r, shell) => {
      expect(r.failures, shell).toEqual([]);
      expect(r.warnings.length, shell).toBe(2);
      expect(r.warnings.join("\n"), shell).toContain("generated NSIS script not found");
      expect(r.warnings.join("\n"), shell).toContain("generated WiX source not found");
    });
  });

  it("fails when an installer was built but the generated file its mode is read from is not there (the bundler moved it)", () => {
    each("generated-files-absent-but-installers-built", (r, shell) => {
      expect(r.warnings, shell).toEqual([]);
      expect(r.failures.length, shell).toBe(2);
      expect(r.failures.join("\n"), shell).toContain("generated NSIS script not found");
      expect(r.failures.join("\n"), shell).toContain("although the NSIS setup was built");
      expect(r.failures.join("\n"), shell).toContain("although the MSI was built");
    });
    // Each kind on its own: the NSIS setup built, the MSI not (and the other way round).
    each("nsis-absent-but-built", (r, shell) => {
      expect(r.warnings, shell).toEqual([]);
      expect(r.failures.length, shell).toBe(1);
      expect(r.failures[0], shell).toContain("although the NSIS setup was built");
    });
    each("wix-absent-but-built", (r, shell) => {
      expect(r.warnings, shell).toEqual([]);
      expect(r.failures.length, shell).toBe(1);
      expect(r.failures[0], shell).toContain("although the MSI was built");
    });
  });

  it("requires the bootstrapper file to exist, to be 1 to 3 MB and to be signed by Microsoft Corporation", () => {
    each("file-missing", (r, shell) => {
      expect(r.failures.length, shell).toBe(1);
      expect(r.failures[0], shell).toContain("is not a file");
    });
    each("file-unsigned", (r, shell) => {
      // Right size, no signature: one failure, about the signature (and not the size).
      expect(r.failures.length, shell).toBe(1);
      expect(r.failures[0], shell).toMatch(/has a (NotSigned|UnknownError) Authenticode signature, expected Valid/);
    });
    each("file-unsigned-too-small", (r, shell) => {
      expect(r.failures.length, shell).toBe(2);
      expect(r.failures.join("\n"), shell).toContain("is 900 bytes, expected 1000000 to 3000000");
      expect(r.failures.join("\n"), shell).toMatch(/Authenticode signature, expected Valid/);
    });
    each("file-unsigned-too-large", (r, shell) => {
      expect(r.failures.join("\n"), shell).toContain("is 3100000 bytes, expected 1000000 to 3000000");
    });
    each("file-microsoft-signed", (r, shell) => expect(r.failures, shell).toEqual([]));
    each("file-microsoft-signed-wrong-size", (r, shell) => {
      expect(r.failures.length, shell).toBe(1);
      expect(r.failures[0], shell).toContain("bytes, expected 90000000 to 100000000");
    });
  });

  it("takes a signer only when its organisation is Microsoft Corporation", () => {
    each("subjects", (r, shell) => expect(r.failures, shell).toEqual(Object.entries(SUBJECTS).map(([s, ok]) => `${s}=${ok ? "True" : "False"}`)));
  });

  it("finds the bootstrapper in the extracted NSIS setup, and refuses a missing or a different file", () => {
    each("nsis-carries-it", (r, shell) => expect(r.failures, shell).toEqual([]));
    each("nsis-carries-it-unchecked", (r, shell) => expect(r.failures, shell).toEqual([]));
    each("nsis-carries-another-file", (r, shell) => expect(r.failures, shell).toEqual(["the WebView2 bootstrapper inside the NSIS setup is not the file the build used"]));
    each("nsis-without-it", (r, shell) => {
      expect(r.failures.length, shell).toBe(1);
      expect(r.failures[0], shell).toContain("does not contain MicrosoftEdgeWebview2Setup.exe");
    });
  });

  it("finds the bootstrapper among the streams 7-Zip lists for the MSI, with the size of the file the build used", () => {
    each("msi-sizes", (r, shell) => {
      expect([...r.failures].sort(), shell).toEqual(["!Binary=504", "Binary.MicrosoftEdgeWebview2Setup.exe=1800000", "Binary.WixUI_Bmp_New=318", "a.cab=14"].sort());
    });
    each("msi-carries-it", (r, shell) => expect(r.failures, shell).toEqual([]));
    each("msi-carries-it-unchecked", (r, shell) => expect(r.failures, shell).toEqual([]));
    each("msi-carries-another-file", (r, shell) => expect(r.failures, shell).toEqual(["the WebView2 bootstrapper inside the MSI is 1700000 bytes, not the 1800000 of the file the build used"]));
    each("msi-without-it", (r, shell) => {
      expect(r.failures.length, shell).toBe(1);
      expect(r.failures[0], shell).toContain("does not contain the Binary MicrosoftEdgeWebview2Setup.exe");
    });
  });
});
