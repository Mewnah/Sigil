//! Tauri build script: ACL manifests for the app commands and every inlined plugin.
//!
//! Plugin command lists are data-driven (P0-S0): each inlined plugin has one file
//! `plugin-commands/<plugin-name>.txt` listing its IPC commands, one per line (`#` starts a
//! comment). The file stem is the plugin name used in `Builder::new("<plugin-name>")` and in the
//! capability id `"<plugin-name>:default"`. Every plugin gets `DefaultPermissionRule::AllowAllCommands`.
//!
//! To add a command: add the `#[command]`, list it in the plugin's `generate_handler!`, and add a
//! line to `plugin-commands/<plugin-name>.txt`. A command missing here compiles fine but fails at
//! runtime with a "not allowed" ACL error. A new plugin also needs its own
//! `capabilities/<plugin-name>.json` granting `"<plugin-name>:default"` to the `main` window.
//!
//! The AppManifest list (commands registered directly in main.rs) stays inline below.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const PLUGIN_COMMANDS_DIR: &str = "plugin-commands";

/// Parse one `plugin-commands/<plugin>.txt` file: trimmed non-empty lines, `#` comments stripped.
fn parse_command_list(plugin: &str, text: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut commands = Vec::new();
    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let valid = line
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            && !line.starts_with(|c: char| c.is_ascii_digit());
        if !valid {
            panic!(
                "{PLUGIN_COMMANDS_DIR}/{plugin}.txt:{}: invalid command name {line:?} (expected snake_case)",
                line_no + 1
            );
        }
        if !seen.insert(line.to_string()) {
            panic!("{PLUGIN_COMMANDS_DIR}/{plugin}.txt:{}: duplicate command {line:?}", line_no + 1);
        }
        commands.push(line.to_string());
    }
    if commands.is_empty() {
        // tauri-build emits no default permission for a plugin without commands, so its
        // "<plugin>:default" capability entry would fail to resolve.
        panic!("{PLUGIN_COMMANDS_DIR}/{plugin}.txt lists no commands");
    }
    commands
}

/// Read every `plugin-commands/*.txt` (sorted by plugin name) as `(plugin, commands)`.
/// Strings are leaked because tauri-build takes `&'static` slices; this runs once per build.
fn load_inlined_plugins() -> Vec<(&'static str, tauri_build::InlinedPlugin)> {
    let dir = Path::new(PLUGIN_COMMANDS_DIR);
    println!("cargo:rerun-if-changed={PLUGIN_COMMANDS_DIR}");
    let mut files: Vec<_> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {PLUGIN_COMMANDS_DIR}/: {e}"))
        .map(|entry| entry.expect("plugin-commands entry").path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("txt"))
        .collect();
    files.sort();

    let mut plugins = Vec::with_capacity(files.len());
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let plugin = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_else(|| panic!("non-UTF-8 plugin command file name: {}", path.display()))
            .to_string();
        if plugin.is_empty()
            || !plugin
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        {
            panic!("{}: plugin names are lower-case kebab-case", path.display());
        }
        let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let commands: Vec<&'static str> = parse_command_list(&plugin, &text)
            .into_iter()
            .map(|c| -> &'static str { Box::leak(c.into_boxed_str()) })
            .collect();
        let commands: &'static [&'static str] = Vec::leak(commands);
        let name: &'static str = Box::leak(plugin.into_boxed_str());
        plugins.push((
            name,
            tauri_build::InlinedPlugin::new()
                .commands(commands)
                .default_permission(tauri_build::DefaultPermissionRule::AllowAllCommands),
        ));
    }
    plugins
}

fn main() {
    tauri_build::try_build(
        tauri_build::Attributes::new()
            .app_manifest(tauri_build::AppManifest::new().commands(&[
                "get_native_features",
                "get_port",
                "app_close",
                "oauth_loopback_start",
            ]))
            .plugins(load_inlined_plugins()),
    )
    .expect("failed to run tauri-build");
}
