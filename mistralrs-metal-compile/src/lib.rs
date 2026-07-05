use std::env;
use std::path::PathBuf;
use std::process::Command;

#[cfg(feature = "metal")]
use candle_metal_kernels::metal::{Device, Library};
#[cfg(feature = "metal")]
use objc2_metal::{MTLCompileOptions, MTLLanguageVersion, MTLMathMode};
#[cfg(feature = "metal")]
use std::collections::{HashMap, HashSet};

pub const DEFAULT_METAL_STD: &str = "metal3.1";

const PRECOMPILE_ENV: &str = "MISTRALRS_METAL_PRECOMPILE";
const PLATFORMS_ENV: &str = "MISTRALRS_METAL_PLATFORMS";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalPlatform {
    MacOS,
    Ios,
    TvOS,
    VisionOS,
    WatchOS,
}

impl MetalPlatform {
    const ALL: [Self; 5] = [
        Self::MacOS,
        Self::Ios,
        Self::TvOS,
        Self::VisionOS,
        Self::WatchOS,
    ];

    fn sdk(self) -> &'static str {
        match self {
            Self::MacOS => "macosx",
            Self::Ios => "iphoneos",
            Self::TvOS => "appletvos",
            Self::VisionOS => "xros",
            Self::WatchOS => "watchos",
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            Self::MacOS => "",
            Self::Ios => "_ios",
            Self::TvOS => "_tvos",
            Self::VisionOS => "_visionos",
            Self::WatchOS => "_watchos",
        }
    }

    /// The Metal language standard used for this platform.
    ///
    /// visionOS only supports Metal starting with Metal 4 on visionOS 26+,
    /// so the configured standard is overridden there.
    /// See: https://support.apple.com/en-us/102894
    fn metal_std(self, configured: &'static str) -> &'static str {
        match self {
            Self::VisionOS => "metal4.0",
            _ => configured,
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "macos" | "macosx" => Some(Self::MacOS),
            "ios" | "iphoneos" => Some(Self::Ios),
            "tvos" | "appletvos" => Some(Self::TvOS),
            "visionos" | "xros" => Some(Self::VisionOS),
            "watchos" => Some(Self::WatchOS),
            _ => None,
        }
    }
}

pub struct MetalSource {
    pub name: &'static str,
    pub source: &'static str,
}

pub struct MetalSourceSet {
    pub library_name: &'static str,
    pub source_dir: &'static str,
    pub metal_sources: &'static [&'static str],
    pub header_sources: &'static [&'static str],
    pub include_only_sources: &'static [&'static str],
    pub embedded_sources: &'static [MetalSource],
    pub metal_std: &'static str,
}

#[macro_export]
macro_rules! metal_source_set {
    (
        $(#[$meta:meta])*
        $vis:vis const $name:ident;
        library_name: $library_name:literal,
        source_dir: $source_dir:literal,
        metal_sources: [$($metal_source:literal),* $(,)?],
        header_sources: [$($header_source:literal),* $(,)?],
        include_only_sources: [$($include_only_source:literal),* $(,)?],
        metal_std: $metal_std:expr $(,)?
    ) => {
        $(#[$meta])*
        $vis const $name: $crate::MetalSourceSet = $crate::MetalSourceSet {
            library_name: $library_name,
            source_dir: $source_dir,
            metal_sources: &[$($metal_source),*],
            header_sources: &[$($header_source),*],
            include_only_sources: &[$($include_only_source),*],
            embedded_sources: &[
                $(
                    $crate::MetalSource {
                        name: concat!($metal_source, ".metal"),
                        source: include_str!(concat!(
                            env!("CARGO_MANIFEST_DIR"),
                            "/",
                            $source_dir,
                            "/",
                            $metal_source,
                            ".metal"
                        )),
                    },
                )*
                $(
                    $crate::MetalSource {
                        name: concat!($header_source, ".metal"),
                        source: include_str!(concat!(
                            env!("CARGO_MANIFEST_DIR"),
                            "/",
                            $source_dir,
                            "/",
                            $header_source,
                            ".metal"
                        )),
                    },
                )*
                $(
                    $crate::MetalSource {
                        name: concat!($include_only_source, ".metal"),
                        source: include_str!(concat!(
                            env!("CARGO_MANIFEST_DIR"),
                            "/",
                            $source_dir,
                            "/",
                            $include_only_source,
                            ".metal"
                        )),
                    },
                )*
            ],
            metal_std: $metal_std,
        };
    };
    (
        $(#[$meta:meta])*
        $vis:vis const $name:ident;
        library_name: $library_name:literal,
        source_dir: $source_dir:literal,
        metal_sources: [$($metal_source:literal),* $(,)?],
        header_sources: [$($header_source:literal),* $(,)?],
        include_only_sources: [$($include_only_source:literal),* $(,)?] $(,)?
    ) => {
        $crate::metal_source_set! {
            $(#[$meta])*
            $vis const $name;
            library_name: $library_name,
            source_dir: $source_dir,
            metal_sources: [$($metal_source),*],
            header_sources: [$($header_source),*],
            include_only_sources: [$($include_only_source),*],
            metal_std: $crate::DEFAULT_METAL_STD,
        }
    };
}

pub fn compile_metallibs(config: &MetalSourceSet) -> Result<(), String> {
    emit_rerun_directives(config);
    if should_skip_precompile() {
        println!("cargo:warning=Skipping Metal kernel precompilation ({PRECOMPILE_ENV}=0)");
        write_dummy_metallibs(config)?;
        return Ok(());
    }

    invalidate_on_metal_toolchain_change(config)?;

    let platforms = selected_platforms()?;
    ensure_target_platform_selected(&platforms)?;
    for platform in platforms {
        compile_platform(config, platform)?;
    }
    Ok(())
}

/// Track the Metal toolchain version and invalidate cached metallibs on change.
///
/// Cargo only re-runs build scripts when files listed in rerun-if-changed are
/// modified. It does NOT track the Xcode / Metal compiler version. If the user
/// upgrades Xcode (e.g. from a beta with an air-lld bug to a fixed release),
/// the old broken .metallib files stay cached and get silently baked into the
/// binary via include_bytes!.
///
/// We query `xcrun metal --version` and write the output to a stamp file in
/// OUT_DIR. If the version string changes, the stale metallibs are removed so
/// they get recompiled.
fn invalidate_on_metal_toolchain_change(config: &MetalSourceSet) -> Result<(), String> {
    let out_dir = out_dir()?;
    let stamp = out_dir.join(".metal_toolchain_version");

    let current_version = Command::new("xcrun")
        .args(["metal", "--version"])
        .output()
        .map(|o| {
            format!(
                "{}{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            )
        })
        .unwrap_or_default();

    let needs_rebuild = match std::fs::read_to_string(&stamp) {
        Ok(prev) => prev != current_version,
        Err(_) => true,
    };

    if needs_rebuild {
        let _ = std::fs::write(&stamp, &current_version);
        for platform in MetalPlatform::ALL {
            let _ = std::fs::remove_file(out_dir.join(metallib_name(config, platform)));
        }
        println!("cargo:warning=Metal toolchain version changed — recompiling Metal kernels");
    }
    Ok(())
}

#[cfg(feature = "metal")]
pub fn compile_runtime_library(
    device: &Device,
    config: &MetalSourceSet,
) -> Result<Library, String> {
    let main_source = build_runtime_source(config)?;
    let compile_options = {
        let opts = MTLCompileOptions::new();
        opts.setLanguageVersion(MTLLanguageVersion::Version3_1);
        opts.setMathMode(MTLMathMode::Fast);
        opts
    };
    device
        .new_library_with_source(&main_source, Some(&compile_options))
        .map_err(|err| format!("Failed to compile Metal kernels at runtime: {err}"))
}

#[cfg(feature = "metal")]
fn build_runtime_source(config: &MetalSourceSet) -> Result<String, String> {
    let file_system = source_map(config)?;
    validate_embedded_sources(config, &file_system)?;

    let mut included_files = HashSet::new();
    let mut include_stack = Vec::new();
    let mut main_source = runtime_prelude();

    for source in config.metal_sources {
        let file = metal_file_name(source);
        if included_files.contains(file.as_str()) {
            continue;
        }

        let content = file_system
            .get(file.as_str())
            .ok_or_else(|| format!("Missing embedded Metal source `{file}`"))?;
        let processed = preprocess_includes(
            content,
            &file,
            &file_system,
            &mut included_files,
            &mut include_stack,
        )?;
        included_files.insert(file.clone());
        main_source.push_str(&format!("\n// ===== {file} =====\n"));
        main_source.push_str(&processed);
    }

    Ok(main_source)
}

fn emit_rerun_directives(config: &MetalSourceSet) {
    for source in config
        .metal_sources
        .iter()
        .chain(config.header_sources)
        .chain(config.include_only_sources)
    {
        println!(
            "cargo::rerun-if-changed={}/{}.metal",
            config.source_dir, source
        );
    }
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed={PRECOMPILE_ENV}");
    println!("cargo:rerun-if-env-changed={PLATFORMS_ENV}");
}

fn should_skip_precompile() -> bool {
    env::var(PRECOMPILE_ENV)
        .map(|value| {
            let value = value.to_ascii_lowercase();
            matches!(value.as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

fn selected_platforms() -> Result<Vec<MetalPlatform>, String> {
    let raw = match env::var(PLATFORMS_ENV) {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => return Ok(MetalPlatform::ALL.to_vec()),
        Err(err) => return Err(format!("Failed to read {PLATFORMS_ENV}: {err}")),
    };

    let mut platforms = Vec::new();
    for token in raw
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        if token.eq_ignore_ascii_case("all") {
            return Ok(MetalPlatform::ALL.to_vec());
        }
        let platform = MetalPlatform::parse(token).ok_or_else(|| {
            format!(
                "Invalid {PLATFORMS_ENV} entry `{token}`; expected macos, ios, tvos, visionos, watchos, or all"
            )
        })?;
        if !platforms.contains(&platform) {
            platforms.push(platform);
        }
    }

    if platforms.is_empty() {
        Err(format!(
            "{PLATFORMS_ENV} must contain macos, ios, tvos, visionos, watchos, or all"
        ))
    } else {
        Ok(platforms)
    }
}

fn ensure_target_platform_selected(platforms: &[MetalPlatform]) -> Result<(), String> {
    let Some(target_platform) = target_platform()? else {
        return Ok(());
    };
    if platforms.contains(&target_platform) {
        return Ok(());
    }
    Err(format!(
        "{PLATFORMS_ENV} does not include the current target platform: {:?}",
        target_platform
    ))
}

fn target_platform() -> Result<Option<MetalPlatform>, String> {
    let target_os = match env::var("CARGO_CFG_TARGET_OS") {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => return Ok(None),
        Err(err) => return Err(format!("Failed to read CARGO_CFG_TARGET_OS: {err}")),
    };
    Ok(match target_os.as_str() {
        "macos" => Some(MetalPlatform::MacOS),
        "ios" => Some(MetalPlatform::Ios),
        "tvos" => Some(MetalPlatform::TvOS),
        "visionos" => Some(MetalPlatform::VisionOS),
        "watchos" => Some(MetalPlatform::WatchOS),
        _ => None,
    })
}

fn write_dummy_metallibs(config: &MetalSourceSet) -> Result<(), String> {
    let out_dir = out_dir()?;
    for platform in MetalPlatform::ALL {
        std::fs::write(out_dir.join(metallib_name(config, platform)), [])
            .map_err(|err| format!("Failed to write dummy metallib: {err}"))?;
    }
    Ok(())
}

fn compile_platform(config: &MetalSourceSet, platform: MetalPlatform) -> Result<(), String> {
    let current_dir = env::current_dir().map_err(|err| format!("Failed to get cwd: {err}"))?;
    let out_dir = out_dir()?;
    let working_directory = out_dir.to_string_lossy().to_string();
    let sources = current_dir.join(config.source_dir);
    let metal_std = platform.metal_std(config.metal_std);

    let mut compile_air_cmd = Command::new("xcrun");
    compile_air_cmd
        .arg("--sdk")
        .arg(platform.sdk())
        .arg("metal")
        .arg(format!("-std={metal_std}"))
        .arg(format!("-working-directory={working_directory}"))
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-O3")
        .arg("-c")
        .arg("-w");
    for metal_file in config.metal_sources.iter().chain(config.header_sources) {
        compile_air_cmd.arg(sources.join(format!("{metal_file}.metal")));
    }
    run_command(
        &mut compile_air_cmd,
        &format!("Compiling metal -> air ({} SDK)", platform.sdk()),
    )?;

    // CRITICAL: Pass `--sdk` and `-std` to the link step too!
    //
    // Without `-std=<metal_version>`, `xcrun metal` (the linker) falls back to
    // the Metal version implied by MACOSX_DEPLOYMENT_TARGET. When Cargo
    // cross-compiles for x86_64-apple-darwin it sets
    // MACOSX_DEPLOYMENT_TARGET=10.12 (or 10.13), which implies Metal 2.0 /
    // AIR 2.0. The .air files were compiled with a newer -std, so air-lld
    // silently rejects every .air input and produces a ~92-byte empty
    // metallib (exit code 0). At runtime, every Metal function cannot be
    // found.
    let metallib = out_dir.join(metallib_name(config, platform));
    let mut compile_metallib_cmd = Command::new("xcrun");
    compile_metallib_cmd
        .arg("--sdk")
        .arg(platform.sdk())
        .arg("metal")
        .arg(format!("-std={metal_std}"))
        .arg("-o")
        .arg(&metallib);
    for metal_file in config.metal_sources.iter().chain(config.header_sources) {
        compile_metallib_cmd.arg(out_dir.join(format!("{metal_file}.air")));
    }
    run_command(
        &mut compile_metallib_cmd,
        &format!("Compiling air -> metallib ({} SDK)", platform.sdk()),
    )?;

    validate_metallib(config, platform, &metallib)
}

/// Validate that the linked metallib is not empty.
///
/// Xcode 26 betas had a bug where air-lld would warn about AIR version
/// mismatches and silently produce a ~92-byte empty metallib (exit code 0).
/// A valid metallib with several kernel sources should be well over 4 KB.
fn validate_metallib(
    config: &MetalSourceSet,
    platform: MetalPlatform,
    metallib: &std::path::Path,
) -> Result<(), String> {
    const MIN_METALLIB_SIZE: u64 = 4096;
    let lib_name = metallib_name(config, platform);
    match std::fs::metadata(metallib) {
        Ok(meta) if meta.len() < MIN_METALLIB_SIZE => Err(format!(
            "Generated {lib_name} is only {} bytes (expected >= {MIN_METALLIB_SIZE}).\n\
             The metallib is likely empty — all .air inputs may have been \
             silently discarded by air-lld due to AIR version mismatch.\n\
             This happens when MACOSX_DEPLOYMENT_TARGET is set to an old \
             value (e.g. 10.13) which implies Metal 2.0 / AIR 2.0, but \
             the .air files were compiled with a newer -std.\n\
             Try: rm -rf target/*/build/{}-* && cargo build\n\
             Or update your Xcode installation.",
            meta.len(),
            config.library_name,
        )),
        Ok(_) => Ok(()),
        Err(e) => Err(format!("Cannot stat generated {lib_name}: {e}")),
    }
}

fn out_dir() -> Result<PathBuf, String> {
    env::var("OUT_DIR")
        .map(PathBuf::from)
        .map_err(|_| "OUT_DIR not set".to_string())
}

fn metallib_name(config: &MetalSourceSet, platform: MetalPlatform) -> String {
    format!("{}{}.metallib", config.library_name, platform.suffix())
}

fn run_command(cmd: &mut Command, action: &str) -> Result<(), String> {
    let output = cmd
        .output()
        .map_err(|err| format!("{action} failed to start: {err}"))?;
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        return Err(format!(
            "{action} failed with status: {}\nstderr:\n{stderr}",
            output.status
        ));
    }

    // Detect air-lld version-mismatch warnings. Xcode 26 betas had a bug
    // where air-lld would warn:
    //   "ignoring file '...air', file AIR version (2.6) is bigger
    //    than the one of the target being linked (2.0)"
    // and silently produce a ~92-byte empty metallib (exit code 0).
    if stderr.contains("air-lld: warning: ignoring file") {
        return Err(format!(
            "air-lld discarded .air files due to AIR version mismatch while {action}.\n\
             This typically means the metallib link step is using a lower Metal \
             standard than the compile step, often caused by \
             MACOSX_DEPLOYMENT_TARGET being set to an old value.\n\
             Try:\n\
             1. Clean the build cache for this crate under target/*/build/\n\
             2. Set {PRECOMPILE_ENV}=0 to skip precompilation (Metal shaders \
                will be compiled at runtime).\n\n\
             air-lld stderr:\n{stderr}"
        ));
    }

    for line in stderr.lines() {
        println!("cargo:warning=[{action}] {line}");
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn runtime_prelude() -> String {
    [
        "#include <metal_stdlib>",
        "#include <metal_common>",
        "#include <metal_math>",
        "#include <metal_integer>",
        "#include <metal_simdgroup>",
        "#include <metal_simdgroup_matrix>",
        "",
        "using namespace metal;",
        "",
    ]
    .join("\n")
}

#[cfg(feature = "metal")]
fn source_map(config: &MetalSourceSet) -> Result<HashMap<&'static str, &'static str>, String> {
    let mut file_system = HashMap::new();
    for source in config.embedded_sources {
        if file_system.insert(source.name, source.source).is_some() {
            return Err(format!("Duplicate embedded Metal source `{}`", source.name));
        }
    }
    Ok(file_system)
}

#[cfg(feature = "metal")]
fn validate_embedded_sources(
    config: &MetalSourceSet,
    file_system: &HashMap<&'static str, &'static str>,
) -> Result<(), String> {
    for source in config
        .metal_sources
        .iter()
        .chain(config.header_sources)
        .chain(config.include_only_sources)
    {
        let file = metal_file_name(source);
        if !file_system.contains_key(file.as_str()) {
            return Err(format!("Missing embedded Metal source `{file}`"));
        }
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn preprocess_includes(
    content: &str,
    current_file: &str,
    file_system: &HashMap<&'static str, &'static str>,
    included_files: &mut HashSet<String>,
    include_stack: &mut Vec<String>,
) -> Result<String, String> {
    if include_stack.iter().any(|file| file == current_file) {
        return Err(format!(
            "Circular include detected: {} -> {}",
            include_stack.join(" -> "),
            current_file
        ));
    }

    include_stack.push(current_file.to_string());

    let mut result = String::new();
    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("#include") {
            if let Some(include_file) = quoted_include(trimmed) {
                let included_content = file_system.get(include_file).ok_or_else(|| {
                    format!("Unknown local Metal include `{include_file}` from `{current_file}`")
                })?;
                if included_files.insert(include_file.to_string()) {
                    let processed = preprocess_includes(
                        included_content,
                        include_file,
                        file_system,
                        included_files,
                        include_stack,
                    )?;
                    result.push_str(&format!("\n// ===== Start of {include_file} =====\n"));
                    result.push_str(&processed);
                    result.push_str(&format!("\n// ===== End of {include_file} =====\n"));
                }
                continue;
            }
            result.push_str(line);
            result.push('\n');
            continue;
        }

        if trimmed == "#pragma once" {
            continue;
        }

        if line.ends_with("\\ ") || line.ends_with("\\\t") {
            let cleaned = line.trim_end();
            let without_backslash = cleaned.trim_end_matches('\\');
            result.push_str(without_backslash);
            result.push_str(" \\");
        } else {
            result.push_str(line);
        }
        result.push('\n');
    }

    include_stack.pop();
    Ok(result)
}

#[cfg(feature = "metal")]
fn quoted_include(line: &str) -> Option<&str> {
    let start = line.find('"')?;
    let end = line[start + 1..].find('"')?;
    Some(&line[start + 1..start + 1 + end])
}

#[cfg(feature = "metal")]
fn metal_file_name(source: &str) -> String {
    format!("{source}.metal")
}
