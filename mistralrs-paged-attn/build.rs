use anyhow::Result;

#[cfg(all(feature = "cuda", target_family = "unix"))]
const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn main() -> Result<()> {
    use std::path::PathBuf;

    // Declare expected cfg values for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_fp8)");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda/pagedattention.cuh");
    println!("cargo:rerun-if-changed=src/cuda/copy_blocks_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/reshape_and_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/concat_and_cache_mla_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/gather_mla_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/gather_kv_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer_mla_decode.cu");
    println!("cargo:rerun-if-changed=src/cuda/update_kvscales.cu");
    println!("cargo:rerun-if-changed=src/cuda/flash_attn_sinks.cu");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/cp_async.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/exception.h");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/fastdiv.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/layout.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/math.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/page.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/pos_enc.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/utils.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/vec_dtypes.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/cascade.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/decode.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/default_decode_params.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/state.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/variant_helper.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/variants.cuh");

    let mut builder = cudaforge::KernelBuilder::new()
        .source_glob("src/cuda/*.cu")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("--verbose")
        .arg("--compiler-options")
        .arg("-fPIC");

    let compute_cap = builder.get_compute_cap().unwrap_or(80);
    // Enable FP8 if compute capability >= 8.0 (Ampere and newer)
    let using_fp8 = if compute_cap >= 80 {
        builder = builder.arg("-DENABLE_FP8");
        true
    } else {
        false
    };

    // https://github.com/EricLBuehler/mistral.rs/issues/286
    if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
        builder = builder.arg("--compiler-options");
        builder = builder.arg(cuda_nvcc_flags_env);
    }
    println!("cargo:info={builder:?}");

    let target = std::env::var("TARGET").unwrap();
    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    // https://github.com/EricLBuehler/mistral.rs/issues/588
    let out_file = if target.contains("msvc") {
        // Windows case
        build_dir.join("mistralrspagedattention.lib")
    } else {
        build_dir.join("libmistralrspagedattention.a")
    };
    builder
        .build_lib(out_file)
        .expect("Build paged attention lib failed!");

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=mistralrspagedattention");
    println!("cargo:rustc-link-lib=dylib=cudart");

    if using_fp8 {
        println!("cargo:rustc-cfg=has_fp8");
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn main() -> Result<(), String> {
    use std::path::PathBuf;
    use std::process::Command;
    use std::{env, str};

    // Declare expected cfg values for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_fp8)");

    const METAL_SOURCES: [&str; 5] = [
        "copy_blocks",
        "pagedattention",
        "reshape_and_cache",
        "kv_scale_update",
        "gather_kv_cache",
    ];
    for src in METAL_SOURCES {
        println!("cargo::rerun-if-changed=src/metal/kernels/{src}.metal");
    }
    println!("cargo::rerun-if-changed=src/metal/kernels/utils.metal");
    println!("cargo::rerun-if-changed=src/metal/kernels/float8.metal");
    println!("cargo::rerun-if-changed=build.rs");

    // ── Track Metal toolchain version ────────────────────────────────
    //
    // Cargo only re-runs build scripts when files listed in
    // rerun-if-changed are modified.  It does NOT track the Xcode /
    // Metal compiler version.  If the user upgrades Xcode (e.g. from a
    // beta with an air-lld bug to a fixed release), the old broken
    // .metallib files stay cached and get silently baked into the
    // binary via include_bytes!.
    //
    // We query `xcrun metal --version` and write the output to a
    // stamp file in OUT_DIR.  If the version string changes, the stamp
    // file changes, and Cargo will re-run the build script.
    {
        let out_dir =
            PathBuf::from(std::env::var("OUT_DIR").map_err(|_| "OUT_DIR not set")?);
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
            for name in [
                "mistralrs_paged_attention.metallib",
                "mistralrs_paged_attention_ios.metallib",
                "mistralrs_paged_attention_tvos.metallib",
            ] {
                let _ = std::fs::remove_file(out_dir.join(name));
            }
            println!(
                "cargo:warning=Metal toolchain version changed — recompiling Metal kernels"
            );
        }
    }

    // Check if precompilation should be skipped
    // https://github.com/EricLBuehler/mistral.rs/pull/1311#issuecomment-3001309885
    println!("cargo:rerun-if-env-changed=MISTRALRS_METAL_PRECOMPILE");
    let skip_precompile = env::var("MISTRALRS_METAL_PRECOMPILE")
        .map(|v| v == "0" || v.to_lowercase() == "false")
        .unwrap_or(false);

    if skip_precompile {
        println!(
            "cargo:warning=Skipping Metal kernel precompilation (MISTRALRS_METAL_PRECOMPILE=0)"
        );
        // Write a dummy metallib file to satisfy the include_bytes! macro
        let out_dir = PathBuf::from(std::env::var("OUT_DIR").map_err(|_| "OUT_DIR not set")?);
        std::fs::write(out_dir.join("mistralrs_paged_attention.metallib"), []).unwrap();
        std::fs::write(out_dir.join("mistralrs_paged_attention_ios.metallib"), []).unwrap();
        std::fs::write(out_dir.join("mistralrs_paged_attention_tvos.metallib"), []).unwrap();
        return Ok(());
    }

    enum Platform {
        MacOS,
        Ios,
        TvOS,
    }

    impl Platform {
        fn sdk(&self) -> &str {
            match self {
                Platform::MacOS => "macosx",
                Platform::Ios => "iphoneos",
                Platform::TvOS => "appletvos",
            }
        }

        fn metal_std(&self) -> &str {
            // Use Metal 3.1 unified standard for all platforms.
            // This enables native bfloat16 support (__HAVE_BFLOAT__) which is
            // required for PagedAttention kernels with bf16 models (e.g. Qwen3).
            // Without Metal 3.1, the emulated _MLX_BFloat16 struct is used instead,
            // which can fail on some Metal compiler/runtime combinations.
            // https://github.com/EricLBuehler/mistral.rs/issues/1844
            //
            // Note: Metal 3.1 MSL compiles on all Apple Silicon. The native bfloat
            // type is used on M3+ GPUs; older GPUs use the emulated fallback path
            // in utils.metal, which is still correctly compiled with MSL 3.1.
            match self {
                Platform::MacOS | Platform::Ios | Platform::TvOS => "metal3.1",
            }
        }
    }

    fn compile(platform: Platform) -> Result<(), String> {
        let current_dir = env::current_dir().expect("Failed to get current directory");
        let out_dir = PathBuf::from(std::env::var("OUT_DIR").map_err(|_| "OUT_DIR not set")?);
        let working_directory = out_dir.to_string_lossy().to_string();
        let sources = current_dir.join("src").join("metal").join("kernels");

        // Compile metal to air
        let mut compile_air_cmd = Command::new("xcrun");
        compile_air_cmd
            .arg("--sdk")
            .arg(platform.sdk())
            .arg("metal")
            .arg(format!("-std={}", platform.metal_std()))
            .arg(format!("-working-directory={working_directory}"))
            .arg("-Wall")
            .arg("-Wextra")
            .arg("-O3")
            .arg("-c")
            .arg("-w");
        for metal_file in METAL_SOURCES {
            compile_air_cmd.arg(sources.join(format!("{metal_file}.metal")));
        }
        compile_air_cmd.arg(sources.join("utils.metal"));
        compile_air_cmd.arg(sources.join("float8.metal"));

        let output = compile_air_cmd
            .stderr(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .output()
            .expect("Failed to compile metal -> air");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!(
                "Compiling metal -> air failed for {} SDK.\n\
                 Exit status: {}\nstderr:\n{}",
                platform.sdk(),
                output.status,
                stderr,
            );
        }

        let stderr_str = String::from_utf8_lossy(&output.stderr);
        if !stderr_str.is_empty() {
            for line in stderr_str.lines() {
                println!("cargo:warning=[metal->air {}] {}", platform.sdk(), line);
            }
        }

        // ── Compile air to metallib ──────────────────────────────────
        //
        // CRITICAL: Pass `--sdk` and `-std` here too!
        //
        // Without `-std=<metal_version>`, `xcrun metal` (the linker)
        // falls back to the Metal version implied by
        // MACOSX_DEPLOYMENT_TARGET.  When Cargo cross-compiles for
        // x86_64-apple-darwin it sets MACOSX_DEPLOYMENT_TARGET=10.12
        // (or 10.13), which implies Metal 2.0 / AIR 2.0.  The .air
        // files were compiled with -std=metal3.0 (AIR 2.5), so
        // air-lld silently rejects every .air input and produces a
        // ~92-byte empty metallib (exit code 0).  At runtime, every
        // Metal function cannot be found.
        let lib_name = match platform {
            Platform::MacOS => "mistralrs_paged_attention.metallib",
            Platform::Ios => "mistralrs_paged_attention_ios.metallib",
            Platform::TvOS => "mistralrs_paged_attention_tvos.metallib",
        };
        let metallib = out_dir.join(lib_name);
        let mut compile_metallib_cmd = Command::new("xcrun");
        compile_metallib_cmd
            .arg("--sdk")
            .arg(platform.sdk())
            .arg("metal")
            .arg(format!("-std={}", platform.metal_std()))
            .arg("-o")
            .arg(&metallib);

        for metal_file in METAL_SOURCES {
            compile_metallib_cmd.arg(out_dir.join(format!("{metal_file}.air")));
        }
        compile_metallib_cmd.arg(out_dir.join("utils.air"));
        compile_metallib_cmd.arg(out_dir.join("float8.air"));

        let output = compile_metallib_cmd
            .stderr(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .output()
            .expect("Failed to compile air -> metallib");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!(
                "Compiling air -> metallib failed for {} SDK.\n\
                 Exit status: {}\nstderr:\n{}",
                platform.sdk(),
                output.status,
                stderr,
            );
        }

        // Detect air-lld version-mismatch warnings.
        // Xcode 26 betas had a bug where air-lld would warn:
        //   "ignoring file '...air', file AIR version (2.6) is bigger
        //    than the one of the target being linked (2.0)"
        // and silently produce a ~92-byte empty metallib (exit code 0).
        let stderr_str = String::from_utf8_lossy(&output.stderr);
        if stderr_str.contains("air-lld: warning: ignoring file") {
            panic!(
                "air-lld discarded .air files due to AIR version mismatch \
                 while linking {lib_name} ({} SDK).\n\
                 This typically means the metallib link step is using a \
                 lower Metal standard than the compile step, often caused \
                 by MACOSX_DEPLOYMENT_TARGET being set to an old value.\n\
                 Try:\n\
                 1. Clean the build cache: rm -rf target/*/build/mistralrs-paged-attn-*\n\
                 2. Set MISTRALRS_METAL_PRECOMPILE=0 to skip precompilation \
                    (Metal shaders will be compiled at runtime).\n\n\
                 air-lld stderr:\n{stderr_str}",
                platform.sdk(),
            );
        }

        if !stderr_str.is_empty() {
            for line in stderr_str.lines() {
                println!(
                    "cargo:warning=[air->metallib {}] {}",
                    platform.sdk(),
                    line
                );
            }
        }

        // Validate metallib is not empty.
        // A valid metallib with 5+ kernel sources should be well over 4 KB.
        // An empty metallib header is ~92 bytes.
        const MIN_METALLIB_SIZE: u64 = 4096;
        match std::fs::metadata(&metallib) {
            Ok(meta) => {
                if meta.len() < MIN_METALLIB_SIZE {
                    panic!(
                        "Generated {lib_name} is only {} bytes (expected >= {MIN_METALLIB_SIZE}).\n\
                         The metallib is likely empty — all .air inputs may have been \
                         silently discarded by air-lld due to AIR version mismatch.\n\
                         This happens when MACOSX_DEPLOYMENT_TARGET is set to an old \
                         value (e.g. 10.13) which implies Metal 2.0 / AIR 2.0, but \
                         the .air files were compiled with a newer -std.\n\
                         Try: rm -rf target/*/build/mistralrs-paged-attn-* && cargo build\n\
                         Or update your Xcode installation.",
                        meta.len(),
                    );
                }
            }
            Err(e) => {
                panic!("Cannot stat generated {lib_name}: {e}");
            }
        }

        Ok(())
    }

    compile(Platform::MacOS)?;
    compile(Platform::Ios)?;
    compile(Platform::TvOS)?;

    Ok(())
}

#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")))]
fn main() -> Result<()> {
    // Declare expected cfg values for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_fp8)");
    Ok(())
}
