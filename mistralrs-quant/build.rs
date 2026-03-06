#[cfg(feature = "cuda")]
#[allow(unused)]
fn cuda_version_from_build_system() -> (usize, usize) {
    let output = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .expect("Failed to execute `nvcc`");

    if !output.status.success() {
        panic!(
            "`nvcc --version` failed.\nstdout:\n{}\n\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let version_line = stdout.lines().nth(3).unwrap();
    let release_section = version_line.split(", ").nth(1).unwrap();
    let version_number = release_section.split(' ').nth(1).unwrap();

    match version_number {
        "13.1" => (13, 1),
        "13.0" => (13, 0),
        "12.9" => (12, 9),
        "12.8" => (12, 8),
        "12.6" => (12, 6),
        "12.5" => (12, 5),
        "12.4" => (12, 4),
        "12.3" => (12, 3),
        "12.2" => (12, 2),
        "12.1" => (12, 1),
        "12.0" => (12, 0),
        "11.8" => (11, 8),
        "11.7" => (11, 7),
        "11.6" => (11, 6),
        "11.5" => (11, 5),
        "11.4" => (11, 4),
        v => panic!("Unsupported cuda toolkit version: `{v}`. Please raise a github issue."),
    }
}

fn main() -> Result<(), String> {
    // Declare expected cfg values for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_marlin_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_blockwise_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_scalar_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_vector_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_mxfp4_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_mxfp4_wmma_kernels)");

    #[cfg(feature = "cuda")]
    {
        use std::{path::PathBuf, vec};
        const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

        println!("cargo:rerun-if-changed=build.rs");
        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

        let mut builder = cudaforge::KernelBuilder::new()
            .source_glob("kernels/*/*.cu")
            .out_dir(build_dir.clone())
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
        // ======== Handle optional kernel compilation via rustc-cfg flags
        let cc_over_80 = compute_cap >= 80;

        if cc_over_80 {
            println!("cargo:rustc-cfg=has_marlin_kernels");
            println!("cargo:rustc-cfg=has_blockwise_fp8_kernels");
            println!("cargo:rustc-cfg=has_scalar_fp8_kernels");
            println!("cargo:rustc-cfg=has_vector_fp8_kernels");
            // WMMA tensor core MXFP4 kernel (FP16/BF16 WMMA requires SM >= 80)
            println!("cargo:rustc-cfg=has_mxfp4_wmma_kernels");
        }
        // MXFP4 is always enabled with CUDA (uses LUT-based dequantization)
        println!("cargo:rustc-cfg=has_mxfp4_kernels");

        let excluded_files = if cc_over_80 {
            vec!["dummy_*.cu", "*_dummy.cu"]
        } else {
            vec!["marlin_*.cu", "*_fp8.cu", "*_fp8_gemm.cu", "*_wmma.cu"]
        };
        builder = builder.exclude(&excluded_files);

        // https://github.com/EricLBuehler/mistral.rs/issues/286
        if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
            builder = builder.arg("--compiler-options");
            builder = builder.arg(cuda_nvcc_flags_env);
        }

        let target = std::env::var("TARGET").unwrap();
        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        // https://github.com/EricLBuehler/mistral.rs/issues/588
        let out_file = if target.contains("msvc") {
            // Windows case
            build_dir.join("mistralrsquant.lib")
        } else {
            build_dir.join("libmistralrsquant.a")
        };
        builder
            .build_lib(out_file)
            .expect("Build mistral quant lib failed!");
        println!("cargo:rustc-link-search={}", build_dir.display());
        println!("cargo:rustc-link-lib=mistralrsquant");
        println!("cargo:rustc-link-lib=dylib=cudart");

        if target.contains("msvc") {
            // nothing to link to
        } else if target.contains("apple")
            || target.contains("freebsd")
            || target.contains("openbsd")
        {
            println!("cargo:rustc-link-lib=dylib=c++");
        } else if target.contains("android") {
            println!("cargo:rustc-link-lib=dylib=c++_shared");
        } else {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }

        let (major, minor) = cuda_version_from_build_system();
        println!("cargo:rustc-cfg=feature=\"cuda-{major}0{minor}0\"");

        Ok(())
    }

    #[cfg(feature = "metal")]
    {
        use std::path::PathBuf;
        use std::process::Command;
        use std::{env, str};

        const METAL_SOURCES: [&str; 15] = [
            "bitwise",
            "blockwise_fp8",
            "bnb_dequantize",
            "f8q8",
            "fused_glu",
            "hqq_dequantize",
            "hqq_bitpack",
            "mxfp4",
            "quantized",
            "scalar_fp8",
            "scan",
            "sdpa_with_sinks",
            "softmax_with_sinks",
            "sort",
            "copy",
        ];
        const HEADER_SOURCES: [&str; 5] = ["utils", "bf16", "scan_impl", "sort_impl", "copy_impl"];
        // Include-only headers (not compiled directly, just tracked for changes)
        const INCLUDE_ONLY: [&str; 2] = ["float8", "float4"];
        for src in METAL_SOURCES {
            println!("cargo::rerun-if-changed=src/metal_kernels/{src}.metal");
        }
        for src in HEADER_SOURCES {
            println!("cargo::rerun-if-changed=src/metal_kernels/{src}.metal");
        }
        for src in INCLUDE_ONLY {
            println!("cargo::rerun-if-changed=src/metal_kernels/{src}.metal");
        }
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
                // Write the new version stamp.  Because the .metallib files
                // in OUT_DIR will be stale, delete them so that even if Cargo
                // somehow skips the rebuild, include_bytes! will fail at
                // compile time instead of silently embedding empty libs.
                let _ = std::fs::write(&stamp, &current_version);
                for name in [
                    "mistralrs_quant.metallib",
                    "mistralrs_quant_ios.metallib",
                    "mistralrs_quant_tvos.metallib",
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
            std::fs::write(out_dir.join("mistralrs_quant.metallib"), []).unwrap();
            std::fs::write(out_dir.join("mistralrs_quant_ios.metallib"), []).unwrap();
            std::fs::write(out_dir.join("mistralrs_quant_tvos.metallib"), []).unwrap();
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
                // Explicitly set the Metal standard so Xcode 26+ doesn't fall
                // back to a version that is too low for the features we need.
                // https://github.com/EricLBuehler/mistral.rs/issues/1844
                //
                // macOS: Metal 3.1 — M1/M2 (GPU Family 7/8) can run 3.1
                //        metallibs; M3+ (Family 9) has native bfloat.
                // iOS:   Metal 3.0 — A15/A16 (GPU Family 7/8, e.g. iPhone
                //        14/15/16e) only support Metal 3.0. Using 3.1 causes
                //        "Error while loading function" at runtime on these
                //        GPUs. Native bfloat is unavailable but bf16.metal
                //        provides a software emulation fallback. Shaders
                //        already guard bfloat instantiation behind
                //        `#if __METAL_VERSION__ >= 310`.
                // tvOS:  Metal 3.0 — Apple TV 4K (3rd gen) uses A15 = Metal 3.0.
                match self {
                    Platform::MacOS => "metal3.1",
                    Platform::Ios | Platform::TvOS => "metal3.0",
                }
            }
        }

        fn compile(platform: Platform) -> Result<(), String> {
            let current_dir = env::current_dir().expect("Failed to get current directory");
            let out_dir = PathBuf::from(std::env::var("OUT_DIR").map_err(|_| "OUT_DIR not set")?);
            let working_directory = out_dir.to_string_lossy().to_string();
            let sources = current_dir.join("src").join("metal_kernels");

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
            for metal_file in HEADER_SOURCES {
                compile_air_cmd.arg(sources.join(format!("{metal_file}.metal")));
            }

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

            // Print any warnings from the metal compiler so they're visible
            // in `cargo build -vv` output.
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
            // files were compiled with -std=metal3.1 (AIR 2.6) or
            // -std=metal3.0 (AIR 2.5), so air-lld silently rejects every
            // .air input and produces a ~92-byte empty metallib (exit
            // code 0).  At runtime, `fused_glu_float` and every other
            // Metal function cannot be found.
            let lib_name = match platform {
                Platform::MacOS => "mistralrs_quant.metallib",
                Platform::Ios => "mistralrs_quant_ios.metallib",
                Platform::TvOS => "mistralrs_quant_tvos.metallib",
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
            for metal_file in HEADER_SOURCES {
                compile_metallib_cmd.arg(out_dir.join(format!("{metal_file}.air")));
            }

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

            // ── Detect air-lld version-mismatch warnings ─────────────────
            //
            // Xcode 26 betas had a bug where air-lld would warn:
            //   "ignoring file '...air', file AIR version (2.6) is bigger
            //    than the one of the target being linked (2.0)"
            // and silently produce a ~92-byte empty metallib.  The exit
            // code was still 0.  Detect this and fail loudly.
            let stderr_str = String::from_utf8_lossy(&output.stderr);
            if stderr_str.contains("air-lld: warning: ignoring file") {
                panic!(
                    "air-lld discarded .air files due to AIR version mismatch \
                     while linking {lib_name} ({} SDK).\n\
                     This typically means the metallib link step is using a \
                     lower Metal standard than the compile step, often caused \
                     by MACOSX_DEPLOYMENT_TARGET being set to an old value.\n\
                     Try:\n\
                     1. Clean the build cache: rm -rf target/*/build/mistralrs-quant-*\n\
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

            // ── Validate metallib is not empty ───────────────────────────
            //
            // A valid metallib for mistralrs-quant with 15+ kernel sources
            // should be at least ~1 MB.  An empty header is ~92 bytes.
            // If air-lld silently dropped all inputs (version mismatch,
            // etc.) the file will be tiny.  Catch this before it gets
            // baked into the binary via include_bytes!.
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
                             Try: rm -rf target/*/build/mistralrs-quant-* && cargo build\n\
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

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    Ok(())
}
