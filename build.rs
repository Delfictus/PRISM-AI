// build.rs - Build configuration for PRISM-AI with CUDA and MLIR support

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/kernels/");

    // Detect features
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    let mlir_enabled = env::var("CARGO_FEATURE_MLIR").is_ok();

    if cuda_enabled {
        build_cuda_kernels();
    }

    if mlir_enabled {
        configure_mlir();
    }
}

fn build_cuda_kernels() {
    println!("cargo:warning=Building CUDA kernels...");

    // Find CUDA installation
    let cuda_path = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| {
            // Try common locations
            if Path::new("/usr/local/cuda").exists() {
                "/usr/local/cuda".to_string()
            } else if Path::new("/usr/local/cuda-12.3").exists() {
                "/usr/local/cuda-12.3".to_string()
            } else {
                panic!("CUDA not found! Set CUDA_HOME or install CUDA toolkit");
            }
        });

    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cusparse");
    println!("cargo:rustc-link-lib=cufft");
    println!("cargo:rustc-link-lib=curand");
    println!("cargo:rustc-link-lib=cusolver");

    // Build CUDA kernels if they exist
    let kernel_dir = PathBuf::from("src/kernels");
    if kernel_dir.exists() {
        compile_cuda_files(&kernel_dir, &cuda_path);
    }
}

fn compile_cuda_files(kernel_dir: &Path, cuda_path: &str) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Find all .cu files
    let cu_files: Vec<_> = std::fs::read_dir(kernel_dir)
        .expect("Failed to read kernels directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()? == "cu" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if cu_files.is_empty() {
        println!("cargo:warning=No CUDA kernel files found");
        return;
    }

    // Compile each CUDA file
    for cu_file in cu_files {
        let file_name = cu_file.file_stem().unwrap().to_str().unwrap();
        let ptx_file = out_dir.join(format!("{}.ptx", file_name));
        let obj_file = out_dir.join(format!("{}.o", file_name));

        println!("cargo:warning=Compiling CUDA kernel: {:?}", cu_file);

        // Detect GPU architecture
        let gpu_arch = detect_gpu_arch().unwrap_or_else(|| "sm_86".to_string());

        // Compile to PTX for runtime loading
        let ptx_status = Command::new(format!("{}/bin/nvcc", cuda_path))
            .args(&[
                "-ptx",
                "-arch", &gpu_arch,
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "-I", "src/kernels",
                cu_file.to_str().unwrap(),
                "-o", ptx_file.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to execute nvcc");

        if !ptx_status.success() {
            panic!("Failed to compile CUDA kernel: {:?}", cu_file);
        }

        // Also compile to object file for static linking
        let obj_status = Command::new(format!("{}/bin/nvcc", cuda_path))
            .args(&[
                "-c",
                "-arch", &gpu_arch,
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "--relocatable-device-code=true",
                "-I", "src/kernels",
                cu_file.to_str().unwrap(),
                "-o", obj_file.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to execute nvcc");

        if !obj_status.success() {
            panic!("Failed to compile CUDA kernel to object: {:?}", cu_file);
        }

        // Link the object file
        println!("cargo:rustc-link-arg={}", obj_file.display());
    }

    // Copy PTX files to target directory for runtime loading
    let target_dir = PathBuf::from("target/ptx");
    std::fs::create_dir_all(&target_dir).expect("Failed to create PTX directory");

    for entry in std::fs::read_dir(&out_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("ptx") {
            let dest = target_dir.join(path.file_name().unwrap());
            std::fs::copy(&path, &dest).expect("Failed to copy PTX file");
            println!("cargo:warning=PTX file copied to: {:?}", dest);
        }
    }
}

fn detect_gpu_arch() -> Option<String> {
    // Try to detect GPU compute capability
    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;

    if output.status.success() {
        let cap = String::from_utf8_lossy(&output.stdout)
            .trim()
            .replace('.', "");

        Some(format!("sm_{}", cap))
    } else {
        None
    }
}

fn configure_mlir() {
    println!("cargo:warning=Configuring MLIR...");

    // Find MLIR installation
    let mlir_dir = env::var("MLIR_DIR")
        .unwrap_or_else(|_| "/usr/local".to_string());

    let llvm_dir = env::var("LLVM_DIR")
        .unwrap_or_else(|_| mlir_dir.clone());

    // Compile MLIR runtime C++ file if it exists
    compile_mlir_runtime(&mlir_dir, &llvm_dir);

    // Add MLIR/LLVM libraries
    println!("cargo:rustc-link-search=native={}/lib", mlir_dir);
    println!("cargo:rustc-link-search=native={}/lib", llvm_dir);

    // Link MLIR libraries
    let mlir_libs = [
        "MLIR",
        "MLIRAnalysis",
        "MLIRDialect",
        "MLIRIR",
        "MLIRParser",
        "MLIRPass",
        "MLIRTransforms",
        "MLIRSupport",
        "MLIRTargetLLVMIRExport",
        "MLIRExecutionEngine",
        "MLIRJitRunner",
        "MLIRGPUDialect",
        "MLIRNVVMDialect",
        "MLIRGPUToNVVMTransforms",
    ];

    for lib in mlir_libs.iter() {
        println!("cargo:rustc-link-lib={}", lib);
    }

    // Link LLVM libraries
    let llvm_libs = [
        "LLVMCore",
        "LLVMSupport",
        "LLVMX86CodeGen",
        "LLVMX86AsmParser",
        "LLVMX86Desc",
        "LLVMX86Info",
        "LLVMNVPTXCodeGen",
        "LLVMNVPTXDesc",
        "LLVMNVPTXInfo",
    ];

    for lib in llvm_libs.iter() {
        println!("cargo:rustc-link-lib={}", lib);
    }

    // Generate bindings if bindgen is available
    #[cfg(feature = "bindgen")]
    generate_mlir_bindings(&mlir_dir);
}

fn compile_mlir_runtime(mlir_dir: &str, llvm_dir: &str) {
    let runtime_cpp = PathBuf::from("src/mlir_runtime.cpp");
    if !runtime_cpp.exists() {
        println!("cargo:warning=MLIR runtime C++ file not found, skipping compilation");
        return;
    }

    println!("cargo:warning=Compiling MLIR runtime...");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj_file = out_dir.join("mlir_runtime.o");

    // Compile C++ file
    let status = Command::new("clang++")
        .args(&[
            "-c",
            "-std=c++17",
            "-O3",
            "-fPIC",
            &format!("-I{}/include", mlir_dir),
            &format!("-I{}/include", llvm_dir),
            "-I/usr/local/cuda/include",
            runtime_cpp.to_str().unwrap(),
            "-o", obj_file.to_str().unwrap(),
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-link-arg={}", obj_file.display());
            println!("cargo:warning=MLIR runtime compiled successfully");
        },
        Ok(_) => {
            println!("cargo:warning=MLIR runtime compilation failed, continuing without it");
        },
        Err(e) => {
            println!("cargo:warning=clang++ not found: {}, skipping MLIR runtime", e);
        }
    }
}

#[cfg(feature = "bindgen")]
fn generate_mlir_bindings(mlir_dir: &str) {
    use bindgen::Builder;

    let bindings = Builder::default()
        .header(format!("{}/include/mlir-c/IR.h", mlir_dir))
        .header(format!("{}/include/mlir-c/Pass.h", mlir_dir))
        .header(format!("{}/include/mlir-c/Dialect/GPU.h", mlir_dir))
        .clang_arg(format!("-I{}/include", mlir_dir))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate MLIR bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("mlir_bindings.rs"))
        .expect("Couldn't write bindings");
}