use std::path::PathBuf;

fn main() {
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") => {
            let cuda_home =
                PathBuf::from(std::env::var("CUDA_HOME").unwrap_or("/usr/local/cuda".to_string()));

            let cuda_lib = std::env::var("CUDA_LIB")
                .map(|lib| PathBuf::from(lib))
                .unwrap_or(cuda_home.join("lib64"));

            println!("cargo:rustc-link-search=native={}", cuda_lib.display());

            #[cfg(feature = "ffi")]
            {
                let cuda_include = std::env::var("CUDA_INCLUDE")
                    .map(|include| PathBuf::from(include))
                    .unwrap_or(cuda_home.join("include"));
                ffi_ctx::cuda_driver_api(&cuda_include);
            }
        }
        Ok("windows") => {
            let cuda_home = PathBuf::from(std::env::var("CUDA_PATH").unwrap_or(
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8".to_string(),
            ));

            let cuda_lib = std::env::var("CUDA_LIB")
                .map(|lib| PathBuf::from(lib))
                .unwrap_or(cuda_home.join("lib\\x64"));

            println!("cargo:rustc-link-search=native={}", cuda_lib.display());

            #[cfg(feature = "ffi")]
            {
                let cuda_include = std::env::var("CUDA_INCLUDE")
                    .map(|include| PathBuf::from(include))
                    .unwrap_or(cuda_home.join("include"));
                ffi_ctx::cuda_driver_api(&cuda_include);
            }
        }
        _ => {
            panic!("Unsupported target OS")
        }
    };

    println!("cargo:rustc-link-lib=dylib=cuda");
}

#[cfg(feature = "ffi")]
mod ffi_ctx {
    use std::{path::PathBuf, sync::LazyLock};

    use regex::Regex;

    static PARAM_TAG: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(\\|@)p\s*([\*\w][\w\d_]*)").unwrap());
    static PARAM_BLOCK: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?m)^(?:\s*)[\\@]param\s+(\w+)\s+(.*)$").unwrap());
    static RETURN_BLOCK: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?m)^(?:\s*)[\\@]returns?\s+(.*)$").unwrap());
    static TYPE_REF: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"::([A-Za-z_][A-Za-z0-9_]*)").unwrap());
    static NOTE_BLOCK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\\note\w*").unwrap());

    fn note_replace(comment: &str) -> String {
        let mut result = String::new();
        let mut inserted = false;

        for line in comment.lines() {
            if NOTE_BLOCK.is_match(line) && !inserted {
                result.push_str("### Note:\n");
                inserted = true;
            }

            result.push_str(line);
            result.push('\n');
        }

        result
    }

    pub fn cuda_driver_api(cuda_include: &PathBuf) {
        let mut builder = bindgen_base();
        builder = builder
            .header(format!("{}", cuda_include.join("cuda.h").display()))
            .clang_arg(format!("-I{}", cuda_include.display()))
            .allowlist_type("CU.*")
            .allowlist_type("cuuint(32|64)_t")
            .allowlist_type("cudaError_enum")
            .allowlist_type("cu.*Complex$")
            .allowlist_type("cuda.*")
            .allowlist_type("libraryPropertyType.*")
            .allowlist_var("^CU.*")
            .allowlist_function("^cu.*");

        let file_path = root_dir().join("ffi.rs");

        builder
            .generate()
            .expect("Unable to generate bindings")
            .write_to_file(&file_path)
            .expect("Couldn't write bindings!");
    }

    pub fn root_dir() -> PathBuf {
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("src")
    }

    pub fn bindgen_base() -> bindgen::Builder {
        #[derive(Debug)]
        struct NppFfiCallbacks;

        impl bindgen::callbacks::ParseCallbacks for NppFfiCallbacks {
            fn will_parse_macro(&self, _name: &str) -> bindgen::callbacks::MacroParsingBehavior {
                bindgen::callbacks::MacroParsingBehavior::Ignore
            }

            fn process_comment(&self, comment: &str) -> Option<String> {
                if comment.is_empty() {
                    return None;
                }

                let mut result = comment.to_string();

                result = PARAM_BLOCK.replace_all(&result, "#### $1:\n$2").to_string();
                result = RETURN_BLOCK
                    .replace_all(&result, "### Returns:\n$1")
                    .to_string();
                result = PARAM_TAG.replace_all(&result, "`$2`").to_string();
                result = TYPE_REF
                    .replace_all(&result, "[$1](crate::ffi::$1)")
                    .to_string();

                result = note_replace(&result);
                result = result.replace("\\brief ", "");
                result = result.replace("\\note ", "### Note:\n");
                result = result.replace("\\sa", "### See also:\n");
                result = result.replace("\\notefnerr", "Note that this function may also return error codes from previous, asynchronous launches.\n");
                result = result.replace("\\note_async", "This function exhibits [asynchronous](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html#api-sync-behavior__memcpy-async) behavior for most use cases.\n");
                result = result.replace(
                    "\\note_null_stream",
                    "This function uses standard [default stream](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream) semantics.\n",
                );
                result = result.replace("\\note_memcpy", "Memory regions requested must be either entirely registered with CUDA, or in the case of host pageable transfers, not registered at all. Memory regions spanning over allocations that are both registered and not registered with CUDA are not supported and will return CUDA_ERROR_INVALID_VALUE.\n");
                result = result.replace(
                    "\\note_memset",
                    "See also [memset synchronization details](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html#api-sync-behavior__memset).\n",
                );
                result = result.replace(
                    "\\note_sync",
                    "This function exhibits [synchronous](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync) behavior for most use cases.\n",
                );
                result = result.replace("\\note:", "\n");

                Some(result)
            }
        }

        bindgen::builder()
            .parse_callbacks(Box::new(NppFfiCallbacks))
            .rust_edition(bindgen::RustEdition::Edition2021)
            .ctypes_prefix("::std::ffi")
            .default_enum_style(bindgen::EnumVariation::Rust {
                non_exhaustive: false,
            })
            .layout_tests(false)
            .allowlist_recursively(false)
            .wrap_static_fns(true)
    }
}
