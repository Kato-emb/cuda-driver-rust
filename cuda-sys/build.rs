#[cfg(feature = "ffi")]
use std::path::PathBuf;

fn main() {
    let cuda_home = std::env::var("CUDA_HOME").unwrap_or("/usr/local/cuda".to_string());
    let cuda_lib = std::env::var("CUDA_LIB").unwrap_or(format!("{cuda_home}/lib64"));
    println!("cargo:rustc-link-search=native={cuda_lib}");

    #[cfg(feature = "ffi")]
    {
        let cuda_include = std::env::var("CUDA_INCLUDE").unwrap_or(format!("{cuda_home}/include"));
        cuda_driver_api(&cuda_include);
        println!("cargo:rustc-link-lib=dylib=cuda");
    }
}

#[cfg(feature = "ffi")]
fn cuda_driver_api(cuda_include: &str) {
    let mut builder = bindgen_base();
    builder = builder
        .header(format!("{}/cuda.h", cuda_include))
        .clang_arg(format!("-I{}", cuda_include))
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

#[cfg(feature = "ffi")]
fn root_dir() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("src")
}

#[cfg(feature = "ffi")]
fn bindgen_base() -> bindgen::Builder {
    #[derive(Debug)]
    struct NppFfiCallbacks;

    impl bindgen::callbacks::ParseCallbacks for NppFfiCallbacks {
        fn will_parse_macro(&self, _name: &str) -> bindgen::callbacks::MacroParsingBehavior {
            bindgen::callbacks::MacroParsingBehavior::Ignore
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
