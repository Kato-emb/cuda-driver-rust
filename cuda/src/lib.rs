use error::CudaResult;

pub mod context;
pub mod device;
pub mod error;

pub mod raw;

pub static CUDA_INITIALIZED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Initializes the CUDA driver API.
pub fn init() {
    use crate::raw::init;

    match unsafe { init::init(init::InitFlags::_ZERO) } {
        Ok(_) => {
            CUDA_INITIALIZED.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        Err(e) => {
            panic!("CUDA could not initialized: {:?}", e);
        }
    }
}

#[macro_export]
macro_rules! assert_initialized {
    () => {
        if !crate::CUDA_INITIALIZED.load(std::sync::atomic::Ordering::SeqCst) {
            panic!("CUDA Driver has not been initialized. Call `cuda::init()` first.");
        }
    };
}

#[macro_export]
macro_rules! skip_assert_initialized {
    () => {};
}

pub fn check_api_version() -> CudaResult<i32> {
    unsafe { raw::version::get_version() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_initialization() {
        init();
        assert!(CUDA_INITIALIZED.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn test_cuda_version() {
        let version = check_api_version().unwrap();
        assert!(version > 0, "CUDA version should be greater than 0");
    }
}
