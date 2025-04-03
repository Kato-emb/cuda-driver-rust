use error::CudaResult;

pub mod context;
pub mod device;
pub mod error;
pub mod memory;

pub mod raw;

pub static CUDA_INITIALIZED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn initialized() {
    use crate::raw::init;

    if unsafe { init::init(init::InitFlags::_ZERO) }.is_err() {
        panic!("CUDA not initialized");
    } else {
        CUDA_INITIALIZED.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

#[macro_export]
macro_rules! assert_initialized {
    () => {
        if !crate::CUDA_INITIALIZED.load(std::sync::atomic::Ordering::SeqCst) {
            $crate::initialized();
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
    fn test_cuda_version() {
        let version = check_api_version().unwrap();
        assert!(version > 0, "CUDA version should be greater than 0");
    }
}
