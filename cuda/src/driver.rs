use std::fmt;

use crate::error::CudaResult;

pub mod context;
pub mod device;
pub mod event;
pub mod ipc;
pub mod memory;
pub mod stream;

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

/// This macro initializes the thread context with default setting if it is not already initialized.
#[macro_export]
macro_rules! assert_initialized {
    () => {
        if !crate::driver::CUDA_INITIALIZED.load(std::sync::atomic::Ordering::SeqCst) {
            panic!("CUDA Driver has not been initialized. Call `cuda::driver::init()` first.");
        }
    };
}

/// This macro skips the assertion for CUDA initialization.
#[macro_export]
macro_rules! skip_assert_initialized {
    () => {};
}

/// CUDA driver API version
pub struct DriverVersion(i32);

impl fmt::Debug for DriverVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("driverVersion")
            .field("raw", &self.0)
            .finish()
    }
}

impl fmt::Display for DriverVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major(), self.minor())
    }
}

impl DriverVersion {
    pub fn new() -> CudaResult<Self> {
        let version = unsafe { crate::raw::version::get_version() }?;
        Ok(Self(version))
    }

    /// Return the major version of the CUDA driver.
    pub fn major(&self) -> i32 {
        self.0 / 1000
    }

    /// Return the minor version of the CUDA driver.
    pub fn minor(&self) -> i32 {
        (self.0 % 1000) / 10
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_driver_version() {
        crate::driver::init();
        let version = DriverVersion::new().unwrap();
        assert!(
            version.major() > 0,
            "CUDA driver major version should be greater than 0"
        );
        assert!(
            version.minor() >= 0,
            "CUDA driver minor version should be greater than or equal to 0"
        );

        println!("CUDA Driver Version: {}", version);
    }
}
