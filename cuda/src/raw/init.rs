use cuda_sys::ffi as sys;

use crate::error::{CudaResult, ToResult};

bitflags::bitflags! {
    pub struct InitFlags: u32 {
        const _ZERO = 0;
    }
}

/// Initialize the CUDA driver API Initializes the driver API and must be called before any other function from the driver API in the current process.
/// ## Parameters
/// - `flags`: Initialization flag for CUDA. Currently, only `0` is supported.
pub unsafe fn init(flags: InitFlags) -> CudaResult<()> {
    unsafe { sys::cuInit(flags.bits()) }.to_result()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_raw_init() {
        let result = unsafe { init(InitFlags::_ZERO) };
        assert!(result.is_ok(), "CUDA initialization failed: {:?}", result);
    }
}
