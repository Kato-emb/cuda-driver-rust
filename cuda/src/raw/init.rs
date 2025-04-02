use cuda_sys::ffi as sys;

use crate::error::{CudaResult, ToResult};

bitflags::bitflags! {
    pub struct InitFlags: u32 {
        const _ZERO = 0;
    }
}

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
