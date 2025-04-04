use crate::error::{CudaResult, ToResult};

/// Returns the latest CUDA version supported by driver.
/// ## Returns
/// * `driverVersion` - The version of the CUDA driver. (ex. CUDA 9.2 would be represented by 9020)
pub unsafe fn get_version() -> CudaResult<i32> {
    let mut version = 0;
    unsafe { cuda_sys::ffi::cuDriverGetVersion(&mut version) }.to_result()?;
    Ok(version)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_raw_get_version() {
        let result = unsafe { get_version() };
        assert!(
            result.is_ok(),
            "CUDA version retrieval failed: {:?}",
            result
        );
    }
}
