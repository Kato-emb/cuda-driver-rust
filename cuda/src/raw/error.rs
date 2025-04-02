use crate::{
    error::{CudaResult, ToResult},
    wrap_sys_handle,
};
use cuda_sys::ffi as sys;

wrap_sys_handle!(ErrorCode, sys::CUresult);

pub unsafe fn get_error_string(error: ErrorCode) -> CudaResult<String> {
    let mut ptr = std::ptr::null();
    unsafe { sys::cuGetErrorString(error.0, &mut ptr) }.to_result()?;

    Ok(unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned())
}

pub unsafe fn get_error_name(error: ErrorCode) -> CudaResult<String> {
    let mut ptr = std::ptr::null();
    unsafe { sys::cuGetErrorName(error.0, &mut ptr) }.to_result()?;

    Ok(unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned())
}
