use cuda_sys::ffi as sys;

use crate::{
    error::{CudaResult, ToResult},
    wrap_sys_handle,
};

pub mod array;
pub mod device;
pub mod ordered;
pub mod pinned;
pub mod pool;
pub mod unified;

wrap_sys_handle!(DevicePtr, sys::CUdeviceptr);

pub unsafe fn free(ptr: DevicePtr) -> CudaResult<()> {
    unsafe { sys::cuMemFree_v2(ptr.0) }.to_result()
}

pub unsafe fn get_info() -> CudaResult<(usize, usize)> {
    let mut free = 0;
    let mut total = 0;
    unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) }.to_result()?;

    Ok((free as usize, total as usize))
}
