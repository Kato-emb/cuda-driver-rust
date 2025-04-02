use cuda_sys::ffi as sys;

use super::DevicePtr;
use crate::error::{CudaResult, ToResult};

pub unsafe fn malloc(bytesize: usize) -> CudaResult<DevicePtr> {
    let mut ptr = 0;
    unsafe { sys::cuMemAlloc_v2(&mut ptr, bytesize) }.to_result()?;

    Ok(DevicePtr(ptr))
}

pub unsafe fn malloc_pitch(
    width: usize,
    height: usize,
    element_size: u32,
) -> CudaResult<(DevicePtr, usize)> {
    let mut pitch = 0;
    let mut ptr = 0;
    unsafe { sys::cuMemAllocPitch_v2(&mut ptr, &mut pitch, width, height, element_size) }
        .to_result()?;

    Ok((DevicePtr(ptr), pitch))
}

pub unsafe fn get_address_range(ptr: DevicePtr) -> CudaResult<(DevicePtr, usize)> {
    let mut base = 0;
    let mut size = 0;
    unsafe { sys::cuMemGetAddressRange_v2(&mut base, &mut size, ptr.0) }.to_result()?;

    Ok((DevicePtr(base), size))
}
