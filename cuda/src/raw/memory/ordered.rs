use cuda_sys::ffi as sys;

use crate::{
    error::{CudaResult, ToResult},
    raw::stream::Stream,
};

use super::{DevicePtr, pool::MemoryPool};

pub unsafe fn malloc_async(bytesize: usize, stream: Stream) -> CudaResult<DevicePtr> {
    let mut ptr = 0;
    unsafe { sys::cuMemAllocAsync(&mut ptr, bytesize, stream.0) }.to_result()?;

    Ok(DevicePtr(ptr))
}

pub unsafe fn malloc_from_pool_async(
    bytesize: usize,
    pool: MemoryPool,
    stream: Stream,
) -> CudaResult<DevicePtr> {
    let mut ptr = 0;
    unsafe { sys::cuMemAllocFromPoolAsync(&mut ptr, bytesize, pool.0, stream.0) }.to_result()?;

    Ok(DevicePtr(ptr))
}

pub unsafe fn free_async(ptr: DevicePtr, stream: Stream) -> CudaResult<()> {
    unsafe { sys::cuMemFreeAsync(ptr.0, stream.0) }.to_result()
}
