use std::mem::MaybeUninit;

use cuda_sys::ffi as sys;

use crate::{
    error::{CudaResult, ToResult},
    raw::stream::Stream,
    wrap_sys_handle,
};

use super::{DevicePtr, pool::MemoryPool};

wrap_sys_handle!(MemoryPoolExportData, sys::CUmemPoolPtrExportData);

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

pub unsafe fn export_pointer(device_ptr: DevicePtr) -> CudaResult<MemoryPoolExportData> {
    let mut export_data = MaybeUninit::uninit();
    unsafe { sys::cuMemPoolExportPointer(export_data.as_mut_ptr(), device_ptr.0) }.to_result()?;

    Ok(MemoryPoolExportData(unsafe { export_data.assume_init() }))
}

pub unsafe fn import_pointer(
    pool: MemoryPool,
    share_data: &MemoryPoolExportData,
) -> CudaResult<DevicePtr> {
    let mut device_ptr = 0;
    unsafe {
        sys::cuMemPoolImportPointer(
            &mut device_ptr,
            pool.0,
            &share_data.0 as *const _ as *mut sys::CUmemPoolPtrExportData,
        )
    }
    .to_result()?;

    Ok(DevicePtr(device_ptr))
}
