use std::mem::MaybeUninit;

use cuda_sys::ffi as sys;

use crate::{
    error::{CudaResult, ToResult},
    raw::{pool::MemoryPool, stream::Stream},
    wrap_sys_handle,
};

use super::{CudaPointer, DeviceAccessible, DeviceAllocated};

wrap_sys_handle!(PooledDevicePtr, sys::CUdeviceptr);

impl std::fmt::Debug for PooledDevicePtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Pointer::fmt(&(self.0 as *mut std::ffi::c_void), f)
    }
}

unsafe impl CudaPointer for PooledDevicePtr {
    unsafe fn from_raw_ptr<P: Sized>(ptr: *mut P) -> Self {
        PooledDevicePtr(ptr as sys::CUdeviceptr)
    }

    unsafe fn offset(self, byte_count: isize) -> Self
    where
        Self: Sized,
    {
        let ptr = self.as_device_ptr() as i64;
        let new_ptr = ptr.wrapping_add(byte_count as i64) as sys::CUdeviceptr;
        PooledDevicePtr(new_ptr)
    }
}

impl DeviceAccessible for PooledDevicePtr {
    #[inline(always)]
    fn as_device_ptr(&self) -> sys::CUdeviceptr {
        self.0
    }
}

impl DeviceAllocated for PooledDevicePtr {}

wrap_sys_handle!(PooledPtrExportData, sys::CUmemPoolPtrExportData);

pub unsafe fn malloc_pooled_async(
    bytesize: usize,
    pool: MemoryPool,
    stream: Stream,
) -> CudaResult<PooledDevicePtr> {
    let mut ptr = 0;
    unsafe { sys::cuMemAllocFromPoolAsync(&mut ptr, bytesize, pool.0, stream.0) }.to_result()?;

    Ok(PooledDevicePtr(ptr))
}

pub unsafe fn export(pooled_ptr: PooledDevicePtr) -> CudaResult<PooledPtrExportData> {
    let mut export_data = MaybeUninit::uninit();
    unsafe { sys::cuMemPoolExportPointer(export_data.as_mut_ptr(), pooled_ptr.0) }.to_result()?;

    Ok(PooledPtrExportData(unsafe { export_data.assume_init() }))
}

pub unsafe fn import(
    pool: MemoryPool,
    share_data: &PooledPtrExportData,
) -> CudaResult<PooledDevicePtr> {
    let mut device_ptr = 0;
    unsafe {
        sys::cuMemPoolImportPointer(
            &mut device_ptr,
            pool.0,
            &share_data.0 as *const _ as *mut sys::CUmemPoolPtrExportData,
        )
    }
    .to_result()?;

    Ok(PooledDevicePtr(device_ptr))
}
