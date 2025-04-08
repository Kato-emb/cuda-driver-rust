use cuda_sys::ffi as sys;

use crate::{
    error::{CudaResult, ToResult},
    wrap_sys_enum, wrap_sys_handle,
};

use super::stream::Stream;

pub mod array;
pub mod pinned;
pub mod pooled;
pub mod unified;
pub mod vmm;

pub trait DeviceAccessible {
    fn as_device_ptr(&self) -> sys::CUdeviceptr;
}

/// A trait for device pointers that are managed by CUDA.
///
/// Deallocated by CUDA, call [free()] or [free_async()] to deallocate.
pub trait DeviceManaged: DeviceAccessible {}

wrap_sys_handle!(DevicePtr, sys::CUdeviceptr);

impl DeviceAccessible for DevicePtr {
    #[inline(always)]
    fn as_device_ptr(&self) -> sys::CUdeviceptr {
        self.0
    }
}

impl DeviceManaged for DevicePtr {}

impl std::fmt::Debug for DevicePtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DevicePtr").field("ptr", &self.0).finish()
    }
}

wrap_sys_handle!(Location, sys::CUmemLocation);
wrap_sys_handle!(AccessDesc, sys::CUmemAccessDesc);

wrap_sys_enum!(
    AccessFlags,
    sys::CUmemAccess_flags,
    {
        None = CU_MEM_ACCESS_FLAGS_PROT_NONE,
        Read = CU_MEM_ACCESS_FLAGS_PROT_READ,
        ReadWrite = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        Max = CU_MEM_ACCESS_FLAGS_PROT_MAX,
    }
);

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ShareableHandleFlags: u64 {
        const _ZERO = 0;
    }
}

wrap_sys_enum!(
    AllocationType,
    sys::CUmemAllocationType,
    {
        Invalid = CU_MEM_ALLOCATION_TYPE_INVALID,
        Pinned = CU_MEM_ALLOCATION_TYPE_PINNED,
        Max = CU_MEM_ALLOCATION_TYPE_MAX,
    }
);

wrap_sys_enum!(
    AllocationHandleType,
    sys::CUmemAllocationHandleType,
    {
        None = CU_MEM_HANDLE_TYPE_NONE,
        PosixFD = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        Win32 = CU_MEM_HANDLE_TYPE_WIN32,
        Win32Kmt = CU_MEM_HANDLE_TYPE_WIN32_KMT,
        Fabric = CU_MEM_HANDLE_TYPE_FABRIC,
        Max = CU_MEM_HANDLE_TYPE_MAX,
    }
);

wrap_sys_enum!(
    LocationType,
    sys::CUmemLocationType,
    {
        Invalid = CU_MEM_LOCATION_TYPE_INVALID,
        Device = CU_MEM_LOCATION_TYPE_DEVICE,
        Host = CU_MEM_LOCATION_TYPE_HOST,
        HostNuma = CU_MEM_LOCATION_TYPE_HOST_NUMA,
        HostNumaCurrent = CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT,
        Max = CU_MEM_LOCATION_TYPE_MAX,
    }
);

pub unsafe fn malloc(bytesize: usize) -> CudaResult<DevicePtr> {
    let mut ptr = 0;
    unsafe { sys::cuMemAlloc_v2(&mut ptr, bytesize) }.to_result()?;

    Ok(DevicePtr(ptr))
}

pub unsafe fn malloc_async(bytesize: usize, stream: Stream) -> CudaResult<DevicePtr> {
    let mut ptr = 0;
    unsafe { sys::cuMemAllocAsync(&mut ptr, bytesize, stream.0) }.to_result()?;

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

pub unsafe fn free<P>(ptr: P) -> CudaResult<()>
where
    P: DeviceManaged,
{
    unsafe { sys::cuMemFree_v2(ptr.as_device_ptr()) }.to_result()
}

pub unsafe fn free_async<P>(ptr: P, stream: Stream) -> CudaResult<()>
where
    P: DeviceManaged,
{
    unsafe { sys::cuMemFreeAsync(ptr.as_device_ptr(), stream.0) }.to_result()
}

pub unsafe fn get_address_range(ptr: DevicePtr) -> CudaResult<(Option<DevicePtr>, Option<usize>)> {
    let mut base = 0;
    let mut size = 0;
    unsafe { sys::cuMemGetAddressRange_v2(&mut base, &mut size, ptr.as_device_ptr()) }
        .to_result()?;

    let base = if base == 0 {
        None
    } else {
        Some(DevicePtr(base))
    };

    let size = if size == 0 { None } else { Some(size) };

    Ok((base, size))
}

/// [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g51e719462c04ee90a6b0f8b2a75fe031)
pub unsafe fn get_handle_for_address_range() {}

pub unsafe fn get_info() -> CudaResult<(usize, usize)> {
    let mut free = 0;
    let mut total = 0;
    unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) }.to_result()?;

    Ok((free as usize, total as usize))
}

pub unsafe fn copy<Dst, Src>(dst: Dst, src: Src, byte_count: usize) -> CudaResult<()>
where
    Dst: DeviceAccessible,
    Src: DeviceAccessible,
{
    unsafe { sys::cuMemcpy(dst.as_device_ptr(), src.as_device_ptr(), byte_count) }.to_result()
}

pub unsafe fn copy_async<Dst, Src>(
    dst: Dst,
    src: Src,
    bytesize: usize,
    stream: Stream,
) -> CudaResult<()>
where
    Dst: DeviceAccessible,
    Src: DeviceAccessible,
{
    unsafe { sys::cuMemcpyAsync(dst.as_device_ptr(), src.as_device_ptr(), bytesize, stream.0) }
        .to_result()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::*;

    #[test]
    fn test_cuda_raw_memory_alloc() {
        unsafe { init::init(init::InitFlags::_ZERO) }.unwrap();
        let device = unsafe { device::get_device(0) }.unwrap();
        let ctx = unsafe { context::create(context::ContextFlags::SCHED_AUTO, device) }.unwrap();
        unsafe { context::set_current(ctx) }.unwrap();
        let stream = unsafe { stream::create(stream::StreamFlags::DEFAULT) }.unwrap();
        let bytesize = 1024;

        let ptr = unsafe { malloc_async(bytesize, stream) }.unwrap();
        unsafe { free_async(ptr, stream) }.unwrap();
    }
}
