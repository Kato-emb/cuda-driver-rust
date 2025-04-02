use cuda_sys::ffi as sys;

use crate::{
    error::{CudaResult, ToResult},
    wrap_sys_handle,
};

use super::DevicePtr;

wrap_sys_handle!(PinnedPtr, *mut std::ffi::c_void);

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PinnedFlags: u32 {
        const PORTABLE = 0x01;
        const DEVICEMAP = 0x02;
        const WRITECOMBINED = 0x04;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DevicePointerFlags: u32 {
        const _ZERO = 0x00;
    }
}
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct HostRegisterFlags: u32 {
        const PORTABLE = 0x01;
        const DEVICEMAP = 0x02;
        const IOMEMORY = 0x04;
        const READ_ONLY = 0x08;
    }
}

pub unsafe fn malloc_pinned(bytesize: usize) -> CudaResult<PinnedPtr> {
    let mut ptr = std::ptr::null_mut();
    unsafe { sys::cuMemAllocHost_v2(&mut ptr, bytesize) }.to_result()?;

    Ok(PinnedPtr(ptr))
}

pub unsafe fn malloc_pinned_with_flags(
    bytesize: usize,
    flags: PinnedFlags,
) -> CudaResult<PinnedPtr> {
    let mut ptr = std::ptr::null_mut();
    unsafe { sys::cuMemHostAlloc(&mut ptr, bytesize, flags.bits()) }.to_result()?;

    Ok(PinnedPtr(ptr))
}

pub unsafe fn free_pinned(ptr: PinnedPtr) -> CudaResult<()> {
    unsafe { sys::cuMemFreeHost(ptr.0) }.to_result()
}

pub unsafe fn get_device_pointer(
    host_ptr: PinnedPtr,
    flags: DevicePointerFlags,
) -> CudaResult<DevicePtr> {
    let mut device_ptr = 0;
    unsafe { sys::cuMemHostGetDevicePointer_v2(&mut device_ptr, host_ptr.0, flags.bits()) }
        .to_result()?;

    Ok(DevicePtr(device_ptr))
}

pub unsafe fn get_flags(host_ptr: PinnedPtr) -> CudaResult<PinnedFlags> {
    let mut flags = 0;
    unsafe { sys::cuMemHostGetFlags(&mut flags, host_ptr.0) }.to_result()?;

    Ok(PinnedFlags::from_bits(flags).unwrap_or(PinnedFlags::empty()))
}

pub unsafe fn register(
    host_ptr: PinnedPtr,
    bytesize: usize,
    flags: HostRegisterFlags,
) -> CudaResult<()> {
    unsafe { sys::cuMemHostRegister_v2(host_ptr.0, bytesize, flags.bits()) }.to_result()
}

pub unsafe fn unregister(host_ptr: PinnedPtr) -> CudaResult<()> {
    unsafe { sys::cuMemHostUnregister(host_ptr.0) }.to_result()
}
