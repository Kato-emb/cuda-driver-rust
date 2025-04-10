use cuda_sys::ffi as sys;

use crate::{
    error::{CudaResult, ToResult},
    wrap_sys_handle,
};

use super::{CudaPointer, DeviceAccessible, HostAccessible, HostManaged};

wrap_sys_handle!(PinnedHostPtr, *mut std::ffi::c_void);

impl std::fmt::Debug for PinnedHostPtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedHostPtr")
            .field("ptr", &self.0)
            .finish()
    }
}

unsafe impl CudaPointer for PinnedHostPtr {
    unsafe fn from_raw_ptr<P: Sized>(ptr: *mut P) -> Self {
        PinnedHostPtr(ptr as *mut std::ffi::c_void)
    }

    unsafe fn offset(self, byte_count: isize) -> Self
    where
        Self: Sized,
    {
        let ptr = self.as_host_ptr() as *mut u8;
        let new_ptr = ptr.wrapping_offset(byte_count) as *mut std::ffi::c_void;
        PinnedHostPtr(new_ptr)
    }
}

impl HostAccessible for PinnedHostPtr {
    #[inline(always)]
    fn as_host_ptr(&self) -> *mut std::ffi::c_void {
        self.0
    }
}

impl HostManaged for PinnedHostPtr {}

wrap_sys_handle!(PinnedDevicePtr, sys::CUdeviceptr);

unsafe impl CudaPointer for PinnedDevicePtr {
    unsafe fn from_raw_ptr<P: Sized>(ptr: *mut P) -> Self {
        PinnedDevicePtr(ptr as sys::CUdeviceptr)
    }

    unsafe fn offset(self, byte_count: isize) -> Self
    where
        Self: Sized,
    {
        let ptr = self.as_device_ptr() as i64;
        let new_ptr = ptr.wrapping_add(byte_count as i64) as sys::CUdeviceptr;
        PinnedDevicePtr(new_ptr)
    }
}

impl DeviceAccessible for PinnedDevicePtr {
    #[inline(always)]
    fn as_device_ptr(&self) -> sys::CUdeviceptr {
        self.0
    }
}

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

pub unsafe fn malloc_pinned(bytesize: usize) -> CudaResult<PinnedHostPtr> {
    let mut ptr = std::ptr::null_mut();
    unsafe { sys::cuMemAllocHost_v2(&mut ptr, bytesize) }.to_result()?;

    Ok(PinnedHostPtr(ptr))
}

pub unsafe fn malloc_pinned_with_flags(
    bytesize: usize,
    flags: PinnedFlags,
) -> CudaResult<PinnedHostPtr> {
    let mut ptr = std::ptr::null_mut();
    unsafe { sys::cuMemHostAlloc(&mut ptr, bytesize, flags.bits()) }.to_result()?;

    Ok(PinnedHostPtr(ptr))
}

pub unsafe fn free_pinned<P>(ptr: &mut P) -> CudaResult<()>
where
    P: HostManaged,
{
    unsafe { sys::cuMemFreeHost(ptr.as_host_ptr()) }.to_result()
}

pub unsafe fn get_device_pointer(
    host_ptr: &PinnedHostPtr,
    flags: DevicePointerFlags,
) -> CudaResult<PinnedDevicePtr> {
    let mut device_ptr = 0;
    unsafe { sys::cuMemHostGetDevicePointer_v2(&mut device_ptr, host_ptr.0, flags.bits()) }
        .to_result()?;

    Ok(PinnedDevicePtr(device_ptr))
}

pub unsafe fn get_flags(host_ptr: &PinnedHostPtr) -> CudaResult<PinnedFlags> {
    let mut flags = 0;
    unsafe { sys::cuMemHostGetFlags(&mut flags, host_ptr.0) }.to_result()?;

    Ok(PinnedFlags::from_bits(flags).unwrap_or(PinnedFlags::empty()))
}

pub unsafe fn register(
    host_ptr: &PinnedHostPtr,
    bytesize: usize,
    flags: HostRegisterFlags,
) -> CudaResult<()> {
    unsafe { sys::cuMemHostRegister_v2(host_ptr.0, bytesize, flags.bits()) }.to_result()
}

pub unsafe fn unregister(host_ptr: &PinnedHostPtr) -> CudaResult<()> {
    unsafe { sys::cuMemHostUnregister(host_ptr.0) }.to_result()
}
