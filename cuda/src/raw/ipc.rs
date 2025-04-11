use cuda_sys::ffi as sys;

use std::mem::MaybeUninit;

use crate::{
    error::{CudaResult, ToResult},
    wrap_sys_handle,
};

use super::{event::Event, memory::DevicePtr};

pub trait ShareableHandle {
    fn as_ptr(&self) -> *mut std::ffi::c_void;
}

#[cfg(target_os = "linux")]
impl<T> ShareableHandle for T
where
    T: std::os::fd::AsRawFd,
{
    fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.as_raw_fd() as *mut std::ffi::c_void
    }
}

pub trait CudaShareableHandle<T, const SIZE: usize> {
    fn from_bytes(bytes: &[T; SIZE]) -> Self
    where
        Self: Sized;
    fn to_bytes(&self) -> [T; SIZE];
}

wrap_sys_handle!(IpcMemoryHandle, sys::CUipcMemHandle);
wrap_sys_handle!(IpcEventHandle, sys::CUipcEventHandle);

impl CudaShareableHandle<i8, 64> for IpcMemoryHandle {
    fn from_bytes(bytes: &[i8; 64]) -> Self
    where
        Self: Sized,
    {
        Self(sys::CUipcMemHandle { reserved: *bytes })
    }

    fn to_bytes(&self) -> [i8; 64] {
        self.0.reserved
    }
}

impl CudaShareableHandle<i8, 64> for IpcEventHandle {
    fn from_bytes(bytes: &[i8; 64]) -> Self
    where
        Self: Sized,
    {
        Self(sys::CUipcEventHandle { reserved: *bytes })
    }

    fn to_bytes(&self) -> [i8; 64] {
        self.0.reserved
    }
}

wrap_sys_handle!(IpcDevicePtr, sys::CUdeviceptr);

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct IpcMemFlags: u32 {
        const LAZY_ENABLE_PEER_ACCESS = sys::CUipcMem_flags::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS as u32;
    }
}

pub unsafe fn get_mem_handle(ptr: &DevicePtr) -> CudaResult<IpcMemoryHandle> {
    let mut handle = MaybeUninit::uninit();
    unsafe { sys::cuIpcGetMemHandle(handle.as_mut_ptr(), ptr.0) }.to_result()?;

    Ok(IpcMemoryHandle(unsafe { handle.assume_init() }))
}

pub unsafe fn open_mem_handle(
    handle: &IpcMemoryHandle,
    flags: IpcMemFlags,
) -> CudaResult<IpcDevicePtr> {
    let mut ptr = 0;
    unsafe { sys::cuIpcOpenMemHandle_v2(&mut ptr, handle.0, flags.bits()) }.to_result()?;

    Ok(IpcDevicePtr(ptr))
}

pub unsafe fn close_mem_handle(ptr: IpcDevicePtr) -> CudaResult<()> {
    unsafe { sys::cuIpcCloseMemHandle(ptr.0) }.to_result()
}

pub unsafe fn get_event_handle(event: &Event) -> CudaResult<IpcEventHandle> {
    let mut handle = MaybeUninit::uninit();
    unsafe { sys::cuIpcGetEventHandle(handle.as_mut_ptr(), event.0) }.to_result()?;

    Ok(IpcEventHandle(unsafe { handle.assume_init() }))
}

pub unsafe fn open_event_handle(handle: &IpcEventHandle) -> CudaResult<Event> {
    let mut event = MaybeUninit::uninit();
    unsafe { sys::cuIpcOpenEventHandle(event.as_mut_ptr(), handle.0) }.to_result()?;

    Ok(Event(unsafe { event.assume_init() }))
}
