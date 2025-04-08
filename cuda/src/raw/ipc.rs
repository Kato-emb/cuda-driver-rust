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

pub unsafe fn open_mem_handle() {}
pub unsafe fn close_mem_handle() {}
pub unsafe fn get_mem_handle() {}
pub unsafe fn open_event_handle() {}
pub unsafe fn close_event_handle() {}
pub unsafe fn get_event_handle() {}
