#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
pub use windows::*;

#[cfg(target_os = "linux")]
mod unix;
#[cfg(target_os = "linux")]
pub use unix::*;

pub trait ShareableOsHandle {
    fn as_ptr(&self) -> *mut std::ffi::c_void;
}
