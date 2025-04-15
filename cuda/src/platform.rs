#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
pub use windows::*;

pub trait ShareableOsHandle {
    fn as_ptr(&self) -> *mut std::ffi::c_void;
}
