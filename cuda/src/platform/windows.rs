use super::ShareableOsHandle;

pub type OsRawHandle = std::os::windows::io::RawHandle;
pub type OsOwnedHandle = std::os::windows::io::OwnedHandle;
pub type OsBorrowedHandle<'fd> = std::os::windows::io::BorrowedHandle<'fd>;

impl<T> ShareableOsHandle for T
where
    T: std::os::windows::io::AsRawHandle,
{
    fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.as_raw_handle() as *mut std::ffi::c_void
    }
}
