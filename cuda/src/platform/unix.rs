use super::ShareableOsHandle;

pub type OsRawHandle = std::os::fd::RawFd;
pub type OsOwnedHandle = std::os::fd::OwnedFd;
pub type OsBorrowedHandle<'fd> = std::os::fd::BorrowedFd<'fd>;

impl<T> ShareableOsHandle for T
where
    T: std::os::fd::AsRawFd,
{
    fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.as_raw_fd() as *mut std::ffi::c_void
    }
}
