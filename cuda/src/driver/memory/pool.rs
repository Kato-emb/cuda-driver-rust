use std::os::fd::AsRawFd;

use crate::{error::CudaResult, raw::memory};

pub struct CudaMemoryPool {
    pub(crate) inner: memory::pool::MemoryPool,
}

impl Drop for CudaMemoryPool {
    fn drop(&mut self) {
        if self.inner.0.is_null() {
            return;
        }

        if let Err(e) = unsafe { memory::pool::destroy(self.inner) } {
            log::error!("Failed to destroy memory pool: {:?}", e);
        }
    }
}

impl std::fmt::Debug for CudaMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaMemoryPool")
            .field("ptr", &self.inner.0)
            .field("release_threshold", &self.release_threshold().ok())
            .field("reserved", &self.current_memory_reserved_size().ok())
            .field("used", &self.current_memory_used_size().ok())
            .finish()
    }
}

impl CudaMemoryPool {
    pub fn new(props: &memory::pool::MemoryPoolProps) -> CudaResult<Self> {
        let inner = unsafe { memory::pool::create(props) }?;

        Ok(Self { inner })
    }

    pub fn access(&self, location: &memory::Location) -> CudaResult<memory::AccessFlags> {
        unsafe { memory::pool::get_access(self.inner, location) }
    }

    pub fn set_access(&self, map: &[memory::AccessDesc]) -> CudaResult<()> {
        unsafe { memory::pool::set_access(self.inner, map) }
    }

    #[cfg(target_os = "linux")]
    pub fn export_to_fd(&self) -> CudaResult<std::os::fd::OwnedFd> {
        use std::os::fd::FromRawFd;

        let raw_fd: i32 = unsafe {
            memory::pool::export_to_shareable_handle(
                self.inner,
                memory::AllocationHandleType::PosixFD,
                memory::ShareableHandleFlags::_ZERO,
            )
        }?;

        Ok(unsafe { std::os::fd::OwnedFd::from_raw_fd(raw_fd) })
    }

    pub fn release_threshold(&self) -> CudaResult<u64> {
        unsafe {
            memory::pool::get_attribute(
                self.inner,
                memory::pool::MemoryPoolAttribute::ReleaseThreshold,
            )
        }
    }

    pub fn set_release_threshold(&self, threshold: u64) -> CudaResult<()> {
        unsafe {
            memory::pool::set_attribute(
                self.inner,
                memory::pool::MemoryPoolAttribute::ReleaseThreshold,
                &threshold,
            )
        }
    }

    pub fn reuse_follow_event_dependencies(&self) -> CudaResult<bool> {
        let value = unsafe {
            memory::pool::get_attribute::<i32>(
                self.inner,
                memory::pool::MemoryPoolAttribute::ReuseFollowEventDependencies,
            )
        }?;

        Ok(value != 0)
    }

    pub fn set_reuse_follow_event_dependencies(&self, value: bool) -> CudaResult<()> {
        let value = if value { 1 } else { 0 };
        unsafe {
            memory::pool::set_attribute(
                self.inner,
                memory::pool::MemoryPoolAttribute::ReuseFollowEventDependencies,
                &value,
            )
        }
    }

    pub fn reuse_allow_opportunistic(&self) -> CudaResult<bool> {
        let value = unsafe {
            memory::pool::get_attribute::<i32>(
                self.inner,
                memory::pool::MemoryPoolAttribute::ReuseAllowOpportunistic,
            )
        }?;

        Ok(value != 0)
    }

    pub fn set_reuse_allow_opportunistic(&self, value: bool) -> CudaResult<()> {
        let value = if value { 1 } else { 0 };
        unsafe {
            memory::pool::set_attribute(
                self.inner,
                memory::pool::MemoryPoolAttribute::ReuseAllowOpportunistic,
                &value,
            )
        }
    }

    pub fn reuse_allow_internal_dependencies(&self) -> CudaResult<bool> {
        let value = unsafe {
            memory::pool::get_attribute::<i32>(
                self.inner,
                memory::pool::MemoryPoolAttribute::ReuseAllowInternalDependencies,
            )
        }?;

        Ok(value != 0)
    }

    pub fn set_reuse_allow_internal_dependencies(&self, value: bool) -> CudaResult<()> {
        let value = if value { 1 } else { 0 };
        unsafe {
            memory::pool::set_attribute(
                self.inner,
                memory::pool::MemoryPoolAttribute::ReuseAllowInternalDependencies,
                &value,
            )
        }
    }

    pub fn current_memory_reserved_size(&self) -> CudaResult<u64> {
        unsafe {
            memory::pool::get_attribute::<u64>(
                self.inner,
                memory::pool::MemoryPoolAttribute::ReservedMemCurrent,
            )
        }
    }

    pub fn high_memory_reserved_size(&self) -> CudaResult<u64> {
        unsafe {
            memory::pool::get_attribute::<u64>(
                self.inner,
                memory::pool::MemoryPoolAttribute::ReservedMemHigh,
            )
        }
    }

    pub fn reset_high_memory_reserved_size(&self) -> CudaResult<()> {
        unsafe {
            memory::pool::set_attribute(
                self.inner,
                memory::pool::MemoryPoolAttribute::ReservedMemHigh,
                &0u64,
            )
        }
    }

    pub fn current_memory_used_size(&self) -> CudaResult<u64> {
        unsafe {
            memory::pool::get_attribute::<u64>(
                self.inner,
                memory::pool::MemoryPoolAttribute::UsedMemCurrent,
            )
        }
    }

    pub fn high_memory_used_size(&self) -> CudaResult<u64> {
        unsafe {
            memory::pool::get_attribute::<u64>(
                self.inner,
                memory::pool::MemoryPoolAttribute::UsedMemHigh,
            )
        }
    }

    pub fn reset_high_memory_used_size(&self) -> CudaResult<()> {
        unsafe {
            memory::pool::set_attribute(
                self.inner,
                memory::pool::MemoryPoolAttribute::UsedMemHigh,
                &0u64,
            )
        }
    }

    /// Tries to release memory back to the OS.
    pub fn trim_to(&mut self, keep_size: usize) -> CudaResult<()> {
        unsafe { memory::pool::trim_to(self.inner, keep_size) }
    }
}

pub struct CudaMemoryPoolView<Handle: memory::pool::ShareableHandle> {
    pub(crate) inner: memory::pool::MemoryPool,
    handle: Handle,
}

impl<Handle: memory::pool::ShareableHandle> std::fmt::Debug for CudaMemoryPoolView<Handle> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaMemoryPoolView")
            .field("ptr", &self.inner.0)
            .field("handle", &self.handle.as_ptr())
            .finish()
    }
}

#[cfg(target_os = "linux")]
impl<'fd> CudaMemoryPoolView<std::os::fd::BorrowedFd<'fd>> {
    pub fn import_from_fd(fd: std::os::fd::BorrowedFd<'fd>) -> CudaResult<Self> {
        let handle = fd.as_raw_fd();
        let inner = unsafe {
            memory::pool::import_from_shareable_handle(
                &handle,
                memory::AllocationHandleType::PosixFD,
                memory::ShareableHandleFlags::_ZERO,
            )
        }?;

        Ok(Self { inner, handle: fd })
    }
}

#[cfg(test)]
mod tests {
    use std::{os::fd::AsFd, sync::Arc};

    use crate::{
        driver::{
            context::CudaPrimaryContext, device::CudaDevice, memory::ordered::CudaOrderedMemory,
            stream::CudaStream,
        },
        raw::stream::StreamFlags,
    };

    use super::*;

    #[test]
    fn test_cuda_driver_memory_pool_new() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let context = CudaPrimaryContext::new(device).unwrap();
        context.set_current().unwrap();

        let stream = CudaStream::new(StreamFlags::DEFAULT).unwrap();
        let stream = Arc::new(stream);

        let mut props = memory::pool::MemoryPoolProps::default();
        props.0.allocType = memory::AllocationType::Pinned.into();
        props.0.handleTypes = memory::AllocationHandleType::PosixFD.into();
        props.0.maxSize = 1024 * 1024 * 1024; // 1 GB
        props.0.location.type_ = memory::LocationType::Device.into();
        props.0.location.id = 0;

        let pool = CudaMemoryPool::new(&props).unwrap();
        assert!(!pool.inner.0.is_null());

        pool.set_release_threshold(1024 * 1024).unwrap();

        println!("Memory pool: {:?}", pool);

        let fd = pool.export_to_fd().unwrap();
        println!("Exported file descriptor: {:?}", fd);

        let imported_pool = CudaMemoryPoolView::import_from_fd(fd.as_fd()).unwrap();

        println!("Imported memory pool: {:?}", imported_pool);

        let export_buf = CudaOrderedMemory::alloc_from_pool(1024, &pool, stream.clone()).unwrap();
        println!("Exported buffer: {:?}", export_buf);
        let export_data = export_buf.export().unwrap();

        let imported_buf =
            CudaOrderedMemory::import(&imported_pool, &export_data, stream.clone()).unwrap();

        println!("Imported buffer: {:?}", imported_buf);
        println!("Pool {:?}", pool);
    }
}
