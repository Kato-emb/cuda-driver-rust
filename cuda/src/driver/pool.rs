use crate::{
    error::CudaResult,
    raw::{
        memory::{AllocationHandleType, AllocationType, LocationType},
        pool::*,
    },
};

#[derive(Debug)]
pub struct CudaMemoryPool {
    pub(crate) inner: MemoryPool,
}

impl Drop for CudaMemoryPool {
    fn drop(&mut self) {
        if let Err(e) = unsafe { destroy(self.inner) } {
            log::error!("Failed to destroy CUDA memory pool: {:?}", e);
        }
    }
}

impl CudaMemoryPool {
    pub fn new(max_size: usize) -> CudaResult<Self> {
        let mut props = MemoryPoolProps::default();
        props.0.allocType = AllocationType::Pinned.into();
        props.0.handleTypes = AllocationHandleType::PosixFD.into();
        props.0.maxSize = max_size;
        props.0.location.type_ = LocationType::Device.into();
        props.0.location.id = 0;

        let inner = unsafe { create(&props) }?;
        Ok(Self { inner })
    }

    #[cfg(target_os = "linux")]
    pub fn export(&self) -> CudaResult<std::os::fd::OwnedFd> {
        use crate::raw::memory::{AllocationHandleType, ShareableHandleFlags};

        let handle_type = AllocationHandleType::PosixFD;
        unsafe { export_to_shareable_handle(&self.inner, handle_type, ShareableHandleFlags::_ZERO) }
    }

    #[cfg(target_os = "linux")]
    pub fn import(handle: std::os::fd::BorrowedFd) -> CudaResult<Self> {
        use crate::raw::memory::{AllocationHandleType, ShareableHandleFlags};

        let handle_type = AllocationHandleType::PosixFD;
        let pool = unsafe {
            import_from_shareable_handle(handle, handle_type, ShareableHandleFlags::_ZERO)
        }?;

        Ok(Self { inner: pool })
    }

    pub fn trim_to(&self, keep: usize) -> CudaResult<()> {
        unsafe { trim_to(self.inner, keep) }
    }
}

#[cfg(test)]
mod tests {
    use crate::driver::{context::CudaPrimaryContext, device::CudaDevice};

    use super::*;

    #[test]
    fn test_cuda_driver_pool() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let ctx = CudaPrimaryContext::new(device).unwrap();
        ctx.set_current().unwrap();

        let pool = CudaMemoryPool::new(1024 * 1024).unwrap();
        println!("Memory pool created: {:?}", pool);
    }
}
