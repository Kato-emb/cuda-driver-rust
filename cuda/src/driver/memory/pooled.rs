use crate::{
    driver::stream::CudaStream,
    error::CudaResult,
    raw::{
        memory::{AllocationHandleType, AllocationType, LocationType, pooled::*},
        pool::*,
    },
};

use super::{CudaDeviceBuffer, DeviceRepr};

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

pub type CudaDevicePooledBuffer<Repr> = CudaDeviceBuffer<Repr, PooledDevicePtr>;

impl<Repr: DeviceRepr> CudaDevicePooledBuffer<Repr> {
    pub fn alloc_pooled_async(
        len: usize,
        pool: &CudaMemoryPool,
        stream: &CudaStream,
    ) -> CudaResult<Self> {
        let bytesize = len.wrapping_mul(std::mem::size_of::<Repr>());
        let ptr = unsafe { malloc_pooled_async(bytesize, pool.inner, stream.inner) }?;

        Ok(Self {
            ptr,
            size: bytesize,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn export(&self) -> CudaResult<(PooledPtrExportData, usize)> {
        let data = unsafe { export(self.ptr) }?;
        let len = self.size.wrapping_div(std::mem::size_of::<Repr>());

        Ok((data, len))
    }

    pub fn import(
        data: PooledPtrExportData,
        len: usize,
        pool: &CudaMemoryPool,
    ) -> CudaResult<Self> {
        let ptr = unsafe { import(pool.inner, &data) }?;
        let bytesize = len.wrapping_mul(std::mem::size_of::<Repr>());

        Ok(Self {
            ptr,
            size: bytesize,
            _marker: std::marker::PhantomData,
        })
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
