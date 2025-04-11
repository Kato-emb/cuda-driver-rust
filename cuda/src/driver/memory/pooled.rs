use crate::{
    driver::{device::CudaDevice, stream::CudaStream},
    error::CudaResult,
    raw::memory::{
        AccessDesc, AccessFlags, AllocationHandleType, AllocationType, Location, LocationType,
        pooled::*,
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
    pub fn to_fd(&self) -> CudaResult<std::os::fd::OwnedFd> {
        use crate::raw::memory::{AllocationHandleType, ShareableHandleFlags};

        let handle_type = AllocationHandleType::PosixFD;
        unsafe { export_to_shareable_handle(&self.inner, handle_type, ShareableHandleFlags::_ZERO) }
    }

    #[cfg(target_os = "linux")]
    pub fn from_fd(handle: std::os::fd::BorrowedFd) -> CudaResult<Self> {
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

    pub fn accessibility_from_device(&self, device: &CudaDevice) -> CudaResult<AccessFlags> {
        let mut location = Location::default();
        location.0.type_ = LocationType::Device.into();
        location.0.id = device.as_raw();
        unsafe { get_access(&self.inner, &location) }
    }

    pub fn set_accessibility(
        &mut self,
        device: &CudaDevice,
        access: AccessFlags,
    ) -> CudaResult<()> {
        let mut desc = AccessDesc::default();
        desc.0.location.type_ = LocationType::Device.into();
        desc.0.location.id = device.as_raw();
        desc.0.flags = access.into();
        unsafe { set_access(&self.inner, &[desc]) }
    }

    pub fn alloc_async<Repr: DeviceRepr>(
        &self,
        len: usize,
        stream: &CudaStream,
    ) -> CudaResult<CudaDevicePooledBuffer<Repr>> {
        let bytesize = len.wrapping_mul(std::mem::size_of::<Repr>());
        let ptr = unsafe { malloc_pooled_async(bytesize, &self.inner, &stream.inner) }?;

        Ok(CudaDevicePooledBuffer {
            ptr,
            size: bytesize,
            _marker: std::marker::PhantomData,
        })
    }
}

pub type CudaDevicePooledBuffer<Repr> = CudaDeviceBuffer<Repr, PooledDevicePtr>;

impl<Repr: DeviceRepr> CudaDevicePooledBuffer<Repr> {
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
    use std::os::fd::AsFd;

    use crate::{
        driver::{
            context::CudaPrimaryContext, device::CudaDevice, memory::pinned::CudaHostPinnedBuffer,
        },
        raw::stream::StreamFlags,
    };

    use super::*;

    #[test]
    fn test_cuda_driver_pool() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let ctx = CudaPrimaryContext::new(device).unwrap();
        ctx.set_current().unwrap();

        let stream = CudaStream::new(StreamFlags::NON_BLOCKING).unwrap();

        let pool = CudaMemoryPool::new(1024 * 1024).unwrap();
        println!("Memory pool created: {:?}", pool);

        // pool.set_accessibility(ctx.device(), AccessFlags::Read)
        //     .unwrap(); CudaError { code: 801, name: "CUDA_ERROR_NOT_SUPPORTED" }
        let flags = pool.accessibility_from_device(ctx.device()).unwrap();
        println!("Memory pool accessibility from device: {:?}", flags);

        let fd = pool.to_fd().unwrap();
        println!("Exported memory pool handle: {:?}", fd);

        let imported_pool = CudaMemoryPool::from_fd(fd.as_fd()).unwrap();
        println!("Imported memory pool: {:?}", imported_pool);

        let flags = imported_pool
            .accessibility_from_device(ctx.device())
            .unwrap();
        println!(
            "Imported memory pool accessibility from device: {:?}",
            flags
        );

        let mut pooled_buffer = pool.alloc_async::<u8>(1024, &stream).unwrap();
        pooled_buffer.as_mut_slice().set(u8::MAX).unwrap();

        let (data, size) = pooled_buffer.export().unwrap();

        let imported_buffer =
            CudaDevicePooledBuffer::<u8>::import(data, size, &imported_pool).unwrap();
        let imported_slice = imported_buffer.as_slice();

        let mut pinned_buffer = CudaHostPinnedBuffer::<u8>::alloc(size).unwrap();
        pinned_buffer
            .as_mut_slice()
            .copy_from_device(&imported_slice)
            .unwrap();

        for i in pinned_buffer.as_slice().iter() {
            assert_eq!(*i, u8::MAX);
        }
    }
}
