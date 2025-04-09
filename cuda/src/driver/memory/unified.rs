use crate::{
    driver::stream::CudaStream,
    error::{CudaResult, DropResult},
    raw::memory,
};

use super::DeviceRepr;

pub struct CudaUnifiedPointerOwned<T: DeviceRepr, Ptr: memory::DeviceManaged> {
    inner: CudaDevicePointer<'static, T, Ptr>,
}

impl<T: DeviceRepr, Ptr: memory::DeviceManaged> Drop for CudaUnifiedPointerOwned<T, Ptr> {
    fn drop(&mut self) {
        if self.inner.ptr.as_device_ptr() == 0 {
            return;
        }

        let ptr = std::mem::replace(&mut self.inner.ptr, Ptr::null());
        let old = Self {
            inner: CudaDevicePointer {
                ptr,
                size: 0,
                _marker: std::marker::PhantomData,
            },
        };

        if let Err((_, e)) = old.free() {
            log::error!("Failed to free CUDA unified pointer: {:?}", e);
        }
    }
}

impl<T: DeviceRepr, Ptr: memory::DeviceManaged> std::ops::Deref
    for CudaUnifiedPointerOwned<T, Ptr>
{
    type Target = CudaDevicePointer<'static, T, Ptr>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: DeviceRepr, Ptr: memory::DeviceManaged> std::ops::DerefMut
    for CudaUnifiedPointerOwned<T, Ptr>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T: DeviceRepr, Ptr: memory::DeviceManaged> CudaUnifiedPointerOwned<T, Ptr> {
    pub fn free(mut self) -> DropResult<Self> {
        match unsafe { memory::free(&mut self.inner.ptr) } {
            Ok(_) => {
                // The pointer is now invalid, so we need to drop it.
                std::mem::forget(self);
                Ok(())
            }
            Err(e) => Err((self, e)),
        }
    }

    pub fn free_async(mut self, stream: &CudaStream) -> DropResult<Self> {
        match unsafe { memory::free_async(&mut self.inner.ptr, &stream.inner) } {
            Ok(_) => {
                // The pointer is now invalid, so we need to drop it.
                std::mem::forget(self);
                Ok(())
            }
            Err(e) => Err((self, e)),
        }
    }
}

impl<T: DeviceRepr> CudaUnifiedPointerOwned<T, memory::unified::UnifiedDevicePtr> {
    pub fn new_with_flags(
        len: usize,
        flags: memory::unified::MemoryAttachFlags,
    ) -> CudaResult<Self> {
        let bytesize = len.checked_mul(std::mem::size_of::<T>()).unwrap_or(0);
        let ptr = unsafe { memory::unified::malloc_unified(bytesize, flags) }?;
        Ok(CudaUnifiedPointerOwned {
            inner: CudaDevicePointer {
                ptr,
                size: bytesize,
                _marker: std::marker::PhantomData,
            },
        })
    }

    pub fn advise(
        &self,
        len: usize,
        advice: memory::unified::Advice,
        location: memory::Location,
    ) -> CudaResult<()> {
        let count = len.checked_mul(std::mem::size_of::<T>()).unwrap_or(0);
        unsafe { memory::unified::advise(&self.inner.ptr, count, advice, location) }
    }

    pub fn prefetch_async(
        &self,
        len: usize,
        location: memory::Location,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let count = len.checked_mul(std::mem::size_of::<T>()).unwrap_or(0);
        unsafe { memory::unified::prefetch_async(&self.inner.ptr, count, location, &stream.inner) }
    }
}
