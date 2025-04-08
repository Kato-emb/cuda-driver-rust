use crate::error::{CudaResult, DropResult};

use crate::raw::memory::*;

use super::stream::CudaStream;

// pub mod device;
// pub mod ordered;
// pub mod pool;
pub mod pinned;

#[derive(Debug)]
pub struct CudaDevicePointer<Ptr: DeviceAccessible> {
    ptr: Ptr,
    size: usize,
}

#[derive(Debug)]
pub struct CudaHostPointer<Ptr: HostAccessible> {
    ptr: Ptr,
    size: usize,
}

impl<Ptr: DeviceAccessible> CudaDevicePointer<Ptr> {
    pub fn copy_from_device<Src: DeviceAccessible>(
        &mut self,
        src: &CudaDevicePointer<Src>,
        count: usize,
    ) -> CudaResult<()> {
        unsafe { copy_dtod(&mut self.ptr, &src.ptr, count) }
    }

    pub fn copy_from_device_async<Src: DeviceAccessible>(
        &mut self,
        src: &CudaDevicePointer<Src>,
        count: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        unsafe { copy_dtod_async(&mut self.ptr, &src.ptr, count, &stream.inner) }
    }

    pub fn copy_from_host<Src: HostAccessible>(
        &mut self,
        src: &CudaHostPointer<Src>,
        count: usize,
    ) -> CudaResult<()> {
        unsafe { copy_htod(&mut self.ptr, &src.ptr, count) }
    }

    pub fn copy_from_host_async<Src: HostAccessible>(
        &mut self,
        src: &CudaHostPointer<Src>,
        count: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        unsafe { copy_htod_async(&mut self.ptr, &src.ptr, count, &stream.inner) }
    }

    pub fn set_u8(&mut self, value: u8, count: usize) -> CudaResult<()> {
        debug_assert!(count <= self.size);
        unsafe { set_d8(&mut self.ptr, value, count) }
    }

    pub fn set_u8_async(&mut self, value: u8, count: usize, stream: &CudaStream) -> CudaResult<()> {
        debug_assert!(count <= self.size);
        unsafe { set_d8_async(&mut self.ptr, value, count, &stream.inner) }
    }

    pub fn set_u16(&mut self, value: u16, count: usize) -> CudaResult<()> {
        debug_assert!(count * 2 <= self.size);
        unsafe { set_d16(&mut self.ptr, value, count) }
    }

    pub fn set_u16_async(
        &mut self,
        value: u16,
        count: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        debug_assert!(count * 2 <= self.size);
        unsafe { set_d16_async(&mut self.ptr, value, count, &stream.inner) }
    }

    pub fn set_u32(&mut self, value: u32, count: usize) -> CudaResult<()> {
        debug_assert!(count * 4 <= self.size);
        unsafe { set_d32(&mut self.ptr, value, count) }
    }

    pub fn set_u32_async(
        &mut self,
        value: u32,
        count: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        debug_assert!(count * 4 <= self.size);
        unsafe { set_d32_async(&mut self.ptr, value, count, &stream.inner) }
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl<Ptr: HostAccessible> CudaHostPointer<Ptr> {
    pub fn copy_from_device<Src: DeviceAccessible>(
        &mut self,
        src: &CudaDevicePointer<Src>,
        count: usize,
    ) -> CudaResult<()> {
        unsafe { copy_dtoh(&mut self.ptr, &src.ptr, count) }
    }

    pub fn copy_from_device_async<Src: DeviceAccessible>(
        &mut self,
        src: &CudaDevicePointer<Src>,
        count: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        unsafe { copy_dtoh_async(&mut self.ptr, &src.ptr, count, &stream.inner) }
    }

    pub fn as_slice<T>(&self) -> &[T] {
        let len = self.size.checked_div(std::mem::size_of::<T>()).unwrap_or(0);
        unsafe { std::slice::from_raw_parts(self.ptr.as_host_ptr() as *const T, len) }
    }

    pub fn as_mut_slice<T>(&mut self) -> &mut [T] {
        let len = self.size.checked_div(std::mem::size_of::<T>()).unwrap_or(0);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_host_ptr() as *mut T, len) }
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

#[derive(Debug)]
pub struct CudaDevicePointerOwned<Ptr: DeviceManaged> {
    pub(crate) inner: CudaDevicePointer<Ptr>,
}

impl<Ptr: DeviceManaged> Drop for CudaDevicePointerOwned<Ptr> {
    fn drop(&mut self) {
        if self.inner.ptr.as_device_ptr() == 0 {
            return;
        }

        let ptr = std::mem::replace(&mut self.inner.ptr, Ptr::null());
        let old = Self {
            inner: CudaDevicePointer { ptr, size: 0 },
        };

        if let Err((_, e)) = old.free() {
            log::error!("Failed to free CUDA device pointer: {:?}", e);
        }
    }
}

impl<Ptr: DeviceManaged> std::ops::Deref for CudaDevicePointerOwned<Ptr> {
    type Target = CudaDevicePointer<Ptr>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<Ptr: DeviceManaged> std::ops::DerefMut for CudaDevicePointerOwned<Ptr> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<Ptr: DeviceManaged> CudaDevicePointerOwned<Ptr> {
    pub fn free(mut self) -> DropResult<Self> {
        match unsafe { free(&mut self.inner.ptr) } {
            Ok(_) => {
                // The pointer is now invalid, so we need to drop it.
                std::mem::forget(self);
                Ok(())
            }
            Err(e) => Err((self, e)),
        }
    }

    pub fn free_async(mut self, stream: &CudaStream) -> DropResult<Self> {
        match unsafe { free_async(&mut self.inner.ptr, &stream.inner) } {
            Ok(_) => {
                // The pointer is now invalid, so we need to drop it.
                std::mem::forget(self);
                Ok(())
            }
            Err(e) => Err((self, e)),
        }
    }
}

impl CudaDevicePointerOwned<DevicePtr> {
    pub fn new(bytesize: usize) -> CudaResult<Self> {
        let ptr = unsafe { malloc(bytesize) }?;
        Ok(CudaDevicePointerOwned {
            inner: CudaDevicePointer {
                ptr,
                size: bytesize,
            },
        })
    }

    pub fn new_async(bytesize: usize, stream: &CudaStream) -> CudaResult<Self> {
        let ptr = unsafe { malloc_async(bytesize, stream.inner) }?;
        Ok(CudaDevicePointerOwned {
            inner: CudaDevicePointer {
                ptr,
                size: bytesize,
            },
        })
    }

    pub fn new_pitch(width: usize, height: usize, element_size: u32) -> CudaResult<(Self, usize)> {
        let (ptr, pitch) = unsafe { malloc_pitch(width, height, element_size) }?;
        Ok((
            CudaDevicePointerOwned {
                inner: CudaDevicePointer {
                    ptr,
                    size: (pitch * height),
                },
            },
            pitch,
        ))
    }

    // pub fn address_range(&self) -> CudaResult<>
}

#[cfg(test)]
mod tests {
    use crate::{
        driver::{context::CudaPrimaryContext, device::CudaDevice},
        raw::stream::StreamFlags,
    };

    use super::*;

    #[test]
    fn test_cuda_driver_memory_alloc() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let ctx = CudaPrimaryContext::new(device).unwrap();
        ctx.set_current().unwrap();

        let stream = CudaStream::new(StreamFlags::DEFAULT).unwrap();

        let bytesize = 1024;
        let mut ptr = CudaDevicePointerOwned::new(bytesize).unwrap();
        assert!(ptr.ptr.as_device_ptr() != 0);
        let ptr2 = CudaDevicePointerOwned::new_async(bytesize, &stream).unwrap();
        assert!(ptr2.ptr.as_device_ptr() != 0);
        let (ptr3, pitch) = CudaDevicePointerOwned::new_pitch(1024, 768, 4).unwrap();
        assert!(ptr3.ptr.as_device_ptr() != 0);
        assert!(pitch > 0);

        ptr.set_u8(u8::MAX, 1024).unwrap();
        ptr.set_u16(u16::MAX, 512).unwrap();
        ptr.set_u32(u32::MAX, 256).unwrap();

        ptr.copy_from_device(&ptr2, 1024).unwrap();
        ptr.copy_from_device_async(&ptr2, 1024, &stream).unwrap();

        ptr.free().unwrap();
        ptr2.free_async(&stream).unwrap();
        drop(ptr3);
    }
}
