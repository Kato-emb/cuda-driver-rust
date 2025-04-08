use crate::error::{CudaResult, DropResult};

use crate::raw::memory::*;

use super::stream::CudaStream;

// pub mod device;
// pub mod ordered;
// pub mod pool;
pub mod pinned;

pub unsafe trait DeviceRepr: Copy + 'static {}
unsafe impl DeviceRepr for bool {}
unsafe impl DeviceRepr for i8 {}
unsafe impl DeviceRepr for i16 {}
unsafe impl DeviceRepr for i32 {}
unsafe impl DeviceRepr for i64 {}
unsafe impl DeviceRepr for i128 {}
unsafe impl DeviceRepr for isize {}
unsafe impl DeviceRepr for u8 {}
unsafe impl DeviceRepr for u16 {}
unsafe impl DeviceRepr for u32 {}
unsafe impl DeviceRepr for u64 {}
unsafe impl DeviceRepr for u128 {}
unsafe impl DeviceRepr for usize {}
unsafe impl DeviceRepr for f32 {}
unsafe impl DeviceRepr for f64 {}

#[derive(Debug)]
pub struct CudaDevicePointer<'a, T: DeviceRepr, Ptr: DeviceAccessible> {
    ptr: Ptr,
    size: usize,
    _marker: std::marker::PhantomData<&'a T>,
}

#[derive(Debug)]
pub struct CudaHostPointer<'a, T, Ptr: HostAccessible> {
    ptr: Ptr,
    size: usize,
    _marker: std::marker::PhantomData<&'a T>,
}

impl<T: DeviceRepr, Ptr: DeviceAccessible> CudaDevicePointer<'_, T, Ptr> {
    pub fn copy_from_device<Src: DeviceAccessible>(
        &mut self,
        src: &CudaDevicePointer<T, Src>,
        len: usize,
    ) -> CudaResult<()> {
        let byte_count = len.checked_mul(std::mem::size_of::<T>()).unwrap_or(0);
        unsafe { copy_dtod(&mut self.ptr, &src.ptr, byte_count) }
    }

    pub fn copy_from_device_async<Src: DeviceAccessible>(
        &mut self,
        src: &CudaDevicePointer<T, Src>,
        len: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let byte_count = len.checked_mul(std::mem::size_of::<T>()).unwrap_or(0);
        unsafe { copy_dtod_async(&mut self.ptr, &src.ptr, byte_count, &stream.inner) }
    }

    pub fn copy_from_host<Src: HostAccessible>(
        &mut self,
        src: &CudaHostPointer<T, Src>,
        len: usize,
    ) -> CudaResult<()> {
        let byte_count = len.checked_mul(std::mem::size_of::<T>()).unwrap_or(0);
        unsafe { copy_htod(&mut self.ptr, &src.ptr, byte_count) }
    }

    pub fn copy_from_host_async<Src: HostAccessible>(
        &mut self,
        src: &CudaHostPointer<T, Src>,
        len: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let byte_count = len.checked_mul(std::mem::size_of::<T>()).unwrap_or(0);
        unsafe { copy_htod_async(&mut self.ptr, &src.ptr, byte_count, &stream.inner) }
    }

    pub fn len(&self) -> usize {
        self.size.wrapping_div(std::mem::size_of::<T>())
    }
}

impl<Ptr: DeviceAccessible> CudaDevicePointer<'_, u8, Ptr> {
    pub fn set_u8(&mut self, value: u8, count: usize) -> CudaResult<()> {
        debug_assert!(count <= self.len());
        unsafe { set_d8(&mut self.ptr, value, count) }
    }

    pub fn set_u8_async(&mut self, value: u8, count: usize, stream: &CudaStream) -> CudaResult<()> {
        debug_assert!(count <= self.len());
        unsafe { set_d8_async(&mut self.ptr, value, count, &stream.inner) }
    }
}

impl<Ptr: DeviceAccessible> CudaDevicePointer<'_, u16, Ptr> {
    pub fn set_u16(&mut self, value: u16, count: usize) -> CudaResult<()> {
        debug_assert!(count <= self.len());
        unsafe { set_d16(&mut self.ptr, value, count) }
    }

    pub fn set_u16_async(
        &mut self,
        value: u16,
        count: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        debug_assert!(count <= self.len());
        unsafe { set_d16_async(&mut self.ptr, value, count, &stream.inner) }
    }
}

impl<Ptr: DeviceAccessible> CudaDevicePointer<'_, u32, Ptr> {
    pub fn set_u32(&mut self, value: u32, count: usize) -> CudaResult<()> {
        debug_assert!(count <= self.len(), "count: {}, len: {}", count, self.len());
        unsafe { set_d32(&mut self.ptr, value, count) }
    }

    pub fn set_u32_async(
        &mut self,
        value: u32,
        count: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        debug_assert!(count <= self.len());
        unsafe { set_d32_async(&mut self.ptr, value, count, &stream.inner) }
    }
}

impl<T: DeviceRepr, Ptr: HostAccessible> CudaHostPointer<'_, T, Ptr> {
    pub fn copy_from_device<Src: DeviceAccessible>(
        &mut self,
        src: &CudaDevicePointer<T, Src>,
        len: usize,
    ) -> CudaResult<()> {
        let byte_count = len.checked_mul(std::mem::size_of::<T>()).unwrap_or(0);
        unsafe { copy_dtoh(&mut self.ptr, &src.ptr, byte_count) }
    }

    pub fn copy_from_device_async<Src: DeviceAccessible>(
        &mut self,
        src: &CudaDevicePointer<T, Src>,
        len: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let byte_count = len.checked_mul(std::mem::size_of::<T>()).unwrap_or(0);
        unsafe { copy_dtoh_async(&mut self.ptr, &src.ptr, byte_count, &stream.inner) }
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_host_ptr() as *const T, self.len()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_host_ptr() as *mut T, self.len()) }
    }

    pub fn len(&self) -> usize {
        self.size.wrapping_div(std::mem::size_of::<T>())
    }
}

#[derive(Debug)]
pub struct CudaDevicePointerOwned<T: DeviceRepr, Ptr: DeviceManaged> {
    pub(crate) inner: CudaDevicePointer<'static, T, Ptr>,
}

impl<T: DeviceRepr, Ptr: DeviceManaged> Drop for CudaDevicePointerOwned<T, Ptr> {
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
            log::error!("Failed to free CUDA device pointer: {:?}", e);
        }
    }
}

impl<T: DeviceRepr, Ptr: DeviceManaged> std::ops::Deref for CudaDevicePointerOwned<T, Ptr> {
    type Target = CudaDevicePointer<'static, T, Ptr>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: DeviceRepr, Ptr: DeviceManaged> std::ops::DerefMut for CudaDevicePointerOwned<T, Ptr> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T: DeviceRepr, Ptr: DeviceManaged> CudaDevicePointerOwned<T, Ptr> {
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

impl<T: DeviceRepr> CudaDevicePointerOwned<T, DevicePtr> {
    pub fn new(len: usize) -> CudaResult<Self> {
        let bytesize = len.checked_mul(std::mem::size_of::<T>()).unwrap_or(0);
        let ptr = unsafe { malloc(bytesize) }?;
        Ok(CudaDevicePointerOwned {
            inner: CudaDevicePointer {
                ptr,
                size: bytesize,
                _marker: std::marker::PhantomData,
            },
        })
    }

    pub fn new_async(len: usize, stream: &CudaStream) -> CudaResult<Self> {
        let bytesize = len.checked_mul(std::mem::size_of::<T>()).unwrap_or(0);
        let ptr = unsafe { malloc_async(bytesize, stream.inner) }?;
        Ok(CudaDevicePointerOwned {
            inner: CudaDevicePointer {
                ptr,
                size: bytesize,
                _marker: std::marker::PhantomData,
            },
        })
    }

    pub fn new_pitch(width: usize, height: usize) -> CudaResult<(Self, usize)> {
        let element_size = std::mem::size_of::<T>().try_into().unwrap_or(0);
        let width = width.checked_mul(std::mem::size_of::<T>()).unwrap_or(0);

        let (ptr, pitch) = unsafe { malloc_pitch(width, height, element_size) }?;
        Ok((
            CudaDevicePointerOwned {
                inner: CudaDevicePointer {
                    ptr,
                    size: (pitch * height),
                    _marker: std::marker::PhantomData,
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
        let (mut ptr3, pitch) = CudaDevicePointerOwned::new_pitch(1024, 768).unwrap();
        assert!(ptr3.ptr.as_device_ptr() != 0);
        assert!(pitch > 0);

        ptr.set_u8(u8::MAX, 1024).unwrap();
        ptr3.set_u32(u32::MAX, 256).unwrap();

        ptr.copy_from_device(&ptr2, 1024).unwrap();
        ptr.copy_from_device_async(&ptr2, 1024, &stream).unwrap();

        ptr.free().unwrap();
        ptr2.free_async(&stream).unwrap();
        drop(ptr3);
    }
}
