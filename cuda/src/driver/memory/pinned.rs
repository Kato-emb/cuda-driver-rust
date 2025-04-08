use crate::{
    error::{CudaResult, DropResult},
    raw::memory::{HostAccessible, HostManaged, pinned::*},
};

use super::{CudaDevicePointer, CudaHostPointer};

impl CudaHostPointer<PinnedHostPtr> {
    pub fn as_device_pointer(&self) -> CudaResult<CudaDevicePointer<PinnedDevicePtr>> {
        debug_assert!(self.flags()?.contains(PinnedFlags::DEVICEMAP));
        let device_ptr = unsafe { get_device_pointer(&self.ptr, DevicePointerFlags::_ZERO) }?;

        Ok(CudaDevicePointer {
            ptr: device_ptr,
            size: self.size,
        })
    }

    pub fn flags(&self) -> CudaResult<PinnedFlags> {
        unsafe { get_flags(&self.ptr) }
    }
}

#[derive(Debug)]
pub struct CudaPinnedPointerOwned<Ptr: HostManaged> {
    inner: CudaHostPointer<Ptr>,
}

impl<Ptr: HostManaged> Drop for CudaPinnedPointerOwned<Ptr> {
    fn drop(&mut self) {
        if self.inner.ptr.as_host_ptr().is_null() {
            return;
        }

        let ptr = std::mem::replace(&mut self.inner.ptr, Ptr::null());
        let old = Self {
            inner: CudaHostPointer { ptr, size: 0 },
        };

        if let Err((_, e)) = old.free() {
            log::error!("Failed to free CUDA pinned pointer: {:?}", e);
        }
    }
}

impl<Ptr: HostManaged> std::ops::Deref for CudaPinnedPointerOwned<Ptr> {
    type Target = CudaHostPointer<Ptr>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<Ptr: HostManaged> std::ops::DerefMut for CudaPinnedPointerOwned<Ptr> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<Ptr: HostManaged> CudaPinnedPointerOwned<Ptr> {
    pub fn free(mut self) -> DropResult<Self> {
        match unsafe { free_pinned(&mut self.inner.ptr) } {
            Ok(_) => {
                // The pointer is now invalid, so we need to drop it.
                std::mem::forget(self);
                Ok(())
            }
            Err(e) => Err((self, e)),
        }
    }
}

impl CudaPinnedPointerOwned<PinnedHostPtr> {
    pub fn new(bytesize: usize) -> CudaResult<Self> {
        let ptr = unsafe { malloc_pinned(bytesize) }?;
        Ok(Self {
            inner: CudaHostPointer {
                ptr,
                size: bytesize,
            },
        })
    }

    pub fn new_with_flags(bytesize: usize, flags: PinnedFlags) -> CudaResult<Self> {
        let ptr = unsafe { malloc_pinned_with_flags(bytesize, flags) }?;
        Ok(Self {
            inner: CudaHostPointer {
                ptr,
                size: bytesize,
            },
        })
    }
}

#[derive(Debug)]
pub struct CudaPinnedPointerRegistered<T> {
    inner: CudaHostPointer<PinnedHostPtr>,
    _marker: std::marker::PhantomData<T>,
}

impl<T> Drop for CudaPinnedPointerRegistered<T> {
    fn drop(&mut self) {
        if self.inner.ptr.as_host_ptr().is_null() {
            return;
        }

        let old = std::mem::replace(&mut self.ptr, PinnedHostPtr::null());
        let old = Self {
            inner: CudaHostPointer { ptr: old, size: 0 },
            _marker: std::marker::PhantomData,
        };

        let vec = old.into_vec();
        drop(vec);
    }
}

impl<T> std::ops::Deref for CudaPinnedPointerRegistered<T> {
    type Target = CudaHostPointer<PinnedHostPtr>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> std::ops::DerefMut for CudaPinnedPointerRegistered<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T> CudaPinnedPointerRegistered<T> {
    pub fn new(src: Vec<T>, flags: HostRegisterFlags) -> CudaResult<Self> {
        let bytesize = std::mem::size_of::<T>() * src.capacity();

        let host_ptr = unsafe {
            let src = src.leak();
            let host_ptr = PinnedHostPtr::from_raw(src.as_mut_ptr() as *mut std::ffi::c_void);
            register(&host_ptr, bytesize, flags)?;

            host_ptr
        };

        Ok(Self {
            inner: CudaHostPointer {
                ptr: host_ptr,
                size: bytesize,
            },
            _marker: std::marker::PhantomData,
        })
    }

    pub fn into_vec(self) -> Vec<T> {
        let len = self.size.checked_div(std::mem::size_of::<T>()).unwrap_or(0);
        let vec = unsafe { Vec::from_raw_parts(self.inner.ptr.into_raw() as *mut T, len, len) };

        std::mem::forget(self);
        vec
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        driver::{context::CudaPrimaryContext, device::CudaDevice, memory::CudaDevicePointerOwned},
        raw::memory::HostAccessible,
    };

    use super::*;

    #[test]
    fn test_cuda_driver_memory_pinned_alloc() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let ctx = CudaPrimaryContext::new(device).unwrap();
        ctx.set_current().unwrap();

        let bytesize = 1024;
        let mut host_ptr = CudaPinnedPointerOwned::new(bytesize).unwrap();
        assert!(host_ptr.inner.ptr.as_host_ptr() != std::ptr::null_mut());
        println!("flags: {:?}", host_ptr.flags().unwrap());

        let mut pinned_ptr = host_ptr.as_device_pointer().unwrap();

        let mut device_ptr = CudaDevicePointerOwned::new(bytesize).unwrap();

        device_ptr.set_u8(127, bytesize).unwrap();
        pinned_ptr.copy_from_device(&device_ptr, bytesize).unwrap();

        for (idx, i) in host_ptr.as_slice::<u8>().iter().enumerate() {
            assert_eq!(*i, 127, "Index: {}", idx);
        }

        device_ptr.set_u16(32767, 512).unwrap();
        pinned_ptr.copy_from_device(&device_ptr, bytesize).unwrap();

        for (idx, i) in host_ptr.as_slice::<u16>().iter().enumerate() {
            assert_eq!(*i, 32767, "Index: {}", idx);
        }

        device_ptr.set_u32(u32::MAX, 256).unwrap();
        pinned_ptr.copy_from_device(&device_ptr, bytesize).unwrap();

        for (idx, i) in host_ptr.as_slice::<u32>().iter().enumerate() {
            assert_eq!(*i, u32::MAX, "Index: {}", idx);
        }

        host_ptr.as_mut_slice().copy_from_slice(&[0u8; 1024]);

        for (idx, i) in host_ptr.as_slice::<u8>().iter().enumerate() {
            assert_eq!(*i, 0, "Index: {}", idx);
        }
    }

    #[test]
    fn test_cuda_driver_memory_pinned_register() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let ctx = CudaPrimaryContext::new(device).unwrap();
        ctx.set_current().unwrap();

        let bytesize = 256;
        let host_ptr = CudaPinnedPointerRegistered::<u32>::new(
            vec![0u32; bytesize],
            HostRegisterFlags::PORTABLE | HostRegisterFlags::DEVICEMAP,
        )
        .unwrap();
        assert!(host_ptr.inner.ptr.as_host_ptr() != std::ptr::null_mut());
        println!("flags: {:?}", host_ptr.flags().unwrap());

        let mut pinned_ptr = host_ptr.as_device_pointer().unwrap();
        pinned_ptr.set_u32(12345, bytesize).unwrap();

        let vec = host_ptr.into_vec();

        for (idx, i) in vec.iter().enumerate() {
            assert_eq!(*i, 12345, "Index: {}", idx);
        }
    }
}
