use crate::{
    error::{CudaResult, DropResult},
    raw::memory::{
        CudaPointer, HostAccessible,
        pinned::{PinnedHostPtr, *},
    },
};

use super::{CudaDeviceSlice, CudaDeviceSliceMut, CudaHostSlice, CudaHostSliceMut, DeviceRepr};

pub trait PinnedBuffer<Repr: DeviceRepr> {
    type Ptr: DevicePinned;

    fn as_pinned_ptr(&self) -> Self::Ptr;
    fn size(&self) -> usize;

    fn as_device<'a>(&'a self) -> CudaResult<CudaDeviceSlice<'a, Repr, PinnedDevicePtr>> {
        let device_ptr =
            unsafe { get_device_pointer(&self.as_pinned_ptr(), DevicePointerFlags::_ZERO) }?;
        Ok(CudaDeviceSlice {
            ptr: device_ptr,
            offset: 0,
            len: self.size().wrapping_div(std::mem::size_of::<Repr>()),
            _marker: std::marker::PhantomData,
        })
    }

    fn as_device_mut<'a>(
        &'a mut self,
    ) -> CudaResult<CudaDeviceSliceMut<'a, Repr, PinnedDevicePtr>> {
        let device_ptr =
            unsafe { get_device_pointer(&self.as_pinned_ptr(), DevicePointerFlags::_ZERO) }?;
        Ok(CudaDeviceSliceMut {
            ptr: device_ptr,
            offset: 0,
            len: self.size().wrapping_div(std::mem::size_of::<Repr>()),
            _marker: std::marker::PhantomData,
        })
    }

    fn flags(&self) -> CudaResult<PinnedFlags> {
        unsafe { get_flags(&self.as_pinned_ptr()) }
    }
}

#[derive(Debug)]
pub struct CudaHostPinnedBuffer<Repr: DeviceRepr> {
    ptr: PinnedHostPtr,
    size: usize,
    _marker: std::marker::PhantomData<*mut Repr>,
}

impl<Repr: DeviceRepr> Drop for CudaHostPinnedBuffer<Repr> {
    fn drop(&mut self) {
        if self.ptr.as_host_ptr().is_null() {
            return;
        }

        let ptr = std::mem::replace(&mut self.ptr, unsafe {
            PinnedHostPtr::from_raw_ptr(std::ptr::null_mut::<std::ffi::c_void>())
        });
        let old = Self {
            ptr,
            size: 0,
            _marker: std::marker::PhantomData,
        };

        if let Err((_, e)) = old.free() {
            log::error!("Failed to free CUDA pinned pointer: {:?}", e);
        }
    }
}

impl<Repr: DeviceRepr> CudaHostPinnedBuffer<Repr> {
    pub fn free(mut self) -> DropResult<Self> {
        match unsafe { free_pinned(&mut self.ptr) } {
            Ok(_) => {
                // The pointer is now invalid, so we need to drop it.
                std::mem::forget(self);
                Ok(())
            }
            Err(e) => Err((self, e)),
        }
    }

    pub fn alloc(len: usize) -> CudaResult<Self> {
        let bytesize = len.wrapping_mul(std::mem::size_of::<Repr>());
        let ptr = unsafe { malloc_pinned(bytesize) }?;

        Ok(Self {
            ptr,
            size: bytesize,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn alloc_with_flags(len: usize, flags: PinnedFlags) -> CudaResult<Self> {
        let bytesize = len.wrapping_mul(std::mem::size_of::<Repr>());
        let ptr = unsafe { malloc_pinned_with_flags(bytesize, flags) }?;

        Ok(Self {
            ptr,
            size: bytesize,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn as_slice(&self) -> CudaHostSlice<Repr, PinnedHostPtr> {
        CudaHostSlice {
            ptr: self.ptr,
            offset: 0,
            len: self.size.wrapping_div(std::mem::size_of::<Repr>()),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn as_mut_slice(&mut self) -> CudaHostSliceMut<Repr, PinnedHostPtr> {
        CudaHostSliceMut {
            ptr: self.ptr,
            offset: 0,
            len: self.size.wrapping_div(std::mem::size_of::<Repr>()),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Repr: DeviceRepr> PinnedBuffer<Repr> for CudaHostPinnedBuffer<Repr> {
    type Ptr = PinnedHostPtr;

    fn as_pinned_ptr(&self) -> Self::Ptr {
        self.ptr
    }

    fn size(&self) -> usize {
        self.size
    }
}

#[derive(Debug)]
pub struct CudaHostRegisteredBuffer<Repr: DeviceRepr> {
    ptr: RegisteredHostPtr,
    size: usize,
    _marker: std::marker::PhantomData<*mut Repr>,
}

impl<Repr: DeviceRepr> Drop for CudaHostRegisteredBuffer<Repr> {
    fn drop(&mut self) {
        if self.ptr.as_host_ptr().is_null() {
            return;
        }

        let old = std::mem::replace(&mut self.ptr, unsafe {
            RegisteredHostPtr::from_raw_ptr(std::ptr::null_mut::<*mut Repr>())
        });
        let old = Self {
            ptr: old,
            size: 0,
            _marker: std::marker::PhantomData,
        };

        if let Err(e) = old.unlock() {
            log::error!("Failed to unregister CUDA host pointer: {:?}", e);
        }
    }
}

impl<Repr: DeviceRepr> CudaHostRegisteredBuffer<Repr> {
    pub fn lock_vec(src: Vec<Repr>, flags: HostRegisterFlags) -> CudaResult<Self> {
        let bytesize = std::mem::size_of::<Repr>() * src.capacity();

        let host_ptr = unsafe {
            let src = src.leak();
            let host_ptr = src.as_mut_ptr() as *mut std::ffi::c_void;
            register(host_ptr, bytesize, flags)?;
            RegisteredHostPtr::from_raw(host_ptr)
        };

        Ok(Self {
            ptr: host_ptr,
            size: bytesize,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn unlock(self) -> CudaResult<Vec<Repr>> {
        unsafe { unregister(&self.ptr) }?;
        let len = self.size.wrapping_div(std::mem::size_of::<Repr>());
        let vec = unsafe { Vec::from_raw_parts(self.ptr.into_raw() as *mut Repr, len, len) };

        std::mem::forget(self);
        Ok(vec)
    }
}

impl<Repr: DeviceRepr> PinnedBuffer<Repr> for CudaHostRegisteredBuffer<Repr> {
    type Ptr = RegisteredHostPtr;

    fn as_pinned_ptr(&self) -> Self::Ptr {
        self.ptr
    }

    fn size(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        driver::{context::CudaPrimaryContext, device::CudaDevice, memory::CudaSliceAccess},
        raw::memory::HostAccessible,
    };

    use super::*;

    #[test]
    fn test_cuda_driver_memory_pinned_alloc_u8() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let ctx = CudaPrimaryContext::new(device).unwrap();
        ctx.set_current().unwrap();

        let size = 1024;
        let mut host_buffer = CudaHostPinnedBuffer::alloc(size).unwrap();
        assert!(host_buffer.ptr.as_host_ptr() != std::ptr::null_mut());
        println!("buffer: {:?}", host_buffer);

        let mut device_slice = host_buffer.as_device_mut().unwrap().subslice(100..500);
        println!("slice: {:?}", device_slice);
        device_slice.set(127u8).unwrap();

        for (idx, i) in host_buffer.as_slice().iter().enumerate() {
            if idx >= 100 && idx < 500 {
                assert_eq!(*i, 127, "Index: {}", idx);
            } else {
                assert_eq!(*i, 0, "Index: {}", idx);
            }
        }
    }

    #[test]
    fn test_cuda_driver_memory_pinned_alloc_u16() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let ctx = CudaPrimaryContext::new(device).unwrap();
        ctx.set_current().unwrap();

        let size = 1024;
        let mut host_buffer = CudaHostPinnedBuffer::alloc(size).unwrap();
        assert!(host_buffer.ptr.as_host_ptr() != std::ptr::null_mut());
        println!("buffer: {:?}", host_buffer);

        let mut device_slice = host_buffer.as_device_mut().unwrap().subslice(100..500);
        println!("slice: {:?}", device_slice);
        device_slice.set(12345u16).unwrap();

        for (idx, i) in host_buffer.as_slice().iter().enumerate() {
            if idx >= 100 && idx < 500 {
                assert_eq!(*i, 12345, "Index: {}", idx);
            } else {
                assert_eq!(*i, 0, "Index: {}", idx);
            }
        }
    }

    #[test]
    fn test_cuda_driver_memory_pinned_alloc_u32() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let ctx = CudaPrimaryContext::new(device).unwrap();
        ctx.set_current().unwrap();

        let size = 1024;
        let mut host_buffer = CudaHostPinnedBuffer::alloc(size).unwrap();
        assert!(host_buffer.ptr.as_host_ptr() != std::ptr::null_mut());
        println!("buffer: {:?}", host_buffer);

        let mut device_slice = host_buffer.as_device_mut().unwrap().subslice(100..500);
        println!("slice: {:?}", device_slice);
        device_slice.set(12345u32).unwrap();

        for (idx, i) in host_buffer.as_slice().iter().enumerate() {
            if idx >= 100 && idx < 500 {
                assert_eq!(*i, 12345, "Index: {}", idx);
            } else {
                assert_eq!(*i, 0, "Index: {}", idx);
            }
        }
    }

    #[test]
    fn test_cuda_driver_memory_pinned_register() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let ctx = CudaPrimaryContext::new(device).unwrap();
        ctx.set_current().unwrap();

        let bytesize = 1024;
        let mut reg_buffer = CudaHostRegisteredBuffer::lock_vec(
            vec![0u32; bytesize],
            HostRegisterFlags::PORTABLE | HostRegisterFlags::DEVICEMAP,
        )
        .unwrap();
        assert!(reg_buffer.ptr.as_host_ptr() != std::ptr::null_mut());
        println!("buffer: {:?}", reg_buffer);

        let mut device_slice = reg_buffer.as_device_mut().unwrap().subslice(100..500);
        println!("slice: {:?}", device_slice);
        device_slice.set(12345).unwrap();

        let vec = reg_buffer.unlock().unwrap();

        for (idx, i) in vec.iter().enumerate() {
            if idx >= 100 && idx < 500 {
                assert_eq!(*i, 12345, "Index: {}", idx);
            } else {
                assert_eq!(*i, 0, "Index: {}", idx);
            }
        }
    }
}
