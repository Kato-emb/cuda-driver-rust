use crate::error::{CudaResult, DropResult};

use crate::raw::memory::*;

use super::stream::CudaStream;

// pub mod device;
// pub mod ordered;
// pub mod pool;
// pub mod pinned;
// pub mod unified;

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

pub struct CudaDevicePointer<Repr: DeviceRepr, Ptr: DeviceAccessible> {
    inner: Ptr,
    _marker: std::marker::PhantomData<*mut Repr>,
}

impl<Repr: DeviceRepr, Ptr: DeviceAccessible> std::fmt::Debug for CudaDevicePointer<Repr, Ptr> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ptr = self.inner.as_device_ptr() as *const std::ffi::c_void;
        std::fmt::Pointer::fmt(&ptr, f)
    }
}

impl<Repr: DeviceRepr, Ptr: DeviceAccessible> CudaDevicePointer<Repr, Ptr> {
    pub fn new(inner: Ptr) -> Self {
        CudaDevicePointer {
            inner,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn as_ptr(&self) -> *const Repr {
        self.inner.as_device_ptr() as *const Repr
    }

    pub fn as_mut_ptr(&mut self) -> *mut Repr {
        self.inner.as_device_ptr() as *mut Repr
    }

    pub fn as_raw_ptr(&self) -> u64 {
        self.inner.as_device_ptr()
    }

    pub fn is_null(&self) -> bool {
        self.inner.as_device_ptr() == 0
    }

    pub fn null() -> Self {
        CudaDevicePointer {
            inner: unsafe { Ptr::from_raw_ptr(0) },
            _marker: std::marker::PhantomData,
        }
    }

    pub unsafe fn offset(self, count: isize) -> Self {
        let base = self.inner.as_device_ptr();

        let byte_offset = count
            .checked_mul(std::mem::size_of::<Repr>() as isize)
            .unwrap_or(0);

        let inner = if byte_offset > 0 {
            let ptr = base.checked_add(byte_offset as u64).unwrap_or(base);
            unsafe { Ptr::from_raw_ptr(ptr) }
        } else {
            let ptr = base.checked_sub(byte_offset.abs() as u64).unwrap_or(base);
            unsafe { Ptr::from_raw_ptr(ptr) }
        };

        CudaDevicePointer {
            inner,
            _marker: std::marker::PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct CudaDeviceBuffer<Repr: DeviceRepr, Ptr: DeviceManaged> {
    ptr: CudaDevicePointer<Repr, Ptr>,
    size: usize,
}

impl<Repr: DeviceRepr, Ptr: DeviceManaged> Drop for CudaDeviceBuffer<Repr, Ptr> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        let ptr = std::mem::replace(&mut self.ptr, CudaDevicePointer::null());
        let old = Self { ptr, size: 0 };

        if let Err((_, e)) = old.free() {
            log::error!("Failed to free CUDA device pointer: {:?}", e);
        }
    }
}

impl<Repr: DeviceRepr, Ptr: DeviceManaged> CudaDeviceBuffer<Repr, Ptr> {
    pub fn free(mut self) -> DropResult<Self> {
        match unsafe { free(&mut self.ptr.inner) } {
            Ok(_) => {
                // The pointer is now invalid, so we need to drop it.
                std::mem::forget(self);
                Ok(())
            }
            Err(e) => Err((self, e)),
        }
    }

    pub fn free_async(mut self, stream: &CudaStream) -> DropResult<Self> {
        match unsafe { free_async(&mut self.ptr.inner, &stream.inner) } {
            Ok(_) => {
                // The pointer is now invalid, so we need to drop it.
                std::mem::forget(self);
                Ok(())
            }
            Err(e) => Err((self, e)),
        }
    }

    pub fn as_slice(&self) -> CudaDeviceSlice<Repr, Ptr> {
        CudaDeviceSlice {
            ptr: self.ptr.inner,
            offset: 0,
            len: self.size / std::mem::size_of::<Repr>(),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn as_mut_slice(&mut self) -> CudaDeviceSliceMut<Repr, Ptr> {
        CudaDeviceSliceMut {
            ptr: self.ptr.inner,
            offset: 0,
            len: self.size / std::mem::size_of::<Repr>(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Repr: DeviceRepr> CudaDeviceBuffer<Repr, DevicePtr> {
    pub fn new(len: usize) -> CudaResult<Self> {
        let bytesize = len.checked_mul(std::mem::size_of::<Repr>()).unwrap_or(0);
        let ptr = unsafe { malloc(bytesize) }?;
        Ok(Self {
            ptr: CudaDevicePointer::new(ptr),
            size: bytesize,
        })
    }

    pub fn new_async(len: usize, stream: &CudaStream) -> CudaResult<Self> {
        let bytesize = len.checked_mul(std::mem::size_of::<Repr>()).unwrap_or(0);
        let ptr = unsafe { malloc_async(bytesize, stream.inner) }?;
        Ok(Self {
            ptr: CudaDevicePointer::new(ptr),
            size: bytesize,
        })
    }

    pub fn new_pitch(width: usize, height: usize) -> CudaResult<(Self, usize)> {
        let element_size = std::mem::size_of::<Repr>().try_into().unwrap_or(0);
        let width = width.checked_mul(std::mem::size_of::<Repr>()).unwrap_or(0);

        let (ptr, pitch) = unsafe { malloc_pitch(width, height, element_size) }?;
        Ok((
            Self {
                ptr: CudaDevicePointer::new(ptr),
                size: (pitch * height),
            },
            pitch,
        ))
    }
}

#[derive(Debug)]
pub struct CudaDeviceSlice<'a, Repr: DeviceRepr, Ptr: DeviceAccessible> {
    ptr: Ptr,
    offset: isize,
    len: usize,
    _marker: std::marker::PhantomData<&'a [Repr]>,
}

#[derive(Debug)]
pub struct CudaDeviceSliceMut<'a, Repr: DeviceRepr, Ptr: DeviceAccessible> {
    ptr: Ptr,
    offset: isize,
    len: usize,
    _marker: std::marker::PhantomData<&'a mut [Repr]>,
}

pub trait DeviceSliceAccess<Repr: DeviceRepr> {
    type Ptr;

    fn offset(&self) -> isize;
    fn len(&self) -> usize;
    fn as_ptr(&self) -> Self::Ptr;

    fn byte_offset(&self) -> isize {
        self.offset()
            .wrapping_mul(std::mem::size_of::<Repr>() as isize)
    }

    fn byte_size(&self) -> usize {
        self.len().wrapping_mul(std::mem::size_of::<Repr>())
    }
}

impl<Repr: DeviceRepr, Ptr: DeviceAccessible> DeviceSliceAccess<Repr>
    for CudaDeviceSlice<'_, Repr, Ptr>
{
    type Ptr = Ptr;

    fn offset(&self) -> isize {
        self.offset
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_ptr(&self) -> Self::Ptr {
        let base = self.ptr.as_device_ptr();
        if self.offset > 0 {
            unsafe { Ptr::from_raw_ptr(base.wrapping_add(self.offset as u64)) }
        } else {
            unsafe { Ptr::from_raw_ptr(base.wrapping_sub(self.offset.abs() as u64)) }
        }
    }
}

impl<Repr: DeviceRepr, Ptr: DeviceAccessible> DeviceSliceAccess<Repr>
    for CudaDeviceSliceMut<'_, Repr, Ptr>
{
    type Ptr = Ptr;

    fn offset(&self) -> isize {
        self.offset
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_ptr(&self) -> Self::Ptr {
        let base = self.ptr.as_device_ptr();
        if self.offset > 0 {
            unsafe { Ptr::from_raw_ptr(base.wrapping_add(self.offset as u64)) }
        } else {
            unsafe { Ptr::from_raw_ptr(base.wrapping_sub(self.offset.abs() as u64)) }
        }
    }
}

impl<Repr: DeviceRepr, Dst: DeviceAccessible> CudaDeviceSliceMut<'_, Repr, Dst> {
    pub fn copy_from_device<Src>(
        &mut self,
        src: &impl DeviceSliceAccess<Repr, Ptr = Src>,
    ) -> CudaResult<()>
    where
        Src: DeviceAccessible,
    {
        debug_assert!(src.len() <= self.len());

        let byte_count = src.byte_size();
        unsafe { copy_dtod(&mut self.ptr, &src.as_ptr(), byte_count) }
    }

    pub fn copy_from_device_async<Src>(
        &mut self,
        src: &impl DeviceSliceAccess<Repr, Ptr = Src>,
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        Src: DeviceAccessible,
    {
        debug_assert!(src.len() <= self.len());

        let byte_count = src.byte_size();
        unsafe { copy_dtod_async(&mut self.ptr, &src.as_ptr(), byte_count, &stream.inner) }
    }

    pub fn copy_from_host<Src>(
        &mut self,
        src: &impl DeviceSliceAccess<Repr, Ptr = Src>,
    ) -> CudaResult<()>
    where
        Src: HostAccessible,
    {
        debug_assert!(src.len() <= self.len());

        let byte_count = src.byte_size();
        unsafe { copy_htod(&mut self.ptr, &src.as_ptr(), byte_count) }
    }

    pub fn copy_from_host_async<Src>(
        &mut self,
        src: &impl DeviceSliceAccess<Repr, Ptr = Src>,
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        Src: HostAccessible,
    {
        debug_assert!(src.len() <= self.len());

        let byte_count = src.byte_size();
        unsafe { copy_htod_async(&mut self.ptr, &src.as_ptr(), byte_count, &stream.inner) }
    }

    pub fn set(&mut self, value: Repr) -> CudaResult<()> {
        let size = std::mem::size_of::<Repr>();

        match size {
            1 => unsafe {
                let bits = std::mem::transmute_copy::<Repr, u8>(&value);
                set_d8(&mut self.ptr, bits, self.len)
            },
            2 => unsafe {
                let bits = std::mem::transmute_copy::<Repr, u16>(&value);
                set_d16(&mut self.ptr, bits, self.len)
            },
            4 => unsafe {
                let bits = std::mem::transmute_copy::<Repr, u32>(&value);
                set_d32(&mut self.ptr, bits, self.len)
            },
            _ => panic!("Unsupported size: {}", size),
        }
    }

    pub fn set_async(&mut self, value: Repr, stream: &CudaStream) -> CudaResult<()> {
        let size = std::mem::size_of::<Repr>();

        match size {
            1 => unsafe {
                let bits = std::mem::transmute_copy::<Repr, u8>(&value);
                set_d8_async(&mut self.ptr, bits, self.len, &stream.inner)
            },
            2 => unsafe {
                let bits = std::mem::transmute_copy::<Repr, u16>(&value);
                set_d16_async(&mut self.ptr, bits, self.len, &stream.inner)
            },
            4 => unsafe {
                let bits = std::mem::transmute_copy::<Repr, u32>(&value);
                set_d32_async(&mut self.ptr, bits, self.len, &stream.inner)
            },
            _ => panic!("Unsupported size: {}", size),
        }
    }
}

// #[derive(Debug)]
// pub struct CudaHostSliceInner<Repr: DeviceRepr, Ptr: HostAccessible> {
//     ptr: Ptr,
//     offset: isize,
//     len: usize,
//     _marker: std::marker::PhantomData<Repr>,
// }

// impl<Repr: DeviceRepr, Ptr: HostAccessible> AccessibleSlice<Repr, Ptr>
//     for CudaHostSliceInner<Repr, Ptr>
// {
//     fn as_ptr(&self) -> Ptr {
//         let base = self.ptr.as_host_ptr();
//         Ptr::from_raw_ptr(unsafe { base.offset(self.offset) })
//     }

//     fn len(&self) -> usize {
//         self.len
//     }
// }

// #[derive(Debug)]
// pub struct CudaHostSlice<'a, Repr: DeviceRepr, Ptr: HostAccessible> {
//     inner: &'a CudaHostSliceInner<Repr, Ptr>,
// }

// impl<Repr: DeviceRepr, Ptr: HostAccessible> AccessibleSlice<Repr, Ptr>
//     for CudaHostSlice<'_, Repr, Ptr>
// {
//     fn as_ptr(&self) -> Ptr {
//         self.inner.as_ptr()
//     }

//     fn len(&self) -> usize {
//         self.inner.len()
//     }
// }

// impl<Repr: DeviceRepr, Ptr: HostAccessible> std::ops::Deref for CudaHostSlice<'_, Repr, Ptr> {
//     type Target = CudaHostSliceInner<Repr, Ptr>;

//     fn deref(&self) -> &Self::Target {
//         &self.inner
//     }
// }

// #[derive(Debug)]
// pub struct CudaHostSliceMut<'a, Repr: DeviceRepr, Ptr: HostAccessible> {
//     inner: &'a mut CudaHostSliceInner<Repr, Ptr>,
// }

// impl<Repr: DeviceRepr, Ptr: HostAccessible> AccessibleSlice<Repr, Ptr>
//     for CudaHostSliceMut<'_, Repr, Ptr>
// {
//     fn as_ptr(&self) -> Ptr {
//         self.inner.as_ptr()
//     }

//     fn len(&self) -> usize {
//         self.inner.len()
//     }
// }

// impl<Repr: DeviceRepr, Ptr: HostAccessible> std::ops::Deref for CudaHostSliceMut<'_, Repr, Ptr> {
//     type Target = CudaHostSliceInner<Repr, Ptr>;

//     fn deref(&self) -> &Self::Target {
//         &self.inner
//     }
// }

// impl<Repr: DeviceRepr, Ptr: HostAccessible> std::ops::DerefMut for CudaHostSliceMut<'_, Repr, Ptr> {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.inner
//     }
// }

// impl<Repr: DeviceRepr, Dst: HostAccessible> CudaHostSliceInner<Repr, Dst> {
//     pub fn copy_from_device<Src>(&mut self, src: &impl AccessibleSlice<Repr, Src>) -> CudaResult<()>
//     where
//         Src: DeviceAccessible,
//     {
//         debug_assert!(src.len() <= self.len());

//         let byte_count = src
//             .len()
//             .checked_mul(std::mem::size_of::<Repr>())
//             .unwrap_or(0);
//         unsafe { copy_dtoh(&mut self.as_ptr(), &src.as_ptr(), byte_count) }
//     }

//     pub fn copy_from_device_async<Src>(
//         &mut self,
//         src: &impl AccessibleSlice<Repr, Src>,
//         stream: &CudaStream,
//     ) -> CudaResult<()>
//     where
//         Src: DeviceAccessible,
//     {
//         debug_assert!(src.len() <= self.len());

//         let byte_count = src
//             .len()
//             .checked_mul(std::mem::size_of::<Repr>())
//             .unwrap_or(0);
//         unsafe { copy_dtoh_async(&mut self.as_ptr(), &src.as_ptr(), byte_count, &stream.inner) }
//     }

//     pub fn as_slice(&self) -> &[Repr] {
//         unsafe {
//             std::slice::from_raw_parts(&self.as_ptr() as *const Dst as *const Repr, self.len())
//         }
//     }

//     pub fn as_mut_slice(&mut self) -> &mut [Repr] {
//         unsafe {
//             std::slice::from_raw_parts_mut(&mut self.as_ptr() as *mut Dst as *mut Repr, self.len())
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use crate::driver::{context::CudaPrimaryContext, device::CudaDevice};

    use super::*;

    #[test]
    fn test_cuda_driver_memory_malloc() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let ctx = CudaPrimaryContext::new(device).unwrap();
        ctx.set_current().unwrap();

        let mut buffer = CudaDeviceBuffer::new(1024).unwrap();
        assert!(
            !buffer.ptr.is_null(),
            "Failed to allocate CUDA device buffer"
        );

        let mut slice = buffer.as_mut_slice();
        println!("Slice: {:?}", slice);
        slice.set(u8::MAX).unwrap();
    }
}
