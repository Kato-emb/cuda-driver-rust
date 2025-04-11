use crate::error::{CudaResult, DropResult};

use crate::raw::memory::*;

use super::stream::CudaStream;

pub mod pinned;
pub mod pooled;
pub mod unified;

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

pub trait CudaSliceAccess<Repr: DeviceRepr> {
    type Ptr: CudaPointer;

    fn offset(&self) -> isize;
    fn len(&self) -> usize;
    fn as_raw_ptr(&self) -> Self::Ptr;

    fn as_ptr(&self) -> Self::Ptr {
        unsafe { self.as_raw_ptr().offset(self.byte_offset()) }
    }

    fn byte_offset(&self) -> isize {
        self.offset()
            .wrapping_mul(std::mem::size_of::<Repr>() as isize)
    }

    fn byte_size(&self) -> usize {
        self.len().wrapping_mul(std::mem::size_of::<Repr>())
    }

    fn subslice(self, range: std::ops::Range<usize>) -> Self;
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

impl<Repr: DeviceRepr, Ptr: DeviceAccessible> CudaSliceAccess<Repr>
    for CudaDeviceSlice<'_, Repr, Ptr>
{
    type Ptr = Ptr;

    fn offset(&self) -> isize {
        self.offset
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_raw_ptr(&self) -> Self::Ptr {
        self.ptr
    }

    fn subslice(self, range: std::ops::Range<usize>) -> Self {
        let offset = self.offset + (range.start as isize);
        let len = range.end - range.start;
        debug_assert!(len <= self.len);

        CudaDeviceSlice {
            ptr: self.ptr,
            offset,
            len,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Repr: DeviceRepr, Ptr: DeviceAccessible> CudaSliceAccess<Repr>
    for CudaDeviceSliceMut<'_, Repr, Ptr>
{
    type Ptr = Ptr;

    fn offset(&self) -> isize {
        self.offset
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_raw_ptr(&self) -> Self::Ptr {
        self.ptr
    }

    fn subslice(self, range: std::ops::Range<usize>) -> Self {
        let offset = self.offset + (range.start as isize);
        let len = range.end - range.start;
        debug_assert!(len <= self.len);

        CudaDeviceSliceMut {
            ptr: self.ptr,
            offset,
            len,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Repr: DeviceRepr, Dst: DeviceAccessible> CudaDeviceSliceMut<'_, Repr, Dst> {
    pub fn copy_from_device<Src>(
        &mut self,
        src: &impl CudaSliceAccess<Repr, Ptr = Src>,
    ) -> CudaResult<()>
    where
        Src: DeviceAccessible,
    {
        debug_assert!(src.len() <= self.len());

        let byte_count = src.byte_size();
        unsafe { copy_dtod(&mut self.as_ptr(), &src.as_ptr(), byte_count) }
    }

    pub fn copy_from_device_async<Src>(
        &mut self,
        src: &impl CudaSliceAccess<Repr, Ptr = Src>,
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        Src: DeviceAccessible,
    {
        debug_assert!(src.len() <= self.len());

        let byte_count = src.byte_size();
        unsafe { copy_dtod_async(&mut self.as_ptr(), &src.as_ptr(), byte_count, &stream.inner) }
    }

    pub fn copy_from_host<Src>(
        &mut self,
        src: &impl CudaSliceAccess<Repr, Ptr = Src>,
    ) -> CudaResult<()>
    where
        Src: HostAccessible,
    {
        debug_assert!(src.len() <= self.len());

        let byte_count = src.byte_size();
        unsafe { copy_htod(&mut self.as_ptr(), &src.as_ptr(), byte_count) }
    }

    pub fn copy_from_host_async<Src>(
        &mut self,
        src: &impl CudaSliceAccess<Repr, Ptr = Src>,
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        Src: HostAccessible,
    {
        debug_assert!(src.len() <= self.len());

        let byte_count = src.byte_size();
        unsafe { copy_htod_async(&mut self.as_ptr(), &src.as_ptr(), byte_count, &stream.inner) }
    }

    pub fn set(&mut self, value: Repr) -> CudaResult<()> {
        let size = std::mem::size_of::<Repr>();
        let num_elements = self.len();

        match size {
            1 => unsafe {
                let bits = std::mem::transmute_copy::<Repr, u8>(&value);
                set_d8(&mut self.as_ptr(), bits, num_elements)
            },
            2 => unsafe {
                let bits = std::mem::transmute_copy::<Repr, u16>(&value);
                set_d16(&mut self.as_ptr(), bits, num_elements)
            },
            4 => unsafe {
                let bits = std::mem::transmute_copy::<Repr, u32>(&value);
                set_d32(&mut self.as_ptr(), bits, num_elements)
            },
            _ => panic!("Unsupported size: {}", size),
        }
    }

    pub fn set_async(&mut self, value: Repr, stream: &CudaStream) -> CudaResult<()> {
        let size = std::mem::size_of::<Repr>();
        let num_elements = self.len();

        match size {
            1 => unsafe {
                let bits = std::mem::transmute_copy::<Repr, u8>(&value);
                set_d8_async(&mut self.as_ptr(), bits, num_elements, &stream.inner)
            },
            2 => unsafe {
                let bits = std::mem::transmute_copy::<Repr, u16>(&value);
                set_d16_async(&mut self.as_ptr(), bits, num_elements, &stream.inner)
            },
            4 => unsafe {
                let bits = std::mem::transmute_copy::<Repr, u32>(&value);
                set_d32_async(&mut self.as_ptr(), bits, num_elements, &stream.inner)
            },
            _ => panic!("Unsupported size: {}", size),
        }
    }
}

#[derive(Debug)]
pub struct CudaHostSlice<'a, Repr: DeviceRepr, Ptr: HostAccessible> {
    ptr: Ptr,
    offset: isize,
    len: usize,
    _marker: std::marker::PhantomData<&'a [Repr]>,
}

#[derive(Debug)]
pub struct CudaHostSliceMut<'a, Repr: DeviceRepr, Ptr: HostAccessible> {
    ptr: Ptr,
    offset: isize,
    len: usize,
    _marker: std::marker::PhantomData<&'a mut [Repr]>,
}

impl<Repr: DeviceRepr, Ptr: HostAccessible> std::ops::Deref for CudaHostSlice<'_, Repr, Ptr> {
    type Target = [Repr];

    fn deref(&self) -> &Self::Target {
        unsafe {
            std::slice::from_raw_parts(self.as_ptr().as_host_ptr() as *const Repr, self.len())
        }
    }
}

impl<Repr: DeviceRepr, Ptr: HostAccessible> std::ops::Deref for CudaHostSliceMut<'_, Repr, Ptr> {
    type Target = [Repr];

    fn deref(&self) -> &Self::Target {
        unsafe {
            std::slice::from_raw_parts(self.as_ptr().as_host_ptr() as *const Repr, self.len())
        }
    }
}

impl<Repr: DeviceRepr, Ptr: HostAccessible> std::ops::DerefMut for CudaHostSliceMut<'_, Repr, Ptr> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            std::slice::from_raw_parts_mut(&self.as_ptr() as *const Ptr as *mut Repr, self.len())
        }
    }
}

impl<Repr: DeviceRepr, Ptr: HostAccessible> CudaSliceAccess<Repr> for CudaHostSlice<'_, Repr, Ptr> {
    type Ptr = Ptr;

    fn offset(&self) -> isize {
        self.offset
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_raw_ptr(&self) -> Self::Ptr {
        self.ptr
    }

    fn subslice(self, range: std::ops::Range<usize>) -> Self {
        let offset = self.offset + (range.start as isize);
        let len = range.end - range.start;
        debug_assert!(len <= self.len);

        CudaHostSlice {
            ptr: self.ptr,
            offset,
            len,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Repr: DeviceRepr, Ptr: HostAccessible> CudaSliceAccess<Repr>
    for CudaHostSliceMut<'_, Repr, Ptr>
{
    type Ptr = Ptr;

    fn offset(&self) -> isize {
        self.offset
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_raw_ptr(&self) -> Self::Ptr {
        self.ptr
    }

    fn subslice(self, range: std::ops::Range<usize>) -> Self {
        let offset = self.offset + (range.start as isize);
        let len = range.end - range.start;

        CudaHostSliceMut {
            ptr: self.ptr,
            offset,
            len,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Repr: DeviceRepr, Dst: HostAccessible> CudaHostSliceMut<'_, Repr, Dst> {
    pub fn copy_from_device<Src>(
        &mut self,
        src: &impl CudaSliceAccess<Repr, Ptr = Src>,
    ) -> CudaResult<()>
    where
        Src: DeviceAccessible,
    {
        debug_assert!(src.len() <= self.len());

        let byte_count = self.byte_size();
        unsafe { copy_dtoh(&mut self.as_ptr(), &src.as_ptr(), byte_count) }
    }

    pub fn copy_from_device_async<Src>(
        &mut self,
        src: &impl CudaSliceAccess<Repr, Ptr = Src>,
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        Src: DeviceAccessible,
    {
        debug_assert!(src.len() <= self.len());

        let byte_count = self.byte_size();
        unsafe { copy_dtoh_async(&mut self.as_ptr(), &src.as_ptr(), byte_count, &stream.inner) }
    }
}

#[derive(Debug)]
pub struct CudaDeviceBuffer<Repr: DeviceRepr, Ptr: DeviceAllocated> {
    ptr: Ptr,
    size: usize,
    _marker: std::marker::PhantomData<*mut Repr>,
}

impl<Repr: DeviceRepr, Ptr: DeviceAllocated> Drop for CudaDeviceBuffer<Repr, Ptr> {
    fn drop(&mut self) {
        if self.ptr.as_device_ptr() == 0 {
            return;
        }

        let ptr = std::mem::replace(&mut self.ptr, unsafe {
            Ptr::from_raw_ptr(std::ptr::null_mut::<std::ffi::c_void>())
        });
        let old = Self {
            ptr,
            size: 0,
            _marker: std::marker::PhantomData,
        };

        if let Err((_, e)) = old.free() {
            log::error!("Failed to free CUDA device pointer: {:?}", e);
        }
    }
}

impl<Repr: DeviceRepr, Ptr: DeviceAllocated> CudaDeviceBuffer<Repr, Ptr> {
    pub fn free(mut self) -> DropResult<Self> {
        match unsafe { free(&mut self.ptr) } {
            Ok(_) => {
                // The pointer is now invalid, so we need to drop it.
                std::mem::forget(self);
                Ok(())
            }
            Err(e) => Err((self, e)),
        }
    }

    pub fn free_async(mut self, stream: &CudaStream) -> DropResult<Self> {
        match unsafe { free_async(&mut self.ptr, &stream.inner) } {
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
            ptr: self.ptr,
            offset: 0,
            len: self.size.wrapping_div(std::mem::size_of::<Repr>()),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn as_mut_slice(&mut self) -> CudaDeviceSliceMut<Repr, Ptr> {
        CudaDeviceSliceMut {
            ptr: self.ptr,
            offset: 0,
            len: self.size.wrapping_div(std::mem::size_of::<Repr>()),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Repr: DeviceRepr> CudaDeviceBuffer<Repr, DevicePtr> {
    pub fn alloc(len: usize) -> CudaResult<Self> {
        let bytesize = len.wrapping_div(std::mem::size_of::<Repr>());
        let ptr = unsafe { malloc(bytesize) }?;

        Ok(Self {
            ptr,
            size: bytesize,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn alloc_async(len: usize, stream: &CudaStream) -> CudaResult<Self> {
        let bytesize = len.wrapping_div(std::mem::size_of::<Repr>());
        let ptr = unsafe { malloc_async(bytesize, stream.inner) }?;

        Ok(Self {
            ptr,
            size: bytesize,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn alloc_pitch(width: usize, height: usize) -> CudaResult<(Self, usize)> {
        let element_size = std::mem::size_of::<Repr>().try_into().unwrap_or(0);
        let width = width.checked_mul(std::mem::size_of::<Repr>()).unwrap_or(0);
        let (ptr, pitch) = unsafe { malloc_pitch(width, height, element_size) }?;

        Ok((
            Self {
                ptr,
                size: (pitch * height),
                _marker: std::marker::PhantomData,
            },
            pitch,
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::driver::{context::CudaPrimaryContext, device::CudaDevice};

    use super::*;

    #[test]
    fn test_cuda_driver_memory_alloc() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let ctx = CudaPrimaryContext::new(device).unwrap();
        ctx.set_current().unwrap();

        let mut buffer = CudaDeviceBuffer::alloc(1024).unwrap();
        assert!(buffer.ptr.0 != 0, "Failed to allocate CUDA device buffer");

        let slice = buffer.as_mut_slice();
        let mut subslice = slice.subslice(100..512);
        println!("Slice: {:?}", subslice);
        println!(
            "offset: {}, size: {}",
            subslice.byte_offset(),
            subslice.byte_size()
        );
        subslice.set(u32::MAX).unwrap();
    }
}
