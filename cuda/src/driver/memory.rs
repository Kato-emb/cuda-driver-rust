use crate::error::{CudaResult, DropResult};

use crate::raw::memory::*;

use super::stream::CudaStream;

pub mod pinned;
pub mod pooled;
pub mod unified;

pub unsafe trait DeviceRepr: Copy + 'static {}
pub unsafe trait Align1: DeviceRepr {}
pub unsafe trait Align2: DeviceRepr {}
pub unsafe trait Align4: DeviceRepr {}

unsafe impl DeviceRepr for bool {}
unsafe impl DeviceRepr for i8 {}
unsafe impl Align1 for i8 {}
unsafe impl DeviceRepr for i16 {}
unsafe impl Align2 for i16 {}
unsafe impl DeviceRepr for i32 {}
unsafe impl Align4 for i32 {}
unsafe impl DeviceRepr for i64 {}
unsafe impl DeviceRepr for i128 {}
unsafe impl DeviceRepr for isize {}
unsafe impl DeviceRepr for u8 {}
unsafe impl Align1 for u8 {}
unsafe impl DeviceRepr for u16 {}
unsafe impl Align2 for u16 {}
unsafe impl DeviceRepr for u32 {}
unsafe impl Align4 for u32 {}
unsafe impl DeviceRepr for u64 {}
unsafe impl DeviceRepr for u128 {}
unsafe impl DeviceRepr for usize {}
unsafe impl DeviceRepr for f32 {}
unsafe impl Align4 for f32 {}
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
}

pub trait CudaSliceReadable<Repr: DeviceRepr>: CudaSliceAccess<Repr> + Sized {
    type Target: CudaSliceAccess<Repr>;

    fn index(&self, range: std::ops::Range<usize>) -> Self::Target;

    fn get<R>(&self, range: R) -> Option<Self::Target>
    where
        R: std::ops::RangeBounds<usize>,
    {
        let start = match range.start_bound() {
            std::ops::Bound::Included(&start) => start,
            std::ops::Bound::Excluded(&start) => start + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            std::ops::Bound::Included(&end) => end + 1,
            std::ops::Bound::Excluded(&end) => end,
            std::ops::Bound::Unbounded => self.len(),
        };

        if start >= end || end > self.len() {
            return None;
        }

        Some(self.index(start..end))
    }
}

pub trait CudaSliceWritable<Repr: DeviceRepr>: CudaSliceAccess<Repr> + Sized {
    type Target: CudaSliceAccess<Repr>;

    fn index_mut(&mut self, range: std::ops::Range<usize>) -> Self::Target;

    fn get_mut<R>(&mut self, range: R) -> Option<Self::Target>
    where
        R: std::ops::RangeBounds<usize>,
    {
        let start = match range.start_bound() {
            std::ops::Bound::Included(&start) => start,
            std::ops::Bound::Excluded(&start) => start + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            std::ops::Bound::Included(&end) => end + 1,
            std::ops::Bound::Excluded(&end) => end,
            std::ops::Bound::Unbounded => self.len(),
        };

        if start >= end || end > self.len() {
            return None;
        }

        Some(self.index_mut(start..end))
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
}

impl<'a, Repr: DeviceRepr, Ptr: DeviceAccessible> CudaSliceReadable<Repr>
    for CudaDeviceSlice<'a, Repr, Ptr>
{
    type Target = CudaDeviceSlice<'a, Repr, Ptr>;

    fn index(&self, range: std::ops::Range<usize>) -> Self::Target {
        let offset = self.offset + (range.start as isize);
        let len = range.end - range.start;
        debug_assert!(len <= self.len, "{} > {}", len, self.len);

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
}

impl<'a, Repr: DeviceRepr, Ptr: DeviceAccessible> CudaSliceReadable<Repr>
    for CudaDeviceSliceMut<'a, Repr, Ptr>
{
    type Target = CudaDeviceSlice<'a, Repr, Ptr>;

    fn index(&self, range: std::ops::Range<usize>) -> Self::Target {
        let offset = self.offset + (range.start as isize);
        let len = range.end - range.start;
        debug_assert!(len <= self.len, "{} > {}", len, self.len);

        CudaDeviceSlice {
            ptr: self.ptr,
            offset,
            len,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, Repr: DeviceRepr, Ptr: DeviceAccessible> CudaSliceWritable<Repr>
    for CudaDeviceSliceMut<'a, Repr, Ptr>
{
    type Target = CudaDeviceSliceMut<'a, Repr, Ptr>;

    fn index_mut(&mut self, range: std::ops::Range<usize>) -> Self::Target {
        let offset = self.offset + (range.start as isize);
        let len = range.end - range.start;
        debug_assert!(len <= self.len, "{} > {}", len, self.len);

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

    pub unsafe fn copy_from_device_async<Src>(
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

    pub unsafe fn copy_from_host_async<Src>(
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
}

impl<Repr: Align1, Dst: DeviceAccessible> CudaDeviceSliceMut<'_, Repr, Dst> {
    pub fn set_d8(&mut self, value: Repr) -> CudaResult<()> {
        debug_assert!(std::mem::size_of::<Repr>() == std::mem::size_of::<u8>());
        let num_elements = self.len();

        unsafe {
            let bits = std::mem::transmute_copy::<Repr, u8>(&value);
            set_d8(&mut self.as_ptr(), bits, num_elements)
        }
    }

    pub unsafe fn set_d8_async(&mut self, value: Repr, stream: &CudaStream) -> CudaResult<()> {
        debug_assert!(std::mem::size_of::<Repr>() == std::mem::size_of::<u8>());
        let num_elements = self.len();

        unsafe {
            let bits = std::mem::transmute_copy::<Repr, u8>(&value);
            set_d8_async(&mut self.as_ptr(), bits, num_elements, &stream.inner)
        }
    }
}

impl<Repr: Align2, Dst: DeviceAccessible> CudaDeviceSliceMut<'_, Repr, Dst> {
    pub fn set_d16(&mut self, value: Repr) -> CudaResult<()> {
        debug_assert!(std::mem::size_of::<Repr>() == std::mem::size_of::<u16>());
        let num_elements = self.len();

        unsafe {
            let bits = std::mem::transmute_copy::<Repr, u16>(&value);
            set_d16(&mut self.as_ptr(), bits, num_elements)
        }
    }

    pub unsafe fn set_d16_async(&mut self, value: Repr, stream: &CudaStream) -> CudaResult<()> {
        debug_assert!(std::mem::size_of::<Repr>() == std::mem::size_of::<u16>());
        let num_elements = self.len();

        unsafe {
            let bits = std::mem::transmute_copy::<Repr, u16>(&value);
            set_d16_async(&mut self.as_ptr(), bits, num_elements, &stream.inner)
        }
    }
}

impl<Repr: Align4, Dst: DeviceAccessible> CudaDeviceSliceMut<'_, Repr, Dst> {
    pub fn set_d32(&mut self, value: Repr) -> CudaResult<()> {
        debug_assert!(std::mem::size_of::<Repr>() == std::mem::size_of::<u32>());
        let num_elements = self.len();

        unsafe {
            let bits = std::mem::transmute_copy::<Repr, u32>(&value);
            set_d32(&mut self.as_ptr(), bits, num_elements)
        }
    }

    pub unsafe fn set_d32_async(&mut self, value: Repr, stream: &CudaStream) -> CudaResult<()> {
        debug_assert!(std::mem::size_of::<Repr>() == std::mem::size_of::<u32>());
        let num_elements = self.len();

        unsafe {
            let bits = std::mem::transmute_copy::<Repr, u32>(&value);
            set_d32_async(&mut self.as_ptr(), bits, num_elements, &stream.inner)
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

impl<'a, Repr: DeviceRepr> From<&'a [Repr]> for CudaHostSlice<'a, Repr, HostPtr> {
    fn from(slice: &'a [Repr]) -> Self {
        let len = slice.len();
        let offset = 0;
        let ptr = unsafe { HostPtr::from_raw_ptr::<Repr>(slice.as_ptr() as *mut _) };

        CudaHostSlice {
            ptr,
            offset,
            len,
            _marker: std::marker::PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct CudaHostSliceMut<'a, Repr: DeviceRepr, Ptr: HostAccessible> {
    ptr: Ptr,
    offset: isize,
    len: usize,
    _marker: std::marker::PhantomData<&'a mut [Repr]>,
}

impl<'a, Repr: DeviceRepr> From<&'a mut [Repr]> for CudaHostSlice<'a, Repr, HostPtr> {
    fn from(slice: &'a mut [Repr]) -> Self {
        let len = slice.len();
        let offset = 0;
        let ptr = unsafe { HostPtr::from_raw_ptr::<Repr>(slice.as_ptr() as *mut _) };

        CudaHostSlice {
            ptr,
            offset,
            len,
            _marker: std::marker::PhantomData,
        }
    }
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
}

impl<'a, Repr: DeviceRepr, Ptr: HostAccessible> CudaSliceReadable<Repr>
    for CudaHostSlice<'a, Repr, Ptr>
{
    type Target = CudaHostSlice<'a, Repr, Ptr>;

    fn index(&self, range: std::ops::Range<usize>) -> Self::Target {
        let offset = self.offset + (range.start as isize);
        let len = range.end - range.start;
        debug_assert!(len <= self.len, "{} > {}", len, self.len);

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
}

impl<'a, Repr: DeviceRepr, Ptr: HostAccessible> CudaSliceReadable<Repr>
    for CudaHostSliceMut<'a, Repr, Ptr>
{
    type Target = CudaHostSlice<'a, Repr, Ptr>;

    fn index(&self, range: std::ops::Range<usize>) -> Self::Target {
        let offset = self.offset + (range.start as isize);
        let len = range.end - range.start;
        debug_assert!(len <= self.len, "{} > {}", len, self.len);

        CudaHostSlice {
            ptr: self.ptr,
            offset,
            len,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, Repr: DeviceRepr, Ptr: HostAccessible> CudaSliceWritable<Repr>
    for CudaHostSliceMut<'a, Repr, Ptr>
{
    type Target = CudaHostSliceMut<'a, Repr, Ptr>;

    fn index_mut(&mut self, range: std::ops::Range<usize>) -> Self::Target {
        let offset = self.offset + (range.start as isize);
        let len = range.end - range.start;
        debug_assert!(len <= self.len, "{} > {}", len, self.len);

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

    pub unsafe fn copy_from_device_async<Src>(
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
    pub fn free(self) -> DropResult<Self> {
        match unsafe { free(self.ptr) } {
            Ok(_) => {
                // The pointer is now invalid, so we need to drop it.
                std::mem::forget(self);
                Ok(())
            }
            Err(e) => Err((self, e)),
        }
    }

    pub fn free_async(self, stream: &CudaStream) -> DropResult<Self> {
        match unsafe { free_async(self.ptr, &stream.inner) } {
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
        let bytesize = len.wrapping_mul(std::mem::size_of::<Repr>());
        let ptr = unsafe { malloc(bytesize) }?;

        Ok(Self {
            ptr,
            size: bytesize,
            _marker: std::marker::PhantomData,
        })
    }

    pub unsafe fn alloc_async(len: usize, stream: &CudaStream) -> CudaResult<Self> {
        let bytesize = len.wrapping_mul(std::mem::size_of::<Repr>());
        let ptr = unsafe { malloc_async(bytesize, &stream.inner) }?;

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
        println!("Buffer: {:?}", buffer);

        assert!(buffer.as_slice().get(..1025).is_none());
        assert!(buffer.as_slice().get(..512).is_some_and(|s| s.len() == 512));
        assert!(buffer.as_slice().get(..).is_some_and(|s| s.len() == 1024));

        let mut slice = buffer.as_mut_slice().get_mut(..512).unwrap();
        println!("Slice: {:?}", slice);
        println!(
            "offset: {}, size: {}",
            slice.byte_offset(),
            slice.byte_size()
        );
        slice.set_d32(u32::MAX).unwrap();
    }
}
