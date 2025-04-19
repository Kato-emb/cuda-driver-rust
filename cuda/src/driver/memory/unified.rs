use crate::{
    driver::stream::CudaStream,
    error::CudaResult,
    raw::memory::{Location, LocationType, unified::*},
};

use super::{
    CudaDeviceBuffer, CudaDeviceSlice, CudaDeviceSliceMut, CudaHostSlice, CudaHostSliceMut,
    CudaSliceAccess, DeviceRepr,
};

pub type CudaDeviceUnifiedBuffer<Repr> = CudaDeviceBuffer<Repr, UnifiedDevicePtr>;

impl<Repr: DeviceRepr> CudaDeviceUnifiedBuffer<Repr> {
    pub fn alloc_unified(len: usize) -> CudaResult<Self> {
        Self::alloc_unified_with_flags(len, MemoryAttachFlags::GLOBAL)
    }

    pub fn alloc_unified_with_flags(len: usize, flags: MemoryAttachFlags) -> CudaResult<Self> {
        let bytesize = len.wrapping_div(std::mem::size_of::<Repr>());
        let ptr = unsafe { malloc_unified(bytesize, flags) }?;

        Ok(Self {
            ptr,
            size: bytesize,
            _marker: std::marker::PhantomData,
        })
    }
}

pub trait UnifiedSlice<Repr: DeviceRepr>: CudaSliceAccess<Repr, Ptr = UnifiedDevicePtr> {
    unsafe fn to_host_async(
        self,
        stream: &CudaStream,
    ) -> CudaResult<CudaHostSlice<Repr, UnifiedDevicePtr>>
    where
        Self: Sized,
    {
        let device = stream.device()?;
        let count = self.byte_size();

        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#system-requirements-for-unified-memory
        if device.concurrent_managed_access()? {
            let mut location = Location::default();
            location.0.type_ = LocationType::Host.into();
            location.0.id = 0;

            unsafe { prefetch_async(&self.as_ptr(), count, location, &stream.inner) }?;
        } else {
            stream.attach_memory_async(&self, count, MemoryAttachFlags::HOST)?;
        }

        Ok(CudaHostSlice {
            ptr: self.as_raw_ptr(),
            offset: self.offset(),
            len: self.len(),
            _marker: std::marker::PhantomData,
        })
    }

    fn to_device_async(
        self,
        stream: &CudaStream,
    ) -> CudaResult<CudaDeviceSlice<Repr, UnifiedDevicePtr>>
    where
        Self: Sized,
    {
        let device = stream.device()?;
        let count = self.byte_size();

        if device.concurrent_managed_access()? {
            let mut location = Location::default();
            location.0.type_ = LocationType::Device.into();
            location.0.id = device.as_raw();

            unsafe { prefetch_async(&self.as_ptr(), count, location, &stream.inner) }?;
        } else {
            stream.attach_memory_async(&self, count, MemoryAttachFlags::GLOBAL)?;
        }

        Ok(CudaDeviceSlice {
            ptr: self.as_raw_ptr(),
            offset: self.offset(),
            len: self.len(),
            _marker: std::marker::PhantomData,
        })
    }
}

pub trait UnifiedSliceMut<Repr: DeviceRepr>: CudaSliceAccess<Repr, Ptr = UnifiedDevicePtr> {
    unsafe fn to_host_mut_async(
        self,
        stream: &CudaStream,
    ) -> CudaResult<CudaHostSliceMut<Repr, UnifiedDevicePtr>>
    where
        Self: Sized,
    {
        let device = stream.device()?;
        let count = self.byte_size();

        if device.concurrent_managed_access()? {
            let mut location = Location::default();
            location.0.type_ = LocationType::Host.into();
            location.0.id = 0;

            unsafe { prefetch_async(&self.as_ptr(), count, location, &stream.inner) }?;
        } else {
            stream.attach_memory_async(&self, count, MemoryAttachFlags::HOST)?;
        }

        Ok(CudaHostSliceMut {
            ptr: self.as_raw_ptr(),
            offset: self.offset(),
            len: self.len(),
            _marker: std::marker::PhantomData,
        })
    }

    unsafe fn to_device_mut_async(
        self,
        stream: &CudaStream,
    ) -> CudaResult<CudaDeviceSliceMut<Repr, UnifiedDevicePtr>>
    where
        Self: Sized,
    {
        let device = stream.device()?;
        let count = self.byte_size();

        if device.concurrent_managed_access()? {
            let mut location = Location::default();
            location.0.type_ = LocationType::Device.into();
            location.0.id = device.as_raw();

            unsafe { prefetch_async(&self.as_ptr(), count, location, &stream.inner) }?;
        } else {
            stream.attach_memory_async(&self, count, MemoryAttachFlags::GLOBAL)?;
        }

        Ok(CudaDeviceSliceMut {
            ptr: self.as_raw_ptr(),
            offset: self.offset(),
            len: self.len(),
            _marker: std::marker::PhantomData,
        })
    }
}

impl<Repr: DeviceRepr> UnifiedSlice<Repr> for CudaDeviceSlice<'_, Repr, UnifiedDevicePtr> {}
impl<Repr: DeviceRepr> UnifiedSlice<Repr> for CudaDeviceSliceMut<'_, Repr, UnifiedDevicePtr> {}
impl<Repr: DeviceRepr> UnifiedSliceMut<Repr> for CudaDeviceSliceMut<'_, Repr, UnifiedDevicePtr> {}

impl<Repr: DeviceRepr> UnifiedSlice<Repr> for CudaHostSlice<'_, Repr, UnifiedDevicePtr> {}
impl<Repr: DeviceRepr> UnifiedSlice<Repr> for CudaHostSliceMut<'_, Repr, UnifiedDevicePtr> {}
impl<Repr: DeviceRepr> UnifiedSliceMut<Repr> for CudaHostSliceMut<'_, Repr, UnifiedDevicePtr> {}

#[cfg(test)]
mod tests {
    use crate::{
        driver::{
            context::CudaPrimaryContext,
            device::CudaDevice,
            memory::{CudaSliceWritable, pinned::CudaHostPinnedBuffer},
        },
        raw::stream::StreamFlags,
    };

    use super::*;

    #[test]
    fn test_cuda_driver_memory_unified_alloc_prefetch() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        if !device.unified_addressing().unwrap() {
            return;
        }

        let ctx = CudaPrimaryContext::new(device).unwrap();
        ctx.set_current().unwrap();

        let stream = CudaStream::new(StreamFlags::NON_BLOCKING).unwrap();

        let mut unified_buffer = CudaDeviceUnifiedBuffer::<u8>::alloc_unified(128).unwrap();

        let mut device_slice =
            unsafe { unified_buffer.as_mut_slice().to_device_mut_async(&stream) }.unwrap();
        stream.synchronize().unwrap();
        device_slice.set_d8(127).unwrap();
        let host_slice = unsafe { device_slice.to_host_mut_async(&stream) }.unwrap();
        stream.synchronize().unwrap();
        println!("host slice: {:?}", host_slice);

        for i in host_slice.iter() {
            assert_eq!(*i, 127);
        }

        let mut device_slice = unsafe { host_slice.to_device_mut_async(&stream) }.unwrap();
        stream.synchronize().unwrap();
        println!("device slice: {:?}", device_slice);

        device_slice.get_mut(100..128).unwrap().set_d8(0).unwrap();

        let mut pinned_buffer = CudaHostPinnedBuffer::alloc(128).unwrap();
        let mut pinned_slice = pinned_buffer.as_mut_slice();

        pinned_slice
            .copy_from_device(&unified_buffer.as_slice())
            .unwrap();
        println!("pinned slice: {:?}", pinned_slice);

        for (idx, i) in pinned_slice.iter().enumerate() {
            if idx >= 100 {
                assert_eq!(*i, 0, "Index: {}", idx);
            } else {
                assert_eq!(*i, 127, "Index: {}", idx);
            }
        }
    }
}
