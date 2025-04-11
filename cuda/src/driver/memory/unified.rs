use crate::{
    driver::{context::CudaContext, stream::CudaStream},
    error::CudaResult,
    raw::memory::{Location, LocationType, unified::*},
};

use super::{
    CudaDeviceBuffer, CudaDeviceSlice, CudaDeviceSliceMut, CudaHostSlice, CudaHostSliceMut,
    CudaSliceAccess, DeviceRepr,
};

pub type CudaDeviceUnifiedBuffer<Repr> = CudaDeviceBuffer<Repr, UnifiedDevicePtr>;

impl<Repr: DeviceRepr> CudaDeviceUnifiedBuffer<Repr> {
    pub fn alloc_unified(len: usize, flags: MemoryAttachFlags) -> CudaResult<Self> {
        let bytesize = len.wrapping_div(std::mem::size_of::<Repr>());
        let ptr = unsafe { malloc_unified(bytesize, flags) }?;

        Ok(Self {
            ptr,
            size: bytesize,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn context(&self) -> CudaResult<CudaContext> {
        let ctx = unsafe { pointer_get_attribute(PointerAttribute::Context, &self.ptr) }?;
        Ok(CudaContext { inner: ctx })
    }
}

pub trait UnifiedSlice<Repr: DeviceRepr>: CudaSliceAccess<Repr, Ptr = UnifiedDevicePtr> {
    fn prefetch_host_async(
        &self,
        stream: &CudaStream,
    ) -> CudaResult<CudaHostSlice<Repr, UnifiedDevicePtr>> {
        let count = self.byte_size();
        let mut location = Location::default();
        location.0.type_ = LocationType::Host.into();
        location.0.id = 0;

        unsafe { prefetch_async(&self.as_ptr(), count, location, &stream.inner) }?;

        Ok(CudaHostSlice {
            ptr: self.as_raw_ptr(),
            offset: self.offset(),
            len: self.len(),
            _marker: std::marker::PhantomData,
        })
    }

    fn prefetch_device_async(
        &self,
        stream: &CudaStream,
    ) -> CudaResult<CudaDeviceSlice<Repr, UnifiedDevicePtr>> {
        let device = stream.device()?;
        let count = self.byte_size();
        let mut location = Location::default();
        location.0.type_ = LocationType::Device.into();
        location.0.id = device.as_raw();

        unsafe { prefetch_async(&self.as_ptr(), count, location, &stream.inner) }?;

        Ok(CudaDeviceSlice {
            ptr: self.as_raw_ptr(),
            offset: self.offset(),
            len: self.len(),
            _marker: std::marker::PhantomData,
        })
    }
}

pub trait UnifiedSliceMut<Repr: DeviceRepr>: CudaSliceAccess<Repr, Ptr = UnifiedDevicePtr> {
    fn prefetch_host_mut_async(
        &mut self,
        stream: &CudaStream,
    ) -> CudaResult<CudaHostSliceMut<Repr, UnifiedDevicePtr>> {
        let count = self.byte_size();
        let mut location = Location::default();
        location.0.type_ = LocationType::Host.into();
        location.0.id = 0;

        unsafe { prefetch_async(&self.as_ptr(), count, location, &stream.inner) }?;

        Ok(CudaHostSliceMut {
            ptr: self.as_raw_ptr(),
            offset: self.offset(),
            len: self.len(),
            _marker: std::marker::PhantomData,
        })
    }

    fn prefetch_device_mut_async(
        &mut self,
        stream: &CudaStream,
    ) -> CudaResult<CudaDeviceSliceMut<Repr, UnifiedDevicePtr>> {
        let device = stream.device()?;
        let count = self.byte_size();
        let mut location = Location::default();
        location.0.type_ = LocationType::Device.into();
        location.0.id = device.as_raw();

        unsafe { prefetch_async(&self.as_ptr(), count, location, &stream.inner) }?;

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
            context::CudaPrimaryContext, device::CudaDevice, memory::pinned::CudaHostPinnedBuffer,
        },
        raw::stream::StreamFlags,
    };

    use super::*;

    #[test]
    fn test_cuda_driver_memory_unified_alloc() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let ctx = CudaPrimaryContext::new(device).unwrap();
        ctx.set_current().unwrap();

        let stream = CudaStream::new(StreamFlags::NON_BLOCKING).unwrap();

        let mut unified_buffer =
            CudaDeviceUnifiedBuffer::<u8>::alloc_unified(128, MemoryAttachFlags::GLOBAL).unwrap();
        let ptr_ctx = unified_buffer.context().unwrap();
        assert_eq!(ptr_ctx.inner.0, ctx.inner.0);

        let mut device_slice = unified_buffer.as_mut_slice();
        device_slice.set(127).unwrap();
        let mut host_slice = device_slice.prefetch_host_mut_async(&stream).unwrap();
        stream.synchronize().unwrap();
        println!("host slice: {:?}", host_slice);

        for i in host_slice.iter() {
            assert_eq!(*i, 127);
        }

        let device_slice = host_slice.prefetch_device_mut_async(&stream).unwrap();
        stream.synchronize().unwrap();
        println!("device slice: {:?}", device_slice);

        device_slice.subslice(100..128).set(0).unwrap();

        let mut pinned_buffer = CudaHostPinnedBuffer::alloc(128).unwrap();
        let mut pinned_slice = pinned_buffer.as_mut_slice();

        pinned_slice
            .copy_from_device_async(&unified_buffer.as_slice(), &stream)
            .unwrap();
        stream.synchronize().unwrap();
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
