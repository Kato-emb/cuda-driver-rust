use crate::{
    driver::{context::CudaContext, stream::CudaStream},
    error::CudaResult,
    raw::memory::{Location, LocationType, unified::*},
};

use super::{CudaDeviceBuffer, CudaDeviceSlice, CudaDeviceSliceMut, CudaSliceAccess, DeviceRepr};

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
    fn prefetch_async(&self, location_type: LocationType, stream: &CudaStream) -> CudaResult<()> {
        let count = self.byte_size();
        let mut location = Location::default();
        location.0.type_ = location_type.into();
        // ToDo. IDを設定できるようにする（Hostの場合0固定）
        location.0.id = 0;

        unsafe { prefetch_async(&self.as_raw_ptr(), count, location, &stream.inner) }
    }
}

impl<Repr: DeviceRepr> UnifiedSlice<Repr> for CudaDeviceSlice<'_, Repr, UnifiedDevicePtr> {}
impl<Repr: DeviceRepr> UnifiedSlice<Repr> for CudaDeviceSliceMut<'_, Repr, UnifiedDevicePtr> {}

#[cfg(test)]
mod tests {
    use crate::{
        driver::{context::CudaPrimaryContext, device::CudaDevice},
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
            CudaDeviceUnifiedBuffer::<u8>::alloc_unified(1024, MemoryAttachFlags::GLOBAL).unwrap();
        let ptr_ctx = unified_buffer.context().unwrap();
        assert_eq!(ptr_ctx.inner.0, ctx.inner.0);

        let device_slice = unified_buffer.as_mut_slice();
        device_slice
            .prefetch_async(LocationType::Device, &stream)
            .unwrap();
    }
}
