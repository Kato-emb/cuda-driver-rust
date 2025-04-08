use crate::{
    error::CudaResult,
    raw::memory::{DeviceManaged, unified::*},
};

use super::{CudaDevicePointer, DeviceRepr};

pub struct CudaUnifiedPointerOwned<T: DeviceRepr, Ptr: DeviceManaged> {
    inner: CudaDevicePointer<'static, T, Ptr>,
}

impl<T: DeviceRepr> CudaUnifiedPointerOwned<T, UnifiedDevicePtr> {
    pub fn new_with_flags(len: usize, flags: MemoryAttachFlags) -> CudaResult<Self> {
        let bytesize = len.checked_mul(std::mem::size_of::<T>()).unwrap_or(0);
        let ptr = unsafe { malloc_unified(bytesize, flags) }?;
        Ok(CudaUnifiedPointerOwned {
            inner: CudaDevicePointer {
                ptr,
                size: bytesize,
                _marker: std::marker::PhantomData,
            },
        })
    }
}
