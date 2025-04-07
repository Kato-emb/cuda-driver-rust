use crate::{error::CudaResult, raw::memory};

pub mod device;
pub mod ordered;
pub mod pool;

#[derive(Debug)]
pub struct CudaDevicePointer {
    pub(crate) inner: memory::DevicePtr,
}

impl CudaDevicePointer {
    pub fn len(&self) -> CudaResult<usize> {
        unsafe { memory::device::get_address_range(self.inner) }
            .map(|(base_ptr, size)| {
                let offset = (self.inner.0 - base_ptr.0).try_into().unwrap_or(0);
                size - offset
            })
            .map_err(|e| e.into())
    }
}
