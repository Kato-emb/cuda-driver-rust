use crate::{error::CudaResult, raw::event::*};

use super::stream::CudaStream;

#[derive(Debug)]
pub struct CudaEvent {
    pub(crate) inner: Event,
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if let Err(e) = unsafe { destroy(self.inner) } {
            log::error!("Failed to destroy CUDA event: {:?}", e);
        }
    }
}

impl CudaEvent {
    pub fn new(flags: EventFlags) -> CudaResult<Self> {
        let event = unsafe { create(flags) }?;
        Ok(CudaEvent { inner: event })
    }

    pub fn elapsed(&self, earlier: &CudaEvent) -> CudaResult<f32> {
        unsafe { elapsed_time(&earlier.inner, &self.inner) }
    }

    pub fn query(&self) -> CudaResult<bool> {
        unsafe { query(&self.inner) }
    }

    pub fn record(&self, stream: &CudaStream) -> CudaResult<()> {
        unsafe { record(&self.inner, &stream.inner) }
    }

    pub fn record_with_flags(
        &self,
        stream: &CudaStream,
        flags: EventRecordFlags,
    ) -> CudaResult<()> {
        unsafe { record_with_flags(&self.inner, &stream.inner, flags) }
    }

    pub fn synchronize(&self) -> CudaResult<()> {
        unsafe { synchronize(&self.inner) }
    }
}
