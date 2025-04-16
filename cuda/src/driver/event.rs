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

    pub fn elapsed(&self, earlier: &CudaEvent) -> CudaResult<f32> {
        unsafe { elapsed_time(&earlier.inner, &self.inner) }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        driver::{context::CudaPrimaryContext, device::CudaDevice},
        raw::stream::StreamFlags,
    };

    use super::*;

    #[test]
    fn test_cuda_driver_event() {
        crate::driver::init();
        let ctx = CudaPrimaryContext::new(CudaDevice::new(0).unwrap()).unwrap();
        ctx.set_current().unwrap();
        let stream = CudaStream::new(StreamFlags::NON_BLOCKING).unwrap();

        let event = CudaEvent::new(EventFlags::DEFAULT).unwrap();
        event.record(&stream).unwrap();
        assert!(!event.query().unwrap());
        event.synchronize().unwrap();
        assert!(event.query().unwrap());

        let event2 = CudaEvent::new(EventFlags::DEFAULT).unwrap();
        event2.record(&stream).unwrap();
        event2.synchronize().unwrap();
        assert!(event2.query().unwrap());

        let elapsed = event2.elapsed(&event).unwrap();
        assert!(elapsed >= 0.0, "Elapsed time should be non-negative");
    }
}
