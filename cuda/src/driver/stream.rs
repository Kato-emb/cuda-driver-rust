use crate::{
    error::CudaResult,
    raw::{event::EventWaitFlags, stream::*},
};

use super::{device::CudaDevice, event::CudaEvent};

#[derive(Debug)]
pub struct CudaStream {
    pub(crate) inner: Stream,
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if let Err(e) = unsafe { destroy(self.inner) } {
            log::error!("Failed to destroy stream: {:?}", e);
        }
    }
}

impl CudaStream {
    pub fn new(flags: StreamFlags) -> CudaResult<Self> {
        let inner = unsafe { create(flags) }?;
        Ok(Self { inner })
    }

    pub fn new_with_priority(flags: StreamFlags, priority: i32) -> CudaResult<Self> {
        let inner = unsafe { create_with_priority(flags, priority) }?;
        Ok(Self { inner })
    }

    pub fn id(&self) -> CudaResult<u64> {
        unsafe { get_id(self.inner) }
    }

    pub fn device(&self) -> CudaResult<CudaDevice> {
        let device = unsafe { get_device(self.inner) }?;
        Ok(CudaDevice { inner: device })
    }

    // pub fn context(&self) {}

    pub fn flags(&self) -> CudaResult<StreamFlags> {
        unsafe { get_flags(self.inner) }
    }

    pub fn priority(&self) -> CudaResult<i32> {
        unsafe { get_priority(self.inner) }
    }

    pub fn query(&self) -> CudaResult<bool> {
        unsafe { query(self.inner) }
    }

    pub fn synchronize(&self) -> CudaResult<()> {
        unsafe { synchronize(self.inner) }
    }

    pub fn wait_event(&self, event: &CudaEvent, flags: EventWaitFlags) -> CudaResult<()> {
        unsafe { wait_event(self.inner, event.inner, flags) }
    }
}
