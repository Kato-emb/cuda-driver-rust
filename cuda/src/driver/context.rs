use std::ops::Deref;

use crate::{driver::event::CudaEvent, error::CudaResult, raw::context::*};

use super::device::CudaDevice;

#[derive(Debug)]
pub struct CudaContext {
    pub(crate) inner: Context,
}

impl CudaContext {
    pub fn id(&self) -> CudaResult<u64> {
        unsafe { get_id(Some(self.inner)) }
    }

    pub fn api_version(&self) -> CudaResult<u32> {
        unsafe { get_api_version(self.inner) }
    }

    pub fn record_event(&self, event: &CudaEvent) -> CudaResult<()> {
        unsafe { record_event(self.inner, event.inner) }
    }

    pub fn wait_event(&self, event: &CudaEvent) -> CudaResult<()> {
        unsafe { wait_event(self.inner, event.inner) }
    }

    pub fn set_current(&self) -> CudaResult<()> {
        unsafe { set_current(self.inner) }
    }

    pub fn push_current(&self) -> CudaResult<()> {
        unsafe { push_current(self.inner) }
    }
}

#[derive(Debug)]
pub struct CurrentContext;

impl CurrentContext {
    pub fn context(&self) -> CudaResult<CudaContext> {
        let ctx = unsafe { get_current() }?;
        Ok(CudaContext { inner: ctx })
    }

    pub fn id(&self) -> CudaResult<u64> {
        unsafe { get_id(None) }
    }

    pub fn flags(&self) -> CudaResult<ContextFlags> {
        unsafe { get_flags() }
    }

    pub fn set_flags(&self, flags: ContextFlags) -> CudaResult<()> {
        unsafe { set_flags(flags) }
    }

    pub fn cache_config(&self) -> CudaResult<CachePreference> {
        unsafe { get_cache_config() }
    }

    pub fn set_cache_config(&self, config: CachePreference) -> CudaResult<()> {
        unsafe { set_cache_config(config) }
    }

    pub fn device(&self) -> CudaResult<CudaDevice> {
        let device = unsafe { get_device() }?;
        Ok(CudaDevice { inner: device })
    }

    pub fn synchronize(&self) -> CudaResult<()> {
        unsafe { synchronize() }
    }
}

#[derive(Debug)]
pub struct PrimaryContext {
    ctx: CudaContext,
    device: CudaDevice,
}

impl Drop for PrimaryContext {
    fn drop(&mut self) {
        if let Err(e) = unsafe { primary::release(self.device.inner) } {
            log::error!("Failed to release CUDA context: {:?}", e);
        }
    }
}

impl Deref for PrimaryContext {
    type Target = CudaContext;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

impl PrimaryContext {
    pub fn new(device: CudaDevice) -> CudaResult<Self> {
        let ctx = unsafe { primary::retain(device.inner) }?;
        Ok(Self {
            ctx: CudaContext { inner: ctx },
            device,
        })
    }

    pub fn bind_to_thread(&self) -> CudaResult<()> {
        self.ctx.set_current()
    }

    pub fn state(&self) -> CudaResult<(ContextFlags, bool)> {
        unsafe { primary::get_state(self.device.inner) }
    }

    /// Destroy all allocations and reset all state on the primary context.
    /// ## Safety
    /// - ensure that no other module in the process is using the device any more.
    pub unsafe fn reset(&self) -> CudaResult<()> {
        unsafe { primary::reset(self.device.inner) }
    }

    pub fn set_flags(&self, flags: ContextFlags) -> CudaResult<()> {
        unsafe { primary::set_flags(self.device.inner, flags) }
    }
}

pub struct OwnedContext {
    ctx: CudaContext,
}

impl Drop for OwnedContext {
    fn drop(&mut self) {
        if let Err(e) = unsafe { destroy(self.ctx.inner) } {
            log::error!("Failed to destroy CUDA context: {:?}", e);
        }
    }
}

impl Deref for OwnedContext {
    type Target = CudaContext;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_driver_context_primary_new() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let result = PrimaryContext::new(device);
        assert!(
            result.is_ok(),
            "CUDA primary context creation failed: {:?}",
            result
        );

        let ctx = result.unwrap();
        assert!(ctx.state().is_ok_and(|(_flags, active)| active));

        ctx.bind_to_thread().unwrap();
    }

    // #[test]
    // fn test_cuda_driver_context_custom_new() {
    //     crate::driver::init();
    //     let device = CudaDevice::new(0).unwrap();
    //     let result = CustomContext::new(device, ContextFlags::SCHED_AUTO);
    //     assert!(
    //         result.is_ok(),
    //         "CUDA custom context creation failed: {:?}",
    //         result
    //     );
    // }
}
