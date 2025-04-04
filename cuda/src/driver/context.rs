use crate::{
    driver::{device::CudaDevice, event::CudaEvent},
    error::CudaResult,
    raw::context::*,
};

pub trait ContextOps: Sized {
    fn release(ctx: &CudaContext<Self>) -> CudaResult<()>;
}

#[derive(Debug)]
pub struct Primary;
impl ContextOps for Primary {
    fn release(ctx: &CudaContext<Self>) -> CudaResult<()> {
        unsafe { primary::release(ctx.device.inner) }
    }
}

#[derive(Debug)]
pub struct Custom;
impl ContextOps for Custom {
    fn release(ctx: &CudaContext<Self>) -> CudaResult<()> {
        unsafe { destroy(ctx.inner) }
    }
}

#[derive(Debug)]
pub struct CudaContext<Ctx: ContextOps> {
    inner: Context,
    device: CudaDevice,
    _marker: std::marker::PhantomData<Ctx>,
}

impl<Ctx: ContextOps> Drop for CudaContext<Ctx> {
    fn drop(&mut self) {
        if let Err(e) = Ctx::release(self) {
            log::error!("Failed to release CUDA context: {:?}", e);
        }
    }
}

impl<Ctx: ContextOps> CudaContext<Ctx> {
    pub fn id(&self) -> CudaResult<u64> {
        unsafe { get_id(self.inner) }
    }

    pub fn api_version(&self) -> CudaResult<u32> {
        unsafe { get_api_version(self.inner) }
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn record_event(&self, event: &CudaEvent) -> CudaResult<()> {
        unsafe { record_event(self.inner, event.inner) }
    }

    pub fn wait_event(&self, event: &CudaEvent) -> CudaResult<()> {
        unsafe { wait_event(self.inner, event.inner) }
    }
}

pub type PrimaryContext = CudaContext<Primary>;
pub type CustomContext = CudaContext<Custom>;

unsafe impl Send for CustomContext {}
unsafe impl Sync for CustomContext {}

impl PrimaryContext {
    pub fn new(device: CudaDevice) -> CudaResult<Self> {
        let ctx = unsafe { primary::retain(device.inner) }?;
        Ok(Self {
            inner: ctx,
            device,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn bind_to_thread(&self) -> CudaResult<()> {
        unsafe { set_current(self.inner) }
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

impl CustomContext {
    pub fn new(device: CudaDevice, flags: ContextFlags) -> CudaResult<Self> {
        let ctx = unsafe { create(flags, device.inner) }?;
        Ok(Self {
            inner: ctx,
            device,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn push_to_thread(&self) -> CudaResult<()> {
        unsafe { push_current(self.inner) }
    }
}

pub struct CurrentContext;

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

    #[test]
    fn test_cuda_driver_context_custom_new() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let result = CustomContext::new(device, ContextFlags::SCHED_AUTO);
        assert!(
            result.is_ok(),
            "CUDA custom context creation failed: {:?}",
            result
        );
    }
}
