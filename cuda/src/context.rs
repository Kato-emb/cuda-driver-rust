use crate::raw::context::*;
use crate::{device::CudaDevice, error::CudaResult, raw::context::Context};

pub trait Ctx: Sized {
    fn release(ctx: &CudaContext<Self>) -> CudaResult<()>;
}

#[derive(Debug)]
pub struct Primary;
impl Ctx for Primary {
    fn release(ctx: &CudaContext<Self>) -> CudaResult<()> {
        unsafe { primary::release(ctx.device.inner) }
    }
}

#[derive(Debug)]
pub struct Local;
impl Ctx for Local {
    fn release(ctx: &CudaContext<Self>) -> CudaResult<()> {
        unsafe { destroy(ctx.inner) }
    }
}

#[derive(Debug)]
pub struct CudaContext<C: Ctx> {
    inner: Context,
    device: CudaDevice,
    _marker: std::marker::PhantomData<C>,
}

impl<C: Ctx> Drop for CudaContext<C> {
    fn drop(&mut self) {
        if let Err(e) = C::release(self) {
            log::error!("Failed to release CUDA context: {:?}", e);
        }
    }
}

pub type CudaPrimaryContext = CudaContext<Primary>;
pub type CudaLocalContext = CudaContext<Local>;

impl CudaPrimaryContext {
    pub fn new(device: CudaDevice) -> CudaResult<Self> {
        let inner = unsafe { primary::retain(device.inner) }?;
        unsafe { set_current(inner) }?;

        Ok(Self {
            inner,
            device,
            _marker: std::marker::PhantomData,
        })
    }
}

impl CudaLocalContext {
    pub fn new(device: CudaDevice, flags: ContextFlags) -> CudaResult<Self> {
        let inner = unsafe { create(flags, device.inner) }?;
        Ok(Self {
            inner,
            device,
            _marker: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_context_create_primary() {
        let device = CudaDevice::new(0).unwrap();
        let result = CudaPrimaryContext::new(device);
        assert!(result.is_ok(), "CUDA context creation failed: {:?}", result);
    }

    #[test]
    fn test_cuda_context_create_local() {
        let device = CudaDevice::new(0).unwrap();
        let result = CudaLocalContext::new(device, ContextFlags::SCHED_AUTO);
        assert!(result.is_ok(), "CUDA context creation failed: {:?}", result);
    }
}
