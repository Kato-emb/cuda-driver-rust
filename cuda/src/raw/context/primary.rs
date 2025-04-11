use std::mem::MaybeUninit;

use cuda_sys::ffi as sys;

use crate::{
    error::{CudaResult, ToResult},
    raw::device::Device,
};

use super::{Context, ContextFlags};

pub unsafe fn get_state(device: &Device) -> CudaResult<(ContextFlags, bool)> {
    let mut flags = 0;
    let mut active = 0;
    unsafe { sys::cuDevicePrimaryCtxGetState(device.0, &mut flags, &mut active) }.to_result()?;

    let flags = ContextFlags::from_bits(flags).unwrap_or(ContextFlags::empty());
    let active = active == 1;

    Ok((flags, active))
}

pub unsafe fn retain(device: &Device) -> CudaResult<Context> {
    let mut context = MaybeUninit::uninit();
    unsafe { sys::cuDevicePrimaryCtxRetain(context.as_mut_ptr(), device.0) }.to_result()?;

    Ok(Context(unsafe { context.assume_init() }))
}

pub unsafe fn reset(device: &Device) -> CudaResult<()> {
    unsafe { sys::cuDevicePrimaryCtxReset_v2(device.0) }.to_result()
}

pub unsafe fn release(device: &Device) -> CudaResult<()> {
    unsafe { sys::cuDevicePrimaryCtxRelease_v2(device.0) }.to_result()
}

pub unsafe fn set_flags(device: &Device, flags: ContextFlags) -> CudaResult<()> {
    unsafe { sys::cuDevicePrimaryCtxSetFlags_v2(device.0, flags.bits()) }.to_result()
}

#[cfg(test)]
mod tests {
    use crate::raw::{context, device, init};

    use super::*;

    #[test]
    fn test_cuda_raw_context_primary_create() {
        unsafe { init::init(init::InitFlags::_ZERO) }.unwrap();
        let device = unsafe { device::get_device(0) }.unwrap();
        let result = unsafe { retain(&device) };
        assert!(result.is_ok(), "CUDA context creation failed: {:?}", result);

        let context = result.unwrap();
        let result = unsafe { context::set_current(&context) };
        assert!(
            result.is_ok(),
            "CUDA context set current failed: {:?}",
            result
        );

        let result = unsafe { get_state(&device) };
        assert!(
            result.is_ok(),
            "CUDA context state retrieval failed: {:?}",
            result
        );
        let (flags, active) = result.unwrap();
        assert!(active, "CUDA context is not active");
        assert_eq!(
            flags,
            ContextFlags::empty(),
            "CUDA context flags do not match"
        );

        let result = unsafe { release(&device) };
        assert!(result.is_ok(), "CUDA context release failed: {:?}", result);
    }
}
