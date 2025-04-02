use std::mem::MaybeUninit;

use cuda_sys::ffi as sys;

use crate::{
    error::{CudaResult, ToResult},
    wrap_sys_handle,
};

use super::{
    context::{Context, GreenContext},
    device::Device,
    event::EventWaitFlags,
};

wrap_sys_handle!(Stream, sys::CUstream);

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct StreamFlags: u32 {
        const DEFAULT = sys::CUstream_flags_enum::CU_STREAM_DEFAULT as u32;
        const NON_BLOCKING = sys::CUstream_flags_enum::CU_STREAM_NON_BLOCKING as u32;
    }
}

pub unsafe fn create(flags: StreamFlags) -> CudaResult<Stream> {
    let mut stream = MaybeUninit::uninit();
    unsafe { sys::cuStreamCreate(stream.as_mut_ptr(), flags.bits()) }.to_result()?;

    Ok(Stream(unsafe { stream.assume_init() }))
}

pub unsafe fn create_with_priority(flags: StreamFlags, priority: i32) -> CudaResult<Stream> {
    let mut stream = MaybeUninit::uninit();
    unsafe { sys::cuStreamCreateWithPriority(stream.as_mut_ptr(), flags.bits(), priority) }
        .to_result()?;

    Ok(Stream(unsafe { stream.assume_init() }))
}

pub unsafe fn destroy(stream: Stream) -> CudaResult<()> {
    unsafe { sys::cuStreamDestroy_v2(stream.0) }.to_result()
}

pub unsafe fn get_context(stream: Stream) -> CudaResult<(Context, Option<GreenContext>)> {
    let mut ctx = MaybeUninit::uninit();
    let mut green_ctx = MaybeUninit::uninit();
    unsafe { sys::cuStreamGetCtx_v2(stream.0, ctx.as_mut_ptr(), green_ctx.as_mut_ptr()) }
        .to_result()?;

    let ctx = unsafe { ctx.assume_init() };
    let green_ctx = unsafe { green_ctx.assume_init() };

    match green_ctx.is_null() {
        true => Ok((Context(ctx), None)),
        false => Ok((Context(ctx), Some(GreenContext(green_ctx)))),
    }
}

pub unsafe fn get_device(stream: Stream) -> CudaResult<Device> {
    let mut device = MaybeUninit::uninit();
    unsafe { sys::cuStreamGetDevice(stream.0, device.as_mut_ptr()) }.to_result()?;

    Ok(Device(unsafe { device.assume_init() }))
}

pub unsafe fn get_flags(stream: Stream) -> CudaResult<StreamFlags> {
    let mut flags = 0;
    unsafe { sys::cuStreamGetFlags(stream.0, &mut flags) }.to_result()?;

    Ok(StreamFlags::from_bits(flags).unwrap_or(StreamFlags::empty()))
}

pub unsafe fn get_id(stream: Stream) -> CudaResult<u64> {
    let mut id = 0;
    unsafe { sys::cuStreamGetId(stream.0, &mut id) }.to_result()?;

    Ok(id)
}

pub unsafe fn get_priority(stream: Stream) -> CudaResult<i32> {
    let mut priority = 0;
    unsafe { sys::cuStreamGetPriority(stream.0, &mut priority) }.to_result()?;

    Ok(priority)
}

pub unsafe fn query(stream: Stream) -> CudaResult<bool> {
    let ret = unsafe { sys::cuStreamQuery(stream.0) }.to_result();

    match ret {
        Ok(_) => Ok(true),
        Err(e) if e.inner.0 == sys::cudaError_enum::CUDA_ERROR_NOT_READY => Ok(false),
        Err(e) => Err(e),
    }
}

pub unsafe fn synchronize(stream: Stream) -> CudaResult<()> {
    unsafe { sys::cuStreamSynchronize(stream.0) }.to_result()
}

pub unsafe fn wait_event(
    stream: Stream,
    event: sys::CUevent,
    flags: EventWaitFlags,
) -> CudaResult<()> {
    unsafe { sys::cuStreamWaitEvent(stream.0, event, flags.bits()) }.to_result()
}
