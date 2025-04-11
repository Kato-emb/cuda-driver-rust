use std::mem::MaybeUninit;

use cuda_sys::ffi as sys;

use crate::{
    error::{CudaResult, ToResult},
    wrap_sys_handle,
};

use super::stream::Stream;

wrap_sys_handle!(Event, sys::CUevent);

impl std::fmt::Debug for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Pointer::fmt(&(self.0 as *mut std::ffi::c_void), f)
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct EventFlags: u32 {
        const DEFAULT = sys::CUevent_flags_enum::CU_EVENT_DEFAULT as u32;
        const BLOCKING_SYNC = sys::CUevent_flags_enum::CU_EVENT_BLOCKING_SYNC as u32;
        const DISABLE_TIMING = sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING as u32;
        const INTERPROCESS = sys::CUevent_flags_enum::CU_EVENT_INTERPROCESS as u32;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct EventRecordFlags: u32 {
        const DEFAULT = sys::CUevent_record_flags_enum::CU_EVENT_RECORD_DEFAULT as u32;
        const EXTERNAL = sys::CUevent_record_flags_enum::CU_EVENT_RECORD_EXTERNAL as u32;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct EventWaitFlags: u32 {
        const DEFAULT = sys::CUevent_wait_flags_enum::CU_EVENT_WAIT_DEFAULT as u32;
        const EXTERNAL = sys::CUevent_wait_flags_enum::CU_EVENT_WAIT_EXTERNAL as u32;
    }
}

pub unsafe fn create(flags: EventFlags) -> CudaResult<Event> {
    let mut event = MaybeUninit::uninit();
    unsafe { sys::cuEventCreate(event.as_mut_ptr(), flags.bits()) }.to_result()?;

    Ok(Event(unsafe { event.assume_init() }))
}

pub unsafe fn destroy(event: Event) -> CudaResult<()> {
    unsafe { sys::cuEventDestroy_v2(event.0) }.to_result()
}

pub unsafe fn elapsed_time(start: &Event, end: &Event) -> CudaResult<f32> {
    let mut time = 0.0;
    unsafe { sys::cuEventElapsedTime_v2(&mut time, start.0, end.0) }.to_result()?;

    Ok(time)
}

pub unsafe fn query(event: &Event) -> CudaResult<bool> {
    let ret = unsafe { sys::cuEventQuery(event.0) }.to_result();

    match ret {
        Ok(_) => Ok(true),
        Err(e) if e.inner.0 == sys::cudaError_enum::CUDA_ERROR_NOT_READY => Ok(false),
        Err(e) => Err(e),
    }
}

pub unsafe fn record(event: &Event, stream: &Stream) -> CudaResult<()> {
    unsafe { sys::cuEventRecord(event.0, stream.0) }.to_result()
}

pub unsafe fn record_with_flags(
    event: &Event,
    stream: &Stream,
    flags: EventRecordFlags,
) -> CudaResult<()> {
    unsafe { sys::cuEventRecordWithFlags(event.0, stream.0, flags.bits()) }.to_result()
}

pub unsafe fn synchronize(event: &Event) -> CudaResult<()> {
    unsafe { sys::cuEventSynchronize(event.0) }.to_result()
}

#[cfg(test)]
mod tests {
    use crate::raw::{context, device, init};

    use super::*;

    #[test]
    fn test_cuda_raw_event_create() {
        unsafe { init::init(init::InitFlags::_ZERO) }.unwrap();
        let device = unsafe { device::get_device(0) }.unwrap();
        let ctx = unsafe { context::create(context::ContextFlags::SCHED_AUTO, device) }.unwrap();

        let result = unsafe { create(EventFlags::DEFAULT) };
        assert!(result.is_ok(), "CUDA event creation failed: {:?}", result);

        let event = result.unwrap();
        let result = unsafe { query(&event) }.unwrap();
        assert!(result, "CUDA event query return false");

        let result = unsafe { destroy(event) };
        assert!(
            result.is_ok(),
            "CUDA event destruction failed: {:?}",
            result
        );

        unsafe { context::destroy(ctx) }.unwrap();
    }
}
