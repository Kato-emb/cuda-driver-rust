use std::mem::MaybeUninit;

use cuda_sys::ffi as sys;

use crate::{
    error::{CudaResult, ToResult},
    wrap_sys_enum, wrap_sys_handle,
};

use super::device::Device;

pub mod primary;

wrap_sys_handle!(Context, sys::CUcontext);
wrap_sys_handle!(GreenContext, sys::CUgreenCtx);

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ContextFlags: u32 {
        const SCHED_AUTO = sys::CUctx_flags_enum::CU_CTX_SCHED_AUTO as u32;
        const SCHED_SPIN = sys::CUctx_flags_enum::CU_CTX_SCHED_SPIN as u32;
        const SCHED_YIELD = sys::CUctx_flags_enum::CU_CTX_SCHED_YIELD as u32;
        const SCHED_BLOCKING_SYNC = sys::CUctx_flags_enum::CU_CTX_SCHED_BLOCKING_SYNC as u32;
        const SCHED_MASK = sys::CUctx_flags_enum::CU_CTX_SCHED_MASK as u32;
        const MAP_HOST = sys::CUctx_flags_enum::CU_CTX_MAP_HOST as u32;
        const LMEM_RESIZE_TO_MAX = sys::CUctx_flags_enum::CU_CTX_LMEM_RESIZE_TO_MAX as u32;
        const COREDUMP_ENABLE = sys::CUctx_flags_enum::CU_CTX_COREDUMP_ENABLE as u32;
        const USER_COREDUMP_ENABLE = sys::CUctx_flags_enum::CU_CTX_USER_COREDUMP_ENABLE as u32;
        const SYNC_MEMOPS = sys::CUctx_flags_enum::CU_CTX_SYNC_MEMOPS as u32;
        const FLAGS_MASK = sys::CUctx_flags_enum::CU_CTX_FLAGS_MASK as u32;
    }
}

wrap_sys_enum!(
    Limit,
    sys::CUlimit,
    {
        StackSize = CU_LIMIT_STACK_SIZE,
        PrintfFifoSize = CU_LIMIT_PRINTF_FIFO_SIZE,
        MallocHeapSize = CU_LIMIT_MALLOC_HEAP_SIZE,
        DevRuntimeSyncDepth = CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH,
        DevRuntimePendingLaunchCount = CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT,
        MaxL2FetchGranularity = CU_LIMIT_MAX_L2_FETCH_GRANULARITY,
        PersitingL2CacheSize = CU_LIMIT_PERSISTING_L2_CACHE_SIZE,
        ShmemSize = CU_LIMIT_SHMEM_SIZE,
        CigEnabled = CU_LIMIT_CIG_ENABLED,
        CigShmemFallbackEnabled = CU_LIMIT_CIG_SHMEM_FALLBACK_ENABLED,
        Max = CU_LIMIT_MAX,
    }
);

pub unsafe fn create(flags: ContextFlags, device: Device) -> CudaResult<Context> {
    let mut ctx = MaybeUninit::uninit();
    unsafe { sys::cuCtxCreate_v2(ctx.as_mut_ptr(), flags.bits(), device.0) }.to_result()?;

    Ok(Context(unsafe { ctx.assume_init() }))
}

pub unsafe fn create_v3() {}
pub unsafe fn create_v4() {}

pub unsafe fn destroy(ctx: Context) -> CudaResult<()> {
    unsafe { sys::cuCtxDestroy_v2(ctx.0) }.to_result()
}

pub unsafe fn set_current(ctx: Context) -> CudaResult<()> {
    unsafe { sys::cuCtxSetCurrent(ctx.0) }.to_result()
}

pub unsafe fn get_current() -> CudaResult<Context> {
    let mut ctx = MaybeUninit::uninit();
    unsafe { sys::cuCtxGetCurrent(ctx.as_mut_ptr()) }.to_result()?;

    Ok(Context(unsafe { ctx.assume_init() }))
}

pub unsafe fn push_current(ctx: Context) -> CudaResult<()> {
    unsafe { sys::cuCtxPushCurrent_v2(ctx.0) }.to_result()
}

pub unsafe fn pop_current() -> CudaResult<Context> {
    let mut ctx = MaybeUninit::uninit();
    unsafe { sys::cuCtxPopCurrent_v2(ctx.as_mut_ptr()) }.to_result()?;

    Ok(Context(unsafe { ctx.assume_init() }))
}

pub unsafe fn set_flags(flags: ContextFlags) -> CudaResult<()> {
    unsafe { sys::cuCtxSetFlags(flags.bits()) }.to_result()
}

pub unsafe fn get_flags() -> CudaResult<ContextFlags> {
    let mut flags = 0;
    unsafe { sys::cuCtxGetFlags(&mut flags) }.to_result()?;

    Ok(ContextFlags::from_bits(flags).unwrap_or(ContextFlags::empty()))
}

pub unsafe fn get_api_version(ctx: Context) -> CudaResult<u32> {
    let mut version = 0;
    unsafe { sys::cuCtxGetApiVersion(ctx.0, &mut version) }.to_result()?;

    Ok(version)
}

pub unsafe fn synchronize() -> CudaResult<()> {
    unsafe { sys::cuCtxSynchronize() }.to_result()
}

pub unsafe fn get_id(ctx: Context) -> CudaResult<u64> {
    let mut id = 0;
    unsafe { sys::cuCtxGetId(ctx.0, &mut id) }.to_result()?;

    Ok(id)
}

pub unsafe fn get_limit(limit: Limit) -> CudaResult<usize> {
    let mut value = 0;
    unsafe { sys::cuCtxGetLimit(&mut value, limit.into()) }.to_result()?;

    Ok(value)
}

pub unsafe fn set_limit(limit: Limit, value: usize) -> CudaResult<()> {
    unsafe { sys::cuCtxSetLimit(limit.into(), value) }.to_result()
}

pub unsafe fn get_stream_priority_range() -> CudaResult<(i32, i32)> {
    let mut min = 0;
    let mut max = 0;
    unsafe { sys::cuCtxGetStreamPriorityRange(&mut min, &mut max) }.to_result()?;

    Ok((min, max))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::{device, init::*};

    #[test]
    fn test_cuda_raw_context_create() {
        unsafe { init(InitFlags::_ZERO) }.unwrap();
        let device = unsafe { device::get_device(0) }.unwrap();
        let result = unsafe { create(ContextFlags::SCHED_AUTO, device) };
        assert!(result.is_ok(), "CUDA context creation failed: {:?}", result);

        let ctx = result.unwrap();
        let destroy_result = unsafe { destroy(ctx) };
        assert!(
            destroy_result.is_ok(),
            "CUDA context destruction failed: {:?}",
            destroy_result
        );
    }
}
