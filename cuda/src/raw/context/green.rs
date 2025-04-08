use std::mem::MaybeUninit;

use cuda_sys::ffi as sys;

use crate::{
    error::{CudaResult, ToResult},
    raw::{
        device::Device,
        event::Event,
        stream::{Stream, StreamFlags},
    },
    wrap_sys_enum, wrap_sys_handle,
};

wrap_sys_handle!(GreenContext, sys::CUgreenCtx);
wrap_sys_handle!(ResourceDesc, sys::CUdevResourceDesc);
wrap_sys_handle!(Resource, sys::CUdevResource);

impl std::fmt::Debug for Resource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resource")
            .field("type", &self.0.type_)
            .finish()
    }
}

wrap_sys_handle!(SmResource, sys::CUdevSmResource);

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct GreenCtxCreateFlags: u32 {
        const DEFAULT_STREAM = sys::CUgreenCtxCreate_flags::CU_GREEN_CTX_DEFAULT_STREAM as u32;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct SmResourceSplitFlags: u32 {
        const IgnoreSmCoscheduling = sys::CUdevSmResourceSplit_flags::CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING as u32;
        const MaxPotentialClusterSize = sys::CUdevSmResourceSplit_flags::CU_DEV_SM_RESOURCE_SPLIT_MAX_POTENTIAL_CLUSTER_SIZE as u32;
    }
}

wrap_sys_enum!(
    ResourceType,
    sys::CUdevResourceType,
    {
        Invalid = CU_DEV_RESOURCE_TYPE_INVALID,
        Sm = CU_DEV_RESOURCE_TYPE_SM,
    }
);

pub unsafe fn create(
    desc: ResourceDesc,
    device: Device,
    flags: GreenCtxCreateFlags,
) -> CudaResult<GreenContext> {
    let mut ctx = MaybeUninit::uninit();
    unsafe { sys::cuGreenCtxCreate(ctx.as_mut_ptr(), desc.0, device.0, flags.bits()) }
        .to_result()?;

    Ok(GreenContext(unsafe { ctx.assume_init() }))
}

pub unsafe fn destroy(ctx: GreenContext) -> CudaResult<()> {
    unsafe { sys::cuGreenCtxDestroy(ctx.0) }.to_result()
}

pub unsafe fn get_resource(ctx: GreenContext, resource_type: ResourceType) -> CudaResult<Resource> {
    let mut resource = MaybeUninit::uninit();
    unsafe { sys::cuGreenCtxGetDevResource(ctx.0, resource.as_mut_ptr(), resource_type.into()) }
        .to_result()?;

    Ok(Resource(unsafe { resource.assume_init() }))
}

pub unsafe fn record_event(ctx: GreenContext, event: Event) -> CudaResult<()> {
    unsafe { sys::cuGreenCtxRecordEvent(ctx.0, event.0) }.to_result()
}

pub unsafe fn wait_event(ctx: GreenContext, event: Event) -> CudaResult<()> {
    unsafe { sys::cuGreenCtxWaitEvent(ctx.0, event.0) }.to_result()
}

pub unsafe fn stream_create(
    ctx: GreenContext,
    flags: StreamFlags,
    priority: i32,
) -> CudaResult<Stream> {
    debug_assert!(flags.contains(StreamFlags::NON_BLOCKING));
    let mut stream = MaybeUninit::uninit();
    unsafe { sys::cuGreenCtxStreamCreate(stream.as_mut_ptr(), ctx.0, flags.bits(), priority) }
        .to_result()?;

    Ok(Stream(unsafe { stream.assume_init() }))
}

pub unsafe fn get_green_ctx(stream: Stream) -> CudaResult<Option<GreenContext>> {
    let mut ctx = MaybeUninit::uninit();
    unsafe { sys::cuStreamGetGreenCtx(stream.0, ctx.as_mut_ptr()) }.to_result()?;

    let ctx = unsafe { ctx.assume_init() };

    if ctx.is_null() {
        Ok(None)
    } else {
        Ok(Some(GreenContext(ctx)))
    }
}

pub unsafe fn generate_desc(resources: &[Resource]) -> CudaResult<ResourceDesc> {
    let num_resources = resources.len().try_into().unwrap_or(0);
    let mut desc = MaybeUninit::uninit();
    unsafe {
        sys::cuDevResourceGenerateDesc(
            desc.as_mut_ptr(),
            resources.as_ptr() as *mut _,
            num_resources,
        )
    }
    .to_result()?;

    Ok(ResourceDesc(unsafe { desc.assume_init() }))
}

pub unsafe fn sm_resource_split_by_count(
    input: &Resource,
    remaining: &mut [Resource],
    flags: SmResourceSplitFlags,
    min_count: u32,
) -> CudaResult<Vec<Resource>> {
    let mut result = MaybeUninit::uninit();
    let mut num_groups = 0;

    unsafe {
        sys::cuDevSmResourceSplitByCount(
            result.as_mut_ptr(),
            &mut num_groups,
            &input.0,
            remaining.as_mut_ptr() as *mut _,
            flags.bits(),
            min_count,
        )
    }
    .to_result()?;

    let resources = unsafe {
        let mut result = Resource(result.assume_init());
        let length = num_groups.try_into().unwrap_or(0);
        Vec::from_raw_parts(&mut result as *mut Resource, length, length)
    };

    Ok(resources)
}
