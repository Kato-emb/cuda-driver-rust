use cuda_sys::ffi as sys;

use crate::{
    error::{CudaResult, ToResult},
    raw::stream::Stream,
    wrap_sys_enum, wrap_sys_handle,
};

use super::DevicePtr;

wrap_sys_handle!(Location, sys::CUmemLocation);

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct MemoryAttachFlags: u32 {
        const GLOBAL = 0x01;
        const HOST = 0x02;
        const SINGLE = 0x04;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PrefetchFlags: u32 {
        const _ZERO = 0x00;
    }
}

wrap_sys_enum!(
    Advice,
    sys::CUmem_advise,
    {
        SetReadMostly = CU_MEM_ADVISE_SET_READ_MOSTLY,
        UnsetReadMostly = CU_MEM_ADVISE_UNSET_READ_MOSTLY,
        SetPreferredLocation = CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
        UnsetPreferredLocation = CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION,
        SetAccessedBy = CU_MEM_ADVISE_SET_ACCESSED_BY,
        UnsetAccessedBy = CU_MEM_ADVISE_UNSET_ACCESSED_BY,
    }
);

wrap_sys_enum!(
    RangeAttribute,
    sys::CUmem_range_attribute,
    {
        ReadMostly = CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY,
        PreferredLocation = CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
        AccessedBy = CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY,
        LastPrefetchLocation = CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
        PreferredLocationType = CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_TYPE,
        PreferredLocationId = CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_ID,
        LastPrefetchLocationType = CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_TYPE,
        LastPrefetchLocationId = CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_ID,
    }
);

wrap_sys_enum!(
    PointerAttribute,
    sys::CUpointer_attribute,
    {
        Context = CU_POINTER_ATTRIBUTE_CONTEXT,
        MemoryType = CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        DevicePointer = CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
        HostPointer = CU_POINTER_ATTRIBUTE_HOST_POINTER,
        P2PTokens = CU_POINTER_ATTRIBUTE_P2P_TOKENS,
        SyncMemops = CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
        BufferId = CU_POINTER_ATTRIBUTE_BUFFER_ID,
        IsManaged = CU_POINTER_ATTRIBUTE_IS_MANAGED,
        DeviceOrdinal = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
        IsRegacyCudaIpcCapable = CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE,
        RangeStartAddr = CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
        RangeSize = CU_POINTER_ATTRIBUTE_RANGE_SIZE,
        Mapped = CU_POINTER_ATTRIBUTE_MAPPED,
        AllowedHandleTypes = CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,
        IsGpuDirectRdmaCapable = CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE,
        AccessFlags = CU_POINTER_ATTRIBUTE_ACCESS_FLAGS,
        MempoolHandle = CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
        MappingSize = CU_POINTER_ATTRIBUTE_MAPPING_SIZE,
        MappingBaseAddr = CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR,
        MemoryBlockId = CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID,
        IsHwDecompressCapable = CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE,
    }
);

pub unsafe fn malloc_unified(bytesize: usize, flags: MemoryAttachFlags) -> CudaResult<DevicePtr> {
    let mut ptr = 0;
    unsafe { sys::cuMemAllocManaged(&mut ptr, bytesize, flags.bits()) }.to_result()?;

    Ok(DevicePtr(ptr))
}

pub unsafe fn advise(
    device_ptr: DevicePtr,
    count: usize,
    advice: Advice,
    location: Location,
) -> CudaResult<()> {
    unsafe { sys::cuMemAdvise_v2(device_ptr.0, count, advice.into(), location.0) }.to_result()
}

pub unsafe fn prefetch_async(
    device_ptr: DevicePtr,
    count: usize,
    location: Location,
    stream: Stream,
) -> CudaResult<()> {
    let flags = PrefetchFlags::_ZERO;
    unsafe { sys::cuMemPrefetchAsync_v2(device_ptr.0, count, location.0, flags.bits(), stream.0) }
        .to_result()
}

pub unsafe fn range_get_attribute<T: Sized>(
    attribute: RangeAttribute,
    device_ptr: DevicePtr,
    count: usize,
) -> CudaResult<T> {
    let mut data = std::mem::MaybeUninit::<T>::uninit();
    let size = std::mem::size_of::<T>();

    unsafe {
        sys::cuMemRangeGetAttribute(
            data.as_mut_ptr() as *mut std::ffi::c_void,
            size,
            attribute.into(),
            device_ptr.0,
            count,
        )
    }
    .to_result()?;

    Ok(unsafe { data.assume_init() })
}

pub unsafe fn range_get_attributes() {}

pub unsafe fn pointer_get_attribute<T>(
    attribute: PointerAttribute,
    device_ptr: DevicePtr,
) -> CudaResult<T> {
    let mut data = std::mem::MaybeUninit::<T>::uninit();

    unsafe {
        sys::cuPointerGetAttribute(
            data.as_mut_ptr() as *mut std::ffi::c_void,
            attribute.into(),
            device_ptr.0,
        )
    }
    .to_result()?;

    Ok(unsafe { data.assume_init() })
}

pub unsafe fn pointer_get_attributes() {}

pub unsafe fn pointer_set_attribute<T>(
    value: *const T,
    attribute: PointerAttribute,
    device_ptr: DevicePtr,
) -> CudaResult<()> {
    unsafe {
        sys::cuPointerSetAttribute(
            value as *const std::ffi::c_void,
            attribute.into(),
            device_ptr.0,
        )
    }
    .to_result()
}
