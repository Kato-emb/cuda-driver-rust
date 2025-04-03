use std::mem::MaybeUninit;

use cuda_sys::ffi as sys;

use crate::{
    error::{CudaResult, ToResult},
    wrap_sys_enum, wrap_sys_handle,
};

use super::{
    DevicePtr,
    unified::{AccessDesc, Location},
};

wrap_sys_handle!(MemoryPool, sys::CUmemoryPool);
wrap_sys_handle!(MemoryPoolProps, sys::CUmemPoolProps);
wrap_sys_handle!(MemPoolExportData, sys::CUmemPoolPtrExportData);

wrap_sys_enum!(
    AllocationType,
    sys::CUmemAllocationType,
    {
        Invalid = CU_MEM_ALLOCATION_TYPE_INVALID,
        Pinned = CU_MEM_ALLOCATION_TYPE_PINNED,
        Max = CU_MEM_ALLOCATION_TYPE_MAX,
    }
);

wrap_sys_enum!(
    AllocationHandleType,
    sys::CUmemAllocationHandleType,
    {
        None = CU_MEM_HANDLE_TYPE_NONE,
        Posix = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        Win32 = CU_MEM_HANDLE_TYPE_WIN32,
        Win32Kmt = CU_MEM_HANDLE_TYPE_WIN32_KMT,
        Fabric = CU_MEM_HANDLE_TYPE_FABRIC,
        Max = CU_MEM_HANDLE_TYPE_MAX,
    }
);

wrap_sys_enum!(
    AccessFlags,
    sys::CUmemAccess_flags,
    {
        None = CU_MEM_ACCESS_FLAGS_PROT_NONE,
        Read = CU_MEM_ACCESS_FLAGS_PROT_READ,
        ReadWrite = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        Max = CU_MEM_ACCESS_FLAGS_PROT_MAX,
    }
);

wrap_sys_enum!(
    MemoryPoolAttribute,
    sys::CUmemPool_attribute,
    {
        ReuseFollowEventDependencies = CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
        ReuseAllowOpportunistic = CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
        ReuseAllowInternalDependencies = CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
        ReleaseThreshold = CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
        ReservedMemCurrent = CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
        ReservedMemHigh = CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
        UsedMemCurrent = CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
        UsedMemHigh = CU_MEMPOOL_ATTR_USED_MEM_HIGH,
    }
);

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ShareableHandleFlags: u64 {
        const _ZERO = 0;
    }
}

pub unsafe fn create(props: &MemoryPoolProps) -> CudaResult<MemoryPool> {
    let mut pool = MaybeUninit::uninit();
    unsafe { sys::cuMemPoolCreate(pool.as_mut_ptr(), &props.0) }.to_result()?;

    Ok(MemoryPool(unsafe { pool.assume_init() }))
}

pub unsafe fn destroy(pool: MemoryPool) -> CudaResult<()> {
    unsafe { sys::cuMemPoolDestroy(pool.0) }.to_result()
}

pub unsafe fn get_access(pool: MemoryPool, location: &Location) -> CudaResult<AccessFlags> {
    let mut flags = unsafe { std::mem::zeroed() };
    unsafe {
        sys::cuMemPoolGetAccess(
            &mut flags,
            pool.0,
            &location.0 as *const _ as *mut sys::CUmemLocation,
        )
    }
    .to_result()?;

    Ok(flags.into())
}

pub unsafe fn set_access(pool: MemoryPool, map: &[AccessDesc]) -> CudaResult<()> {
    let count = map.len();
    unsafe { sys::cuMemPoolSetAccess(pool.0, map.as_ptr() as *const _, count) }.to_result()
}

pub unsafe fn get_attribute<T>(pool: MemoryPool, attr: MemoryPoolAttribute) -> CudaResult<T> {
    let mut value = MaybeUninit::<T>::uninit();
    unsafe {
        sys::cuMemPoolGetAttribute(
            pool.0,
            attr.into(),
            value.as_mut_ptr() as *mut std::ffi::c_void,
        )
    }
    .to_result()?;

    Ok(unsafe { value.assume_init() })
}

pub unsafe fn set_attribute<T>(
    pool: MemoryPool,
    attr: MemoryPoolAttribute,
    value: &T,
) -> CudaResult<()> {
    unsafe {
        sys::cuMemPoolSetAttribute(
            pool.0,
            attr.into(),
            value as *const T as *mut std::ffi::c_void,
        )
    }
    .to_result()
}

pub unsafe fn export_pointer(device_ptr: DevicePtr) -> CudaResult<MemPoolExportData> {
    let mut export_data = MaybeUninit::uninit();
    unsafe { sys::cuMemPoolExportPointer(export_data.as_mut_ptr(), device_ptr.0) }.to_result()?;

    Ok(MemPoolExportData(unsafe { export_data.assume_init() }))
}

pub unsafe fn import_pointer(
    pool: MemoryPool,
    share_data: &MemPoolExportData,
) -> CudaResult<DevicePtr> {
    let mut device_ptr = 0;
    unsafe {
        sys::cuMemPoolImportPointer(
            &mut device_ptr,
            pool.0,
            &share_data.0 as *const _ as *mut sys::CUmemPoolPtrExportData,
        )
    }
    .to_result()?;

    Ok(DevicePtr(device_ptr))
}

pub unsafe fn export_to_shareable_handle<T>(
    pool: MemoryPool,
    handle_type: AllocationHandleType,
    flags: ShareableHandleFlags,
) -> CudaResult<T> {
    let mut handle = MaybeUninit::<T>::uninit();
    unsafe {
        sys::cuMemPoolExportToShareableHandle(
            handle.as_mut_ptr() as *mut std::ffi::c_void,
            pool.0,
            handle_type.into(),
            flags.bits(),
        )
    }
    .to_result()?;

    Ok(unsafe { handle.assume_init() })
}

pub unsafe fn import_from_shareable_handle<T>(
    handle: &T,
    handle_type: AllocationHandleType,
    flags: ShareableHandleFlags,
) -> CudaResult<MemoryPool> {
    let mut pool = MaybeUninit::uninit();
    unsafe {
        sys::cuMemPoolImportFromShareableHandle(
            pool.as_mut_ptr(),
            handle as *const _ as *mut std::ffi::c_void,
            handle_type.into(),
            flags.bits(),
        )
    }
    .to_result()?;

    Ok(MemoryPool(unsafe { pool.assume_init() }))
}

pub unsafe fn trim_to(pool: MemoryPool, keep: usize) -> CudaResult<()> {
    unsafe { sys::cuMemPoolTrimTo(pool.0, keep) }.to_result()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::{context, device, init};

    #[test]
    fn test_cuda_raw_memory_pool() {
        unsafe { init::init(init::InitFlags::_ZERO) }.unwrap();
        let device = unsafe { device::get_device(0) }.unwrap();
        let ctx = unsafe { context::create(context::ContextFlags::SCHED_AUTO, device) }.unwrap();

        let mut props = MemoryPoolProps(unsafe { std::mem::zeroed() });
        props.0.allocType = AllocationType::Pinned.into();
        props.0.handleTypes = AllocationHandleType::Posix.into();
        props.0.location = sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: device.0,
        };

        let pool = unsafe { create(&props) }.unwrap();

        let mut location = Location(unsafe { std::mem::zeroed() });
        location.0.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        location.0.id = device.0;
        let access_flags = unsafe { get_access(pool, &mut location) }.unwrap();
        assert_eq!(access_flags, AccessFlags::ReadWrite);

        let attr =
            unsafe { get_attribute::<u64>(pool, MemoryPoolAttribute::ReleaseThreshold) }.unwrap();
        assert_eq!(attr, 0);

        unsafe { destroy(pool) }.unwrap();
        unsafe { context::destroy(ctx) }.unwrap();
    }
}
