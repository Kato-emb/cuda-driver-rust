use cuda_sys::ffi as sys;

use crate::{
    error::{CudaResult, ToResult},
    wrap_sys_enum, wrap_sys_handle,
};

pub mod array;
pub mod device;
pub mod ordered;
pub mod pinned;
pub mod pool;
pub mod unified;
pub mod vmm;

wrap_sys_handle!(DevicePtr, sys::CUdeviceptr);

impl std::fmt::Debug for DevicePtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DevicePtr").field("ptr", &self.0).finish()
    }
}

wrap_sys_handle!(Location, sys::CUmemLocation);
wrap_sys_handle!(AccessDesc, sys::CUmemAccessDesc);

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

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ShareableHandleFlags: u64 {
        const _ZERO = 0;
    }
}

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
        PosixFD = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        Win32 = CU_MEM_HANDLE_TYPE_WIN32,
        Win32Kmt = CU_MEM_HANDLE_TYPE_WIN32_KMT,
        Fabric = CU_MEM_HANDLE_TYPE_FABRIC,
        Max = CU_MEM_HANDLE_TYPE_MAX,
    }
);

wrap_sys_enum!(
    LocationType,
    sys::CUmemLocationType,
    {
        Invalid = CU_MEM_LOCATION_TYPE_INVALID,
        Device = CU_MEM_LOCATION_TYPE_DEVICE,
        Host = CU_MEM_LOCATION_TYPE_HOST,
        HostNuma = CU_MEM_LOCATION_TYPE_HOST_NUMA,
        HostNumaCurrent = CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT,
        Max = CU_MEM_LOCATION_TYPE_MAX,
    }
);

pub unsafe fn free(ptr: DevicePtr) -> CudaResult<()> {
    unsafe { sys::cuMemFree_v2(ptr.0) }.to_result()
}

pub unsafe fn get_info() -> CudaResult<(usize, usize)> {
    let mut free = 0;
    let mut total = 0;
    unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) }.to_result()?;

    Ok((free as usize, total as usize))
}
