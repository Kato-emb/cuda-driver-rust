use std::mem::MaybeUninit;

use cuda_sys::ffi as sys;

use crate::{
    error::{CudaResult, ToResult},
    raw::ipc::ShareableHandle,
    wrap_sys_enum, wrap_sys_handle,
};

use super::{
    AccessDesc, AccessFlags, AllocationHandleType, DeviceAccessible, DevicePtr, Location,
    ShareableHandleFlags,
};

wrap_sys_handle!(VirtualDevicePtr, sys::CUdeviceptr);

impl DeviceAccessible for VirtualDevicePtr {
    fn as_device_ptr(&self) -> sys::CUdeviceptr {
        self.0
    }
}

wrap_sys_handle!(DeviceHandle, sys::CUmemGenericAllocationHandle);

impl std::fmt::Debug for DeviceHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceHandle")
            .field("handle", &self.0)
            .finish()
    }
}

wrap_sys_handle!(AllocationProp, sys::CUmemAllocationProp);

impl std::fmt::Debug for AllocationProp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AllocationProp")
            .field("handle", &self.0)
            .finish()
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct AddressReserveFlags: u64 {
        const _ZERO = 0;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct AllocationHandleFlags: u64 {
        const _ZERO = 0;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct MappedFlags: u64 {
        const _ZERO = 0;
    }
}

wrap_sys_enum!(
    AllocationGranularityFlags,
    sys::CUmemAllocationGranularity_flags,
    {
        Minimum = CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        Recommended = CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
    }
);

pub unsafe fn address_reserve(
    size: usize,
    alignment: usize,
    addr: DevicePtr,
    flags: AddressReserveFlags,
) -> CudaResult<VirtualDevicePtr> {
    let mut ptr = 0;
    unsafe { sys::cuMemAddressReserve(&mut ptr, size, alignment, addr.0, flags.bits()) }
        .to_result()?;

    Ok(VirtualDevicePtr(ptr))
}

pub unsafe fn address_free(ptr: VirtualDevicePtr, size: usize) -> CudaResult<()> {
    unsafe { sys::cuMemAddressFree(ptr.0, size) }.to_result()
}

pub unsafe fn create(
    size: usize,
    prop: &AllocationProp,
    flags: AllocationHandleFlags,
) -> CudaResult<DeviceHandle> {
    let mut device_handle = 0;
    unsafe { sys::cuMemCreate(&mut device_handle, size, &prop.0 as *const _, flags.bits()) }
        .to_result()?;

    Ok(DeviceHandle(device_handle))
}

// ToDo. addrの元パラメータが何かわからない
// pub unsafe fn retain<T>(addr: &T) -> CudaResult<DeviceHandle> {
//     let mut device_handle = 0;
//     unsafe {
//         sys::cuMemRetainAllocationHandle(
//             &mut device_handle,
//             addr as *const _ as *mut std::ffi::c_void,
//         )
//     }
//     .to_result()?;

//     Ok(DeviceHandle(device_handle))
// }

pub unsafe fn release(handle: DeviceHandle) -> CudaResult<()> {
    unsafe { sys::cuMemRelease(handle.0) }.to_result()
}

pub unsafe fn map(
    device_ptr: VirtualDevicePtr,
    size: usize,
    offset: usize,
    device_handle: DeviceHandle,
    flags: MappedFlags,
) -> CudaResult<()> {
    unsafe { sys::cuMemMap(device_ptr.0, size, offset, device_handle.0, flags.bits()) }.to_result()
}

pub unsafe fn unmap(device_ptr: VirtualDevicePtr, size: usize) -> CudaResult<()> {
    unsafe { sys::cuMemUnmap(device_ptr.0, size) }.to_result()
}

pub unsafe fn export_to_shareable_handle<Handle: ShareableHandle>(
    device_handle: &DeviceHandle,
    handle_type: AllocationHandleType,
    flags: ShareableHandleFlags,
) -> CudaResult<Handle> {
    let mut handle = MaybeUninit::<Handle>::uninit();
    unsafe {
        sys::cuMemExportToShareableHandle(
            handle.as_mut_ptr() as *mut std::ffi::c_void,
            device_handle.0,
            handle_type.into(),
            flags.bits(),
        )
    }
    .to_result()?;

    Ok(unsafe { handle.assume_init() })
}

pub unsafe fn import_from_shareable_handle<Handle: ShareableHandle>(
    os_handle: &Handle,
    handle_type: AllocationHandleType,
) -> CudaResult<DeviceHandle> {
    let mut device_handle = 0;
    unsafe {
        sys::cuMemImportFromShareableHandle(
            &mut device_handle,
            os_handle.as_ptr(),
            handle_type.into(),
        )
    }
    .to_result()?;

    Ok(DeviceHandle(device_handle))
}

pub unsafe fn get_access(
    location: &Location,
    device_ptr: VirtualDevicePtr,
) -> CudaResult<AccessFlags> {
    let mut flags: sys::CUmemAccess_flags = unsafe { std::mem::zeroed() };
    unsafe {
        sys::cuMemGetAccess(
            &mut flags as *mut _ as *mut u64,
            &location.0 as *const _,
            device_ptr.0,
        )
    }
    .to_result()?;

    Ok(flags.into())
}

pub unsafe fn set_access(
    device_ptr: VirtualDevicePtr,
    size: usize,
    desc: &[AccessDesc],
) -> CudaResult<()> {
    let count = desc.len();
    unsafe { sys::cuMemSetAccess(device_ptr.0, size, desc.as_ptr() as *const _, count) }.to_result()
}

pub unsafe fn get_allocation_granularity(
    prop: &AllocationProp,
    option: AllocationGranularityFlags,
) -> CudaResult<usize> {
    let mut granularity = 0;
    unsafe {
        sys::cuMemGetAllocationGranularity(&mut granularity, &prop.0 as *const _, option.into())
    }
    .to_result()?;

    Ok(granularity)
}

pub unsafe fn get_allocation_properties_from_handle(
    device_handle: &DeviceHandle,
) -> CudaResult<AllocationProp> {
    let mut prop = AllocationProp::default();
    unsafe { sys::cuMemGetAllocationPropertiesFromHandle(&mut prop.0 as *mut _, device_handle.0) }
        .to_result()?;

    Ok(prop)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::raw::{
        context, device, init,
        memory::{AllocationType, LocationType},
    };

    #[test]
    fn test_cuda_raw_memory_vmm_mapping() {
        unsafe { init::init(init::InitFlags::_ZERO) }.unwrap();
        let device = unsafe { device::get_device(0) }.unwrap();
        let ctx = unsafe { context::primary::retain(device) }.unwrap();
        unsafe { context::set_current(ctx) }.unwrap();
        println!("Current device: {:?}", device);
        println!("Current context: {:?}", unsafe { context::get_current() });

        let mut prop = AllocationProp::default();
        prop.0.type_ = AllocationType::Pinned.into();
        prop.0.location.type_ = LocationType::Device.into();
        prop.0.requestedHandleTypes = AllocationHandleType::PosixFD.into();

        let page_size =
            unsafe { get_allocation_granularity(&prop, AllocationGranularityFlags::Minimum) }
                .unwrap();
        println!("Page size: {}", page_size);

        let reserve_ptr =
            unsafe { address_reserve(page_size, 0, DevicePtr(0), AddressReserveFlags::_ZERO) }
                .unwrap();

        let device_handle =
            unsafe { create(page_size, &prop, AllocationHandleFlags::_ZERO) }.unwrap();
        println!("Device handle: {:?}", device_handle);

        let result = unsafe { map(reserve_ptr, page_size, 0, device_handle, MappedFlags::_ZERO) };
        assert!(result.is_ok());

        let prop = unsafe { get_allocation_properties_from_handle(&device_handle) }.unwrap();
        println!("Allocation properties: {:?}", prop);

        let mut desc = AccessDesc::default();
        desc.0.location.type_ = LocationType::Device.into();
        desc.0.location.id = device.into();
        desc.0.flags = AccessFlags::ReadWrite.into();
        unsafe { set_access(reserve_ptr, page_size, &[desc]) }.unwrap();

        let fd = unsafe {
            export_to_shareable_handle::<i32>(
                &device_handle,
                AllocationHandleType::PosixFD,
                ShareableHandleFlags::_ZERO,
            )
        }
        .unwrap();

        println!("Exported FD: {}", fd);

        let import_handle =
            unsafe { import_from_shareable_handle::<i32>(&fd, AllocationHandleType::PosixFD) }
                .unwrap();
        println!("Imported handle: {:?}", import_handle);
        unsafe { release(import_handle) }.unwrap();

        let result = unsafe { unmap(reserve_ptr, page_size) };
        assert!(result.is_ok());

        unsafe { release(device_handle) }.unwrap();
        unsafe { address_free(reserve_ptr, page_size) }.unwrap();
        unsafe { context::primary::release(device) }.unwrap();
    }
}
