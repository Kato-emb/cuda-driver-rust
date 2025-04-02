use std::mem::MaybeUninit;

use cuda_sys::ffi as sys;

use crate::error::{CudaResult, ToResult};
use crate::wrap_sys_handle;

wrap_sys_handle!(Device, sys::CUdevice);
wrap_sys_handle!(Uuid, sys::CUuuid);

pub unsafe fn get_device(ordinal: i32) -> CudaResult<Device> {
    let mut device = 0;
    unsafe { sys::cuDeviceGet(&mut device, ordinal) }.to_result()?;

    Ok(Device(device))
}

// ToDo. `CU_DEVICE_ATTRIBUTE_*`を作成する
pub unsafe fn get_attribute() {}

pub unsafe fn get_count() -> CudaResult<i32> {
    let mut count = 0;
    unsafe { sys::cuDeviceGetCount(&mut count) }.to_result()?;

    Ok(count)
}

pub unsafe fn get_exec_affinity_support() {}

#[cfg(target_os = "windows")]
pub unsafe fn get_luid(device: Device) -> CudaResult<([u8; 8], u32)> {
    let mut luid = [0i8; 8];
    let mut mask = 0;
    unsafe { sys::cuDeviceGetLuid(luid.as_mut_ptr(), &mut mask, device.0) }.to_result()?;

    Ok((luid.map(|b| b as u8), mask))
}

pub unsafe fn get_name(device: Device) -> CudaResult<String> {
    let mut name_buf = [0i8; 256];
    let size = name_buf.len() as i32;
    unsafe { sys::cuDeviceGetName(name_buf.as_mut_ptr() as *mut i8, size, device.0) }
        .to_result()?;

    let name = unsafe { std::ffi::CStr::from_ptr(name_buf.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Ok(name)
}

pub unsafe fn get_nv_sci_sync_attribute() {}
pub unsafe fn get_texture_1d_linear_max_width() {}

pub unsafe fn get_uuid(device: Device) -> CudaResult<Uuid> {
    let mut uuid = MaybeUninit::uninit();
    unsafe { sys::cuDeviceGetUuid_v2(uuid.as_mut_ptr(), device.0) }.to_result()?;

    Ok(Uuid(unsafe { uuid.assume_init() }))
}

pub unsafe fn total_mem(device: Device) -> CudaResult<usize> {
    let mut total = 0;
    unsafe { sys::cuDeviceTotalMem_v2(&mut total, device.0) }.to_result()?;

    Ok(total as usize)
}

pub unsafe fn get_mem_pool() {}
pub unsafe fn set_mem_pool() {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::init::*;

    #[test]
    fn test_cuda_raw_device_get_device() {
        unsafe { init(InitFlags::_ZERO) }.unwrap();

        let result = unsafe { get_device(0) };
        assert!(result.is_ok(), "CUDA device retrieval failed: {:?}", result);
    }

    #[test]
    fn test_cuda_raw_device_get_count() {
        unsafe { init(InitFlags::_ZERO) }.unwrap();

        let result = unsafe { get_count() };
        assert!(
            result.is_ok(),
            "CUDA device count retrieval failed: {:?}",
            result
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn test_cuda_raw_device_get_luid() {
        unsafe { init(InitFlags::_ZERO) }.unwrap();

        let device = unsafe { get_device(0).unwrap() };
        let result = unsafe { get_luid(device) };
        assert!(
            result.is_ok(),
            "CUDA device LUID retrieval failed: {:?}",
            result
        );
    }

    #[test]
    fn test_cuda_raw_device_get_name() {
        unsafe { init(InitFlags::_ZERO) }.unwrap();

        let device = unsafe { get_device(0).unwrap() };
        let result = unsafe { get_name(device) };
        assert!(
            result.is_ok(),
            "CUDA device name retrieval failed: {:?}",
            result
        );
    }

    #[test]
    fn test_cuda_raw_device_get_uuid() {
        unsafe { init(InitFlags::_ZERO) }.unwrap();

        let device = unsafe { get_device(0).unwrap() };
        let result = unsafe { get_uuid(device) };
        assert!(
            result.is_ok(),
            "CUDA device UUID retrieval failed: {:?}",
            result
        );
    }

    #[test]
    fn test_cuda_raw_device_total_mem() {
        unsafe { init(InitFlags::_ZERO) }.unwrap();

        let device = unsafe { get_device(0).unwrap() };
        let result = unsafe { total_mem(device) };
        assert!(
            result.is_ok(),
            "CUDA device total memory retrieval failed: {:?}",
            result
        );
    }
}
