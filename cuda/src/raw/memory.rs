use cuda_sys::ffi as sys;

use crate::wrap_sys_handle;

wrap_sys_handle!(DevicePtr, sys::CUdeviceptr);
