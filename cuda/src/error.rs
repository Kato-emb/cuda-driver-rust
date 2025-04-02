use std::fmt;

use crate::raw::error::*;

#[derive(Clone, Copy)]
pub struct CudaError {
    pub(crate) inner: ErrorCode,
}

impl fmt::Debug for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaError")
            .field("code", &(self.inner.0 as u32))
            .field("name", &self.name())
            .finish()
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name(), self.description())
    }
}

impl CudaError {
    pub fn name(&self) -> String {
        unsafe { get_error_name(self.inner) }.unwrap()
    }

    pub fn description(&self) -> String {
        unsafe { get_error_string(self.inner) }.unwrap()
    }
}

impl std::error::Error for CudaError {}

pub type CudaResult<T> = Result<T, CudaError>;

pub trait ToResult {
    fn to_result(self) -> CudaResult<()>;
}

impl ToResult for cuda_sys::ffi::CUresult {
    fn to_result(self) -> CudaResult<()> {
        if self == cuda_sys::ffi::cudaError_enum::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(CudaError {
                inner: ErrorCode(self),
            })
        }
    }
}
