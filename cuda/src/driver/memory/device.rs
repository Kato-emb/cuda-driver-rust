use crate::error::CudaResult;
use crate::raw::memory;

use super::*;

pub struct CudaDeviceMemory {
    pub(crate) ptr: CudaDevicePointer,
}

impl std::fmt::Debug for CudaDeviceMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDeviceMemory")
            .field("ptr", &self.ptr)
            .field("size", &self.ptr.len().ok())
            .finish()
    }
}

impl Drop for CudaDeviceMemory {
    fn drop(&mut self) {
        if self.ptr.inner.0 == 0 {
            return;
        }

        if let Err(e) = unsafe { memory::free(self.ptr.inner) } {
            log::error!("Failed to free device memory: {:?}", e);
        }
    }
}

impl CudaDeviceMemory {
    pub fn alloc(bytesize: usize) -> CudaResult<Self> {
        let inner = unsafe { memory::device::malloc(bytesize) }?;

        Ok(Self {
            ptr: CudaDevicePointer { inner },
        })
    }

    pub fn alloc_with_pitch(
        width: usize,
        height: usize,
        element_size: u32,
    ) -> CudaResult<(Self, usize)> {
        let (inner, pitch) = unsafe { memory::device::malloc_pitch(width, height, element_size) }?;

        Ok((
            Self {
                ptr: CudaDevicePointer { inner },
            },
            pitch,
        ))
    }

    pub fn len(&self) -> usize {
        self.ptr.len().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use crate::driver::{context::CudaPrimaryContext, device::CudaDevice};

    use super::*;

    #[test]
    fn test_cuda_device_memory_alloc() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let context = CudaPrimaryContext::new(device).unwrap();
        context.set_current().unwrap();

        let size = 1024;
        let device_memory = CudaDeviceMemory::alloc(size).unwrap();
        println!("Allocated device memory: {:?}", device_memory);
        println!("Device memory length: {}", device_memory.len());
    }

    #[test]
    fn test_cuda_device_memory_alloc_with_pitch() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let context = CudaPrimaryContext::new(device).unwrap();
        context.set_current().unwrap();

        let width = 960;
        let height = 1024;
        let element_size = 4; // Assuming 4 bytes per element (e.g., float)
        let (device_memory, pitch) =
            CudaDeviceMemory::alloc_with_pitch(width, height, element_size).unwrap();
        println!(
            "Allocated device memory with pitch: {:?}, pitch: {}",
            device_memory, pitch
        );
    }
}
