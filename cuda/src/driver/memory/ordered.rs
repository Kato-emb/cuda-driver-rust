use std::sync::Arc;

use crate::driver::stream::CudaStream;
use crate::error::CudaResult;
use crate::raw::memory;

use super::device::CudaDeviceMemory;
use super::pool::{CudaMemoryPool, CudaMemoryPoolView};
use super::*;

pub struct CudaOrderedMemory {
    ptr: CudaDevicePointer,
    stream: Arc<CudaStream>,
}

impl std::fmt::Debug for CudaOrderedMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaOrderedMemory")
            .field("ptr", &self.ptr)
            .field("size", &self.ptr.len().ok())
            .field("stream", &self.stream)
            .finish()
    }
}

impl Drop for CudaOrderedMemory {
    fn drop(&mut self) {
        if self.ptr.inner.0 == 0 {
            return;
        }

        if let Err(e) = unsafe { memory::ordered::free_async(self.ptr.inner, self.stream.inner) } {
            log::error!("Failed to free ordered memory: {:?}", e);
        }
    }
}

impl CudaOrderedMemory {
    pub fn alloc(bytesize: usize, stream: Arc<CudaStream>) -> CudaResult<Self> {
        let inner = unsafe { memory::ordered::malloc_async(bytesize, stream.inner) }?;

        Ok(Self {
            ptr: CudaDevicePointer { inner },
            stream,
        })
    }

    pub fn alloc_from_pool(
        bytesize: usize,
        pool: &CudaMemoryPool,
        stream: Arc<CudaStream>,
    ) -> CudaResult<Self> {
        let inner =
            unsafe { memory::ordered::malloc_from_pool_async(bytesize, pool.inner, stream.inner) }?;

        Ok(Self {
            ptr: CudaDevicePointer { inner },
            stream,
        })
    }

    pub fn export(&self) -> CudaResult<memory::ordered::MemoryPoolExportData> {
        unsafe { memory::ordered::export_pointer(self.ptr.inner) }
    }

    pub fn import<Handle: memory::pool::ShareableHandle>(
        pool: &CudaMemoryPoolView<Handle>,
        data: &memory::ordered::MemoryPoolExportData,
        stream: Arc<CudaStream>,
    ) -> CudaResult<Self> {
        let inner = unsafe { memory::ordered::import_pointer(pool.inner, data) }?;

        Ok(Self {
            ptr: CudaDevicePointer { inner },
            stream,
        })
    }

    pub fn len(&self) -> usize {
        self.ptr.len().unwrap_or(0)
    }

    pub fn into_sync(mut self) -> CudaDeviceMemory {
        let ptr = self.ptr.inner;
        self.ptr.inner.0 = 0; // Prevent double free

        CudaDeviceMemory {
            ptr: CudaDevicePointer { inner: ptr },
        }
    }
}

impl CudaDeviceMemory {
    pub fn into_async(mut self, stream: Arc<CudaStream>) -> CudaOrderedMemory {
        let ptr = self.ptr.inner;
        self.ptr.inner.0 = 0; // Prevent double free

        CudaOrderedMemory {
            ptr: CudaDevicePointer { inner: ptr },
            stream,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        driver::{context::CudaPrimaryContext, device::CudaDevice},
        raw::stream::StreamFlags,
    };

    use super::*;

    #[test]
    fn test_cuda_ordered_memory() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let context = CudaPrimaryContext::new(device).unwrap();
        context.set_current().unwrap();
        let stream = Arc::new(CudaStream::new(StreamFlags::NON_BLOCKING).unwrap());
        let result = CudaOrderedMemory::alloc(1024, stream.clone());

        assert!(result.is_ok(), "Failed to allocate ordered memory");
        let memory = result.unwrap();
        assert_eq!(memory.len(), 1024, "Allocated memory size mismatch");

        println!("Allocated ordered memory: {:?}", memory);

        let memory = memory.into_sync();
        assert_eq!(memory.len(), 1024, "Converted memory size mismatch");
    }
}
