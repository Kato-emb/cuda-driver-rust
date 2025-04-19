use std::collections::HashSet;

use crate::{
    assert_initialized,
    error::CudaResult,
    raw::{device::*, memory::AllocationHandleType},
};

use super::memory::pooled::CudaMemoryPool;

pub struct CudaDevice {
    pub(crate) inner: Device,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDevice")
            .field("handle", &self.inner)
            .field("name", &self.name().ok())
            .field("uuid", &self.uuid().ok())
            .field("total_memory", &self.total_memory().ok())
            .finish()
    }
}

impl CudaDevice {
    pub fn count() -> CudaResult<i32> {
        assert_initialized!();
        unsafe { get_count() }
    }

    /// Creates a new Device handle from device number. Default GPU is `0`.
    pub fn new(ordinal: i32) -> CudaResult<Self> {
        assert_initialized!();
        debug_assert!((0..=Self::count()?).contains(&ordinal));
        let inner = unsafe { get_device(ordinal) }?;
        Ok(Self { inner })
    }

    pub fn as_raw(&self) -> i32 {
        self.inner.0
    }

    pub fn name(&self) -> CudaResult<String> {
        unsafe { get_name(self.inner) }
    }

    pub fn uuid(&self) -> CudaResult<uuid::Uuid> {
        let c_uuid = unsafe { get_uuid(self.inner) }?;
        let raw = c_uuid.0.bytes.map(|b| b as u8);

        Ok(uuid::Uuid::from_bytes(raw))
    }

    pub fn total_memory(&self) -> CudaResult<usize> {
        unsafe { total_mem(self.inner) }
    }

    pub fn attribute(&self, attr: DeviceAttribute) -> CudaResult<i32> {
        unsafe { get_attribute(attr, self.inner) }
    }

    pub fn unified_addressing(&self) -> CudaResult<bool> {
        self.attribute(DeviceAttribute::UnifiedAddressing)
            .map(|v| v == 1)
    }

    /// Device can coherently access managed memory concurrently with the CPU.
    /// ### Returns
    /// - `true` : Device can access managed memory concurrently with the CPU.
    /// - `false`: Device cannot access managed memory concurrently with the CPU. Not prefetchable.
    pub fn concurrent_managed_access(&self) -> CudaResult<bool> {
        self.attribute(DeviceAttribute::ConcurrentManagedAccess)
            .map(|v| v == 1)
    }

    pub fn pageable_memory_access(&self) -> CudaResult<bool> {
        self.attribute(DeviceAttribute::PageableMemoryAccess)
            .map(|v| v == 1)
    }

    pub fn memory_pool_supported(&self) -> CudaResult<bool> {
        self.attribute(DeviceAttribute::MemoryPoolsSupported)
            .map(|v| v == 1)
    }

    pub fn virtual_memory_management_supported(&self) -> CudaResult<bool> {
        self.attribute(DeviceAttribute::VirtualAddressManagementSupported)
            .map(|v| v == 1)
    }

    pub fn handle_type_supported(&self, handle_type: AllocationHandleType) -> CudaResult<bool> {
        match handle_type {
            AllocationHandleType::None => Ok(true),
            AllocationHandleType::PosixFD => self
                .attribute(DeviceAttribute::HandleTypePosixFileDescriptorSupported)
                .map(|v| v == 1),
            AllocationHandleType::Win32 => self
                .attribute(DeviceAttribute::HandleTypeWin32HandleSupported)
                .map(|v| v == 1),
            AllocationHandleType::Win32Kmt => self
                .attribute(DeviceAttribute::HandleTypeWin32KmtHandleSupported)
                .map(|v| v == 1),
            AllocationHandleType::Fabric => self
                .attribute(DeviceAttribute::HandleTypeFabricSupported)
                .map(|v| v == 1),
            AllocationHandleType::Max | AllocationHandleType::__Unknown(_) => Ok(false),
        }
    }

    pub fn mempool_supported_handle_types(&self) -> CudaResult<HashSet<AllocationHandleType>> {
        let mut types = HashSet::new();

        let flags = self.attribute(DeviceAttribute::MempoolSupportedHandleTypes)? as u32;

        if flags & AllocationHandleType::PosixFD.as_raw() as u32 != 0 {
            types.insert(AllocationHandleType::PosixFD);
        }

        if flags & AllocationHandleType::Win32.as_raw() as u32 != 0 {
            types.insert(AllocationHandleType::Win32);
        }

        if flags & AllocationHandleType::Win32Kmt.as_raw() as u32 != 0 {
            types.insert(AllocationHandleType::Win32Kmt);
        }

        if flags & AllocationHandleType::Fabric.as_raw() as u32 != 0 {
            types.insert(AllocationHandleType::Fabric);
        }

        if types.is_empty() {
            types.insert(AllocationHandleType::None);
        }

        Ok(types)
    }

    pub fn affinity_supported(&self) -> CudaResult<bool> {
        unsafe { get_exec_affinity_support(AffinityType::SmCount, self.inner) }.map(|v| v == 1)
    }

    pub fn default_memory_pool(&self) -> CudaResult<CudaMemoryPool> {
        let pool = unsafe { get_mem_pool(&self.inner) }?;
        Ok(CudaMemoryPool { inner: pool })
    }

    pub fn set_default_memory_pool(&self, pool: &CudaMemoryPool) -> CudaResult<()> {
        unsafe { set_mem_pool(&self.inner, &pool.inner) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_driver_device_count() {
        crate::driver::init();
        let result = CudaDevice::count();
        assert!(
            result.is_ok(),
            "CUDA device count retrieval failed: {:?}",
            result
        );
    }

    #[test]
    fn test_cuda_driver_device_new() {
        crate::driver::init();
        let result = CudaDevice::new(0);
        assert!(result.is_ok(), "CUDA device creation failed: {:?}", result);
    }

    #[test]
    fn test_cuda_driver_device_name() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let result = device.name();
        assert!(
            result.is_ok(),
            "CUDA device name retrieval failed: {:?}",
            result
        );
    }

    #[test]
    fn test_cuda_driver_device_uuid() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let result = device.uuid();
        assert!(
            result.is_ok(),
            "CUDA device UUID retrieval failed: {:?}",
            result
        );
    }

    #[test]
    fn test_cuda_driver_device_total_memory() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let result = device.total_memory();
        assert!(
            result.is_ok(),
            "CUDA device total memory retrieval failed: {:?}",
            result
        );
    }

    #[test]
    fn test_cuda_driver_device_attribute() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let result = device.attribute(DeviceAttribute::VirtualAddressManagementSupported);
        assert!(
            result.is_ok(),
            "CUDA device attribute retrieval failed: {:?}",
            result
        );
    }

    #[test]
    fn test_cuda_driver_device_affinity() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let result = device.affinity_supported();
        assert!(
            result.is_ok(),
            "CUDA device affinity support check failed: {:?}",
            result
        );
    }
}
