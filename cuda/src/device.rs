use uuid::Uuid;

use crate::{assert_initialized, error::CudaResult, raw::device::*};

#[derive(Debug, Clone, Copy)]
pub struct CudaDevice {
    pub(crate) inner: Device,
}

impl CudaDevice {
    pub fn count() -> CudaResult<i32> {
        assert_initialized!();

        unsafe { get_count() }
    }

    pub fn new(ordinal: i32) -> CudaResult<Self> {
        assert_initialized!();

        debug_assert!((0..=Self::count()?).contains(&ordinal));
        let inner = unsafe { get_device(ordinal) }?;
        Ok(Self { inner })
    }

    pub fn name(&self) -> CudaResult<String> {
        unsafe { get_name(self.inner) }
    }

    pub fn uuid(&self) -> CudaResult<Uuid> {
        let c_uuid = unsafe { get_uuid(self.inner) }?;
        let raw = c_uuid.0.bytes.map(|b| b as u8);

        Ok(Uuid::from_bytes(raw))
    }

    pub fn total_memory(&self) -> CudaResult<usize> {
        unsafe { total_mem(self.inner) }
    }

    pub fn attribute(&self, attr: DeviceAttribute) -> CudaResult<i32> {
        unsafe { get_attribute(attr, self.inner) }
    }

    // pub fn default_memory_pool(&self) -> CudaResult<> {}

    pub fn is_affinity_supported(&self) -> CudaResult<bool> {
        unsafe { get_exec_affinity_support(AffinityType::SmCount, self.inner) }.map(|v| v == 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_device_count() {
        crate::init();
        let result = CudaDevice::count();
        assert!(
            result.is_ok(),
            "CUDA device count retrieval failed: {:?}",
            result
        );
    }

    #[test]
    fn test_cuda_device_new() {
        crate::init();
        let result = CudaDevice::new(0);
        assert!(result.is_ok(), "CUDA device creation failed: {:?}", result);

        let device = result.unwrap();
        let name_result = device.name();
        println!("Device Name: {:?}", name_result);
        assert!(
            name_result.is_ok(),
            "CUDA device name retrieval failed: {:?}",
            name_result
        );

        let uuid = device.uuid();
        println!("Device UUID: {:?}", uuid);
        assert!(
            uuid.is_ok(),
            "CUDA device UUID retrieval failed: {:?}",
            uuid
        );

        let total_memory = device.total_memory();
        println!("Total Memory: {:?}", total_memory);
        assert!(
            total_memory.is_ok(),
            "CUDA device total memory retrieval failed: {:?}",
            total_memory
        );
        let attribute = device.attribute(DeviceAttribute::VirtualAddressManagementSupported);
        println!("Attribute: {:?}", attribute);
        assert!(
            attribute.is_ok(),
            "CUDA device attribute retrieval failed: {:?}",
            attribute
        );

        let affinity = device.is_affinity_supported();
        println!("Affinity Supported: {:?}", affinity);
        assert!(
            affinity.is_ok(),
            "CUDA device affinity support check failed: {:?}",
            affinity
        );
    }
}
