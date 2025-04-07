use crate::{assert_initialized, error::CudaResult, raw::device::*};

pub struct CudaDevice {
    pub(crate) inner: Device,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDevice")
            .field("handle", &self.inner.0)
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

    // pub fn default_memory_pool(&self) -> CudaResult<> {}

    pub fn is_affinity_supported(&self) -> CudaResult<bool> {
        unsafe { get_exec_affinity_support(AffinityType::SmCount, self.inner) }.map(|v| v == 1)
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
        let result = device.is_affinity_supported();
        assert!(
            result.is_ok(),
            "CUDA device affinity support check failed: {:?}",
            result
        );
    }
}
