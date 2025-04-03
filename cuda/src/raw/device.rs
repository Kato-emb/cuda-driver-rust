use std::mem::MaybeUninit;

use cuda_sys::ffi as sys;

use crate::error::{CudaResult, ToResult};
use crate::{wrap_sys_enum, wrap_sys_handle};

wrap_sys_handle!(Device, sys::CUdevice);
wrap_sys_handle!(Uuid, sys::CUuuid);

wrap_sys_enum!(
    DeviceAttribute,
    sys::CUdevice_attribute,
    {
        MaxThreadsPerBlock = CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        MaxBlockDimX = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
        MaxBlockDimY = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
        MaxBlockDimZ = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
        MaxGridDimX = CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
        MaxGridDimY = CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
        MaxGridDimZ = CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
        SharedMemPerBlock = CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK,
        TotalConstantMemory = CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
        WarpSize = CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        MaxPitch = CU_DEVICE_ATTRIBUTE_MAX_PITCH,
        MaxRegistersPerBlock = CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
        ClockRate = CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
        TextureAlignment = CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
        GpuOverlap = CU_DEVICE_ATTRIBUTE_GPU_OVERLAP,
        MultiProcessorCount = CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        KernelExecTimeout = CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
        Integrated = CU_DEVICE_ATTRIBUTE_INTEGRATED,
        CanMapHostMemory = CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
        ComputeMode = CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
        MaxTexture1DWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH,
        MaxTexture2DWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
        MaxTexture2DHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT,
        MaxTexture3DWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,
        MaxTexture3DHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,
        MaxTexture3DDepth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH,
        MaxTexture2DLayeredWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH,
        MaxTexture2DLayeredHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,
        MaxTexture2DLayeredLayers = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,
        SurfaceAlignment = CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT,
        ConcurrentKernels = CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
        EccEnabled = CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
        PciBusId = CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
        PciDeviceId = CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
        TccDriver = CU_DEVICE_ATTRIBUTE_TCC_DRIVER,
        MemoryClockRate = CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
        GlobalMemoryBusWidth = CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
        L2CacheSize = CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        MaxThreadsPerMultiProcessor = CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        AsyncEngineCount = CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
        UnifiedAddressing = CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
        MaxTexture1DLayeredWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH,
        MaxTexture1DLayeredLayers = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS,
        CanTex2DGather = CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER,
        MaxTexture2DGatherWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH,
        MaxTexture2DGatherHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT,
        MaxTexture3DWidthAlt = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE,
        MaxTexture3DHeightAlt = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE,
        MaxTexture3DDepthAlt = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE,
        PciDomainId = CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
        TexturePitchAlignment = CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT,
        MaxTextureCubemapWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH,
        MaxTextureCubemapLayeredWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH,
        MaxTextureCubemapLayeredLayers = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS,
        MaxSurface1DWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH,
        MaxSurface2DWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH,
        MaxSurface2DHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT,
        MaxSurface3DWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH,
        MaxSurface3DHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT,
        MaxSurface3DDepth = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH,
        MaxSurface1DLayeredWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH,
        MaxSurface1DLayeredLayers = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS,
        MaxSurface2DLayeredWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH,
        MaxSurface2DLayeredHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT,
        MaxSurface2DLayeredLayers = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS,
        MaxSurfaceCubemapWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH,
        MaxSurfaceCubemapLayeredWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH,
        MaxSurfaceCubemapLayeredLayers = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS,
        MaxTexture1DLinearWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH,
        MaxTexture2DLinearWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH,
        MaxTexture2DLinearHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT,
        MaxTexture2DLinearPitch = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH,
        MaxTexture2DMipmappedWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH,
        MaxTexture2DMipmappedHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT,
        ComputeCapabilityMajor = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        ComputeCapabilityMinor = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        MaxTexture1DMipmappedWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH,
        StreamPrioritiesSupported = CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED,
        GlobalL1CacheSupported = CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED,
        LocalL1CacheSupported = CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED,
        MaxSharedMemoryPerMultiprocessor = CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        MaxRegistersPerMultiprocessor = CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
        ManagedMemory = CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
        MultiGPuBoard = CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD,
        MultiGPuBoardGroupID = CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID,
        HostNativeAtomicSupported = CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED,
        SingleToDoublePrecisionPerfRatio = CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO,
        PageableMemoryAccess = CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
        ConcurrentManagedAccess = CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
        ComputePreemptionSupported = CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED,
        CanUseHostPointerForRegisteredMem = CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM,
        CanuseStreamMemOpsV1 = CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1,
        CanUse64BitStreamMemOpsV1 = CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1,
        CanUseStreamWaitValueNorV1 = CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1,
        CooperativeLaunch = CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
        CooperativeMultiDeviceLaunch = CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH,
        MaxSharedMemoryPerBlockOptin = CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        CanFlushRemoteWrites = CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES,
        HostRegisterSupported = CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED,
        PageableMemoryAccessUsesHostPageTables = CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
        DirectManagedMemAccessFromHost = CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST,
        VirtualAddressManagementSupported = CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
        HandleTypePosixFileDescriptorSupported = CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
        HandleTypeWin32HandleSupported = CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED,
        HandleTypeWin32KmtHandleSupported = CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED,
        MaxBlocksPerMultiprocessor = CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
        GenericCompressionSupported = CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED,
        MaxPersistingL2CacheSize = CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
        MaxAccessPolicyWindowSize = CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE,
        GpuDirectRdmaWithCudaVmmSupported = CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
        ReservedSharedMemooryPerBlock = CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK,
        SparseCudaArraySupported = CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED,
        ReadOnlyHostRegisterSupported = CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED,
        TimelineSemaphoreInteropSupported = CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED,
        MemoryPoolsSupported = CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
        GpuDirectRdmaSupported = CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED,
        GpuDirectRdmaFlushWritesOptions = CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS,
        GpuDirectRdmaWritesOrdering = CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
        MempoolSupportedHandleTypes = CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES,
        ClusterLaunch = CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH,
        DeferredMappingCudaArraySupported = CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED,
        CanUse64BitStreamMemOps = CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,
        CanUseStreamWaitValueNor = CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR,
        DmaBufSupported = CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
        IpcEventSupported = CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED,
        MemSyncDomainCount = CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT,
        TensorMapAccessSupported = CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED,
        HandleTypeFabricSupported = CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
        UnifiedFunctionPointers = CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS,
        NumaConfig = CU_DEVICE_ATTRIBUTE_NUMA_CONFIG,
        NumaId = CU_DEVICE_ATTRIBUTE_NUMA_ID,
        MulticastSupported = CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
        MpsEnabled = CU_DEVICE_ATTRIBUTE_MPS_ENABLED,
        HostNumaId = CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
        D3d12CigSupported = CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED,
        MemDecompressAlgorithmMask = CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK,
        MemDecompressMaximumLength = CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH,
        GpuPciDeviceId = CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID,
        GpuPciSubsystemId = CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID,
        HostNumaMultinodeIpcSupported = CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED,
        Max = CU_DEVICE_ATTRIBUTE_MAX,
    }
);

wrap_sys_enum!(
    AffinityType,
    sys::CUexecAffinityType,
    {
        SmCount = CU_EXEC_AFFINITY_TYPE_SM_COUNT,
        Max = CU_EXEC_AFFINITY_TYPE_MAX,
    }
);

pub unsafe fn get_device(ordinal: i32) -> CudaResult<Device> {
    let mut device = 0;
    unsafe { sys::cuDeviceGet(&mut device, ordinal) }.to_result()?;

    Ok(Device(device))
}

pub unsafe fn get_attribute(attr: DeviceAttribute, device: Device) -> CudaResult<i32> {
    let mut value = 0;
    unsafe { sys::cuDeviceGetAttribute(&mut value, attr.into(), device.0) }.to_result()?;

    Ok(value)
}

pub unsafe fn get_count() -> CudaResult<i32> {
    let mut count = 0;
    unsafe { sys::cuDeviceGetCount(&mut count) }.to_result()?;

    Ok(count)
}

pub unsafe fn get_exec_affinity_support(type_: AffinityType, device: Device) -> CudaResult<i32> {
    let mut value = 0;
    unsafe { sys::cuDeviceGetExecAffinitySupport(&mut value, type_.into(), device.0) }
        .to_result()?;

    Ok(value)
}

#[cfg(target_os = "windows")]
pub unsafe fn get_luid(device: Device) -> CudaResult<([u8; 8], u32)> {
    let mut luid = [0i8; 8];
    let mut mask = 0;
    unsafe { sys::cuDeviceGetLuid(luid.as_mut_ptr(), &mut mask, device.0) }.to_result()?;

    Ok((luid.map(|b| b as u8), mask))
}

pub unsafe fn get_name(device: Device) -> CudaResult<String> {
    let mut name_buf = [0i8; 256];
    let len = name_buf.len() as i32;
    unsafe { sys::cuDeviceGetName(name_buf.as_mut_ptr() as *mut i8, len, device.0) }.to_result()?;

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

pub unsafe fn get_by_pci_bus_id(pci_bus_id: &str) -> CudaResult<Device> {
    let mut device = 0;
    let pci_bus_id = std::ffi::CString::new(pci_bus_id).unwrap_or_default();
    unsafe { sys::cuDeviceGetByPCIBusId(&mut device, pci_bus_id.as_ptr()) }.to_result()?;

    Ok(Device(device))
}

pub unsafe fn get_pci_bus_id(device: Device) -> CudaResult<String> {
    let mut pci_bus_id_buf = [0i8; 256];
    let len = pci_bus_id_buf.len() as i32;
    unsafe { sys::cuDeviceGetPCIBusId(pci_bus_id_buf.as_mut_ptr(), len, device.0) }.to_result()?;

    let pci_bus_id = unsafe { std::ffi::CStr::from_ptr(pci_bus_id_buf.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    Ok(pci_bus_id)
}

pub unsafe fn register_async_notification() {}
pub unsafe fn unregister_async_notification() {}

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

    #[test]
    fn test_cuda_raw_device_get_pci_bus_id() {
        unsafe { init(InitFlags::_ZERO) }.unwrap();

        let device = unsafe { get_device(0).unwrap() };
        let result = unsafe { get_pci_bus_id(device) };
        assert!(
            result.is_ok(),
            "CUDA device PCI bus ID retrieval failed: {:?}",
            result
        );

        let pci_bus_id = result.unwrap();
        let result = unsafe { get_by_pci_bus_id(&pci_bus_id) };
        assert!(
            result.is_ok(),
            "CUDA device retrieval by PCI bus ID failed: {:?}",
            result
        );
    }
}
