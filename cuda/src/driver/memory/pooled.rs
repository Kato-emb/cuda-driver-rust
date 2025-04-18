use crate::{
    driver::{device::CudaDevice, stream::CudaStream},
    error::CudaResult,
    platform::{OsBorrowedHandle, OsOwnedHandle},
    raw::memory::{
        AccessDesc, AccessFlags, AllocationHandleType, AllocationType, Location, LocationType,
        pooled::*,
    },
};

use super::{CudaDeviceBuffer, DeviceRepr};

pub struct CudaMemoryPool {
    inner: MemoryPool,
}

impl std::fmt::Debug for CudaMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let memory_details = format!(
            "reserved: {}/{}, used: {}/{}",
            self.current_memory_reserved_size().ok().unwrap_or(0),
            self.high_memory_reserved_size().ok().unwrap_or(0),
            self.current_memory_used_size().ok().unwrap_or(0),
            self.high_memory_used_size().ok().unwrap_or(0),
        );

        f.debug_struct("CudaMemoryPool")
            .field("release threshold", &self.release_threshold().ok())
            .field("memory (current/high)", &memory_details)
            .field("reuse event", &self.reuse_follow_event_dependencies().ok())
            .field("reuse memory", &self.reuse_allow_opportunistic().ok())
            .field(
                "reuse internal",
                &self.reuse_allow_internal_dependencies().ok(),
            )
            .finish()
    }
}

impl Drop for CudaMemoryPool {
    fn drop(&mut self) {
        if let Err(e) = unsafe { destroy(self.inner) } {
            log::error!("Failed to destroy CUDA memory pool: {:?}", e);
        }
    }
}

impl CudaMemoryPool {
    // max_size = 0 -> default
    pub fn new(max_size: usize, device: &CudaDevice) -> CudaResult<Self> {
        let mut props = MemoryPoolProps::default();
        props.0.allocType = AllocationType::Pinned.into();

        #[cfg(target_os = "linux")]
        {
            props.0.handleTypes = AllocationHandleType::PosixFD.into();
        }

        #[cfg(target_os = "windows")]
        {
            props.0.handleTypes = AllocationHandleType::Win32.into();
            let mut sa = Box::new(windows::Win32::Security::SECURITY_ATTRIBUTES::default());
            sa.nLength =
                std::mem::size_of::<windows::Win32::Security::SECURITY_ATTRIBUTES>() as u32;
            sa.bInheritHandle = true.into();
            props.0.win32SecurityAttributes = Box::leak(sa) as *mut _ as *mut std::ffi::c_void;
        }

        props.0.maxSize = max_size;
        props.0.location.type_ = LocationType::Device.into();
        props.0.location.id = device.as_raw();

        Self::new_with_props(&props)
    }

    pub fn new_with_props(props: &MemoryPoolProps) -> CudaResult<Self> {
        let inner = unsafe { create(props) }?;
        Ok(Self { inner })
    }

    pub fn to_fd(&self) -> CudaResult<OsOwnedHandle> {
        use crate::raw::memory::{AllocationHandleType, ShareableHandleFlags};

        let handle_type = if cfg!(target_os = "linux") {
            AllocationHandleType::PosixFD
        } else if cfg!(target_os = "windows") {
            AllocationHandleType::Win32
        } else {
            panic!("Unsupported OS for CudaMemoryPool")
        };

        unsafe { export_to_shareable_handle(&self.inner, handle_type, ShareableHandleFlags::_ZERO) }
    }

    pub fn trim_to(&mut self, keep: usize) -> CudaResult<()> {
        unsafe { trim_to(&mut self.inner, keep) }
    }

    // ToDo. viewでも実行可能なため、トレイト境界によってviewにも実装する
    pub fn accessibility_from_device(&self, device: &CudaDevice) -> CudaResult<AccessFlags> {
        let mut location = Location::default();
        location.0.type_ = LocationType::Device.into();
        location.0.id = device.as_raw();
        unsafe { get_access(&self.inner, &location) }
    }

    pub fn set_accessibility(
        &mut self,
        device: &CudaDevice,
        access: AccessFlags,
    ) -> CudaResult<()> {
        let mut desc = AccessDesc::default();
        desc.0.location.type_ = LocationType::Device.into();
        desc.0.location.id = device.as_raw();
        desc.0.flags = access.into();
        unsafe { set_access(&self.inner, &[desc]) }
    }

    pub unsafe fn alloc_async<Repr: DeviceRepr>(
        &self,
        len: usize,
        stream: &CudaStream,
    ) -> CudaResult<CudaDevicePooledBuffer<Repr>> {
        let bytesize = len.wrapping_mul(std::mem::size_of::<Repr>());
        let ptr = unsafe { malloc_pooled_async(bytesize, &self.inner, &stream.inner) }?;

        Ok(CudaDevicePooledBuffer {
            ptr,
            size: bytesize,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn release_threshold(&self) -> CudaResult<u64> {
        unsafe { get_attribute(&self.inner, MemoryPoolAttribute::ReleaseThreshold) }
    }

    pub fn set_release_threshold(&self, threshold: u64) -> CudaResult<()> {
        unsafe {
            set_attribute(
                &self.inner,
                MemoryPoolAttribute::ReleaseThreshold,
                &threshold,
            )
        }
    }

    pub fn reuse_follow_event_dependencies(&self) -> CudaResult<bool> {
        let value = unsafe {
            get_attribute::<i32>(
                &self.inner,
                MemoryPoolAttribute::ReuseFollowEventDependencies,
            )
        }?;

        Ok(value != 0)
    }

    pub fn set_reuse_follow_event_dependencies(&self, value: bool) -> CudaResult<()> {
        let value: i32 = if value { 1 } else { 0 };
        unsafe {
            set_attribute(
                &self.inner,
                MemoryPoolAttribute::ReuseFollowEventDependencies,
                &value,
            )
        }
    }

    pub fn reuse_allow_opportunistic(&self) -> CudaResult<bool> {
        let value = unsafe {
            get_attribute::<i32>(&self.inner, MemoryPoolAttribute::ReuseAllowOpportunistic)
        }?;

        Ok(value != 0)
    }

    pub fn set_reuse_allow_opportunistic(&self, value: bool) -> CudaResult<()> {
        let value: i32 = if value { 1 } else { 0 };
        unsafe {
            set_attribute(
                &self.inner,
                MemoryPoolAttribute::ReuseAllowOpportunistic,
                &value,
            )
        }
    }

    pub fn reuse_allow_internal_dependencies(&self) -> CudaResult<bool> {
        let value = unsafe {
            get_attribute::<i32>(
                &self.inner,
                MemoryPoolAttribute::ReuseAllowInternalDependencies,
            )
        }?;

        Ok(value != 0)
    }

    pub fn set_reuse_allow_internal_dependencies(&self, value: bool) -> CudaResult<()> {
        let value: i32 = if value { 1 } else { 0 };
        unsafe {
            set_attribute(
                &self.inner,
                MemoryPoolAttribute::ReuseAllowInternalDependencies,
                &value,
            )
        }
    }

    pub fn current_memory_reserved_size(&self) -> CudaResult<u64> {
        unsafe { get_attribute(&self.inner, MemoryPoolAttribute::ReservedMemCurrent) }
    }

    pub fn high_memory_reserved_size(&self) -> CudaResult<u64> {
        unsafe { get_attribute(&self.inner, MemoryPoolAttribute::ReservedMemHigh) }
    }

    pub fn reset_high_memory_reserved_size(&self) -> CudaResult<()> {
        unsafe { set_attribute(&self.inner, MemoryPoolAttribute::ReservedMemHigh, &0u64) }
    }

    pub fn current_memory_used_size(&self) -> CudaResult<u64> {
        unsafe { get_attribute(&self.inner, MemoryPoolAttribute::UsedMemCurrent) }
    }

    pub fn high_memory_used_size(&self) -> CudaResult<u64> {
        unsafe { get_attribute(&self.inner, MemoryPoolAttribute::UsedMemHigh) }
    }

    pub fn reset_high_memory_used_size(&self) -> CudaResult<()> {
        unsafe { set_attribute(&self.inner, MemoryPoolAttribute::UsedMemHigh, &0u64) }
    }
}

#[derive(Debug)]
pub struct CudaMemoryPoolView<'fd> {
    inner: MemoryPool,
    _handle: OsBorrowedHandle<'fd>,
}

impl<'fd> CudaMemoryPoolView<'fd> {
    pub fn from_fd(handle: OsBorrowedHandle<'fd>) -> CudaResult<Self> {
        use crate::raw::memory::{AllocationHandleType, ShareableHandleFlags};

        let handle_type = if cfg!(target_os = "linux") {
            AllocationHandleType::PosixFD
        } else if cfg!(target_os = "windows") {
            AllocationHandleType::Win32
        } else {
            panic!("Unsupported OS for CudaMemoryPoolView")
        };

        let pool = unsafe {
            import_from_shareable_handle(handle, handle_type, ShareableHandleFlags::_ZERO)
        }?;

        Ok(Self {
            inner: pool,
            _handle: handle,
        })
    }

    pub fn import<Repr: DeviceRepr>(
        &self,
        data: &PooledPtrExportData,
        byte_count: usize,
    ) -> CudaResult<CudaDevicePooledBuffer<Repr>> {
        let ptr = unsafe { import(&self.inner, &data) }?;

        Ok(CudaDevicePooledBuffer {
            ptr,
            size: byte_count,
            _marker: std::marker::PhantomData,
        })
    }
}

pub type CudaDevicePooledBuffer<Repr> = CudaDeviceBuffer<Repr, PooledDevicePtr>;

impl<Repr: DeviceRepr> CudaDevicePooledBuffer<Repr> {
    pub fn export(&self) -> CudaResult<PooledPtrExportData> {
        unsafe { export(self.ptr) }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "linux")]
    use std::os::fd::AsFd;
    #[cfg(target_os = "windows")]
    use std::os::windows::io::AsHandle;

    use crate::{
        driver::{
            context::CudaPrimaryContext,
            device::CudaDevice,
            memory::{
                CudaDeviceSlice, CudaSliceAccess, CudaSliceReadable, CudaSliceWritable,
                pinned::CudaHostPinnedBuffer,
            },
        },
        platform::ShareableOsHandle,
        raw::stream::StreamFlags,
    };

    use super::*;

    #[test]
    fn test_cuda_driver_pool() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let ctx = CudaPrimaryContext::new(device).unwrap();
        ctx.set_current().unwrap();

        let stream = CudaStream::new(StreamFlags::NON_BLOCKING).unwrap();

        let pool = CudaMemoryPool::new(1024 * 1024, ctx.device()).unwrap();
        println!("Memory pool created: {:?}", pool);

        // pool.set_accessibility(ctx.device(), AccessFlags::Read)
        //     .unwrap(); CudaError { code: 801, name: "CUDA_ERROR_NOT_SUPPORTED" }
        let flags = pool.accessibility_from_device(ctx.device()).unwrap();
        assert_eq!(flags, AccessFlags::ReadWrite);

        let fd = pool.to_fd().unwrap();
        assert!(!fd.as_ptr().is_null());

        #[cfg(target_os = "linux")]
        let pool_view = CudaMemoryPoolView::from_fd(fd.as_fd()).unwrap();
        #[cfg(target_os = "windows")]
        let pool_view = CudaMemoryPoolView::from_fd(fd.as_handle()).unwrap();
        println!("Imported memory pool: {:?}", pool_view);

        let mut pooled_buffer = unsafe { pool.alloc_async::<u8>(1024, &stream) }.unwrap();
        assert_eq!(pool.current_memory_used_size().unwrap(), 1024);

        pooled_buffer
            .as_mut_slice()
            .get_mut(300..500)
            .unwrap()
            .set_d8(u8::MAX)
            .unwrap();

        let data = pooled_buffer.export().unwrap();

        let imported_buffer = pool_view
            .import::<u8>(&data, pooled_buffer.as_slice().byte_size())
            .unwrap();
        let imported_slice = imported_buffer.as_slice().get(300..500).unwrap();
        println!("Imported slice: {:?}", imported_slice);
        let ptr = imported_slice.as_ptr();
        let len = imported_slice.len();

        let handle = std::thread::spawn(move || {
            let device = CudaDevice::new(0).unwrap();
            let ctx = CudaPrimaryContext::new(device).unwrap();
            ctx.set_current().unwrap();

            let slice = unsafe { CudaDeviceSlice::<u8, _>::from_raw(ptr, len) };
            assert!(slice.is_some());

            println!("Slice: {:?}", slice);
        });

        handle.join().unwrap();

        let mut pinned_buffer = CudaHostPinnedBuffer::<u8>::alloc(imported_slice.len()).unwrap();
        pinned_buffer
            .as_mut_slice()
            .copy_from_device(&imported_slice)
            .unwrap();

        for i in pinned_buffer.as_slice().iter() {
            assert_eq!(*i, u8::MAX);
        }

        println!("Memory pool created: {:?}", pool);
    }
}
