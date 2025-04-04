use std::mem::MaybeUninit;

use cuda_sys::ffi as sys;

use crate::{
    error::{CudaResult, ToResult},
    raw::{device::Device, stream::Stream},
    wrap_sys_enum, wrap_sys_handle,
};

wrap_sys_handle!(Array, sys::CUarray);

impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Array").field("handle", &self.0).finish()
    }
}

wrap_sys_handle!(ArrayDescriptor, sys::CUDA_ARRAY_DESCRIPTOR);

impl std::fmt::Debug for ArrayDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayDescriptor")
            .field("handle", &self.0)
            .finish()
    }
}

wrap_sys_handle!(Array3DDescriptor, sys::CUDA_ARRAY3D_DESCRIPTOR);
wrap_sys_handle!(ArrayMemoryRequirements, sys::CUDA_ARRAY_MEMORY_REQUIREMENTS);
wrap_sys_handle!(ArraySparseProperties, sys::CUDA_ARRAY_SPARSE_PROPERTIES);

wrap_sys_handle!(Memcpy2D, sys::CUDA_MEMCPY2D);
wrap_sys_handle!(Memcpy3D, sys::CUDA_MEMCPY3D);

wrap_sys_enum!(
    ArrayFormat,
    sys::CUarray_format_enum,
    {
        U8 = CU_AD_FORMAT_UNSIGNED_INT8,
        U16 = CU_AD_FORMAT_UNSIGNED_INT16,
        U32 = CU_AD_FORMAT_UNSIGNED_INT32,
        I8 = CU_AD_FORMAT_SIGNED_INT8,
        I16 = CU_AD_FORMAT_SIGNED_INT16,
        I32 = CU_AD_FORMAT_SIGNED_INT32,
        F16 = CU_AD_FORMAT_HALF,
        F32 = CU_AD_FORMAT_FLOAT,
        NV12 = CU_AD_FORMAT_NV12,
        NormalizedU8C1 = CU_AD_FORMAT_UNORM_INT8X1,
        NormalizedU8C2 = CU_AD_FORMAT_UNORM_INT8X2,
        NormalizedU8C4 = CU_AD_FORMAT_UNORM_INT8X4,
        NormalizedU16C1 = CU_AD_FORMAT_UNORM_INT16X1,
        NormalizedU16C2 = CU_AD_FORMAT_UNORM_INT16X2,
        NormalizedU16C4 = CU_AD_FORMAT_UNORM_INT16X4,
        NormalizedI8C1 = CU_AD_FORMAT_SNORM_INT8X1,
        NormalizedI8C2 = CU_AD_FORMAT_SNORM_INT8X2,
        NormalizedI8C4 = CU_AD_FORMAT_SNORM_INT8X4,
        NormalizedI16C1 = CU_AD_FORMAT_SNORM_INT16X1,
        NormalizedI16C2 = CU_AD_FORMAT_SNORM_INT16X2,
        NormalizedI16C4 = CU_AD_FORMAT_SNORM_INT16X4,
        NormalizedUBC1 = CU_AD_FORMAT_BC1_UNORM,
        NormalizedUBC1SRGB = CU_AD_FORMAT_BC1_UNORM_SRGB,
        NormalizedUBC2 = CU_AD_FORMAT_BC2_UNORM,
        NormalizedUBC2SRGB = CU_AD_FORMAT_BC2_UNORM_SRGB,
        NormalizedUBC3 = CU_AD_FORMAT_BC3_UNORM,
        NormalizedUBC3SRGB = CU_AD_FORMAT_BC3_UNORM_SRGB,
        NormalizedUBC4 = CU_AD_FORMAT_BC4_UNORM,
        NormalizedIBC4 = CU_AD_FORMAT_BC4_SNORM,
        NormalizedUBC5 = CU_AD_FORMAT_BC5_UNORM,
        NormalizedIBC5 = CU_AD_FORMAT_BC5_SNORM,
        UF16BC6H  = CU_AD_FORMAT_BC6H_UF16,
        SF16BC6H  = CU_AD_FORMAT_BC6H_SF16,
        NormalizedUBC7 = CU_AD_FORMAT_BC7_UNORM,
        NormalizedUBC7SRGB = CU_AD_FORMAT_BC7_UNORM_SRGB,
        P010 = CU_AD_FORMAT_P010,
        P016 = CU_AD_FORMAT_P016,
        NV16 = CU_AD_FORMAT_NV16,
        P210 = CU_AD_FORMAT_P210,
        P216 = CU_AD_FORMAT_P216,
        YUY2 = CU_AD_FORMAT_YUY2,
        Y210 = CU_AD_FORMAT_Y210,
        Y216 = CU_AD_FORMAT_Y216,
        AYUV = CU_AD_FORMAT_AYUV,
        Y410 = CU_AD_FORMAT_Y410,
        Y416 = CU_AD_FORMAT_Y416,
        Y444Planar8 = CU_AD_FORMAT_Y444_PLANAR8,
        Y444Planar10 = CU_AD_FORMAT_Y444_PLANAR10,
        YUV444SemiPlanar8 = CU_AD_FORMAT_YUV444_8bit_SemiPlanar,
        YUV444SemiPlanar16 = CU_AD_FORMAT_YUV444_16bit_SemiPlanar,
        NormalizedURGB1010102 = CU_AD_FORMAT_UNORM_INT_101010_2,
        Max = CU_AD_FORMAT_MAX,
    }
);

pub unsafe fn create(desc: &ArrayDescriptor) -> CudaResult<Array> {
    let mut array = MaybeUninit::uninit();
    unsafe { sys::cuArrayCreate_v2(array.as_mut_ptr(), &desc.0) }.to_result()?;

    Ok(Array(unsafe { array.assume_init() }))
}

pub unsafe fn create_3d(desc: &Array3DDescriptor) -> CudaResult<Array> {
    let mut array = MaybeUninit::uninit();
    unsafe { sys::cuArray3DCreate_v2(array.as_mut_ptr(), &desc.0) }.to_result()?;

    Ok(Array(unsafe { array.assume_init() }))
}

pub unsafe fn destroy(array: Array) -> CudaResult<()> {
    unsafe { sys::cuArrayDestroy(array.0) }.to_result()
}

pub unsafe fn get_descriptor(array: Array) -> CudaResult<ArrayDescriptor> {
    let mut desc = MaybeUninit::uninit();
    unsafe { sys::cuArrayGetDescriptor_v2(desc.as_mut_ptr(), array.0) }.to_result()?;

    Ok(ArrayDescriptor(unsafe { desc.assume_init() }))
}

pub unsafe fn get_3d_descriptor(array: Array) -> CudaResult<Array3DDescriptor> {
    let mut desc = MaybeUninit::uninit();
    unsafe { sys::cuArray3DGetDescriptor_v2(desc.as_mut_ptr(), array.0) }.to_result()?;

    Ok(Array3DDescriptor(unsafe { desc.assume_init() }))
}

pub unsafe fn get_memory_requirements(
    array: Array,
    device: Device,
) -> CudaResult<ArrayMemoryRequirements> {
    let mut req = MaybeUninit::uninit();
    unsafe { sys::cuArrayGetMemoryRequirements(req.as_mut_ptr(), array.0, device.0) }
        .to_result()?;

    Ok(ArrayMemoryRequirements(unsafe { req.assume_init() }))
}

pub unsafe fn get_plane(array: Array, plane_idx: u32) -> CudaResult<Array> {
    let mut plane = MaybeUninit::uninit();
    unsafe { sys::cuArrayGetPlane(plane.as_mut_ptr(), array.0, plane_idx) }.to_result()?;

    Ok(Array(unsafe { plane.assume_init() }))
}

pub unsafe fn get_sparse_properties(array: Array) -> CudaResult<ArraySparseProperties> {
    let mut props = MaybeUninit::uninit();
    unsafe { sys::cuArrayGetSparseProperties(props.as_mut_ptr(), array.0) }.to_result()?;

    Ok(ArraySparseProperties(unsafe { props.assume_init() }))
}

pub unsafe fn memcpy_2d(params: &Memcpy2D) -> CudaResult<()> {
    unsafe { sys::cuMemcpy2D_v2(&params.0) }.to_result()
}

pub unsafe fn memcpy_2d_async(params: &Memcpy2D, stream: Stream) -> CudaResult<()> {
    unsafe { sys::cuMemcpy2DAsync_v2(&params.0, stream.0) }.to_result()
}

pub unsafe fn memcpy_2d_unaligned(params: &Memcpy2D) -> CudaResult<()> {
    unsafe { sys::cuMemcpy2DUnaligned_v2(&params.0) }.to_result()
}

pub unsafe fn memcpy_3d(params: &Memcpy3D) -> CudaResult<()> {
    unsafe { sys::cuMemcpy3D_v2(&params.0) }.to_result()
}

pub unsafe fn memcpy_3d_async(params: &Memcpy3D, stream: Stream) -> CudaResult<()> {
    unsafe { sys::cuMemcpy3DAsync_v2(&params.0, stream.0) }.to_result()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::{context, device, init};

    #[test]
    fn test_cuda_raw_array_create() {
        unsafe { init::init(init::InitFlags::_ZERO) }.unwrap();
        let device = unsafe { device::get_device(0) }.unwrap();
        let ctx = unsafe { context::create(context::ContextFlags::SCHED_AUTO, device) }.unwrap();
        unsafe { context::set_current(ctx) }.unwrap();

        let desc = ArrayDescriptor {
            0: sys::CUDA_ARRAY_DESCRIPTOR {
                Width: 1024,
                Height: 768,
                Format: ArrayFormat::U8.into(),
                NumChannels: 1,
            },
        };

        let result = unsafe { create(&desc) };
        assert!(result.is_ok(), "CUDA array creation failed: {:?}", result);

        let array = result.unwrap();
        let desc_result = unsafe { get_descriptor(array) };
        assert!(
            desc_result.is_ok(),
            "CUDA array descriptor retrieval failed: {:?}",
            desc_result
        );

        let desc = desc_result.unwrap();
        assert_eq!(desc.0.Width, 1024);
        assert_eq!(desc.0.Height, 768);
        assert_eq!(desc.0.Format, ArrayFormat::U8.into());
        assert_eq!(desc.0.NumChannels, 1);

        unsafe { destroy(result.unwrap()) }.unwrap();
        unsafe { context::destroy(ctx) }.unwrap();
    }
}
