pub mod error;

pub mod driver;
pub mod platform;
pub mod raw;

// /// This macro initializes the thread context with default setting if it is not already initialized.
// #[macro_export]
// macro_rules! assert_thread_context_initialized {
//     () => {
//         if !$crate::context::has_current_context().unwrap_or(false) {
//             panic!("CUDA context has not been initialized in this thread. Call `` first.");
//         }
//     };
// }

// thread_local! {
//     static PRIMARY_CONTEXT: std::cell::RefCell<Option<CudaPrimaryContext>> = std::cell::RefCell::new(None);
// }

// pub fn ensure_primary_context(ordinal: i32) -> CudaResult<()> {
//     PRIMARY_CONTEXT.with(|cell| {
//         if cell.borrow().is_none() {
//             let device = CudaDevice::new(ordinal)?;
//             let ctx = CudaPrimaryContext::new(device)?;
//             ctx.bind_to_thread()?;
//             *cell.borrow_mut() = Some(ctx);
//         }

//         Ok(())
//     })
// }

// pub fn with_primary_context<F, R>(ordinal: i32, f: F) -> CudaResult<R>
// where
//     F: FnOnce(&CudaPrimaryContext) -> CudaResult<R>,
// {
//     ensure_primary_context(ordinal)?;
//     PRIMARY_CONTEXT.with(|cell| f(&cell.borrow().as_ref().unwrap()))
// }

// pub fn check_api_version() -> CudaResult<i32> {
//     unsafe { raw::version::get_version() }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_cuda_initialization() {
//         init();
//         assert!(CUDA_INITIALIZED.load(std::sync::atomic::Ordering::SeqCst));
//     }

//     #[test]
//     fn test_cuda_version() {
//         let version = check_api_version().unwrap();
//         assert!(version > 0, "CUDA version should be greater than 0");
//     }
// }
