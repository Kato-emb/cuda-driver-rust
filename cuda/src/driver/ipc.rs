use crate::raw::ipc::*;

use crate::{error::CudaResult, raw::ipc::IpcEventHandle};

use super::event::CudaEvent;

pub struct CudaEventHandle {
    inner: IpcEventHandle,
}

impl std::fmt::Debug for CudaEventHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaEventHandle")
            .field("handle", &self.inner.0)
            .finish()
    }
}

impl std::ops::Deref for CudaEventHandle {
    type Target = [u8; 64];

    fn deref(&self) -> &Self::Target {
        unsafe { &*(&self.inner.0.reserved as *const [i8; 64] as *const [u8; 64]) }
    }
}

impl From<&[u8; 64]> for CudaEventHandle {
    fn from(value: &[u8; 64]) -> Self {
        let bytes = value.map(|b| b as i8);

        CudaEventHandle {
            inner: IpcEventHandle::from_bytes(&bytes),
        }
    }
}

impl From<&CudaEventHandle> for [u8; 64] {
    fn from(value: &CudaEventHandle) -> Self {
        value.inner.to_bytes().map(|b| b as u8)
    }
}

impl CudaEvent {
    pub fn to_handle(&self) -> CudaResult<CudaEventHandle> {
        let inner = unsafe { get_event_handle(&self.inner) }?;
        Ok(CudaEventHandle { inner })
    }

    pub fn from_handle(handle: CudaEventHandle) -> CudaResult<CudaEvent> {
        let inner = unsafe { open_event_handle(&handle.inner) }?;
        Ok(CudaEvent { inner })
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, Read, Write};

    use crate::{
        driver::{context::CudaPrimaryContext, device::CudaDevice, stream::CudaStream},
        raw::{
            event::{EventFlags, EventWaitFlags},
            stream::StreamFlags,
        },
    };

    use super::*;

    #[test]
    fn test_cuda_driver_ipc_event() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let context = CudaPrimaryContext::new(device).unwrap();
        context.set_current().unwrap();

        let stream = CudaStream::new(StreamFlags::NON_BLOCKING).unwrap();

        let event = CudaEvent::new(EventFlags::INTERPROCESS | EventFlags::DISABLE_TIMING).unwrap();
        event.record(&stream).unwrap();

        let handle = event.to_handle().unwrap();
        println!("Event handle: {:?}", handle.as_ref());

        let exec = std::env::current_exe().unwrap();
        let mut child = std::process::Command::new(&exec)
            .args(["child_process", "--nocapture"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .expect("Failed to start child process");

        let stdin = child.stdin.take().unwrap();
        let mut writer = std::io::BufWriter::new(stdin);
        let stdout = child.stdout.take().unwrap();
        let mut reader = std::io::BufReader::new(stdout);
        writer.write_all(handle.as_slice()).unwrap();
        writer.flush().unwrap();

        let mut buf = String::new();
        while let Ok(n) = reader.read_line(&mut buf) {
            if n == 0 {
                break;
            }
            println!("[Child]{}", buf.trim());
            buf.clear();
        }

        let status = child.wait().unwrap();
        assert_eq!(status.success(), true);
        println!("Child process exited with status: {:?}", status);
    }

    #[test]
    fn child_process() {
        crate::driver::init();
        let device = CudaDevice::new(0).unwrap();
        let context = CudaPrimaryContext::new(device).unwrap();
        context.set_current().unwrap();

        let stream = CudaStream::new(StreamFlags::NON_BLOCKING).unwrap();

        let event_local = CudaEvent::new(EventFlags::DEFAULT).unwrap();
        event_local.record(&stream).unwrap();

        let mut stdin = std::io::stdin();
        let mut input = [0u8; 64];
        println!("Enter the event handle (64 bytes):");
        stdin.read_exact(&mut input).unwrap();

        let handle = CudaEventHandle::from(&input);
        println!("Received event handle: {:?}", handle);
        let event = CudaEvent::from_handle(handle).unwrap();
        println!("Opened event: {:?}", event);

        stream.wait_event(&event, EventWaitFlags::DEFAULT).unwrap();
        assert!(event.query().unwrap());
    }
}
