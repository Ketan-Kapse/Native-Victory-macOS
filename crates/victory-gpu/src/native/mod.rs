//! The native host: victory-gpu's work for a host GPU API reached through `victory_backend`'s
//! traits instead of `windows::Win32::Graphics::Direct3D12`. Platform neutral and generic over the
//! backend, so it builds (and is linted) on the Windows target too; on macOS it is instantiated with
//! `victory_metal::metal::Metal` ([`Host`]). The D3D12 host (`host.rs`, `draw.rs`, ...) is untouched.
//!
//! Module selection is by `cfg`, not at run time: `crate::host` and everything under it exist only
//! with `cfg(windows)`, and this module is what a macOS build links the title-facing objects
//! (`list.rs`, `queue.rs`, `fence.rs`) to.
//!
//! [`memory`] title memory as zero-copy pools (no shadow, watch or copy-back), [`descriptors`]
//! the title's views for the backend, [`list`] a command list, [`host`] queues, pipelines, heaps and
//! submission, [`title`] the title's queues, fences and lists on top of them; `glue` (macOS) is
//! the one instance the title-facing objects call, and its [`pool_observer`] is what the runtime's
//! physical-memory observer forwards to.

/// Whether environment variable `$name` is set, read once per call site (draws ask per draw).
macro_rules! env_set {
    ($name:literal) => {{
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os($name).is_some())
    }};
}

/// Whether the pass trace's window is open (`VICTORY_TRACE_PASSES`, see `glue::pass_present`):
/// false unless that variable is set, so every pass-trace line costs one relaxed load otherwise.
static PASS_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn pass_on() -> bool {
    PASS_ON.load(std::sync::atomic::Ordering::Relaxed)
}

pub mod addr;
pub mod console_state;
pub(crate) mod descriptor_trace;
pub mod descriptors;
pub mod freed;
mod entries;
#[cfg(target_os = "macos")]
pub(crate) mod glue;
pub mod host;
pub mod list;
pub mod memory;
pub use memory::set_pool_release_hook;
#[cfg(target_os = "macos")]
pub mod offline;
pub mod shaders;
pub mod state;
pub mod textures;
pub mod title;
pub(crate) mod twin;

#[cfg(target_os = "macos")]
pub type Host = host::NativeHost<victory_metal::metal::Metal>;

#[cfg(target_os = "macos")]
pub use glue::{
    create_layer, frames_presented, last_frame, pool_observer, pools, set_event_hook, set_layer,
    shader_stats,
};

/// Reads `out.len()` bytes at `at` from this process, false if they cannot be read (unmapped
/// memory: the kernel checks, nothing faults).
pub fn safe_read(at: *const u8, out: &mut [u8]) -> bool {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            static mach_task_self_: u32;
            fn mach_vm_read_overwrite(
                task: u32,
                address: u64,
                size: u64,
                data: u64,
                out_size: *mut u64,
            ) -> i32;
        }
        let mut got = 0u64;
        // SAFETY: `out` is writable for its length; the kernel validates the source range.
        let r = unsafe {
            mach_vm_read_overwrite(
                mach_task_self_,
                at as u64,
                out.len() as u64,
                out.as_mut_ptr() as u64,
                &mut got,
            )
        };
        r == 0 && got == out.len() as u64
    }
    #[cfg(windows)]
    {
        crate::host::read_into(at, out)
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        let _ = (at, out);
        false
    }
}

/// Whether all of `[at, at + len)` is mapped readable in this process now (one kernel query a
/// region; nothing is read). `false` where the host cannot ask.
pub fn readable(at: u64, len: u64) -> bool {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            static mach_task_self_: u32;
            fn mach_vm_region(
                task: u32,
                address: *mut u64,
                size: *mut u64,
                flavor: i32,
                info: *mut i32,
                count: *mut u32,
                object_name: *mut u32,
            ) -> i32;
        }
        /// `VM_REGION_BASIC_INFO_64` and its count of 32-bit words; `VM_PROT_READ`.
        const BASIC_INFO_64: i32 = 9;
        const BASIC_INFO_COUNT_64: u32 = 9;
        const PROT_READ: i32 = 1;
        let Some(end) = at.checked_add(len) else { return false };
        let mut next = at;
        while next < end {
            let (mut address, mut size, mut info, mut count, mut name) = (next, 0u64, [0i32; 16], BASIC_INFO_COUNT_64, 0u32);
            // SAFETY: out-parameters of the sizes the call writes; the kernel validates the address.
            let r = unsafe {
                mach_vm_region(mach_task_self_, &mut address, &mut size, BASIC_INFO_64, info.as_mut_ptr(), &mut count, &mut name)
            };
            // The region at or after `next`: a later start is a hole. `info[0]` is `protection`.
            if r != 0 || address > next || info[0] & PROT_READ == 0 || size == 0 {
                return false;
            }
            let Some(region_end) = address.checked_add(size) else { return true };
            next = region_end;
        }
        true
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (at, len);
        false
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests;

#[cfg(test)]
mod vertex_tests;
