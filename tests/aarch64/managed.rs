// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//! Native execution of the checked-in AxiomOS controller bytecode.

use super::native::Image;
use core::ffi::c_void;
use rbpf::aarch64::{
    CompileOptions, Invocation, NO_FAULT, RuntimeCallbacks, STATUS_CALLBACK_ERROR, STATUS_OK,
};
use std::ops::Range;

const CONSERVATIVE: &[u8] = include_bytes!("../fixtures/axiomos/conservative_obstacle.bin");
const SLOW: &[u8] = include_bytes!("../fixtures/axiomos/slow_approach.bin");
const STREAK: &[u8] = include_bytes!("../fixtures/axiomos/clear_streak.bin");
const STACK_SIZE: usize = 8192;
const HELPER_FAILURE: u32 = 0xBEEF;

fn compile(program: &[u8]) -> Image {
    Image::compile_bytes(
        program,
        CompileOptions {
            max_instructions: program.len() / 8,
            max_code_size: 8192,
            stack_size: STACK_SIZE,
        },
    )
}

struct Runtime {
    wrapper: [u8; 56],
    payload: [u8; 56],
    stack: [u8; STACK_SIZE],
    map_value: u32,
    captured: Option<(i64, i64)>,
    calls: [usize; 3],
    fail_helper: Option<u32>,
    error: Option<u32>,
}

// Resolve only complete, non-wrapping accesses to an explicitly admitted region.
fn region(address: u64, width: usize, start: u64, len: usize) -> Option<Range<usize>> {
    let offset = usize::try_from(address.checked_sub(start)?).ok()?;
    let end = offset.checked_add(width)?;
    (end <= len).then_some(offset..end)
}

impl Runtime {
    fn new() -> Self {
        Self {
            wrapper: [0; 56],
            payload: [0; 56],
            stack: [0; STACK_SIZE],
            map_value: 0,
            captured: None,
            calls: [0; 3],
            fail_helper: None,
            error: None,
        }
    }

    fn stack_range(&self, address: u64, width: usize) -> Option<Range<usize>> {
        region(address, width, self.stack.as_ptr() as u64, STACK_SIZE)
    }

    fn read(&self, address: u64, width: usize) -> Option<u64> {
        if !matches!(width, 1 | 2 | 4 | 8) {
            return None;
        }
        let map_bytes = self.map_value.to_le_bytes();
        for (start, bytes) in [
            (self.wrapper.as_ptr() as u64, self.wrapper.as_slice()),
            (self.payload.as_ptr() as u64, self.payload.as_slice()),
            (self.stack.as_ptr() as u64, self.stack.as_slice()),
            (&self.map_value as *const u32 as u64, map_bytes.as_slice()),
        ] {
            if let Some(range) = region(address, width, start, bytes.len()) {
                let mut value = [0u8; 8];
                value[..width].copy_from_slice(&bytes[range]);
                return Some(u64::from_le_bytes(value));
            }
        }
        None
    }

    fn stack_word(&self, address: u64) -> Option<u32> {
        let range = self.stack_range(address, 4)?;
        Some(u32::from_le_bytes(self.stack[range].try_into().ok()?))
    }

    fn helper(&mut self, id: u32, args: &[u64]) -> Result<u64, u32> {
        let index = match id {
            5 => 0,
            6 => 1,
            1009 => 2,
            _ => return Err(STATUS_CALLBACK_ERROR),
        };
        self.calls[index] += 1;
        if self.fail_helper == Some(id) {
            self.error = Some(HELPER_FAILURE);
            return Err(HELPER_FAILURE);
        }
        match id {
            5 | 6 if args[0] == 1 && self.stack_word(args[1]) == Some(0) => {
                if id == 5 {
                    return Ok(&self.map_value as *const u32 as u64);
                }
                if args[3] != 0 {
                    return Err(STATUS_CALLBACK_ERROR);
                }
                self.map_value = self.stack_word(args[2]).ok_or(STATUS_CALLBACK_ERROR)?;
                Ok(0)
            }
            1009 => {
                let pair = (args[0] as i64, args[1] as i64);
                if self.captured.is_some()
                    || !(-1000..=1000).contains(&pair.0)
                    || !(-1000..=1000).contains(&pair.1)
                {
                    return Err(STATUS_CALLBACK_ERROR);
                }
                self.captured = Some(pair);
                Ok(0)
            }
            _ => Err(STATUS_CALLBACK_ERROR),
        }
    }

    fn invoke(&mut self, image: &Image, sensor: i64, valid: u32) -> (u32, u64, u64) {
        self.stack.fill(0);
        self.captured = None;
        self.calls = [0; 3];
        self.error = None;
        self.wrapper[..8].copy_from_slice(&(self.payload.as_ptr() as u64).to_le_bytes());
        self.payload[..4].copy_from_slice(&1u32.to_le_bytes());
        self.payload[4..8].copy_from_slice(&56u32.to_le_bytes());
        self.payload[40..48].copy_from_slice(&sensor.to_le_bytes());
        self.payload[48..52].copy_from_slice(&valid.to_le_bytes());
        let mut invocation = Invocation {
            context: self.wrapper.as_ptr(),
            stack: self.stack.as_mut_ptr(),
            stack_len: STACK_SIZE,
            opaque: (self as *mut Self).cast(),
            callbacks: &CALLBACKS,
            result: u64::MAX,
            fault_pc: 0,
        };
        let status = unsafe { image.entry()(&mut invocation) };
        (status, invocation.result, invocation.fault_pc)
    }

    fn cycle(&mut self, image: &Image, sensor: i64, valid: u32, drive: i64) {
        assert_eq!(self.invoke(image, sensor, valid), (STATUS_OK, 0, NO_FAULT));
        assert_eq!(self.captured, Some((drive, drive)));
        assert_eq!(self.calls[2], 1);
        assert_eq!(self.error, None);
    }
}

unsafe extern "C" fn helper(opaque: *mut c_void, id: u32, args: *const u64, out: *mut u64) -> u32 {
    let state = unsafe { &mut *opaque.cast::<Runtime>() };
    match state.helper(id, unsafe { core::slice::from_raw_parts(args, 5) }) {
        Ok(value) => {
            unsafe { *out = value };
            STATUS_OK
        }
        Err(code) => code,
    }
}

unsafe extern "C" fn load(
    opaque: *mut c_void,
    base: u64,
    offset: i64,
    width: u32,
    stack_base: u32,
    out: *mut u64,
) -> u32 {
    let state = unsafe { &*opaque.cast::<Runtime>() };
    if stack_base > 1
        || (stack_base == 1 && base != state.stack.as_ptr() as u64 + STACK_SIZE as u64)
    {
        return STATUS_CALLBACK_ERROR;
    }
    match base
        .checked_add_signed(offset)
        .and_then(|address| state.read(address, width as usize))
    {
        Some(value) => {
            unsafe { *out = value };
            STATUS_OK
        }
        None => STATUS_CALLBACK_ERROR,
    }
}

unsafe extern "C" fn store(
    opaque: *mut c_void,
    base: u64,
    offset: i64,
    width: u32,
    stack_base: u32,
    value: u64,
) -> u32 {
    let state = unsafe { &mut *opaque.cast::<Runtime>() };
    if !matches!(width, 1 | 2 | 4 | 8)
        || stack_base > 1
        || (stack_base == 1 && base != state.stack.as_ptr() as u64 + STACK_SIZE as u64)
    {
        return STATUS_CALLBACK_ERROR;
    }
    match base
        .checked_add_signed(offset)
        .and_then(|address| state.stack_range(address, width as usize))
    {
        Some(range) => {
            state.stack[range].copy_from_slice(&value.to_le_bytes()[..width as usize]);
            STATUS_OK
        }
        None => STATUS_CALLBACK_ERROR,
    }
}

static CALLBACKS: RuntimeCallbacks = RuntimeCallbacks {
    helper,
    load,
    store,
};

#[test]
fn controller_trajectories_and_invalid_sensor() {
    for (program, drives, maps) in [
        (CONSERVATIVE, [250, 250, 250, 0, 0, 250, 250, 250], [0; 8]),
        (SLOW, [300, 300, 300, 0, 100, 300, 300, 300], [0; 8]),
        (
            STREAK,
            [0, 0, 180, 0, 0, 0, 0, 180],
            [1, 2, 3, 0, 0, 1, 2, 3],
        ),
    ] {
        let image = compile(program);
        let mut state = Runtime::new();
        for ((sensor, drive), map) in [2000, 2000, 2000, 400, 1000, 2000, 2000, 2000]
            .into_iter()
            .zip(drives)
            .zip(maps)
        {
            state.cycle(&image, sensor, 1, drive);
            assert_eq!(state.map_value, map);
        }
        for (sensor, valid) in [(2000, 0), (-1, 1)] {
            state.cycle(&image, sensor, valid, 0);
            assert_eq!(state.map_value, 0);
        }
    }
}

#[test]
fn relocated_code_reuses_private_instance_maps() {
    let first = compile(STREAK);
    let second = compile(STREAK);
    assert_ne!(unsafe { first.entry() } as usize, unsafe { second.entry() }
        as usize);
    let mut a = Runtime::new();
    let mut b = Runtime::new();
    a.cycle(&first, 2000, 1, 0);
    b.cycle(&first, 2000, 1, 0);
    a.cycle(&second, 2000, 1, 0);
    a.cycle(&first, 2000, 1, 180);
    assert_eq!((a.map_value, b.map_value), (3, 1));
    b.cycle(&second, 2000, 1, 0);
    b.cycle(&first, 2000, 1, 180);
    assert_eq!((a.map_value, b.map_value), (3, 3));
}

#[test]
fn controller_helper_failures_stop_at_the_call() {
    let image = compile(STREAK);
    for (id, pc, calls, map) in [
        (5, 11, [1, 0, 0], 0),
        (6, 24, [1, 1, 0], 0),
        (1009, 30, [1, 1, 1], 1),
    ] {
        let mut state = Runtime::new();
        state.fail_helper = Some(id);
        assert_eq!(
            state.invoke(&image, 2000, 1),
            (STATUS_CALLBACK_ERROR, 0, pc)
        );
        assert_eq!(state.error, Some(HELPER_FAILURE));
        assert_eq!(state.calls, calls);
        assert_eq!(state.map_value, map);
        assert_eq!(state.captured, None);
    }
}
