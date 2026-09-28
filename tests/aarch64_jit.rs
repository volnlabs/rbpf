// SPDX-License-Identifier: (Apache-2.0 OR MIT)
#![cfg(feature = "aarch64-jit")]

use rbpf::aarch64::{Aarch64Compiler, CompileOptions, ErrorKind};
use rbpf::ebpf::{self, Insn};

fn insn(opc: u8, dst: u8, src: u8, off: i16, imm: i32) -> Insn {
    Insn {
        opc,
        dst,
        src,
        off,
        imm,
    }
}

fn bytes(insns: &[Insn]) -> Vec<u8> {
    insns.iter().flat_map(Insn::to_array).collect()
}

fn options(count: usize) -> CompileOptions {
    CompileOptions {
        max_instructions: count.max(1),
        max_code_size: 1024 * 1024,
        stack_size: 512,
    }
}

fn rejected(insns: &[Insn], kind: ErrorKind, pc: Option<usize>) {
    let program = bytes(insns);
    let mut scratch = vec![0; insns.len()];
    let error = match Aarch64Compiler::new(&program, options(insns.len()), &mut scratch) {
        Err(error) => error,
        Ok(_) => panic!("accepted invalid bytecode"),
    };
    assert_eq!(error.kind, kind);
    assert_eq!(error.pc, pc);
}

#[test]
fn rejects_invalid_streams_and_targets() {
    let exit = insn(ebpf::EXIT, 0, 0, 0, 0);
    for program in [vec![], vec![0], vec![0; ebpf::INSN_SIZE + 1]] {
        let mut scratch = [0u32; 2];
        let error = match Aarch64Compiler::new(&program, options(2), &mut scratch) {
            Err(error) => error,
            Ok(_) => panic!("accepted an empty or misaligned stream"),
        };
        assert_eq!((error.kind, error.pc), (ErrorKind::InvalidProgram, None));
    }
    rejected(
        &[insn(ebpf::MOV64_IMM, 0, 0, 0, 1)],
        ErrorKind::InvalidProgram,
        Some(0),
    );
    rejected(
        &[insn(ebpf::LD_DW_IMM, 0, 0, 0, 1)],
        ErrorKind::InvalidProgram,
        Some(0),
    );
    for malformed in [
        insn(ebpf::MOV64_IMM, 10, 0, 0, 1),
        insn(ebpf::ADD64_IMM, 0, 1, 0, 1),
        insn(ebpf::ADD64_REG, 0, 1, 0, 1),
        insn(ebpf::ADD64_IMM, 0, 0, 1, 1),
        insn(ebpf::NEG64, 0, 0, 0, 1),
        insn(ebpf::LD_DW_IMM, 0, 1, 0, 1),
        insn(ebpf::LD_DW_IMM, 0, 0, 1, 1),
    ] {
        rejected(
            &[malformed, exit.clone()],
            ErrorKind::InvalidProgram,
            Some(0),
        );
    }
    for malformed_tail in [
        insn(1, 0, 0, 0, 2),
        insn(0, 1, 0, 0, 2),
        insn(0, 0, 1, 0, 2),
        insn(0, 0, 0, 1, 2),
    ] {
        rejected(
            &[
                insn(ebpf::LD_DW_IMM, 0, 0, 0, 1),
                malformed_tail,
                exit.clone(),
            ],
            ErrorKind::InvalidProgram,
            Some(0),
        );
    }
    for unsupported in [ebpf::LD_ABS_B, ebpf::LD_IND_W, 0xff] {
        rejected(
            &[insn(unsupported, 0, 0, 0, 0), exit.clone()],
            ErrorKind::UnsupportedInstruction,
            Some(0),
        );
    }
    rejected(
        &[insn(ebpf::JA, 0, 0, -1, 0), exit.clone()],
        ErrorKind::InvalidJump,
        Some(0),
    );
    rejected(
        &[insn(ebpf::JA, 0, 0, 9, 0), exit.clone()],
        ErrorKind::InvalidJump,
        Some(0),
    );
    rejected(
        &[insn(ebpf::MOV64_REG, 0, 11, 0, 0), exit.clone()],
        ErrorKind::InvalidProgram,
        Some(0),
    );
    rejected(
        &[insn(ebpf::ST_DW_XADD, 1, 2, 0, 0), exit.clone()],
        ErrorKind::UnsupportedInstruction,
        Some(0),
    );
    rejected(
        &[insn(ebpf::CALL, 0, 1, 0, 1), exit.clone()],
        ErrorKind::UnsupportedInstruction,
        Some(0),
    );

    let wide = [
        insn(ebpf::LD_DW_IMM, 0, 0, 0, 1),
        insn(0, 0, 0, 0, 2),
        exit.clone(),
    ];
    let into_continuation = [
        insn(ebpf::JA, 0, 0, 1, 0),
        wide[0].clone(),
        wide[1].clone(),
        wide[2].clone(),
    ];
    rejected(&into_continuation, ErrorKind::InvalidJump, Some(0));

    let program = bytes(&[insn(ebpf::MOV64_IMM, 0, 0, 0, 1), exit]);
    let mut scratch = [0u32; 2];
    let error = match Aarch64Compiler::new(&program, options(1), &mut scratch) {
        Err(error) => error,
        Ok(_) => panic!("accepted a program above max_instructions"),
    };
    assert_eq!((error.kind, error.pc), (ErrorKind::InvalidProgram, None));
}

#[test]
fn bounded_buffers_and_canaries() {
    let program = bytes(&[
        insn(ebpf::MOV64_IMM, 0, 0, 0, 42),
        insn(ebpf::EXIT, 0, 0, 0, 0),
    ]);
    let mut no_scratch = [];
    let error = match Aarch64Compiler::new(&program, options(2), &mut no_scratch) {
        Err(error) => error,
        Ok(_) => panic!("accepted no offset scratch"),
    };
    assert_eq!(error.kind, ErrorKind::InsufficientScratch);

    let mut scratch = [0u32; 2];
    let compiler = Aarch64Compiler::new(&program, options(2), &mut scratch).unwrap();
    let len = compiler.code_len();
    assert_eq!(len % 4, 0);
    let mut guarded = vec![0xA5; len + 2];
    let error = compiler.emit_into(&mut guarded[1..len]).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InsufficientOutput);
    assert!(guarded.iter().all(|byte| *byte == 0xA5));

    let info = compiler.emit_into(&mut guarded[1..=len]).unwrap();
    assert_eq!(info.code_len, len);
    assert!(info.entry_offset < len);
    assert_eq!(info.entry_offset % 4, 0);
    assert_eq!((guarded[0], guarded[len + 1]), (0xA5, 0xA5));
    assert_eq!(&guarded[1..5], &0xD10283FFu32.to_le_bytes()); // SUB SP, SP, #160.
    assert!(
        guarded[1..=len]
            .chunks_exact(4)
            .any(|word| word == 0x910283FFu32.to_le_bytes())
    );
    // A64 RET is D65F03C0. The complete image also carries callback/error epilogues.
    assert!(
        guarded[1..=len]
            .chunks_exact(4)
            .any(|word| word == 0xD65F03C0u32.to_le_bytes())
    );

    for (cap, expected) in [(len, None), (len - 1, Some(ErrorKind::CodeTooLarge))] {
        let mut scratch = [0u32; 2];
        let opts = CompileOptions {
            max_code_size: cap,
            ..options(2)
        };
        let outcome = Aarch64Compiler::new(&program, opts, &mut scratch);
        match expected {
            None => assert_eq!(outcome.unwrap().code_len(), len),
            Some(kind) => assert_eq!(outcome.err().unwrap().kind, kind),
        }
    }
}

#[test]
fn rejects_invalid_options_and_checks_callback_branch_encoding() {
    let program = bytes(&[
        insn(ebpf::MOV64_IMM, 0, 0, 0, 1),
        insn(ebpf::EXIT, 0, 0, 0, 0),
    ]);
    let mut scratch = [0u32; 2];
    for opts in [
        CompileOptions {
            stack_size: 0,
            ..options(2)
        },
        CompileOptions {
            max_instructions: 0,
            ..options(2)
        },
    ] {
        let error = match Aarch64Compiler::new(&program, opts, &mut scratch) {
            Err(error) => error,
            Ok(_) => panic!("accepted invalid compile options"),
        };
        assert_eq!(error.kind, ErrorKind::InvalidOptions);
    }
    let error = match Aarch64Compiler::new(
        &program,
        CompileOptions {
            max_code_size: 4,
            ..options(2)
        },
        &mut scratch,
    ) {
        Err(error) => error,
        Ok(_) => panic!("accepted an output cap below the prologue"),
    };
    assert_eq!(error.kind, ErrorKind::CodeTooLarge);

    let call = bytes(&[insn(ebpf::CALL, 0, 0, 0, 7), insn(ebpf::EXIT, 0, 0, 0, 0)]);
    let compiler = Aarch64Compiler::new(&call, options(2), &mut scratch).unwrap();
    let mut code = vec![0u8; compiler.code_len()];
    compiler.emit_into(&mut code).unwrap();
    assert!(
        code.chunks_exact(4)
            .any(|word| word == 0xD63F0200u32.to_le_bytes())
    ); // BLR X16.
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
mod native {
    use super::*;
    use core::arch::global_asm;
    use core::ffi::c_void;
    use rbpf::aarch64::{
        Invocation, NO_FAULT, NativeEntry, RuntimeCallbacks, STATUS_CALLBACK_ERROR,
        STATUS_DIVISION_BY_ZERO, STATUS_INVALID_INVOCATION, STATUS_OK,
    };

    #[link(name = "gcc_s")]
    unsafe extern "C" {
        fn __clear_cache(start: *mut u8, end: *mut u8);
    }

    // The caller seeds every callee-saved general register, calls generated code,
    // reports the number changed, and restores the host's original registers.
    global_asm!(
        r#"
        .global rbpf_test_preserved
        .type rbpf_test_preserved, %function
    rbpf_test_preserved:
        sub sp, sp, #128
        stp x19, x20, [sp, #0]
        stp x21, x22, [sp, #16]
        stp x23, x24, [sp, #32]
        stp x25, x26, [sp, #48]
        stp x27, x28, [sp, #64]
        stp x29, x30, [sp, #80]
        str x2, [sp, #96]
        str x0, [sp, #104]
        str x1, [sp, #112]
        mov x19, #0x101
        mov x20, #0x102
        mov x21, #0x103
        mov x22, #0x104
        mov x23, #0x105
        mov x24, #0x106
        mov x25, #0x107
        mov x26, #0x108
        mov x27, #0x109
        mov x28, #0x10a
        mov x29, #0x10b
        ldr x16, [sp, #104]
        ldr x0, [sp, #112]
        blr x16
        ldr x15, [sp, #96]
        str w0, [x15]
        mov w0, #0
        cmp x19, #0x101
        cset w1, ne
        add w0, w0, w1
        cmp x20, #0x102
        cset w1, ne
        add w0, w0, w1
        cmp x21, #0x103
        cset w1, ne
        add w0, w0, w1
        cmp x22, #0x104
        cset w1, ne
        add w0, w0, w1
        cmp x23, #0x105
        cset w1, ne
        add w0, w0, w1
        cmp x24, #0x106
        cset w1, ne
        add w0, w0, w1
        cmp x25, #0x107
        cset w1, ne
        add w0, w0, w1
        cmp x26, #0x108
        cset w1, ne
        add w0, w0, w1
        cmp x27, #0x109
        cset w1, ne
        add w0, w0, w1
        cmp x28, #0x10a
        cset w1, ne
        add w0, w0, w1
        cmp x29, #0x10b
        cset w1, ne
        add w0, w0, w1
        ldp x19, x20, [sp, #0]
        ldp x21, x22, [sp, #16]
        ldp x23, x24, [sp, #32]
        ldp x25, x26, [sp, #48]
        ldp x27, x28, [sp, #64]
        ldp x29, x30, [sp, #80]
        add sp, sp, #128
        ret
    "#
    );

    unsafe extern "C" {
        fn rbpf_test_preserved(
            entry: NativeEntry,
            invocation: *mut Invocation,
            status: *mut u32,
        ) -> u32;
    }

    struct Image {
        ptr: *mut c_void,
        len: usize,
        entry_offset: usize,
    }

    impl Image {
        fn compile(insns: &[Insn]) -> Self {
            Self::compile_with(insns, options(insns.len()))
        }

        fn compile_with(insns: &[Insn], opts: CompileOptions) -> Self {
            let program = bytes(insns);
            let mut scratch = vec![0u32; insns.len()];
            let compiler = Aarch64Compiler::new(&program, opts, &mut scratch).unwrap();
            let mut code = vec![0u8; compiler.code_len()];
            let info = compiler.emit_into(&mut code).unwrap();
            assert_eq!(info.code_len, code.len());
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            assert!(page_size > 0);
            let len = code.len().div_ceil(page_size as usize) * page_size as usize;
            let ptr = unsafe {
                libc::mmap(
                    core::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert_ne!(
                ptr,
                libc::MAP_FAILED,
                "mmap failed: {}",
                std::io::Error::last_os_error()
            );
            unsafe {
                core::ptr::copy_nonoverlapping(code.as_ptr(), ptr.cast::<u8>(), code.len());
                __clear_cache(ptr.cast::<u8>(), ptr.cast::<u8>().add(code.len()));
            }
            let protected = unsafe { libc::mprotect(ptr, len, libc::PROT_READ | libc::PROT_EXEC) };
            assert_eq!(
                protected,
                0,
                "mprotect failed: {}",
                std::io::Error::last_os_error()
            );
            Self {
                ptr,
                len,
                entry_offset: info.entry_offset,
            }
        }

        unsafe fn entry(&self) -> NativeEntry {
            unsafe { core::mem::transmute(self.ptr.cast::<u8>().add(self.entry_offset)) }
        }
    }

    impl Drop for Image {
        fn drop(&mut self) {
            assert_eq!(unsafe { libc::munmap(self.ptr, self.len) }, 0);
        }
    }

    #[derive(Default)]
    struct State {
        bytes: [u8; 16],
        helper_calls: usize,
        load_calls: usize,
        store_calls: usize,
        fail: bool,
        bad_call: bool,
        stack_top: u64,
        expected_helper: Option<(u32, [u64; 5], u64)>,
        helper_failure_code: Option<u32>,
    }

    unsafe extern "C" fn helper(
        opaque: *mut c_void,
        id: u32,
        args: *const u64,
        out: *mut u64,
    ) -> u32 {
        let state = unsafe { &mut *opaque.cast::<State>() };
        state.helper_calls += 1;
        if state.fail {
            return STATUS_CALLBACK_ERROR;
        }
        if id != state.expected_helper.map_or(7, |expected| expected.0) {
            state.bad_call = true;
            return STATUS_CALLBACK_ERROR;
        }
        let args = unsafe { core::slice::from_raw_parts(args, 5) };
        if let Some((_, expected_args, _)) = state.expected_helper
            && args != expected_args
        {
            state.bad_call = true;
            return STATUS_CALLBACK_ERROR;
        }
        if let Some(code) = state.helper_failure_code {
            return code;
        }
        unsafe {
            *out = state.expected_helper.map_or_else(
                || args.iter().fold(0u64, |sum, arg| sum.wrapping_add(*arg)),
                |expected| expected.2,
            );
        }
        STATUS_OK
    }

    unsafe extern "C" fn load(
        opaque: *mut c_void,
        base: u64,
        offset: i64,
        width: u32,
        is_frame_pointer: u32,
        out: *mut u64,
    ) -> u32 {
        let state = unsafe { &mut *opaque.cast::<State>() };
        state.load_calls += 1;
        if state.fail {
            return STATUS_CALLBACK_ERROR;
        }
        if is_frame_pointer == 1 {
            if base != state.stack_top || offset != -8 || width != 8 {
                state.bad_call = true;
                return STATUS_CALLBACK_ERROR;
            }
            let mut value = [0u8; 8];
            unsafe {
                core::ptr::copy_nonoverlapping((base - 8) as *const u8, value.as_mut_ptr(), 8);
            }
            unsafe {
                *out = u64::from_le_bytes(value);
            }
            return STATUS_OK;
        }
        let Ok(start) = usize::try_from(offset) else {
            state.bad_call = true;
            return STATUS_CALLBACK_ERROR;
        };
        let Some(end) = start.checked_add(width as usize) else {
            state.bad_call = true;
            return STATUS_CALLBACK_ERROR;
        };
        if (base, is_frame_pointer) != (7, 0)
            || !matches!(width, 1 | 2 | 4 | 8)
            || end > state.bytes.len()
        {
            state.bad_call = true;
            return STATUS_CALLBACK_ERROR;
        }
        let mut value = [0u8; 8];
        value[..width as usize].copy_from_slice(&state.bytes[start..end]);
        unsafe {
            *out = u64::from_le_bytes(value);
            // R0 lives in X9. A memory callback may clobber this volatile native register.
            core::arch::asm!("mov x9, #0x777", out("x9") _, options(nostack, nomem));
        }
        STATUS_OK
    }

    unsafe extern "C" fn store(
        opaque: *mut c_void,
        base: u64,
        offset: i64,
        width: u32,
        is_frame_pointer: u32,
        value: u64,
    ) -> u32 {
        let state = unsafe { &mut *opaque.cast::<State>() };
        state.store_calls += 1;
        if state.fail {
            return STATUS_CALLBACK_ERROR;
        }
        if is_frame_pointer == 1 {
            if base != state.stack_top || offset != -8 || width != 8 {
                state.bad_call = true;
                return STATUS_CALLBACK_ERROR;
            }
            unsafe {
                core::ptr::copy_nonoverlapping(
                    value.to_le_bytes().as_ptr(),
                    (base - 8) as *mut u8,
                    8,
                );
            }
            return STATUS_OK;
        }
        let Ok(start) = usize::try_from(offset) else {
            state.bad_call = true;
            return STATUS_CALLBACK_ERROR;
        };
        let Some(end) = start.checked_add(width as usize) else {
            state.bad_call = true;
            return STATUS_CALLBACK_ERROR;
        };
        if (base, is_frame_pointer) != (7, 0)
            || !matches!(width, 1 | 2 | 4 | 8)
            || end > state.bytes.len()
        {
            state.bad_call = true;
            return STATUS_CALLBACK_ERROR;
        }
        state.bytes[start..end].copy_from_slice(&value.to_le_bytes()[..width as usize]);
        STATUS_OK
    }

    static CALLBACKS: RuntimeCallbacks = RuntimeCallbacks {
        helper,
        load,
        store,
    };

    fn run(insns: &[Insn], state: &mut State) -> (u32, u64, u64) {
        let image = Image::compile(insns);
        let mut stack = [0u8; 512];
        let mut context = [0u8; 8];
        let mut invocation = Invocation {
            context: context.as_mut_ptr(),
            stack: stack.as_mut_ptr(),
            stack_len: stack.len(),
            opaque: (state as *mut State).cast(),
            callbacks: &CALLBACKS,
            result: 0xBAD,
            fault_pc: NO_FAULT,
        };
        state.stack_top = (stack.as_mut_ptr() as u64) + stack.len() as u64;
        let status = unsafe { (image.entry())(&mut invocation) };
        (status, invocation.result, invocation.fault_pc)
    }

    fn value(insns: &[Insn], expected: u64) {
        let (status, result, fault) = run(insns, &mut State::default());
        assert_eq!((status, result, fault), (STATUS_OK, expected, NO_FAULT));
    }

    fn alu_expected(op: u8, lhs: u64, rhs: u64, wide: bool) -> u64 {
        let lhs = if wide { lhs } else { lhs as u32 as u64 };
        let rhs = if wide { rhs } else { rhs as u32 as u64 };
        let shift = if wide { rhs & 63 } else { rhs & 31 } as u32;
        let result = match op {
            ebpf::BPF_ADD => lhs.wrapping_add(rhs),
            ebpf::BPF_SUB => lhs.wrapping_sub(rhs),
            ebpf::BPF_MUL => lhs.wrapping_mul(rhs),
            ebpf::BPF_DIV => lhs / rhs,
            ebpf::BPF_MOD => lhs % rhs,
            ebpf::BPF_OR => lhs | rhs,
            ebpf::BPF_AND => lhs & rhs,
            ebpf::BPF_XOR => lhs ^ rhs,
            ebpf::BPF_LSH => lhs.wrapping_shl(shift),
            ebpf::BPF_RSH => lhs.wrapping_shr(shift),
            ebpf::BPF_ARSH => {
                if wide {
                    ((lhs as i64) >> shift) as u64
                } else {
                    ((lhs as u32 as i32) >> shift) as u32 as u64
                }
            }
            ebpf::BPF_MOV => rhs,
            _ => unreachable!(),
        };
        if wide { result } else { result as u32 as u64 }
    }

    #[test]
    fn every_alu_opcode_immediate_and_register_at_both_widths() {
        for op in [
            ebpf::BPF_ADD,
            ebpf::BPF_SUB,
            ebpf::BPF_MUL,
            ebpf::BPF_DIV,
            ebpf::BPF_MOD,
            ebpf::BPF_OR,
            ebpf::BPF_AND,
            ebpf::BPF_XOR,
            ebpf::BPF_LSH,
            ebpf::BPF_RSH,
            ebpf::BPF_ARSH,
            ebpf::BPF_MOV,
        ] {
            for wide in [false, true] {
                for register in [false, true] {
                    let opcode = op
                        | if wide { ebpf::BPF_ALU64 } else { ebpf::BPF_ALU }
                        | if register { ebpf::BPF_X } else { ebpf::BPF_K };
                    let lhs = 0x1234u64;
                    let rhs = 3u64;
                    let program = [
                        insn(ebpf::MOV64_IMM, 0, 0, 0, lhs as i32),
                        insn(ebpf::MOV64_IMM, 2, 0, 0, rhs as i32),
                        insn(
                            opcode,
                            0,
                            if register { 2 } else { 0 },
                            0,
                            if register { 0 } else { rhs as i32 },
                        ),
                        insn(ebpf::EXIT, 0, 0, 0, 0),
                    ];
                    value(&program, alu_expected(op, lhs, rhs, wide));
                }
            }
        }
        value(
            &[
                insn(ebpf::MOV64_IMM, 0, 0, 0, 5),
                insn(ebpf::NEG64, 0, 0, 0, 0),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            0u64.wrapping_sub(5),
        );
        value(
            &[
                insn(ebpf::MOV64_IMM, 0, 0, 0, 5),
                insn(ebpf::NEG32, 0, 0, 0, 0),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            0u32.wrapping_sub(5) as u64,
        );
    }

    fn branch_expected(op: u8, lhs: u64, rhs: u64, wide: bool) -> bool {
        let lhs = if wide { lhs } else { lhs as u32 as u64 };
        let rhs = if wide { rhs } else { rhs as u32 as u64 };
        let (signed_lhs, signed_rhs) = if wide {
            (lhs as i64, rhs as i64)
        } else {
            (lhs as u32 as i32 as i64, rhs as u32 as i32 as i64)
        };
        match op {
            ebpf::BPF_JEQ => lhs == rhs,
            ebpf::BPF_JNE => lhs != rhs,
            ebpf::BPF_JGT => lhs > rhs,
            ebpf::BPF_JGE => lhs >= rhs,
            ebpf::BPF_JLT => lhs < rhs,
            ebpf::BPF_JLE => lhs <= rhs,
            ebpf::BPF_JSET => lhs & rhs != 0,
            ebpf::BPF_JSGT => signed_lhs > signed_rhs,
            ebpf::BPF_JSGE => signed_lhs >= signed_rhs,
            ebpf::BPF_JSLT => signed_lhs < signed_rhs,
            ebpf::BPF_JSLE => signed_lhs <= signed_rhs,
            _ => unreachable!(),
        }
    }

    #[test]
    fn all_conditional_branches_taken_and_not_taken_at_both_widths() {
        let cases = [
            (ebpf::BPF_JEQ, 3i32, 2i32),
            (ebpf::BPF_JNE, 2, 3),
            (ebpf::BPF_JGT, 4, 2),
            (ebpf::BPF_JGE, 3, 2),
            (ebpf::BPF_JLT, 2, 4),
            (ebpf::BPF_JLE, 3, 4),
            (ebpf::BPF_JSET, 2, 4),
            (ebpf::BPF_JSGT, 4, -1),
            (ebpf::BPF_JSGE, 3, -1),
            (ebpf::BPF_JSLT, -1, 4),
            (ebpf::BPF_JSLE, -1, 4),
        ];
        for (op, taken, skipped) in cases {
            for wide in [false, true] {
                for register in [false, true] {
                    for (lhs, expected) in [(taken, true), (skipped, false)] {
                        assert_eq!(branch_expected(op, lhs as i64 as u64, 3, wide), expected);
                        let opcode = op
                            | if wide { ebpf::BPF_JMP } else { ebpf::BPF_JMP32 }
                            | if register { ebpf::BPF_X } else { ebpf::BPF_K };
                        value(
                            &[
                                insn(ebpf::MOV64_IMM, 0, 0, 0, lhs),
                                insn(ebpf::MOV64_IMM, 2, 0, 0, 3),
                                insn(
                                    opcode,
                                    0,
                                    if register { 2 } else { 0 },
                                    2,
                                    if register { 0 } else { 3 },
                                ),
                                insn(ebpf::MOV64_IMM, 0, 0, 0, 0),
                                insn(ebpf::EXIT, 0, 0, 0, 0),
                                insn(ebpf::MOV64_IMM, 0, 0, 0, 1),
                                insn(ebpf::EXIT, 0, 0, 0, 0),
                            ],
                            u64::from(expected),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn alu_widths_wide_immediates_and_forward_control_flow() {
        for (op, bits, expected) in [
            (ebpf::LE, 16, 0x0708),
            (ebpf::LE, 32, 0x05060708),
            (ebpf::LE, 64, 0x0102030405060708),
            (ebpf::BE, 16, 0x0807),
            (ebpf::BE, 32, 0x08070605),
            (ebpf::BE, 64, 0x0807060504030201),
        ] {
            value(
                &[
                    insn(ebpf::LD_DW_IMM, 0, 0, 0, 0x05060708),
                    insn(0, 0, 0, 0, 0x01020304),
                    insn(op, 0, 0, 0, bits),
                    insn(ebpf::EXIT, 0, 0, 0, 0),
                ],
                expected,
            );
        }
        for (op, expected) in [(ebpf::JGT_IMM, 1), (ebpf::JGT_IMM32, 0)] {
            value(
                &[
                    insn(ebpf::LD_DW_IMM, 0, 0, 0, 0),
                    insn(0, 0, 0, 0, 1),
                    insn(op, 0, 0, 2, 3),
                    insn(ebpf::MOV64_IMM, 0, 0, 0, 0),
                    insn(ebpf::EXIT, 0, 0, 0, 0),
                    insn(ebpf::MOV64_IMM, 0, 0, 0, 1),
                    insn(ebpf::EXIT, 0, 0, 0, 0),
                ],
                expected,
            );
        }
        value(
            &[
                insn(ebpf::LD_DW_IMM, 0, 0, 0, -1),
                insn(0, 0, 0, 0, -1),
                insn(ebpf::ADD32_IMM, 0, 0, 0, 1),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            0,
        );
        value(
            &[
                insn(ebpf::MOV64_IMM, 0, 0, 0, -1),
                insn(ebpf::ADD64_IMM, 0, 0, 0, 2),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            1,
        );
        value(
            &[
                insn(ebpf::MOV64_IMM, 0, 0, 0, -1),
                insn(ebpf::JSLT_IMM32, 0, 0, 1, 0),
                insn(ebpf::MOV64_IMM, 0, 0, 0, 99),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            u64::MAX,
        );
        value(
            &[
                insn(ebpf::MOV64_IMM, 0, 0, 0, 0),
                insn(ebpf::JEQ_IMM, 0, 0, 1, 0),
                insn(ebpf::MOV64_IMM, 0, 0, 0, 99),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            0,
        );
        value(
            &[
                insn(ebpf::MOV64_IMM, 0, 0, 0, -2),
                insn(ebpf::ARSH32_IMM, 0, 0, 0, 1),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            u32::MAX as u64,
        );
        value(
            &[
                insn(ebpf::MOV64_IMM, 0, 0, 0, 1),
                insn(ebpf::MOV64_IMM, 2, 0, 0, 65),
                insn(ebpf::LSH64_REG, 0, 2, 0, 0),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            2,
        );
        value(
            &[
                insn(ebpf::MOV64_IMM, 0, 0, 0, 0x1234),
                insn(ebpf::BE, 0, 0, 0, 16),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            0x3412,
        );
        value(
            &[
                insn(ebpf::MOV64_IMM, 0, 0, 0, -1),
                insn(ebpf::JSLT_IMM, 0, 0, 1, 0),
                insn(ebpf::MOV64_IMM, 0, 0, 0, 99),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            u64::MAX,
        );
        value(
            &[
                insn(ebpf::MOV64_IMM, 0, 0, 0, 1),
                insn(ebpf::JEQ_IMM, 0, 0, 1, 1),
                insn(ebpf::EXIT, 0, 0, 0, 0),
                insn(ebpf::MOV64_IMM, 0, 0, 0, 2),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            2,
        );
    }

    #[test]
    fn arithmetic_failure_stops_execution() {
        let (status, result, fault) = run(
            &[
                insn(ebpf::MOV64_IMM, 0, 0, 0, 3),
                insn(ebpf::MOV64_IMM, 2, 0, 0, 0),
                insn(ebpf::DIV64_REG, 0, 2, 0, 0),
                insn(ebpf::ST_B_IMM, 1, 0, 0, 1),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            &mut State::default(),
        );
        assert_eq!((status, result, fault), (STATUS_DIVISION_BY_ZERO, 0, 2));

        // A 64-bit divisor with a zero low word is still nonzero.
        for (opcode, expected) in [(ebpf::DIV64_REG, 1), (ebpf::MOD64_REG, 10)] {
            value(
                &[
                    insn(ebpf::LD_DW_IMM, 0, 0, 0, 10),
                    insn(0, 0, 0, 0, 1),
                    insn(ebpf::LD_DW_IMM, 2, 0, 0, 0),
                    insn(0, 0, 0, 0, 1),
                    insn(opcode, 0, 2, 0, 0),
                    insn(ebpf::EXIT, 0, 0, 0, 0),
                ],
                expected,
            );
        }
        for (opcode, expected) in [
            (ebpf::DIV64_REG, 1),
            (ebpf::MOD64_REG, 0),
            (ebpf::DIV32_REG, 1),
            (ebpf::MOD32_REG, 0),
        ] {
            value(
                &[
                    insn(ebpf::MOV64_IMM, 0, 0, 0, 17),
                    insn(opcode, 0, 0, 0, 0),
                    insn(ebpf::EXIT, 0, 0, 0, 0),
                ],
                expected,
            );
        }
        value(
            &[
                insn(ebpf::MOV64_IMM, 0, 0, 0, -7),
                insn(ebpf::ADD64_IMM, 0, 0, 0, -5),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            (-12i64) as u64,
        );
    }

    #[test]
    fn callbacks_return_values_effects_and_failures() {
        let mut state = State::default();
        let (status, result, fault) = run(
            &[
                insn(ebpf::MOV64_IMM, 6, 0, 0, 40),
                insn(ebpf::MOV64_IMM, 1, 0, 0, 7),
                insn(ebpf::ST_H_IMM, 1, 0, 1, 0x1234),
                insn(ebpf::LD_B_REG, 2, 1, 2, 0),
                insn(ebpf::MOV64_REG, 3, 2, 0, 0),
                insn(ebpf::MOV64_IMM, 2, 0, 0, 2),
                insn(ebpf::CALL, 0, 0, 0, 7),
                insn(ebpf::ADD64_REG, 0, 6, 0, 0),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            &mut state,
        );
        assert_eq!(
            (status, result, fault),
            (STATUS_OK, 7 + 2 + 0x12 + 40, NO_FAULT)
        );
        assert_eq!(&state.bytes[1..3], &[0x34, 0x12]);
        assert_eq!(
            (state.helper_calls, state.load_calls, state.store_calls),
            (1, 1, 1)
        );
        assert!(!state.bad_call);

        let mut failed = State {
            fail: true,
            ..State::default()
        };
        let (status, result, fault) = run(
            &[
                insn(ebpf::MOV64_IMM, 1, 0, 0, 7),
                insn(ebpf::LD_W_REG, 0, 1, 0, 0),
                insn(ebpf::ST_B_IMM, 1, 0, 0, 1),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            &mut failed,
        );
        assert_eq!((status, result, fault), (STATUS_CALLBACK_ERROR, 0, 1));
        assert_eq!((failed.load_calls, failed.store_calls), (1, 0));

        let mut failed = State {
            fail: true,
            ..State::default()
        };
        let (status, result, fault) = run(
            &[
                insn(ebpf::MOV64_IMM, 1, 0, 0, 7),
                insn(ebpf::ST_B_IMM, 1, 0, 0, 1),
                insn(ebpf::CALL, 0, 0, 0, 7),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            &mut failed,
        );
        assert_eq!((status, result, fault), (STATUS_CALLBACK_ERROR, 0, 1));
        assert_eq!((failed.store_calls, failed.helper_calls), (1, 0));
    }

    #[test]
    fn helper_callback_keeps_argument_order_and_callee_saved_registers() {
        let helper_result = 0x1234_5678_9abc_0000u64;
        let image = Image::compile(&[
            insn(ebpf::MOV64_IMM, 1, 0, 0, -17),
            insn(ebpf::MOV64_IMM, 2, 0, 0, 0x1234_5678),
            insn(ebpf::MOV64_IMM, 3, 0, 0, 3),
            insn(ebpf::MOV64_IMM, 4, 0, 0, 0x40000),
            insn(ebpf::MOV64_IMM, 5, 0, 0, -5),
            insn(ebpf::MOV64_IMM, 6, 0, 0, 6),
            insn(ebpf::MOV64_IMM, 7, 0, 0, 7),
            insn(ebpf::MOV64_IMM, 8, 0, 0, 8),
            insn(ebpf::MOV64_IMM, 9, 0, 0, 9),
            insn(ebpf::CALL, 0, 0, 0, 1009),
            insn(ebpf::ADD64_REG, 0, 6, 0, 0),
            insn(ebpf::ADD64_REG, 0, 7, 0, 0),
            insn(ebpf::ADD64_REG, 0, 8, 0, 0),
            insn(ebpf::ADD64_REG, 0, 9, 0, 0),
            insn(ebpf::EXIT, 0, 0, 0, 0),
        ]);
        let mut state = State {
            expected_helper: Some((
                1009,
                [(-17i64) as u64, 0x1234_5678, 3, 0x40000, (-5i64) as u64],
                helper_result,
            )),
            ..State::default()
        };
        let context = [0u8; 8];
        let mut stack = [0u8; 512];
        let mut invocation = Invocation {
            context: context.as_ptr(),
            stack: stack.as_mut_ptr(),
            stack_len: stack.len(),
            opaque: (&mut state as *mut State).cast(),
            callbacks: &CALLBACKS,
            result: 0xBAD,
            fault_pc: 0,
        };
        let mut status = u32::MAX;
        let changed = unsafe { rbpf_test_preserved(image.entry(), &mut invocation, &mut status) };
        assert_eq!(
            changed, 0,
            "generated helper call changed a host callee-saved register"
        );
        assert_eq!(
            (status, invocation.result, invocation.fault_pc),
            (STATUS_OK, helper_result + 6 + 7 + 8 + 9, NO_FAULT)
        );
        assert_eq!((state.helper_calls, state.bad_call), (1, false));
    }

    #[test]
    fn helper_failure_stops_before_store_or_second_call() {
        let mut state = State {
            expected_helper: Some((1009, [7, 0, 0, 0, 0], 0)),
            helper_failure_code: Some(0xBEEF),
            ..State::default()
        };
        let (status, result, fault) = run(
            &[
                insn(ebpf::MOV64_IMM, 0, 0, 0, 42),
                insn(ebpf::MOV64_IMM, 1, 0, 0, 7),
                insn(ebpf::CALL, 0, 0, 0, 1009),
                insn(ebpf::ST_B_IMM, 1, 0, 0, 1),
                insn(ebpf::CALL, 0, 0, 0, 1009),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            &mut state,
        );
        assert_eq!((status, result, fault), (STATUS_CALLBACK_ERROR, 0, 2));
        assert_eq!(
            (state.helper_calls, state.store_calls, state.bad_call),
            (1, 0, false)
        );
        assert_eq!(state.bytes, [0; 16]);
    }

    #[test]
    fn memory_callback_preserves_live_r0_and_all_widths() {
        for (store_op, load_op, width, expected) in [
            (ebpf::ST_B_REG, ebpf::LD_B_REG, 1, 0x78),
            (ebpf::ST_H_REG, ebpf::LD_H_REG, 2, 0x5678),
            (ebpf::ST_W_REG, ebpf::LD_W_REG, 4, 0x12345678),
            (ebpf::ST_DW_REG, ebpf::LD_DW_REG, 8, 0x12345678),
        ] {
            let mut state = State::default();
            let (status, result, fault) = run(
                &[
                    insn(ebpf::MOV64_IMM, 0, 0, 0, 33),
                    insn(ebpf::MOV64_IMM, 1, 0, 0, 7),
                    insn(ebpf::MOV64_IMM, 2, 0, 0, 0x12345678),
                    insn(store_op, 1, 2, 1, 0),
                    insn(load_op, 3, 1, 1, 0),
                    insn(ebpf::ADD64_REG, 0, 3, 0, 0),
                    insn(ebpf::EXIT, 0, 0, 0, 0),
                ],
                &mut state,
            );
            assert_eq!(
                (status, result, fault),
                (STATUS_OK, expected + 33, NO_FAULT)
            );
            assert_eq!(
                &state.bytes[1..1 + width],
                &0x12345678u64.to_le_bytes()[..width]
            );
            assert_eq!((state.load_calls, state.store_calls), (1, 1));
            assert!(!state.bad_call);
        }
        let mut state = State::default();
        let (status, result, fault) = run(
            &[
                insn(ebpf::ST_DW_IMM, 10, 0, -8, 0x12345678),
                insn(ebpf::LD_DW_REG, 0, 10, -8, 0),
                insn(ebpf::EXIT, 0, 0, 0, 0),
            ],
            &mut state,
        );
        assert_eq!((status, result, fault), (STATUS_OK, 0x12345678, NO_FAULT));
        assert_eq!((state.load_calls, state.store_calls), (1, 1));
        assert!(!state.bad_call);
    }

    #[test]
    fn generated_entry_preserves_host_registers() {
        let image = Image::compile(&[
            insn(ebpf::MOV64_IMM, 0, 0, 0, 5),
            insn(ebpf::MOV64_IMM, 6, 0, 0, 6),
            insn(ebpf::ADD64_REG, 0, 6, 0, 0),
            insn(ebpf::EXIT, 0, 0, 0, 0),
        ]);
        let mut state = State::default();
        let mut context = [0u8; 8];
        let mut stack = [0u8; 512];
        let mut invocation = Invocation {
            context: context.as_mut_ptr(),
            stack: stack.as_mut_ptr(),
            stack_len: stack.len(),
            opaque: (&mut state as *mut State).cast(),
            callbacks: &CALLBACKS,
            result: 0,
            fault_pc: NO_FAULT,
        };
        let mut status = u32::MAX;
        let changed = unsafe { rbpf_test_preserved(image.entry(), &mut invocation, &mut status) };
        assert_eq!(changed, 0);
        assert_eq!(
            (status, invocation.result, invocation.fault_pc),
            (STATUS_OK, 11, NO_FAULT)
        );

        invocation.stack_len = 511;
        invocation.result = 99;
        invocation.fault_pc = 0;
        let status = unsafe { (image.entry())(&mut invocation) };
        assert_eq!(
            (status, invocation.result, invocation.fault_pc),
            (STATUS_INVALID_INVOCATION, 0, NO_FAULT)
        );

        invocation.stack_len = stack.len();
        invocation.stack = core::ptr::null_mut();
        invocation.result = 99;
        invocation.fault_pc = 0;
        let status = unsafe { (image.entry())(&mut invocation) };
        assert_eq!(
            (status, invocation.result, invocation.fault_pc),
            (STATUS_INVALID_INVOCATION, 0, NO_FAULT)
        );

        invocation.stack = (usize::MAX - 256) as *mut u8;
        invocation.result = 99;
        invocation.fault_pc = 0;
        let status = unsafe { (image.entry())(&mut invocation) };
        assert_eq!(
            (status, invocation.result, invocation.fault_pc),
            (STATUS_INVALID_INVOCATION, 0, NO_FAULT)
        );

        invocation.stack = stack.as_mut_ptr();
        invocation.callbacks = core::ptr::null();
        invocation.result = 99;
        invocation.fault_pc = 0;
        let status = unsafe { (image.entry())(&mut invocation) };
        assert_eq!(
            (status, invocation.result, invocation.fault_pc),
            (STATUS_INVALID_INVOCATION, 0, NO_FAULT)
        );

        invocation.callbacks = &CALLBACKS;
        for _ in 0..3 {
            invocation.result = 99;
            invocation.fault_pc = 0;
            let status = unsafe { (image.entry())(&mut invocation) };
            assert_eq!(
                (status, invocation.result, invocation.fault_pc),
                (STATUS_OK, 11, NO_FAULT)
            );
        }
    }

    #[test]
    fn eight_kibibyte_stack_places_r10_at_compiled_top() {
        let program = [
            insn(ebpf::ST_DW_IMM, 10, 0, -8, 0x1234),
            insn(ebpf::LD_DW_REG, 0, 10, -8, 0),
            insn(ebpf::EXIT, 0, 0, 0, 0),
        ];
        let image = Image::compile_with(
            &program,
            CompileOptions {
                stack_size: 8192,
                ..options(program.len())
            },
        );
        let mut state = State::default();
        let mut stack = [0u8; 8192];
        let context = [0u8; 8];
        state.stack_top = stack.as_mut_ptr() as u64 + stack.len() as u64;
        let mut invocation = Invocation {
            context: context.as_ptr(),
            stack: stack.as_mut_ptr(),
            stack_len: stack.len(),
            opaque: (&mut state as *mut State).cast(),
            callbacks: &CALLBACKS,
            result: 99,
            fault_pc: 0,
        };
        let status = unsafe { (image.entry())(&mut invocation) };
        assert_eq!(
            (status, invocation.result, invocation.fault_pc),
            (STATUS_OK, 0x1234, NO_FAULT)
        );
        assert_eq!(
            (state.load_calls, state.store_calls, state.bad_call),
            (1, 1, false)
        );
    }
}
