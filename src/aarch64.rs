// SPDX-License-Identifier: (Apache-2.0 OR MIT)

//! Allocation-free AArch64 code generation for a normalized, forward-only BPF subset.
//!
//! The compiler is available on any host with the aarch64-jit feature. It writes
//! little-endian AArch64 instructions into caller-owned, non-executable storage.
//! It neither allocates pages nor changes their permissions. The embedder must
//! verify pointer safety, own the BPF stack and runtime callbacks, establish W^X,
//! synchronize instruction caches, and retain the image until execution ends.
//!
//! Arithmetic and branches run natively; memory accesses and external helpers
//! call the embedder. Local calls, loops, atomics and packet-load modes reject.
//! Division and remainder by zero trap instead of using Linux's zero-divisor
//! result convention. This matches the intended managed-runtime error policy.

use crate::ebpf::{self, Insn};
use core::{ffi::c_void, fmt};

/// Native execution completed successfully.
pub const STATUS_OK: u32 = 0;
/// A callback returned a nonzero status. Its detailed error belongs to the embedder.
pub const STATUS_CALLBACK_ERROR: u32 = 1;
/// Division or remainder encountered a zero divisor.
pub const STATUS_DIVISION_BY_ZERO: u32 = 2;
/// The supplied stack or callback-table pointer was invalid.
pub const STATUS_INVALID_INVOCATION: u32 = 3;
/// No BPF instruction caused a fault (also used for invalid invocation arguments).
pub const NO_FAULT: u64 = u64::MAX;

/// External-helper callback: opaque state, helper ID, five arguments, output.
///
/// The arguments and output pointers are valid only during the call. Return zero
/// on success. Do not retain these pointers, modify arguments, or unwind.
pub type HelperCallback = unsafe extern "C" fn(*mut c_void, u32, *const u64, *mut u64) -> u32;
/// Checked read: opaque state, base, signed offset, byte width, R10-base flag, output.
///
/// Width is 1, 2, 4 or 8. The flag is 1 only when the instruction names R10
/// as its base; a stack pointer copied to another register has flag 0. Validate
/// the effective address and its complete readable extent regardless of the flag.
/// On success write the zero-extended value and return zero. Do not retain output
/// or unwind.
pub type LoadCallback = unsafe extern "C" fn(*mut c_void, u64, i64, u32, u32, *mut u64) -> u32;
/// Checked write: opaque state, base, signed offset, byte width, R10-base flag, value.
///
/// The R10-base flag describes the instruction's base register, not pointer
/// provenance. Validate the effective address, permissions and complete writable
/// extent regardless of the flag; store the low width bytes. Return zero on
/// success and do not unwind.
pub type StoreCallback = unsafe extern "C" fn(*mut c_void, u64, i64, u32, u32, u64) -> u32;

/// Trusted callbacks, borrowed for an entire native invocation.
#[repr(C)]
pub struct RuntimeCallbacks {
    /// Resolve a helper under the embedder's verified helper/effect policy.
    pub helper: HelperCallback,
    /// Perform a checked memory read.
    pub load: LoadCallback,
    /// Perform a checked memory write.
    pub store: StoreCallback,
}

/// Arguments and results exchanged with generated little-endian AArch64 code.
///
/// The ABI requires a 64-bit little-endian target. The caller owns all pointed-to
/// objects for the duration of execution and must avoid conflicting Rust borrows.
/// The callbacks must not mutate this frame or the generated image.
#[repr(C)]
pub struct Invocation {
    /// Initial BPF R1 (the embedder's context wrapper, not necessarily its payload).
    pub context: *const u8,
    /// Borrowed writable BPF stack, already initialized by the embedder.
    pub stack: *mut u8,
    /// Accessible stack bytes; must cover the compiled stack_size. R10 uses the
    /// compiled extent even when this length is larger.
    pub stack_len: usize,
    /// Invocation-local state passed to callbacks.
    pub opaque: *mut c_void,
    /// Non-null pointer to valid callbacks.
    pub callbacks: *const RuntimeCallbacks,
    /// BPF R0 on success; zero on failure.
    pub result: u64,
    /// Original BPF instruction slot on failure, or NO_FAULT.
    pub fault_pc: u64,
}

/// Signature of an emitted image's entry point.
///
/// # Safety
///
/// Call only on little-endian AArch64 after external semantic verification,
/// read-only executable publication and cache synchronization. The image,
/// invocation frame, stack, callback table and callback-owned regions must all
/// remain valid and exclusively accessible as required for the entire call.
/// Callback code must implement the verifier's memory/helper policy and cannot
/// unwind. The compiler's structural validation alone does not establish safety.
/// The caller must enforce callback latency bounds externally.
pub type NativeEntry = unsafe extern "C" fn(*mut Invocation) -> u32;

/// Embedder-owned resource bounds. No memory is allocated by the compiler.
#[derive(Clone, Copy, Debug)]
pub struct CompileOptions {
    /// Maximum encoded BPF slots, including wide-immediate continuation slots.
    pub max_instructions: usize,
    /// Maximum emitted bytes, including the native prologue and epilogue.
    pub max_code_size: usize,
    /// BPF stack extent used to initialize R10; must be nonzero.
    pub stack_size: usize,
}

/// Structural or resource failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// Noncanonical encoding, invalid register, truncation or missing termination.
    InvalidProgram,
    /// Opcode/mode outside the supported subset.
    UnsupportedInstruction,
    /// Backward, out-of-program or continuation-slot jump target.
    InvalidJump,
    /// Invalid compiler options.
    InvalidOptions,
    /// Fewer scratch entries than encoded BPF slots.
    InsufficientScratch,
    /// Output is smaller than the measured image.
    InsufficientOutput,
    /// Code exceeds the supplied bound or architectural displacement range.
    CodeTooLarge,
}

/// Allocation-free compiler error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JitError {
    /// Failure category.
    pub kind: ErrorKind,
    /// Original BPF slot, when the error belongs to an instruction.
    pub pc: Option<usize>,
}

impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.pc {
            Some(pc) => write!(f, "AArch64 JIT {:?} at BPF slot {}", self.kind, pc),
            None => write!(f, "AArch64 JIT {:?}", self.kind),
        }
    }
}

impl core::error::Error for JitError {}

/// Description of bytes emitted into caller storage, not an executable owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodeInfo {
    /// Emitted image length.
    pub code_len: usize,
    /// Entry offset from the image start (currently zero).
    pub entry_offset: usize,
}

const CONTINUATION: u32 = u32::MAX;
// Every branch is within one image. Keeping the entire image below +128 MiB
// makes unconditional B range checks independent of instruction expansion.
const MAX_CODE_SIZE: usize = (1 << 27) - 4;
const FRAME_SIZE: u16 = 160;
const SAVED_R0: usize = 96;
const SAVED_INVOCATION: usize = 104;
const CALLBACK_RESULT: usize = 112;
const HELPER_ARGS: usize = 120;
const SP: u8 = 31;
const R0: u8 = 9;
const TMP: u8 = 10;
const TMP2: u8 = 11;
const CALL_TARGET: u8 = 16;

// These are target offsets, not the compiling host's pointer-sized layout.
const CTX: usize = 0;
const STACK: usize = 8;
const STACK_LEN: usize = 16;
const OPAQUE: usize = 24;
const CALLBACKS: usize = 32;
const RESULT: usize = 40;
const FAULT: usize = 48;

// Guard against accidental ABI layout drift on supported execution targets.
#[cfg(all(target_pointer_width = "64", target_endian = "little"))]
const _: () = {
    use core::mem::offset_of;

    assert!(offset_of!(Invocation, context) == CTX);
    assert!(offset_of!(Invocation, stack) == STACK);
    assert!(offset_of!(Invocation, stack_len) == STACK_LEN);
    assert!(offset_of!(Invocation, opaque) == OPAQUE);
    assert!(offset_of!(Invocation, callbacks) == CALLBACKS);
    assert!(offset_of!(Invocation, result) == RESULT);
    assert!(offset_of!(Invocation, fault_pc) == FAULT);
    assert!(offset_of!(RuntimeCallbacks, helper) == 0);
    assert!(offset_of!(RuntimeCallbacks, load) == 8);
    assert!(offset_of!(RuntimeCallbacks, store) == 16);
};

fn error(kind: ErrorKind, pc: Option<usize>) -> JitError {
    JitError { kind, pc }
}

fn reg(bpf: u8) -> u8 {
    // BPF R1..R10 live in native callee-saved X19..X28. R0 is spilled around
    // callbacks. X29 is a normal native frame pointer and X18 is untouched.
    if bpf == 0 { R0 } else { 18 + bpf }
}

/// Measured program borrowing its bytecode and offset scratch until emission.
pub struct Aarch64Compiler<'program, 'scratch> {
    program: &'program [u8],
    options: CompileOptions,
    offsets: &'scratch [u32],
    epilogue: usize,
    code_len: usize,
}

impl<'program, 'scratch> Aarch64Compiler<'program, 'scratch> {
    /// Validate and measure a program without allocation.
    ///
    /// The scratch needs one u32 per encoded instruction slot. Its contents may
    /// change on failure. Requiring a final EXIT plus forward valid targets
    /// ensures every structurally accepted control-flow path terminates.
    pub fn new(
        program: &'program [u8],
        options: CompileOptions,
        offsets: &'scratch mut [u32],
    ) -> Result<Self, JitError> {
        if options.stack_size == 0
            || options.stack_size > isize::MAX as usize
            || options.max_instructions == 0
        {
            return Err(error(ErrorKind::InvalidOptions, None));
        }
        if program.is_empty() || !program.len().is_multiple_of(ebpf::INSN_SIZE) {
            return Err(error(ErrorKind::InvalidProgram, None));
        }
        let count = program.len() / ebpf::INSN_SIZE;
        if count > options.max_instructions {
            return Err(error(ErrorKind::InvalidProgram, None));
        }
        let offsets = offsets
            .get_mut(..count)
            .ok_or(error(ErrorKind::InsufficientScratch, None))?;
        offsets.fill(CONTINUATION);
        let mut pc = 0;
        while pc < count {
            let slots = validate(program, pc)?;
            offsets[pc] = 0;
            pc += slots;
        }
        if offsets[count - 1] == CONTINUATION
            || ebpf::get_insn(program, count - 1).opc != ebpf::EXIT
        {
            return Err(error(ErrorKind::InvalidProgram, Some(count - 1)));
        }
        for pc in 0..count {
            if offsets[pc] != CONTINUATION
                && let Some(target) = jump_target(ebpf::get_insn(program, pc), pc, count)?
                && offsets[target] == CONTINUATION
            {
                return Err(error(ErrorKind::InvalidJump, Some(pc)));
            }
        }
        let limit = options.max_code_size.min(MAX_CODE_SIZE);
        let mut sink = Sink::new(None, limit);
        prologue(&mut sink, options.stack_size, 0)?;
        pc = 0;
        while pc < count {
            offsets[pc] =
                u32::try_from(sink.pos).map_err(|_| error(ErrorKind::CodeTooLarge, Some(pc)))?;
            emit_insn(&mut sink, program, pc, offsets, 0).map_err(|mut e| {
                e.pc = Some(pc);
                e
            })?;
            pc += if ebpf::get_insn(program, pc).opc == ebpf::LD_DW_IMM {
                2
            } else {
                1
            };
        }
        let epilogue = sink.pos;
        emit_epilogue(&mut sink)?;
        Ok(Self {
            program,
            options,
            offsets,
            epilogue,
            code_len: sink.pos,
        })
    }

    /// Exact image length, including the native call frame.
    pub fn code_len(&self) -> usize {
        self.code_len
    }

    /// Emit the measured image. A short output is rejected before any writes.
    ///
    /// End the mutable buffer borrow before mapping it executable. Bytes after
    /// code_len are untouched. The entry point is at image offset zero.
    pub fn emit_into(&self, output: &mut [u8]) -> Result<CodeInfo, JitError> {
        if output.len() < self.code_len {
            return Err(error(ErrorKind::InsufficientOutput, None));
        }
        let mut sink = Sink::new(Some(output), self.code_len);
        prologue(&mut sink, self.options.stack_size, self.epilogue)?;
        let mut pc = 0;
        while pc < self.offsets.len() {
            emit_insn(&mut sink, self.program, pc, self.offsets, self.epilogue).map_err(
                |mut e| {
                    e.pc = Some(pc);
                    e
                },
            )?;
            pc += if ebpf::get_insn(self.program, pc).opc == ebpf::LD_DW_IMM {
                2
            } else {
                1
            };
        }
        emit_epilogue(&mut sink)?;
        Ok(CodeInfo {
            code_len: sink.pos,
            entry_offset: 0,
        })
    }
}

fn validate(program: &[u8], pc: usize) -> Result<usize, JitError> {
    let i = ebpf::get_insn(program, pc);
    let bad = || error(ErrorKind::InvalidProgram, Some(pc));
    let unsupported = || error(ErrorKind::UnsupportedInstruction, Some(pc));
    if i.dst > 10 || i.src > 10 {
        return Err(bad());
    }
    if i.opc == ebpf::LD_DW_IMM {
        if i.dst == 10 || i.src != 0 || i.off != 0 || pc + 1 >= program.len() / 8 {
            return Err(bad());
        }
        let tail = ebpf::get_insn(program, pc + 1);
        if tail.opc != 0 || tail.dst != 0 || tail.src != 0 || tail.off != 0 {
            return Err(bad());
        }
        return Ok(2);
    }
    let class = i.opc & 7;
    let op = i.opc & 0xf0;
    let register = i.opc & ebpf::BPF_X != 0;
    let canonical_source = if register { i.imm == 0 } else { i.src == 0 };
    match class {
        ebpf::BPF_ALU | ebpf::BPF_ALU64 => {
            if i.dst == 10 || i.off != 0 {
                return Err(bad());
            }
            match op {
                ebpf::BPF_END => {
                    if class != ebpf::BPF_ALU || i.src != 0 || !matches!(i.imm, 16 | 32 | 64) {
                        return Err(bad());
                    }
                }
                ebpf::BPF_NEG => {
                    if register || i.src != 0 || i.imm != 0 {
                        return Err(bad());
                    }
                }
                ebpf::BPF_ADD
                | ebpf::BPF_SUB
                | ebpf::BPF_MUL
                | ebpf::BPF_DIV
                | ebpf::BPF_OR
                | ebpf::BPF_AND
                | ebpf::BPF_LSH
                | ebpf::BPF_RSH
                | ebpf::BPF_MOD
                | ebpf::BPF_XOR
                | ebpf::BPF_MOV
                | ebpf::BPF_ARSH => {
                    if !canonical_source {
                        return Err(bad());
                    }
                }
                _ => return Err(unsupported()),
            }
        }
        ebpf::BPF_JMP | ebpf::BPF_JMP32 => match op {
            ebpf::BPF_CALL => {
                if i.opc != ebpf::CALL || i.dst != 0 || i.src != 0 || i.off != 0 {
                    return Err(unsupported());
                }
            }
            ebpf::BPF_EXIT => {
                if i.opc != ebpf::EXIT || i.dst != 0 || i.src != 0 || i.off != 0 || i.imm != 0 {
                    return Err(bad());
                }
            }
            ebpf::BPF_JA => {
                if i.opc != ebpf::JA || i.dst != 0 || i.src != 0 || i.imm != 0 {
                    return Err(unsupported());
                }
            }
            ebpf::BPF_JEQ
            | ebpf::BPF_JGT
            | ebpf::BPF_JGE
            | ebpf::BPF_JSET
            | ebpf::BPF_JNE
            | ebpf::BPF_JSGT
            | ebpf::BPF_JSGE
            | ebpf::BPF_JLT
            | ebpf::BPF_JLE
            | ebpf::BPF_JSLT
            | ebpf::BPF_JSLE => {
                if !canonical_source {
                    return Err(bad());
                }
            }
            _ => return Err(unsupported()),
        },
        ebpf::BPF_LDX | ebpf::BPF_ST | ebpf::BPF_STX => {
            if i.opc & 0xe0 != ebpf::BPF_MEM {
                return Err(unsupported());
            }
            if (class == ebpf::BPF_LDX && i.dst == 10)
                || (class != ebpf::BPF_ST && i.imm != 0)
                || (class == ebpf::BPF_ST && i.src != 0)
            {
                return Err(bad());
            }
        }
        _ => return Err(unsupported()),
    }
    Ok(1)
}

fn jump_target(i: Insn, pc: usize, count: usize) -> Result<Option<usize>, JitError> {
    let class = i.opc & 7;
    if !matches!(class, ebpf::BPF_JMP | ebpf::BPF_JMP32)
        || matches!(i.opc & 0xf0, ebpf::BPF_CALL | ebpf::BPF_EXIT)
    {
        return Ok(None);
    }
    if i.off < 0 {
        return Err(error(ErrorKind::InvalidJump, Some(pc)));
    }
    let target = pc
        .checked_add(1)
        .and_then(|n| n.checked_add(i.off as usize));
    match target {
        Some(target) if target < count => Ok(Some(target)),
        _ => Err(error(ErrorKind::InvalidJump, Some(pc))),
    }
}

struct Sink<'a> {
    output: Option<&'a mut [u8]>,
    pos: usize,
    limit: usize,
}

impl<'a> Sink<'a> {
    fn new(output: Option<&'a mut [u8]>, limit: usize) -> Self {
        Self {
            output,
            pos: 0,
            limit,
        }
    }

    fn word(&mut self, word: u32) -> Result<(), JitError> {
        let end = self
            .pos
            .checked_add(4)
            .filter(|end| *end <= self.limit)
            .ok_or(error(ErrorKind::CodeTooLarge, None))?;
        if let Some(output) = &mut self.output {
            output
                .get_mut(self.pos..end)
                .ok_or(error(ErrorKind::InsufficientOutput, None))?
                .copy_from_slice(&word.to_le_bytes());
        }
        self.pos = end;
        Ok(())
    }

    fn imm(&mut self, dst: u8, value: u64, wide: bool) -> Result<(), JitError> {
        // One MOVN covers values whose upper 48 bits are all ones.
        if wide && value >> 16 == u64::MAX >> 16 {
            return self.word(0x92800000 | ((!value as u32 & 0xffff) << 5) | u32::from(dst));
        }
        let sf = if wide { 0x80000000 } else { 0 };
        for part in 0..if wide { 4 } else { 2 } {
            // MOVZ clears the other halfwords; only nonzero parts need MOVK.
            if part != 0 && (value >> (part * 16)) & 0xffff == 0 {
                continue;
            }
            let op = if part == 0 { 0x52800000 } else { 0x72800000 };
            self.word(
                sf | op
                    | (part << 21)
                    | (((value >> (part * 16)) as u32 & 0xffff) << 5)
                    | u32::from(dst),
            )?;
        }
        Ok(())
    }

    fn mov(&mut self, dst: u8, src: u8, wide: bool) -> Result<(), JitError> {
        self.word(
            if wide { 0xaa0003e0 } else { 0x2a0003e0 } | (u32::from(src) << 16) | u32::from(dst),
        )
    }

    fn add_imm(&mut self, dst: u8, src: u8, value: u16) -> Result<(), JitError> {
        self.word(0x91000000 | (u32::from(value) << 10) | (u32::from(src) << 5) | u32::from(dst))
    }

    fn load(&mut self, dst: u8, base: u8, offset: usize) -> Result<(), JitError> {
        self.word(
            0xf9400000 | ((offset as u32 / 8) << 10) | (u32::from(base) << 5) | u32::from(dst),
        )
    }

    fn store(&mut self, src: u8, base: u8, offset: usize) -> Result<(), JitError> {
        self.word(
            0xf9000000 | ((offset as u32 / 8) << 10) | (u32::from(base) << 5) | u32::from(src),
        )
    }

    fn binary(&mut self, op: u32, dst: u8, lhs: u8, rhs: u8) -> Result<(), JitError> {
        self.word(op | (u32::from(rhs) << 16) | (u32::from(lhs) << 5) | u32::from(dst))
    }

    fn branch(&mut self, target: usize) -> Result<(), JitError> {
        let delta = target as i64 - self.pos as i64;
        if self.output.is_some() && (delta % 4 != 0 || !(-(1 << 27)..(1 << 27)).contains(&delta)) {
            return Err(error(ErrorKind::CodeTooLarge, None));
        }
        self.word(0x14000000 | ((delta / 4) as u32 & 0x03ffffff))
    }

    fn patch_cond(&mut self, at: usize, target: usize, cond: u8) -> Result<(), JitError> {
        let delta = target as i64 - at as i64;
        if delta % 4 != 0 || !(-(1 << 20)..(1 << 20)).contains(&delta) {
            return Err(error(ErrorKind::CodeTooLarge, None));
        }
        if let Some(output) = &mut self.output {
            let word = 0x54000000 | (((delta / 4) as u32 & 0x7ffff) << 5) | u32::from(cond);
            output
                .get_mut(at..at + 4)
                .ok_or(error(ErrorKind::InsufficientOutput, None))?
                .copy_from_slice(&word.to_le_bytes());
        }
        Ok(())
    }

    fn fail(&mut self, status: u32, pc: u64, epilogue: usize) -> Result<(), JitError> {
        self.load(TMP, SP, SAVED_INVOCATION)?;
        self.imm(TMP2, pc, true)?;
        self.store(TMP2, TMP, FAULT)?;
        self.imm(0, u64::from(status), false)?;
        self.branch(epilogue)
    }

    fn require(
        &mut self,
        valid_cond: u8,
        status: u32,
        pc: u64,
        epilogue: usize,
    ) -> Result<(), JitError> {
        let at = self.pos;
        self.word(0)?;
        self.fail(status, pc, epilogue)?;
        self.patch_cond(at, self.pos, valid_cond)
    }
}

fn prologue(s: &mut Sink<'_>, stack_size: usize, epilogue: usize) -> Result<(), JitError> {
    s.word(0xd10003ff | (u32::from(FRAME_SIZE) << 10))?; // SUB SP, SP, frame
    for pair in 0..6u32 {
        let first = 19 + pair * 2;
        s.word(0xa9000000 | ((pair * 2) << 15) | ((first + 1) << 10) | (31 << 5) | first)?;
    }
    s.add_imm(29, SP, 80)?; // FP points at the saved FP/LR record
    s.store(0, SP, SAVED_INVOCATION)?;
    s.store(31, 0, RESULT)?;
    s.imm(TMP, NO_FAULT, true)?;
    s.store(TMP, 0, FAULT)?;
    s.load(TMP, 0, CALLBACKS)?;
    s.binary(0xeb000000, 31, TMP, 31)?;
    s.require(1, STATUS_INVALID_INVOCATION, NO_FAULT, epilogue)?; // non-null callbacks
    s.load(TMP, 0, STACK)?;
    s.binary(0xeb000000, 31, TMP, 31)?;
    s.require(1, STATUS_INVALID_INVOCATION, NO_FAULT, epilogue)?;
    s.load(TMP, 0, STACK_LEN)?;
    s.imm(TMP2, stack_size as u64, true)?;
    s.binary(0xeb000000, 31, TMP, TMP2)?;
    s.require(2, STATUS_INVALID_INVOCATION, NO_FAULT, epilogue)?; // unsigned >=
    s.load(TMP, 0, STACK)?;
    s.binary(0xab000000, reg(10), TMP, TMP2)?; // ADDS: detect stack-end wrap
    s.require(3, STATUS_INVALID_INVOCATION, NO_FAULT, epilogue)?; // carry clear
    for bpf in 0..10 {
        s.mov(reg(bpf), 31, true)?;
    }
    s.load(reg(1), 0, CTX)
}

fn emit_epilogue(s: &mut Sink<'_>) -> Result<(), JitError> {
    for pair in 0..6u32 {
        let first = 19 + pair * 2;
        s.word(0xa9400000 | ((pair * 2) << 15) | ((first + 1) << 10) | (31 << 5) | first)?;
    }
    s.add_imm(SP, SP, FRAME_SIZE)?;
    s.word(0xd65f03c0)
}

fn emit_insn(
    s: &mut Sink<'_>,
    program: &[u8],
    pc: usize,
    offsets: &[u32],
    epilogue: usize,
) -> Result<(), JitError> {
    let i = ebpf::get_insn(program, pc);
    if i.opc == ebpf::LD_DW_IMM {
        let hi = ebpf::get_insn(program, pc + 1).imm as u32 as u64;
        return s.imm(reg(i.dst), (hi << 32) | i.imm as u32 as u64, true);
    }
    match i.opc & 7 {
        ebpf::BPF_ALU | ebpf::BPF_ALU64 => emit_alu(s, i, pc, epilogue),
        ebpf::BPF_JMP | ebpf::BPF_JMP32 => {
            if i.opc == ebpf::EXIT {
                s.load(TMP, SP, SAVED_INVOCATION)?;
                s.store(R0, TMP, RESULT)?;
                s.imm(0, u64::from(STATUS_OK), false)?;
                return s.branch(epilogue);
            }
            if i.opc == ebpf::CALL {
                return emit_callback(s, i, pc, epilogue);
            }
            let target = offsets[pc + 1 + i.off as usize] as usize;
            if i.opc == ebpf::JA {
                return s.branch(target);
            }
            let wide = i.opc & 7 == ebpf::BPF_JMP;
            let rhs = source(s, &i, wide)?;
            let op = i.opc & 0xf0;
            let flags = if op == ebpf::BPF_JSET {
                0x6a000000
            } else {
                0x6b000000
            };
            s.binary(
                flags | if wide { 0x80000000 } else { 0 },
                31,
                reg(i.dst),
                rhs,
            )?;
            let condition = match op {
                ebpf::BPF_JEQ => 0,
                ebpf::BPF_JGT => 8,
                ebpf::BPF_JGE => 2,
                ebpf::BPF_JSET | ebpf::BPF_JNE => 1,
                ebpf::BPF_JSGT => 12,
                ebpf::BPF_JSGE => 10,
                ebpf::BPF_JLT => 3,
                ebpf::BPF_JLE => 9,
                ebpf::BPF_JSLT => 11,
                ebpf::BPF_JSLE => 13,
                _ => return Err(error(ErrorKind::UnsupportedInstruction, Some(pc))),
            };
            s.word(0x54000040 | (condition ^ 1))?; // inverse condition skips B
            s.branch(target)
        }
        _ => emit_callback(s, i, pc, epilogue),
    }
}

fn source(s: &mut Sink<'_>, i: &Insn, wide: bool) -> Result<u8, JitError> {
    if i.opc & ebpf::BPF_X != 0 {
        Ok(reg(i.src))
    } else {
        s.imm(TMP, i.imm as i64 as u64, wide)?;
        Ok(TMP)
    }
}

fn emit_alu(s: &mut Sink<'_>, i: Insn, pc: usize, epilogue: usize) -> Result<(), JitError> {
    let dst = reg(i.dst);
    let op = i.opc & 0xf0;
    if op == ebpf::BPF_END {
        if i.opc & ebpf::BPF_X != 0 {
            let rev = match i.imm {
                16 => 0x5ac00400,
                32 => 0x5ac00800,
                _ => 0xdac00c00,
            };
            s.word(rev | (u32::from(dst) << 5) | u32::from(dst))?;
        }
        return match i.imm {
            16 => s.word(0x53003c00 | (u32::from(dst) << 5) | u32::from(dst)),
            32 => s.mov(dst, dst, false),
            _ => Ok(()),
        };
    }
    let wide = i.opc & 7 == ebpf::BPF_ALU64;
    let sf = if wide { 0x80000000 } else { 0 };
    if op == ebpf::BPF_NEG {
        return s.binary(sf | 0x4b000000, dst, 31, dst);
    }
    let rhs = source(s, &i, wide)?;
    if matches!(op, ebpf::BPF_DIV | ebpf::BPF_MOD) {
        s.binary(sf | 0x6b000000, 31, rhs, 31)?;
        s.require(1, STATUS_DIVISION_BY_ZERO, pc as u64, epilogue)?;
        // require's failure block only executes on a zero divisor. The valid
        // branch skips it, preserving TMP when it contains an immediate.
    }
    let code = match op {
        ebpf::BPF_ADD => 0x0b000000,
        ebpf::BPF_SUB => 0x4b000000,
        ebpf::BPF_MUL => 0x1b007c00,
        ebpf::BPF_DIV => 0x1ac00800,
        ebpf::BPF_OR => 0x2a000000,
        ebpf::BPF_AND => 0x0a000000,
        ebpf::BPF_LSH => 0x1ac02000,
        ebpf::BPF_RSH => 0x1ac02400,
        ebpf::BPF_XOR => 0x4a000000,
        ebpf::BPF_ARSH => 0x1ac02800,
        ebpf::BPF_MOV => return s.mov(dst, rhs, wide),
        ebpf::BPF_MOD => {
            s.binary(sf | 0x1ac00800, TMP2, dst, rhs)?;
            return s.word(
                sf | 0x1b008000
                    | (u32::from(rhs) << 16)
                    | (u32::from(dst) << 10)
                    | (u32::from(TMP2) << 5)
                    | u32::from(dst),
            );
        }
        _ => return Err(error(ErrorKind::UnsupportedInstruction, Some(pc))),
    };
    s.binary(sf | code, dst, dst, rhs)
}

fn emit_callback(s: &mut Sink<'_>, i: Insn, pc: usize, epilogue: usize) -> Result<(), JitError> {
    // ponytail: checked callbacks cost a call per memory access; inline only
    // after profiling and equivalent bounds/permission tests justify it.
    s.store(R0, SP, SAVED_R0)?;
    s.store(31, SP, CALLBACK_RESULT)?;
    let class = i.opc & 7;
    let is_helper = i.opc == ebpf::CALL;
    let callback_offset;
    if is_helper {
        for arg in 0..5 {
            s.store(reg(arg as u8 + 1), SP, HELPER_ARGS + arg * 8)?;
        }
        s.imm(1, i.imm as u32 as u64, false)?;
        s.add_imm(2, SP, HELPER_ARGS as u16)?;
        s.add_imm(3, SP, CALLBACK_RESULT as u16)?;
        callback_offset = 0;
    } else {
        let base = if class == ebpf::BPF_LDX { i.src } else { i.dst };
        s.mov(1, reg(base), true)?;
        s.imm(2, i.off as i64 as u64, true)?;
        let width = match i.opc & 0x18 {
            ebpf::BPF_B => 1,
            ebpf::BPF_H => 2,
            ebpf::BPF_W => 4,
            _ => 8,
        };
        s.imm(3, width, false)?;
        s.imm(4, u64::from(base == 10), false)?;
        if class == ebpf::BPF_LDX {
            s.add_imm(5, SP, CALLBACK_RESULT as u16)?;
            callback_offset = 8;
        } else {
            if class == ebpf::BPF_ST {
                s.imm(5, i.imm as i64 as u64, true)?;
            } else {
                s.mov(5, reg(i.src), true)?;
            }
            callback_offset = 16;
        }
    }
    s.load(TMP, SP, SAVED_INVOCATION)?;
    s.load(0, TMP, OPAQUE)?;
    s.load(TMP, TMP, CALLBACKS)?;
    s.load(CALL_TARGET, TMP, callback_offset)?;
    s.word(0xd63f0200)?; // BLR X16
    s.load(R0, SP, SAVED_R0)?;
    s.binary(0x6b000000, 31, 0, 31)?;
    s.require(0, STATUS_CALLBACK_ERROR, pc as u64, epilogue)?;
    if is_helper {
        s.load(R0, SP, CALLBACK_RESULT)?;
    } else if class == ebpf::BPF_LDX {
        s.load(reg(i.dst), SP, CALLBACK_RESULT)?;
    }
    Ok(())
}
