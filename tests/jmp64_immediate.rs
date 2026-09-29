// SPDX-License-Identifier: (Apache-2.0 OR MIT)

use rbpf::ebpf::{
    Insn, EXIT, JEQ_IMM, JEQ_IMM32, JGE_IMM, JGT_IMM, JLE_IMM, JLT_IMM, JNE_IMM, LD_DW_IMM,
    MOV64_IMM,
};

fn jump_taken(opc: u8, lhs: u64, imm: i32) -> bool {
    #[rustfmt::skip]
    let insns = [
        Insn { opc: LD_DW_IMM, dst: 1, src: 0, off: 0, imm: lhs as i32 },
        Insn { opc: 0,         dst: 0, src: 0, off: 0, imm: (lhs >> 32) as i32 },
        Insn { opc,            dst: 1, src: 0, off: 2, imm },
        Insn { opc: MOV64_IMM, dst: 0, src: 0, off: 0, imm: 0 },
        Insn { opc: EXIT,      dst: 0, src: 0, off: 0, imm: 0 },
        Insn { opc: MOV64_IMM, dst: 0, src: 0, off: 0, imm: 1 },
        Insn { opc: EXIT,      dst: 0, src: 0, off: 0, imm: 0 },
    ];
    let prog = insns.iter().flat_map(Insn::to_array).collect::<Vec<_>>();
    let vm = rbpf::EbpfVmNoData::new(Some(&prog)).unwrap();
    vm.execute_program().unwrap() == 1
}

#[test]
fn unsigned_jmp64_immediates_sign_extend_before_comparison() {
    let minus_one = u64::MAX;
    let min_i32 = i32::MIN as i64 as u64;
    let zero_extended_min_i32 = i32::MIN as u32 as u64;

    for (opc, lhs, imm, expected) in [
        (JEQ_IMM, minus_one, -1, true),
        (JNE_IMM, minus_one, -1, false),
        (JGT_IMM, 1_u64 << 32, i32::MIN, false),
        (JGE_IMM, min_i32, i32::MIN, true),
        (JLT_IMM, zero_extended_min_i32, i32::MIN, true),
        (JLE_IMM, min_i32, i32::MIN, true),
        (JEQ_IMM, zero_extended_min_i32, i32::MIN, false),
        (JNE_IMM, zero_extended_min_i32, i32::MIN, true),
        (JGT_IMM, 8, 7, true),
        (JLE_IMM, 8, 7, false),
        // JMP32 compares the low 32 bits and retains its existing behavior.
        (JEQ_IMM32, min_i32, i32::MIN, true),
        (JEQ_IMM32, zero_extended_min_i32, i32::MIN, true),
    ] {
        assert_eq!(
            jump_taken(opc, lhs, imm),
            expected,
            "opcode {opc:#x}, lhs {lhs:#x}, imm {imm}"
        );
    }
}
