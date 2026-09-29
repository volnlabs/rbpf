// SPDX-License-Identifier: (Apache-2.0 OR MIT)

use rbpf::ebpf::{self, Insn};

fn insn(opc: u8, off: i16, imm: i32) -> Insn {
    Insn {
        opc,
        dst: 0,
        src: 0,
        off,
        imm,
    }
}

fn execute(insns: &[Insn]) -> u64 {
    let program = insns.iter().flat_map(Insn::to_array).collect::<Vec<_>>();
    let vm = rbpf::EbpfVmNoData::new(Some(&program)).unwrap();
    vm.execute_program().unwrap()
}

#[test]
fn forward_jumps_keep_the_full_program_counter() {
    for pc in [32766, 32767, 32768, 65536] {
        for opc in [ebpf::JA, ebpf::JEQ_IMM, ebpf::JEQ_IMM32] {
            let mut program = vec![insn(ebpf::MOV64_IMM, 0, 42); pc];
            program.extend([
                insn(opc, 1, if opc == ebpf::JA { 0 } else { 42 }),
                insn(ebpf::MOV64_IMM, 0, 99),
                insn(ebpf::EXIT, 0, 0),
            ]);
            assert_eq!(execute(&program), 42, "pc {pc}, opcode {opc:#x}");
        }
    }
}

#[test]
fn backward_jumps_keep_the_full_program_counter() {
    for pc in [32766, 32768, 65536] {
        for opc in [ebpf::JLT_IMM, ebpf::JLT_IMM32] {
            let mut program = vec![insn(ebpf::MOV64_IMM, 0, 0); pc - 1];
            program.extend([
                insn(ebpf::ADD64_IMM, 0, 1),
                // Execute the addition twice, then fall through to EXIT.
                insn(opc, -2, 2),
                insn(ebpf::EXIT, 0, 0),
            ]);
            assert_eq!(execute(&program), 2, "pc {pc}, opcode {opc:#x}");
        }
    }
}
