use std::{io::Write, slice::Iter};

mod structs;
use structs::*;

mod format;
use format::Formattable;

impl Operand {
    fn displacement(displacement: i16) -> Operand {
        Self::Address(None, None, Some(displacement))
    }
}

mod decoder {
    use std::slice::Iter;

    use super::structs::*;

    fn decode_reg(reg: u8, w: u8) -> &'static Register {
        #[rustfmt::skip]
        const W_TO_REG: &'static[Register] = &[
            Register::Al, Register::Cl, Register::Dl, Register::Bl, Register::Ah, Register::Ch , Register::Dh, Register::Bh, // w = 0
            Register::Ax, Register::Cx, Register::Dx, Register::Bx, Register::Sp, Register::Bp, Register::Si, Register::Di, // w = 1
        ];

        &W_TO_REG[usize::from(w) * 8 + usize::from(reg)]
    }

    fn decode_address(content_iter: &mut Iter<u8>, _mod: u8, rm: u8) -> Operand {
        let is_direct_address = _mod == 0b00 && rm == 0b110;
        let displacement_bytes = if is_direct_address { 2 } else { _mod };

        let displacement_number: Option<i16> = match displacement_bytes {
            2 => Some(
                (u16::from(*(content_iter.next().unwrap()))
                    | (u16::from(*(content_iter.next().unwrap())) << 8)) as i16,
            ),
            1 => Some(i16::from(*(content_iter.next().unwrap()) as i8)),
            _ => None,
        };

        if is_direct_address {
            let Some(displacement_number) = displacement_number else {
                panic!("required direct address");
            };
            return Operand::displacement(displacement_number);
        }

        static FIRST_ADDRESS: &'static [Register] = &[
            Register::Bx,
            Register::Bx,
            Register::Bp,
            Register::Bp,
            Register::Si,
            Register::Di,
            Register::Bp,
            Register::Bx,
        ];
        static SECOND_ADDRESS: &'static [Option<Register>] = &[
            Some(Register::Si),
            Some(Register::Di),
            Some(Register::Si),
            Some(Register::Di),
            None,
            None,
            None,
            None,
        ];
        // const FIRST_ADDRESS: &'static [Register] = &["bx+si", "bx+di", "bp+si", "bp+di", "si", "di", "bp", "bx"} // rm2registers
        let a_reg = &FIRST_ADDRESS[usize::from(rm)];
        let b_reg = &SECOND_ADDRESS[usize::from(rm)];

        Operand::Address(Some(*a_reg), *b_reg, displacement_number)
    }

    fn rm_to_reg(
        op_code: OpCode,
        content_iter: &mut Iter<u8>,
        first: &u8,
        second: &u8,
    ) -> Instruction {
        let w = first & 0b1;
        let d = (first >> 1) & 0b1;

        let reg = (second >> 3) & 0b111;
        let rm = second & 0b111;
        let _mod = second >> 6 & 0b11;

        let address_or_reg = match _mod {
            0b00..=0b10 => decode_address(content_iter, _mod, rm),
            0b11 => Operand::Register(*decode_reg(rm, w)),
            _ => {
                panic!("invalid mod processing");
            }
        };

        let reg_name = Operand::Register(*decode_reg(reg, w));

        Instruction {
            left_operand: if d == 0b1 { reg_name } else { address_or_reg },
            right_operand: if d == 0b1 { address_or_reg } else { reg_name },
            flags: if w == 0b1 {
                InstructionFlags::Wide
            } else {
                InstructionFlags::empty()
            },
            op_code,
        }
    }

    fn immediate_to_rm(
        op_code: OpCode,
        first: &u8,
        second: &u8,
        content_iter: &mut Iter<u8>,
        implicit_sign_bit: Option<bool>,
    ) -> Instruction {
        let sign_extend = if let Some(sign_bit) = implicit_sign_bit {
            sign_bit
        } else {
            (first >> 1 & 0b1) == 0b1
        };
        let wide = first & 0b1;

        let _mod = second >> 6 & 0b11;
        let rm = second & 0b111;

        let address = match _mod {
            0b00..=0b10 => decode_address(content_iter, _mod, rm),
            0b11 => Operand::Register(*decode_reg(rm, wide)),
            _ => {
                panic!("invalid mod processing");
            }
        };

        let immediate = {
            let mut init = 0;
            init |= u16::from(*content_iter.next().unwrap());

            if (wide == 0b1 && !sign_extend) {
                init |= u16::from(*content_iter.next().unwrap()) << 8;
            }

            Operand::Immediate(init as i16)
        };

        Instruction {
            op_code,
            flags: if wide == 0b1 {
                InstructionFlags::Wide
            } else {
                InstructionFlags::empty()
            },
            left_operand: address,
            right_operand: immediate,
        }
    }

    // fn format_as_mov(&self) -> String {
    //     let word_type = if self.w { "word" } else { "byte" };
    //     format!("{}, {} {}", self.address, word_type, self.immediate)
    // }
    // fn format_as_arithmetic(&self) -> String {
    //     if self.address.starts_with('[') {
    //         let word_type = if self.s { "word" } else { "byte" };
    //         format!("{} {}, {}", word_type, self.address, self.immediate)
    //     } else {
    //         format!("{}, {}", self.address, self.immediate)
    //     }
    // }

    fn mov_immediate_to_reg(content_iter: &mut Iter<u8>, first: &u8, second: &u8) -> Instruction {
        assert!(first >> 4 == 0b1011);

        let w = first >> 3 & 0b1;
        let reg = first & 0b111;

        let immediate: i16 = if w == 1 {
            (u16::from(*(second)) | (u16::from(*(content_iter.next().unwrap())) << 8)) as i16
        } else {
            i16::from(*(second) as i8)
        };

        let reg_name = decode_reg(reg, w);

        // format!("mov {}, {}", reg_name, data)
        Instruction {
            op_code: OpCode::Mov,
            left_operand: Operand::Register(*reg_name),
            right_operand: Operand::Immediate(immediate),
            flags: if w == 0b1 {
                InstructionFlags::Wide
            } else {
                InstructionFlags::empty()
            },
        }
    }

    fn mov_acc(content_iter: &mut Iter<u8>, first: &u8, second: &u8) -> Instruction {
        assert!(first >> 2 == 0b101000);
        let w = first & 0b1;
        let to_acc = (first >> 1 & 0b1) == 0;

        let displacement_number: i16 = if w == 1 {
            (u16::from(*(second)) | (u16::from(*(content_iter.next().unwrap())) << 8)) as i16
        } else {
            i16::from(*(second) as i8)
        };

        let flags = if w == 0b1 {
            InstructionFlags::Wide
        } else {
            InstructionFlags::empty()
        };
        if to_acc {
            // format!("mov ax, [{}]", displacement_number)
            Instruction {
                op_code: OpCode::Mov,
                left_operand: Operand::Register(Register::Ax),
                flags,
                right_operand: Operand::displacement(displacement_number),
            }
        } else {
            Instruction {
                op_code: OpCode::Mov,
                left_operand: Operand::displacement(displacement_number),
                flags,
                right_operand: Operand::Register(Register::Ax),
            }
            // format!("mov [{}], ax", displacement_number)
        }
    }

    fn im_to_acc(
        op_code: OpCode,
        content_iter: &mut Iter<u8>,
        first: &u8,
        second: &u8,
    ) -> Instruction {
        let w = first & 0b1;

        let immediate: i16 = if w == 1 {
            (u16::from(*(second)) | (u16::from(*(content_iter.next().unwrap())) << 8)) as i16
        } else {
            i16::from(*(second) as i8)
        };

        Instruction {
            left_operand: Operand::Register(if w == 0 { Register::Al } else { Register::Ax }),
            right_operand: Operand::Immediate(immediate),
            flags: if w == 1 {
                InstructionFlags::Wide
            } else {
                InstructionFlags::empty()
            },
            op_code,
        }
        // format!("al, {}", displacement_number)
    }
    // rm - stands for Register or Memory

    // impl ArithmeticInstruction {
    //     fn from(first: &u8, second: &u8) -> Instruction {
    //         let is_immediate_to_rm = first >> 2 == 0b10_0000;
    //         let middle_bits_to_operation = |byte: &u8| match (byte >> 3) & 0b111 {
    //             0b000 => Some(OpCode::Add),
    //             0b101 => Some(OpCode::Sub),
    //             0b111 => Some(OpCode::Cmp),
    //             _ => None,
    //         };
    //         let op_code = middle_bits_to_operation(if is_immediate_to_rm { second } else {first}) ;
    //         if is_immediate_to_rm {
    //             return middle_bits_to_operation(second).map(|it| ArithmeticInstruction {
    //                 t: it,
    //                 action: ArithmeticInstructionAction::ImToRm,
    //             });
    //         }

    //         // exluding middle byte
    //         match first & 0b11_000_111 {
    //             0b00_000_000..=0b00_000_011 => {
    //                 middle_bits_to_operation(first).map(|it| ArithmeticInstruction {
    //                     t: it,
    //                     action: ArithmeticInstructionAction::RmToEither,
    //                 })
    //             }
    //             0b00_000_100..=0b00_000_101 => {
    //                 middle_bits_to_operation(first).map(|it| ArithmeticInstruction {
    //                     t: it,
    //                     action: ArithmeticInstructionAction::ImToAcc,
    //                 })
    //             }
    //             _ => None,
    //         }
    //     }

    //     fn parse_and_format(self, first: &u8, second: &u8, content_iter: &mut Iter<u8>) -> String {
    //         let operation = match &self.t {
    //             ArithmeticInstructionType::Add => "add",
    //             ArithmeticInstructionType::Cmp => "cmp",
    //             ArithmeticInstructionType::Sub => "sub",
    //         };

    //         let content = match self.action {
    //             ArithmeticInstructionAction::ImToAcc => im_to_acc(operation, content_iter, first, second),
    //             ArithmeticInstructionAction::ImToRm => {
    //                 Im2Rm::new(content_iter, first, second, true).format_as_arithmetic()
    //             }
    //             ArithmeticInstructionAction::RmToEither => rm_to_reg(operation, content_iter, first, second),
    //         };

    //         format!("{} {}", operation, content)
    //     }
    // }

    fn parse_jump(op_code: OpCode, second: &u8) -> Instruction {
        let signed_second = *second as i8;

        Instruction {
            flags: InstructionFlags::empty(),
            op_code,
            left_operand: Operand::JumpDisplacement(signed_second),
            right_operand: Operand::Empty,
        }
    }

    pub fn decode(first: &u8, second: &u8, content_iter: &mut Iter<u8>) -> Option<Instruction> {
        let get_arithmetic_op_code = |byte: &u8| match (byte >> 3) & 0b111 {
            0b000 => OpCode::Add,
            0b101 => OpCode::Sub,
            0b111 => OpCode::Cmp,
            _ => panic!("unknown arhithmetic operand {}", byte >> 3 & 0b111),
        };
        match first {
            // movs
            0b1010_0000..=0b1010_0011 => Some(mov_acc(content_iter, first, second)),
            0b1011_0000..=0b1011_1111 => Some(mov_immediate_to_reg(content_iter, first, second)),
            0b1000_1000..=0b1000_1011 => Some(rm_to_reg(OpCode::Mov, content_iter, first, second)),
            0b1100_0110..=0b1100_0111 => Some(immediate_to_rm(
                OpCode::Mov,
                first,
                second,
                content_iter,
                Some(false),
            )),

            0b100000_00..=0b100000_11 => Some(immediate_to_rm(
                get_arithmetic_op_code(second),
                first,
                second,
                content_iter,
                None,
            )),

            0b00_000_0_00..=0b00_000_0_11
            | 0b00_101_0_00..=0b00_101_0_11
            | 0b00_111_0_00..=0b00_111_0_11 => Some(rm_to_reg(
                get_arithmetic_op_code(first),
                content_iter,
                first,
                second,
            )),
            0b00_000_10_0..=0b00_000_10_1
            | 0b00_101_10_0..=0b00_101_10_1
            | 0b00_111_10_0..=0b00_111_10_1 => Some(im_to_acc(
                get_arithmetic_op_code(first),
                content_iter,
                first,
                second,
            )),

            // jumps
            0b0111_0100 => Some(parse_jump(OpCode::Je, second)),
            0b0111_1100 => Some(parse_jump(OpCode::Jl, second)),
            0b0111_1110 => Some(parse_jump(OpCode::Jle, second)),
            0b0111_0010 => Some(parse_jump(OpCode::Jb, second)),
            0b0111_0110 => Some(parse_jump(OpCode::Jbe, second)),
            0b0111_1010 => Some(parse_jump(OpCode::Jp, second)),
            0b0111_0000 => Some(parse_jump(OpCode::Jo, second)),
            0b0111_1000 => Some(parse_jump(OpCode::Js, second)),
            0b0111_0101 => Some(parse_jump(OpCode::Jne, second)),
            0b0111_1101 => Some(parse_jump(OpCode::Jnl, second)),
            0b0111_1111 => Some(parse_jump(OpCode::Jnle, second)),
            0b0111_0011 => Some(parse_jump(OpCode::Jnb, second)),
            0b0111_0111 => Some(parse_jump(OpCode::Jnbe, second)),
            0b0111_1011 => Some(parse_jump(OpCode::Jnp, second)),
            0b0111_0001 => Some(parse_jump(OpCode::Jno, second)),
            0b0111_1001 => Some(parse_jump(OpCode::Jns, second)),
            0b1110_0010 => Some(parse_jump(OpCode::Loop, second)),
            0b1110_0001 => Some(parse_jump(OpCode::Loopz, second)),
            0b1110_0000 => Some(parse_jump(OpCode::Loopnz, second)),
            0b1110_0011 => Some(parse_jump(OpCode::Jcxz, second)),
            _ => None,
        }
    }
}

fn process_binary(mut content_iter: Iter<u8>) -> Vec<Instruction> {
    let mut result: Vec<Instruction> = vec![];
    loop {
        let Some(first) = content_iter.next() else {
            return result;
        };
        let second = content_iter.next().unwrap();
        let Some(instruction) = decoder::decode(first, second, &mut content_iter) else {
            panic!("failed to decode {} {}", first, second)
        };
        result.push(instruction);
    }
}

fn write_instructions<T: Write>(instructions: Vec<Instruction>, out: &mut T) {
    writeln!(out, "bits 16\n").unwrap();
    instructions.iter().for_each(|it| {
        out.write((it.format() + "\n").as_bytes())
            .expect("must write");
    });
}

fn read_file<T: Into<String>>(file_path: T) -> Vec<u8> {
    let Ok(content) = std::fs::read(file_path.into()) else {
        panic!("Failed to read file");
    };

    content
}

struct X86Memory {
    // ax - 0
    // bx - 1
    // cx - 2
    // dx - 3
    // sp - 4
    // bp - 5
    // si - 6
    // di - 7
    arr: [u16; 4096],
}

static start_of_memory: usize = 8;

fn to_index(reg: &Register) -> usize {
    match reg {
        Register::Ax => 0,
        Register::Ah => 0,
        Register::Al => 1,

        Register::Bx => 1,
        Register::Bh => 1 * 2,
        Register::Bl => (1 * 2) + 1,

        Register::Cx => 2,
        Register::Ch => 2 * 2,
        Register::Cl => (2 * 2) + 1,

        Register::Dx => 3,
        Register::Dh => 3 * 2,
        Register::Dl => (3 * 2) + 1,

        Register::Sp => 4,
        Register::Bp => 5,
        Register::Si => 6,
        Register::Di => 7,
    }
}
impl Register {
    fn to_wide(&self) -> Register {
        match self {
            Register::Ah => Register::Ax,
            Register::Al => Register::Ax,

            Register::Bh => Register::Bx,
            Register::Bl => Register::Bx,

            Register::Ch => Register::Cx,
            Register::Cl => Register::Cx,

            Register::Dh => Register::Dx,
            Register::Dl => Register::Dx,

            _ => self.to_owned(),
        }
    }
}

trait SomeTakable<T> {
    fn if_some_ref<F: FnOnce(&T)>(&self, mapper: F);
    fn if_some_ref_mut<F: FnOnce(&mut T)>(&mut self, mapper: F);
}
impl<T> SomeTakable<T> for Option<T> {
    fn if_some_ref<F: FnOnce(&T)>(&self, mapper: F) {
        match self {
            Some(it) => mapper(it),
            None => {}
        }
    }
    fn if_some_ref_mut<F: FnOnce(&mut T)>(&mut self, mapper: F) {
        match self {
            Some(it) => mapper(it),
            None => {}
        }
    }
}

impl X86Memory {
    fn as_mut_bytes(&mut self) -> &mut [u8] {
        let len_in_bytes = std::mem::size_of_val(&self.arr); // 4096 * 2 = 8192
        unsafe { std::slice::from_raw_parts_mut(self.arr.as_mut_ptr().cast::<u8>(), len_in_bytes) }
    }
    fn as_bytes(&self) -> &[u8] {
        let len_in_bytes = std::mem::size_of_val(&self.arr); // 4096 * 2 = 8192
        unsafe { std::slice::from_raw_parts(self.arr.as_ptr().cast::<u8>(), len_in_bytes) }
    }

    fn mov_register<T: Write>(&mut self, instruction: &Instruction, out_opt: &mut Option<&mut T>) {
        let Operand::Register(to_reg) = &instruction.left_operand else {
            panic!("unexpected instruction {}", instruction.format());
        };
        let operand = instruction.right_operand;
        let flags = instruction.flags;

        let is_wide = to_reg.is_wide();
        assert_eq!(flags.contains(InstructionFlags::Wide), is_wide);
        out_opt.if_some_ref_mut(|out| {
            let wide_reg = to_reg.to_wide();
            let prev_value = self.arr[to_index(&wide_reg)];

            out.write(
                format!(
                    "{} ; {} {:#06x} -> ",
                    instruction.format(),
                    wide_reg.format(),
                    prev_value
                )
                .as_bytes(),
            )
            .expect("is ok");
        });
        if is_wide {
            match &operand {
                Operand::Register(from_reg) => {
                    assert!(from_reg.is_wide());

                    self.arr[to_index(to_reg)] = self.arr[to_index(from_reg)];
                }
                Operand::Immediate(data) => {
                    self.arr[to_index(to_reg)] = *data as u16;
                }
                _ => panic!("unexpected operand {}", operand.format()),
            }
        } else {
            match &operand {
                Operand::Register(from_reg) => {
                    assert!(from_reg.is_wide());
                    self.as_mut_bytes()[to_index(to_reg)] = self.as_bytes()[to_index(from_reg)];
                }
                Operand::Immediate(data) => {
                    self.as_mut_bytes()[to_index(to_reg)] = *data as u8;
                }
                _ => panic!("unexpected operand {}", operand.format()),
            }
        }

        out_opt.if_some_ref_mut(|out| {
            let wide_reg = to_reg.to_wide();
            let next_value = self.arr[to_index(&wide_reg)];

            out.write(format!("{:#06x}\n", next_value).as_bytes())
                .expect("is ok");
        });
    }

    fn print_registers<T: Write>(&self, out: &mut T) {
        let registers = [
            Register::Ax,
            Register::Bx,
            Register::Cx,
            Register::Dx,
            Register::Sp,
            Register::Bp,
            Register::Si,
            Register::Di,
        ];

        registers.iter().for_each(|reg| {
            let value = self.arr[to_index(reg)];

            writeln!(out, "{} {:#06x} ({})", reg.format(), value, value).unwrap();
        });
    }
}

fn execute<T: Write>(instructions: Vec<Instruction>, mut out: Option<&mut T>) {
    let mut memory = X86Memory {
        arr: core::array::from_fn(|i| 0u16),
    };
    instructions.iter().for_each(|instr| match instr.op_code {
        OpCode::Mov => match instr.left_operand {
            Operand::Register(_) => {
                memory.mov_register(instr, &mut out);
            }
            _ => {
                panic!("unsupported operand for mov {}", instr.format());
            }
        },
        _ => panic!("unsupported OpCode {}", instr.op_code.format()),
    });
    (&mut out).if_some_ref_mut(|out| {
        writeln!(out, "\nFinal registers: ").unwrap();
        memory.print_registers(out);
    });
}

enum CliMode {
    Disasm,
    Exec,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = match (args.next()) {
        Some(str) if str == "disasm" => CliMode::Disasm,
        Some(str) if str == "exec" => CliMode::Exec,
        Some(str) => {
            std::println!("Unknown mode: {}", str);
            return;
        }
        _ => {
            std::println!("Please at least two args");
            return;
        }
    };
    let Some(path) = args.next() else {
        std::println!("Please provide assembly file path");
        return;
    };

    let content = read_file(path);
    let content_iter = content.iter();
    let instrucitons = process_binary(content_iter);
    let mut stdout = &mut std::io::stdout().lock();
    match mode {
        CliMode::Disasm => {
            write_instructions(instrucitons, &mut std::io::stdout().lock());
        }
        CliMode::Exec => {
            execute(instrucitons, Some(stdout));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::create_dir, fs::File, io::Write, path::Path, process::Command};

    use crate::{process_binary, read_file, write_instructions};

    fn compile_asm_if_not(path: &str, force_recompile: bool) {
        if force_recompile || !Path::new(path).exists() {
            let asm_path = format!("{}.asm", path);
            assert!(Path::new(&asm_path).exists());

            Command::new("nasm").args([asm_path]).output().unwrap();
            assert!(Path::new(&path).exists());
        }
    }

    #[test]
    fn test_parser() {
        let files = [
            "./input/listing_0037_single_register_mov",
            "./input/listing_0038_many_register_mov",
            "./input/listing_0039_more_movs",
            "./input/listing_0040_challenge_movs",
            "./input/listing_0041_add_sub_cmp_jnz",
            "./input/listing_0043_immediate_movs",
            "./input/listing_0044_register_movs",
        ];
        let force_recompilation = std::env::var("RECOMPILE").map_or(false, |it| it == "true");

        for file_path in files {
            compile_asm_if_not(file_path, force_recompilation);
            let file = read_file(file_path);
            let mut res: Vec<u8> = Vec::new();
            println!("checking '{}'", file_path);

            write_instructions(process_binary(file.iter()), &mut res);

            if !Path::new("./.test").exists() {
                create_dir("./.test").unwrap();
            }

            File::create("./.test/test.asm")
                .unwrap()
                .write_all(res.as_slice())
                .unwrap();

            Command::new("nasm")
                .args(["./.test/test.asm"])
                .output()
                .unwrap();

            let nasm_file = read_file("./.test/test".to_string());
            assert!(nasm_file == file);
        }
    }
}
