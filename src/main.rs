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

mod simulator {
    use crate::{
        format::Formattable, main, Instruction, InstructionFlags, IsWide, OpCode, Operand, Register
    };
    use bitflags::bitflags;
    use std::{io::Write, slice::Iter};

    struct X86Memory {
        // most of a X86 and ARM processors are little endian, so I will use
        // little endian as encoding of word size registers
        // ax - (0, 2)
        // bx - (2, 4)
        // cx - (4, 6)
        // dx - (6, 8)
        // sp - (8 - 10)
        // bp - (10 - 12)
        // si - (12 - 14)
        // di - (14 - 16)
        arr: [u8; 8192],

        flags: Flags,
    }
    static start_of_memory: usize = 8;

    bitflags! {
        #[derive(PartialEq, Eq, Clone, Copy)]
        pub struct Flags: u16 {
            // if unsigned overflow happend
            const Carry = 0b0000_0000_0000_0000_0001;
            // checks whenever there is an even amount of 1 bits in the result (checks only 8 low bits)
            const Parity = 0b0000_0000_0000_0000_0100;
            // check for unsigned overflow in low nibble (4 bits)
            const AuxiliaryCarry = 0b0000_0000_0001_0000;
            // if result is zero -> sets to 1
            const Zero = 0b0000_0000_0100_0000;
            // shows sign of the value - in reality it's just a most significant bit of the result
            const Sign = 0b0000_0000_1000_0000;
            const Trace = 0b0000_0001_0000_0000;
            // when this flag is 1 - CPU react to interupts from external devices
            const Interupt = 0b0000_0010_0000_0000;
            // used by some instructions. 0 - processing is done forward, 1 - backward
            const Direction = 0b0000_0100_0000_0000;
            // setted to 1 when signed overflow happened 100 + 50 (out of range).
            // actually it's an derivative of most significant bit
            const Overflow = 0b0000_1000_0000_0000;
        }
    }

    enum ArithmeticOps {
        Add,
        Sub,
        Cmp
    }

    impl Operand {
        fn as_register(self) -> Option<Register> {
            match self {
                Operand::Register(reg) => Some(reg),
                _ => None
            }
        }
    }

    impl X86Memory {
        fn get_word_value(&self, reg: &Register) -> u16 {
            #[cfg(debug_assertions)]
            assert!(reg.reg_type() == RegType::Word);

            if cfg!(not(prefer_native_ops = "false")) && cfg!(target_endian = "little") {
                self.arr.as_u16_ref()[reg.to_word_index()]
            } else {
                (self.arr[reg.to_byte_index()] as u16)
                    | (self.arr[reg.to_byte_index() + 1] as u16) << 8
            }
        }
        fn get_value(&self, reg: &Register) -> u16 {
            if (reg.is_wide()) {
                self.get_word_value(reg)
            } else {
                self.arr[reg.to_byte_index()] as u16
            }
        }
        fn write_word_value(&mut self, reg: &Register, value: u16) {
            #[cfg(debug_assertions)]
            assert!(reg.reg_type() == RegType::Word);

            if cfg!(not(prefer_native_ops = "false")) && cfg!(target_endian = "little") {
                self.arr.as_u16_ref_mut()[reg.to_word_index()] = value;
            } else {
                self.arr[reg.to_byte_index()] = (value & 0x00FF) as u8;
                self.arr[reg.to_byte_index() + 1] = ((value & 0xFF00) >> 8) as u8;
            }
        }

        fn arithmetic_register<T : Write>(
            &mut self,
            instruction: &Instruction,
            out_opt: &mut Option<&mut T>,
        ) {
            let op = match instruction.op_code {
                OpCode::Add => ArithmeticOps::Add,
                OpCode::Sub => ArithmeticOps::Sub,
                OpCode::Cmp => ArithmeticOps::Cmp,
                _ => panic!("wrong opcode {:?}", instruction.op_code),
            };
            let to_reg = instruction.left_operand.as_register()
                .expect("it's arithmetic op for rigister operands");
            let from_reg = instruction.right_operand.as_register()
                .expect("it's arithmetic op for rigister operands");

            let is_wide = to_reg.is_wide();
            assert_eq!(to_reg.is_wide(), from_reg.is_wide());
            assert_eq!(instruction.flags.contains(InstructionFlags::Wide), to_reg.is_wide());

            // what will happen if overflow happen during 8 bit operation?
            let to_reg_value = self.get_value(&to_reg);
            let from_reg_value = self.get_value(&from_reg);

            // [TODO]: add tests
            let flags = {
                let full_result = (to_reg_value as u32) + (from_reg_value as u32);

                let carry_flag = match is_wide {
                    true if (full_result > u16::MAX.into()) => Flags::Carry,
                    false if (full_result > u8::MAX.into()) => Flags::Carry,
                    _ => Flags::empty(),
                };
                let zero_flag = match is_wide {
                    true if (full_result as u16) == 0 => Flags::Zero,
                    false if (full_result as u8) == 0 => Flags::Zero,
                    _ => Flags::empty(),
                };

                let sign_flag = match is_wide {
                    true if (full_result & 0b1000_0000_0000_0000) != 0 => Flags::Sign,
                    false if (full_result & 0b0000_0000_1000_0000) != 0 => Flags::Sign,
                    _ => Flags::empty(),
                };

                let partiy_flag = {
                    let result = full_result as u8;
                    let mut ones_amount = 0;

                    // unrolled loop
                    if (result & (1 << 0)) != 0 { ones_amount += 1; }
                    if (result & (1 << 1)) != 0 { ones_amount += 1; }
                    if (result & (1 << 2)) != 0 { ones_amount += 1; }
                    if (result & (1 << 3)) != 0 { ones_amount += 1; }
                    if (result & (1 << 4)) != 0 { ones_amount += 1; }
                    if (result & (1 << 5)) != 0 { ones_amount += 1; }
                    if (result & (1 << 6)) != 0 { ones_amount += 1; }
                    if (result & (1 << 7)) != 0 { ones_amount += 1; }

                    if ones_amount % 2 == 0 {
                        Flags::Parity
                    } else {
                        Flags::empty()
                    }
                };

                let auxilary_carry_flag = if (to_reg_value & 0xF) + (from_reg_value & 0xF) > 0xF {
                    Flags::AuxiliaryCarry
                } else {
                    Flags::empty()
                };

                let overflow_flag = match is_wide {
                    true if (to_reg_value & (1 << 15)) != ((full_result as u16) & (1 << 15)) => Flags::Overflow,
                    false if (to_reg_value & (1 << 7)) != ((full_result as u16) & (1 << 7)) => Flags::Overflow,
                    _ => Flags::empty()
                };

                carry_flag | zero_flag | sign_flag | partiy_flag | auxilary_carry_flag | overflow_flag;
            };
            // let next_value =

        }

        fn mov_register<T: Write>(
            &mut self,
            instruction: &Instruction,
            out_opt: &mut Option<&mut T>,
        ) {
            let Operand::Register(to_reg) = &instruction.left_operand else {
                panic!("unexpected instruction {}", instruction.format());
            };
            let operand = instruction.right_operand;
            let flags = instruction.flags;

            let is_wide = to_reg.is_wide();
            assert_eq!(flags.contains(InstructionFlags::Wide), is_wide);
            out_opt.if_some_ref_mut(|out| {
                let word_reg = to_reg.to_word();
                let prev_value = self.get_word_value(&word_reg);

                out.write(
                    format!(
                        "{} ; {} {:#06x} -> ",
                        instruction.format(),
                        word_reg.format(),
                        prev_value
                    )
                    .as_bytes(),
                )
                .expect("is ok");
            });

            if is_wide {
                match &operand {
                    Operand::Register(from_reg) => {
                        self.write_word_value(to_reg, self.get_word_value(from_reg));
                    }
                    Operand::Immediate(data) => {
                        self.write_word_value(to_reg, *data as u16);
                    }
                    _ => panic!("unexpected operand {}", operand.format()),
                }
            } else {
                match &operand {
                    Operand::Register(from_reg) => {
                        #[cfg(debug_assertions)]
                        assert!(!from_reg.is_wide());

                        self.arr[to_reg.to_byte_index()] = self.arr[from_reg.to_byte_index()];
                    }
                    Operand::Immediate(data) => {
                        #[cfg(debug_assertions)]
                        assert!(*data <= 0x00FFi16);

                        self.arr[to_reg.to_byte_index()] = (*data) as u8;
                    }
                    _ => panic!("unexpected operand {}", operand.format()),
                }
            }

            out_opt.if_some_ref_mut(|out| {
                let word_reg = to_reg.to_word();
                let next_value = self.get_word_value(&word_reg);

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
                let value = self.get_word_value(reg);

                writeln!(out, "{} {:#06x} ({})", reg.format(), value, value).unwrap();
            });
        }
    }

    pub fn execute<T: Write>(instructions: Vec<Instruction>, mut out: Option<&mut T>) {
        let mut memory = X86Memory {
            arr: core::array::from_fn(|_| 0u8),
            flags: Flags::empty(),
        };
        instructions
            .iter()
            .for_each(|instr| match (&instr.op_code, instr.left_operand) {
                (OpCode::Mov, Operand::Register(_)) => {
                    memory.mov_register(instr, &mut out);
                }
                (OpCode::Cmp, Operand::Register(_)) => {}
                (OpCode::Add, Operand::Register(_)) => {}
                (OpCode::Sub, Operand::Register(_)) => {}
                _ => panic!("unsupported OpCode {}", instr.op_code.format()),
            });
        (&mut out).if_some_ref_mut(|out| {
            writeln!(out, "\nFinal registers: ").unwrap();
            memory.print_registers(out);
        });
    }

    #[derive(PartialEq, Eq)]
    enum RegType {
        Low,
        High,
        Word,
    }
    impl Register {
        fn to_word(&self) -> Register {
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
        fn reg_type(&self) -> RegType {
            match self {
                Register::Ah => RegType::High,
                Register::Bh => RegType::High,
                Register::Ch => RegType::High,
                Register::Dh => RegType::High,

                Register::Al => RegType::Low,
                Register::Bl => RegType::Low,
                Register::Cl => RegType::Low,
                Register::Dl => RegType::Low,

                _ => RegType::Word,
            }
        }
        fn bit_mask(&self) -> u16 {
            match self {
                Register::Ah => 0xFF00,
                Register::Bh => 0xFF00,
                Register::Ch => 0xFF00,
                Register::Dh => 0xFF00,

                Register::Al => 0x00FF,
                Register::Bl => 0x00FF,
                Register::Cl => 0x00FF,
                Register::Dl => 0x00FF,

                _ => 0xFFFF,
            }
        }
        fn bit_shift(&self) -> u16 {
            match self {
                Register::Ah => 8,
                Register::Bh => 8,
                Register::Ch => 8,
                Register::Dh => 8,

                Register::Al => 0,
                Register::Bl => 0,
                Register::Cl => 0,
                Register::Dl => 0,

                _ => 0,
            }
        }

        fn to_word_index(&self) -> usize {
            match self {
                Register::Ax => 0,
                Register::Ah => 0,
                Register::Al => 0,

                Register::Bx => 1,
                Register::Bh => 1,
                Register::Bl => 1,

                Register::Cx => 2,
                Register::Ch => 2,
                Register::Cl => 2,

                Register::Dx => 3,
                Register::Dh => 3,
                Register::Dl => 3,

                Register::Sp => 4,
                Register::Bp => 5,
                Register::Si => 6,
                Register::Di => 7,
            }
        }

        // little endian encoding
        fn to_byte_index(&self) -> usize {
            match self {
                Register::Ax => 0,
                Register::Ah => 1,
                Register::Al => 0,

                Register::Bx => 2,
                Register::Bh => 3,
                Register::Bl => 2,

                Register::Cx => 4,
                Register::Ch => 5,
                Register::Cl => 4,

                Register::Dx => 6,
                Register::Dh => 7,
                Register::Dl => 6,

                Register::Sp => 8,
                Register::Bp => 10,
                Register::Si => 12,
                Register::Di => 14,
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

    trait AsU16 {
        fn as_u16_ref(&self) -> &[u16];
        fn as_u16_ref_mut(&mut self) -> &mut [u16];
    }

    impl AsU16 for [u8] {
        fn as_u16_ref(&self) -> &[u16] {
            let size = size_of_val(self);

            unsafe { std::slice::from_raw_parts(self.as_ptr().cast::<u16>(), size) }
        }

        fn as_u16_ref_mut(&mut self) -> &mut [u16] {
            let size = size_of_val(self);

            unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr().cast::<u16>(), size) }
        }
    }
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
    #[cfg(target_endian = "big")]
    println!("This is a BigEndian system");

    #[cfg(target_endian = "little")]
    println!("This is a Little endian system");

    let mut stdout = &mut std::io::stdout().lock();
    match mode {
        CliMode::Disasm => {
            write_instructions(instrucitons, &mut std::io::stdout().lock());
        }
        CliMode::Exec => {
            use crate::simulator::execute;
            execute(instrucitons, Some(stdout));
        }
    }
}

#[cfg(test)]
mod tests {
    use core::str;
    use std::{fs::create_dir, fs::File, io::Write, path::Path, process::Command};

    use crate::simulator::execute;
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
            "./input/listing_0044_register_movs_extended",
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

    #[test]
    fn test_execution() {
        let files = [
            "./input/listing_0043_immediate_movs",
            "./input/listing_0044_register_movs",
            "./input/listing_0044_register_movs_extended",
        ];
        let force_recompilation = std::env::var("RECOMPILE").map_or(false, |it| it == "true");

        for file_path in files {
            compile_asm_if_not(file_path, force_recompilation);
            let file = read_file(file_path);
            let mut res: Vec<u8> = Vec::new();
            println!("evaluating '{}'", file_path);

            execute(process_binary(file.iter()), Some(&mut res));
            insta::assert_snapshot!(str::from_utf8(res.as_slice()).unwrap());
        }
    }
}
