use std::io::Write;

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
    use std::u8;

    use super::structs::*;

    fn decode_reg(reg: u8, w: u8) -> &'static Register {
        #[rustfmt::skip]
        const W_TO_REG: &'static[Register] = &[
            Register::Al, Register::Cl, Register::Dl, Register::Bl, Register::Ah, Register::Ch , Register::Dh, Register::Bh, // w = 0
            Register::Ax, Register::Cx, Register::Dx, Register::Bx, Register::Sp, Register::Bp, Register::Si, Register::Di, // w = 1
        ];

        &W_TO_REG[usize::from(w) * 8 + usize::from(reg)]
    }

    fn decode_address<T: Iterator<Item = u8>>(content_iter: &mut T, _mod: u8, rm: u8) -> Operand {
        let is_direct_address = _mod == 0b00 && rm == 0b110;
        let displacement_bytes = if is_direct_address { 2 } else { _mod };

        let displacement_number: Option<i16> = match displacement_bytes {
            2 => Some(
                (u16::from((content_iter.next().unwrap()))
                    | (u16::from((content_iter.next().unwrap())) << 8)) as i16,
            ),
            1 => Some(i16::from((content_iter.next().unwrap()) as i8)),
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

    fn rm_to_reg<T: Iterator<Item = u8>>(
        op_code: OpCode,
        content_iter: &mut T,
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

    fn immediate_to_rm<T: Iterator<Item = u8>>(
        op_code: OpCode,
        first: &u8,
        second: &u8,
        content_iter: &mut T,
        sign_extend: bool,
    ) -> Instruction {
        // let sign_extend = explicit_sign_extend.unwrap_or(false);
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
            init |= u16::from((content_iter.next().unwrap()));

            if wide == 0b1 && sign_extend && (init >> 7) & 0b1 == 1 {
                init |= (u8::MAX as u16) << 8;
            } else if wide == 0b1 && !sign_extend {
                init |= u16::from((content_iter.next().unwrap())) << 8;
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

    fn mov_immediate_to_reg<T: Iterator<Item = u8>>(
        content_iter: &mut T,
        first: &u8,
        second: &u8,
    ) -> Instruction {
        assert!(first >> 4 == 0b1011);

        let w = first >> 3 & 0b1;
        let reg = first & 0b111;

        let immediate: i16 = if w == 1 {
            (u16::from(*(second)) | (u16::from((content_iter.next().unwrap())) << 8)) as i16
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

    fn mov_acc<T: Iterator<Item = u8>>(
        content_iter: &mut T,
        first: &u8,
        second: &u8,
    ) -> Instruction {
        assert!(first >> 2 == 0b101000);
        let w = first & 0b1;
        let to_acc = (first >> 1 & 0b1) == 0;

        let displacement_number: i16 = if w == 1 {
            (u16::from(*(second)) | (u16::from((content_iter.next().unwrap())) << 8)) as i16
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

    fn im_to_acc<T: Iterator<Item = u8>>(
        op_code: OpCode,
        content_iter: &mut T,
        first: &u8,
        second: &u8,
    ) -> Instruction {
        let w = first & 0b1;

        let immediate: i16 = if w == 1 {
            (u16::from(*(second)) | (u16::from((content_iter.next().unwrap())) << 8)) as i16
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

    pub fn decode<T: Iterator<Item = u8>>(
        first: &u8,
        second: &u8,
        content_iter: &mut T,
    ) -> Option<Instruction> {
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
                false,
            )),

            0b100000_00..=0b100000_11 => Some(immediate_to_rm(
                get_arithmetic_op_code(second),
                first,
                second,
                content_iter,
                (first >> 1) & 0b1 == 0b1,
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

    pub fn decode_instruction<T: Iterator<Item = u8>>(content_iter: &mut T) -> Option<Instruction> {
        let Some(first) = content_iter.next() else {
            return None;
        };
        let second = content_iter.next().unwrap();
        let Some(instruction) = decode(&first, &second, content_iter) else {
            panic!("failed to decode {} {}", first, second)
        };
        Some(instruction)
    }
}

fn process_binary<T: Iterator<Item = u8>>(mut content_iter: T) -> Vec<Instruction> {
    let mut result: Vec<Instruction> = Vec::with_capacity(2048);
    loop {
        let Some(instr) = decoder::decode_instruction(&mut content_iter) else {
            return result;
        };
        result.push(instr);
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
        decoder, format::Formattable, Instruction, InstructionFlags, IsWide, OpCode, Operand,
        Register,
    };
    use bitflags::bitflags;
    use std::{io::Write, ops::Not};

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
        arr: [u8; 65552],

        instructions: Vec<u8>,
        ip: u16,

        flags: Flags,
    }

    static start_of_memory: usize = 8;

    bitflags! {
        #[derive(PartialEq, Eq, Clone, Copy, Debug)]
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

    impl Formattable for Flags {
        fn format(&self) -> String {
            let mut res = String::with_capacity(9);
            if self.contains(Flags::Carry) {
                res.push_str("C");
            }
            if self.contains(Flags::Parity) {
                res.push_str("P");
            }
            if self.contains(Flags::AuxiliaryCarry) {
                res.push_str("A");
            }
            if self.contains(Flags::Zero) {
                res.push_str("Z");
            }
            if self.contains(Flags::Sign) {
                res.push_str("S");
            }
            if self.contains(Flags::Trace) {
                res.push_str("T");
            }
            if self.contains(Flags::Interupt) {
                res.push_str("I");
            }
            if self.contains(Flags::Direction) {
                res.push_str("D");
            }

            if self.contains(Flags::Overflow) {
                res.push_str("O");
            }
            if self.is_empty() {
                res.push_str("None");
            }

            res
        }
    }

    enum ArithmeticAction {
        Add,
        Sub,
    }
    #[derive(PartialEq, Eq)]
    enum ArithmeticOp {
        Add,
        Sub,
        Cmp,
    }
    impl From<&ArithmeticOp> for ArithmeticAction {
        fn from(from: &ArithmeticOp) -> ArithmeticAction {
            match from {
                ArithmeticOp::Add => ArithmeticAction::Add,
                ArithmeticOp::Cmp | ArithmeticOp::Sub => ArithmeticAction::Sub,
            }
        }
    }

    impl Operand {
        fn as_register(self) -> Option<Register> {
            match self {
                Operand::Register(reg) => Some(reg),
                _ => None,
            }
        }
        fn as_displacement(self) -> Option<i8> {
            match self {
                Operand::JumpDisplacement(displ) => Some(displ),
                _ => None,
            }
        }
    }

    #[derive(PartialEq, Eq)]
    enum Sign {
        Positive,
        Negative,
    }
    fn sign(value: u16, is_wide: bool) -> Sign {
        match is_wide {
            true if (value >> 15 & 0b1) == 1 => Sign::Negative,
            true => Sign::Positive,

            false if (value >> 7 & 0b1) == 1 => Sign::Negative,
            false => Sign::Positive,
        }
    }

    fn produce_math_op_flags(
        before: u16,
        right_sign: Sign,
        after: u16,
        arithm_op: ArithmeticAction,
        is_wide: bool,
    ) -> Flags {
        let carry_flag = match arithm_op {
            // 1xxx -> 0xxx while add
            // 1111
            // 1111
            //11110
            ArithmeticAction::Add if before > after => Flags::Carry,
            // borrow 0xxx -> 1xxx
            ArithmeticAction::Sub if before < after => Flags::Carry,
            _ => Flags::empty(),
        };
        // 1111 + 0001 = 0001_0000
        // 0001_0000 - 0001 = 1111
        let auxilary_carry_flag = match arithm_op {
            // ArithmeticOps::Add if (before & 0xF) + (after & 0xF) > 0xF => Flags::AuxiliaryCarry,
            ArithmeticAction::Add if (before & 0xF) > (after & 0xF) => Flags::AuxiliaryCarry,
            ArithmeticAction::Add => Flags::empty(),
            ArithmeticAction::Sub if (before & 0xF) < (after & 0xF) => Flags::AuxiliaryCarry,
            ArithmeticAction::Sub => Flags::empty(),
        };

        let zero_flag = if after == 0 {
            Flags::Zero
        } else {
            Flags::empty()
        };

        let sign_before = sign(before, is_wide);
        let sign_flag = match sign(after, is_wide) {
            Sign::Positive => Flags::empty(),
            Sign::Negative => Flags::Sign,
        };

        let partiy_flag = {
            let mut ones_amount = 0;

            // unrolled loop
            if (after & (1 << 0)) != 0 {
                ones_amount += 1;
            }
            if (after & (1 << 1)) != 0 {
                ones_amount += 1;
            }
            if (after & (1 << 2)) != 0 {
                ones_amount += 1;
            }
            if (after & (1 << 3)) != 0 {
                ones_amount += 1;
            }
            if (after & (1 << 4)) != 0 {
                ones_amount += 1;
            }
            if (after & (1 << 5)) != 0 {
                ones_amount += 1;
            }
            if (after & (1 << 6)) != 0 {
                ones_amount += 1;
            }
            if (after & (1 << 7)) != 0 {
                ones_amount += 1;
            }

            if ones_amount % 2 == 0 {
                Flags::Parity
            } else {
                Flags::empty()
            }
        };

        let overflow_flag = match (
            arithm_op,
            sign_before,
            right_sign,
            if sign_flag == Flags::Sign {
                Sign::Negative
            } else {
                Sign::Positive
            },
        ) {
            // negative sum of possitive operands
            (ArithmeticAction::Add, Sign::Positive, Sign::Positive, Sign::Negative) => {
                Flags::Overflow
            }
            (ArithmeticAction::Add, Sign::Negative, Sign::Negative, Sign::Positive) => {
                Flags::Overflow
            }
            // positive sum of negative operands
            (ArithmeticAction::Sub, Sign::Negative, Sign::Positive, Sign::Positive) => {
                Flags::Overflow
            }
            (ArithmeticAction::Sub, Sign::Positive, Sign::Negative, Sign::Negative) => {
                Flags::Overflow
            }
            _ => Flags::empty(),
        };

        carry_flag | zero_flag | sign_flag | partiy_flag | auxilary_carry_flag | overflow_flag
    }

    fn execute_add(left: u16, right: u16, is_wide: bool) -> u16 {
        let res = (left as u32) + (right as u32);
        let restructed_op = if is_wide {
            res as u16
        } else {
            (res as u8) as u16
        };

        restructed_op
    }

    #[test]
    fn test_execute_add() {
        assert!(execute_add(0x29, 0x4c, true) == 117);
        assert!(execute_add(0x29, 0x4c, false) == 117);
        assert!(execute_add(0xA9, 0x7c, true) == 293);
        assert!(execute_add(0xA0_09, 0xA0_09, true) == 16402);
        assert!(execute_add(0xA9, 0x7c, false) == 37);
    }

    #[test]
    fn test_produce_flags() {
        {
            let before = 0x29;

            let flags = execute_arithmetic_op(before, 0x4c, &ArithmeticOp::Add, true).1;
            assert!(flags == Flags::AuxiliaryCarry);
        };
        {
            let flags = execute_arithmetic_op(0b1000, 0b001, &ArithmeticOp::Add, true).1;
            assert!(flags == Flags::Parity);
        };

        {
            let flags = execute_arithmetic_op(1, 1, &ArithmeticOp::Sub, true).1;
            assert!(flags == Flags::Parity | Flags::Zero);
        };

        {
            let before = 0x80;
            let flags = execute_arithmetic_op(before, 0xF0, &ArithmeticOp::Add, false).1;
            assert_eq!(flags, Flags::Carry | Flags::Overflow);
        };
    }

    fn execute_sub(left: u16, right: u16, is_wide: bool) -> u16 {
        let res = if right > left {
            // it's fine to have ones in upper byte for non wide operation
            // since we trim them anyway
            ((left as u32) | (0b1 << 16)) - (right as u32)
        } else {
            (left as u32) - (right as u32)
        };
        let restructed_op = if is_wide {
            res as u16
        } else {
            (res as u8) as u16
        };

        restructed_op
    }

    #[test]
    fn test_execute_sub() {
        // borrowing
        assert_eq!(execute_sub(0x29, 0x4c, true), 65501);
        assert!(execute_sub(0x29, 0x4c, false) == 221);

        assert!(execute_sub(0x29, 0x20, true) == 0x09);
        assert!(execute_sub(0x29, 0x20, false) == 0x09);
    }

    fn execute_arithmetic_op(
        left: u16,
        right: u16,
        arithm_op: &ArithmeticOp,
        is_wide: bool,
    ) -> (u16, Flags) {
        let op_result = match arithm_op {
            ArithmeticOp::Add => execute_add(left, right, is_wide),
            ArithmeticOp::Sub | ArithmeticOp::Cmp => execute_sub(left, right, is_wide),
        };
        let flags = produce_math_op_flags(
            left,
            sign(right, is_wide),
            op_result,
            arithm_op.into(),
            is_wide,
        );
        let reg_value = match arithm_op {
            ArithmeticOp::Cmp => left,
            _ => op_result,
        };

        (reg_value, flags)
    }

    struct CountingIter<'a> {
        data: &'a Vec<u8>,
        start_idx: u16,
        movs: u16,
    }

    impl<'a> Iterator for CountingIter<'a> {
        type Item = u8;
        fn next(&mut self) -> Option<Self::Item> {
            let next_idx = (self.start_idx + self.movs) as usize;
            if next_idx >= self.data.len() {
                None
            } else {
                let res = &self.data[next_idx];
                self.movs += 1;
                Some(*res)
            }
        }
    }

    fn print_flags(prev_flags: &Flags, next_flags: &Flags) -> String {
        format!("flags: {} -> {}", prev_flags.format(), next_flags.format())
    }

    impl X86Memory {
        fn from(instructions: Vec<u8>) -> X86Memory {
            X86Memory {
                arr: core::array::from_fn(|_| 0u8),
                instructions,
                ip: 0,
                flags: Flags::empty(),
            }
        }

        fn process_jump<T: Write>(
            &self,
            instruction: &Instruction,
            out_opt: &mut Option<&mut T>,
        ) -> Option<i8> {
            let displacement = instruction.left_operand.as_displacement();
            out_opt.if_some_ref_mut(|out| {
                write!(out, "{} ; ", instruction.format()).expect("should write");
            });
            assert!(displacement.is_some());
            match instruction.op_code {
                OpCode::Je if (self.flags.contains(Flags::Zero)) => displacement,

                OpCode::Jl
                    if (self.flags.contains(Flags::Sign)
                        != self.flags.contains(Flags::Overflow)) =>
                {
                    displacement
                }
                OpCode::Jle
                    if (self.flags.contains(Flags::Sign)
                        != self.flags.contains(Flags::Overflow)
                        || self.flags.contains(Flags::Zero)) =>
                {
                    displacement
                }
                OpCode::Jb if (self.flags.contains(Flags::Carry)) => displacement,
                OpCode::Jbe
                    if (self.flags.contains(Flags::Carry) || self.flags.contains(Flags::Zero)) =>
                {
                    displacement
                }
                OpCode::Jp if (self.flags.contains(Flags::Parity)) => displacement,
                OpCode::Jo if (self.flags.contains(Flags::Overflow)) => displacement,
                OpCode::Js if (self.flags.contains(Flags::Sign)) => displacement,
                OpCode::Jne if (self.flags.not().contains(Flags::Zero)) => displacement,
                OpCode::Jnl
                    if (self.flags.contains(Flags::Sign)
                        == self.flags.contains(Flags::Overflow)) =>
                {
                    displacement
                }
                OpCode::Jnle
                    if (self.flags.contains(Flags::Sign)
                        == self.flags.contains(Flags::Overflow)
                        && self.flags.not().contains(Flags::Zero)) =>
                {
                    displacement
                }

                OpCode::Jnb if (self.flags.not().contains(Flags::Carry)) => displacement,

                OpCode::Jnbe if (self.flags.not().intersects(Flags::Carry | Flags::Zero)) => {
                    displacement
                }
                OpCode::Jnp if (self.flags.not().contains(Flags::Parity)) => displacement,
                OpCode::Jno if (self.flags.not().contains(Flags::Overflow)) => displacement,
                OpCode::Jns if (self.flags.not().contains(Flags::Sign)) => displacement,
                OpCode::Jcxz if (self.get_reg(&Register::Cx) == 0) => displacement,

                _ => None,
            }
        }
        fn process_loop<T: Write>(
            &mut self,
            instruction: &Instruction,
            out_opt: &mut Option<&mut T>,
        ) -> Option<i8> {
            let displacement = instruction.left_operand.as_displacement();
            assert!(displacement.is_some());

            let prev_cx = self.get_reg(&Register::Cx);
            let (next_cx, flags) = execute_arithmetic_op(prev_cx, 1, &ArithmeticOp::Sub, true);

            out_opt.if_some_ref_mut(|out| {
                write!(
                    out,
                    "{} ; cx {:#06x} -> {:#06x} ",
                    instruction.format(),
                    prev_cx,
                    next_cx,
                )
                .expect("write is fine");
            });

            self.arr.write_as_u16(Register::Cx.to_byte_index(), next_cx);
            // it doesn't affect the flags
            // self.flags = flags;

            return match instruction.op_code {
                OpCode::Loop if (next_cx != 0) => displacement,
                OpCode::Loopz if (next_cx != 0 && flags.contains(Flags::Zero)) => displacement,
                OpCode::Loopnz if (next_cx != 0 && flags.not().contains(Flags::Zero)) => {
                    displacement
                }
                _ => None,
            };
        }

        fn address_of_operand(&self, operand: &Operand) -> Option<usize> {
            match operand {
                Operand::Address(reg1, reg2, displ) => {
                    let a = reg1.map(|it| self.get_reg(&it)).unwrap_or(0);
                    let b = reg2.map(|it| self.get_reg(&it)).unwrap_or(0);
                    let c = displ.unwrap_or(0);

                    let address = (a as i32) + (b as i32) + (c as i32);

                    return Some((address as usize) + (start_of_memory as usize));
                }
                _ => None,
            }
        }

        fn run<T: Write>(&mut self, out_opt: &mut Option<&mut T>) {
            loop {
                let mut iter = CountingIter {
                    data: &self.instructions,
                    start_idx: self.ip,
                    movs: 0,
                };
                let Some(instr) = decoder::decode_instruction(&mut iter) else {
                    return;
                };
                let movs = iter.movs;
                match (&instr.op_code, instr.left_operand) {
                    (OpCode::Mov, Operand::Register(_)) => {
                        self.mov_register(&instr, out_opt);
                    }
                    (OpCode::Mov, Operand::Address(_, _, _)) => {
                        self.process_store(&instr, out_opt);
                    }
                    (OpCode::Cmp | OpCode::Add | OpCode::Sub, Operand::Register(_)) => {
                        self.arithmetic_register(&instr, out_opt);
                    }
                    (
                        OpCode::Je
                        | OpCode::Jl
                        | OpCode::Jle
                        | OpCode::Jb
                        | OpCode::Jbe
                        | OpCode::Jp
                        | OpCode::Jo
                        | OpCode::Js
                        | OpCode::Jne
                        | OpCode::Jnl
                        | OpCode::Jnle
                        | OpCode::Jnb
                        | OpCode::Jnbe
                        | OpCode::Jnp
                        | OpCode::Jno
                        | OpCode::Jns
                        | OpCode::Jcxz,
                        _,
                    ) => {
                        let displacement = self.process_jump(&instr, out_opt);
                        match displacement {
                            Some(it) => {
                                let next_ip = if it >= 0 {
                                    self.ip + (it as u16)
                                } else {
                                    self.ip - (it.abs() as u16)
                                };
                                self.ip = next_ip;
                            }
                            None => {}
                        }

                        out_opt.if_some_ref_mut(|out| {
                            writeln!(out, "ip {:#06x} -> {:#06x}", self.ip, self.ip + movs)
                                .expect("should write");
                        });
                    }
                    (OpCode::Loop | OpCode::Loopz | OpCode::Loopnz, _) => {
                        let displacement = self.process_loop(&instr, out_opt);
                        match displacement {
                            Some(it) => {
                                let next_ip = if it >= 0 {
                                    self.ip + (it as u16)
                                } else {
                                    self.ip - (it.abs() as u16)
                                };
                                self.ip = next_ip;
                            }
                            None => {}
                        }

                        out_opt.if_some_ref_mut(|out| {
                            writeln!(out, "ip {:#06x} -> {:#06x}", self.ip, self.ip + movs)
                                .expect("should write");
                        });
                    }
                    _ => panic!("unsupported OpCode {}", instr.op_code.format()),
                }
                self.ip += movs;
                if (self.ip as usize) > self.instructions.len() {
                    return;
                }
            }
        }

        fn get_reg(&self, reg: &Register) -> u16 {
            if reg.is_wide() {
                self.arr.read_as_u16(reg.to_byte_index())
            } else {
                self.arr[reg.to_byte_index()] as u16
            }
        }
        fn set_word_reg(&mut self, reg: &Register, value: u16) {
            #[cfg(debug_assertions)]
            assert!(reg.reg_type() == RegType::Word);
        }
        fn set_byte_reg(&mut self, reg: &Register, value: u8) {
            #[cfg(debug_assertions)]
            assert!(reg.reg_type() != RegType::Word);
            self.arr[reg.to_byte_index()] = value;
        }

        fn set_reg(&mut self, reg: &Register, value: u16) {
            if reg.is_wide() {
                self.arr.write_as_u16(reg.to_byte_index(), value);
            } else {
                self.set_byte_reg(reg, value as u8);
            }
        }

        fn arithmetic_register<T: Write>(
            &mut self,
            instruction: &Instruction,
            out_opt: &mut Option<&mut T>,
        ) {
            let arithm_op = match instruction.op_code {
                OpCode::Add => ArithmeticOp::Add,
                OpCode::Sub => ArithmeticOp::Sub,
                OpCode::Cmp => ArithmeticOp::Cmp,
                _ => panic!("wrong opcode {:?}", instruction.op_code),
            };
            let left_reg = instruction
                .left_operand
                .as_register()
                .expect("it's arithmetic op for rigister operands");

            assert_eq!(
                instruction.flags.contains(InstructionFlags::Wide),
                left_reg.is_wide()
            );
            let right = instruction.right_operand;

            let is_wide = left_reg.is_wide();

            out_opt.if_some_ref_mut(|out| {
                let word_reg = left_reg.to_word();
                let prev_value = self.get_reg(&word_reg);

                out.write(format!("{} ; ", instruction.format(),).as_bytes())
                    .expect("is ok");

                if arithm_op != ArithmeticOp::Cmp {
                    out.write(format!("{} {:#06x} -> ", word_reg.format(), prev_value).as_bytes())
                        .expect("should write");
                }
            });

            let left_value = self.get_reg(&left_reg);
            let right_value = match right {
                Operand::Immediate(value) => value as u16,
                Operand::Register(right_reg) => self.get_reg(&right_reg),
                Operand::Address(_, _, _) => {
                    let address = self.address_of_operand(&right).expect("should be address");

                    if left_reg.is_wide() {
                        self.arr.read_as_u16(address)
                    } else {
                        self.arr[address] as u16
                    }
                }
                _ => panic!("invariant arithm r_val {:#?}", right),
            };

            let (next_value, flags) =
                execute_arithmetic_op(left_value, right_value, &arithm_op, is_wide);

            out_opt.if_some_ref_mut(|out| {
                if arithm_op != ArithmeticOp::Cmp {
                    out.write(format!("{:#06x} ", next_value).as_bytes())
                        .expect("write is ok");
                }

                writeln!(out, "{}", print_flags(&self.flags, &flags)).expect("write is fine");
            });

            self.flags = flags;
            if left_reg.is_wide() {
                self.arr.write_as_u16(left_reg.to_byte_index(), next_value);
            } else {
                self.arr[left_reg.to_byte_index()] = next_value as u8;
            }
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
                let prev_value = self.get_reg(&word_reg);

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
                        let value = self.arr.read_as_u16(from_reg.to_byte_index());
                        self.arr.write_as_u16(to_reg.to_byte_index(), value);
                    }
                    Operand::Immediate(data) => {
                        self.arr.write_as_u16(to_reg.to_byte_index(), *data as u16);
                    }
                    Operand::Address(_, _, _) => {
                        let Some(address) = self.address_of_operand(&operand) else {
                            panic!("invariant");
                        };

                        let value = self.arr.read_as_u16(address as usize);
                        self.arr.write_as_u16(to_reg.to_byte_index(), value);
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
                    Operand::Address(_, _, _) => {
                        let Some(address) = self.address_of_operand(&operand) else {
                            panic!("invariant");
                        };

                        let value = self.arr[address as usize];
                        self.arr[to_reg.to_byte_index()] = value;
                    }
                    _ => panic!("unexpected operand {}", operand.format()),
                }
            }

            out_opt.if_some_ref_mut(|out| {
                let word_reg = to_reg.to_word();
                let next_value = self.get_reg(&word_reg);

                out.write(format!("{:#06x}\n", next_value).as_bytes())
                    .expect("is ok");
            });
        }

        fn process_store<T: Write>(
            &mut self,
            instruction: &Instruction,
            out_opt: &mut Option<&mut T>,
        ) {
            // let Operand::Register(reg) = instruction.right_operand else {
            //     panic!("invariant op ${:?}", instruction.right_operand);
            // };
            let address = self
                .address_of_operand(&instruction.left_operand)
                .expect("address must exist");

            match instruction.right_operand {
                Operand::Register(reg) => {
                    let value = self.get_reg(&reg);
                    out_opt.if_some_ref_mut(|out| {
                        writeln!(
                            out,
                            "{} ; {:#06x} -> ({:#06x})",
                            instruction.format(),
                            value,
                            address
                        )
                        .expect("store should write");
                    });
                    if reg.is_wide() {
                        let value = self.arr.read_as_u16(reg.to_byte_index());

                        self.arr.write_as_u16(address, value);
                    } else {
                        self.arr[address] = self.arr[reg.to_byte_index()];
                    }
                }
                Operand::Immediate(value) => {
                    out_opt.if_some_ref_mut(|out| {
                        writeln!(
                            out,
                            "{} ; {:#06x} -> ({:#06x})",
                            instruction.format(),
                            value,
                            address
                        )
                        .expect("store should write");
                    });

                    if instruction.flags.contains(InstructionFlags::Wide) {
                        self.arr.write_as_u16(address, value as u16);
                    } else {
                        self.arr[address] = value as u8;
                    }
                }
                _ => panic!("invariant {:#?}", instruction),
            }
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
                let value = self.get_reg(reg);

                writeln!(out, "{} {:#06x} ({})", reg.format(), value, value).unwrap();
            });

            writeln!(out, "ip: {:#06x} ({})", self.ip, self.ip).unwrap();
            writeln!(out, "flags: {}", self.flags.format()).unwrap();
        }
    }

    pub fn execute<T: Write>(instructions: Vec<u8>, mut out: Option<&mut T>) {
        let mut memory = X86Memory::from(instructions);
        memory.run(&mut out);
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

        // fn to_word_index(&self) -> usize {
        //     match self {
        //         Register::Ax => 0,
        //         Register::Ah => 0,
        //         Register::Al => 0,

        //         Register::Bx => 1,
        //         Register::Bh => 1,
        //         Register::Bl => 1,

        //         Register::Cx => 2,
        //         Register::Ch => 2,
        //         Register::Cl => 2,

        //         Register::Dx => 3,
        //         Register::Dh => 3,
        //         Register::Dl => 3,

        //         Register::Sp => 4,
        //         Register::Bp => 5,
        //         Register::Si => 6,
        //         Register::Di => 7,
        //     }
        // }

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
        fn read_as_u16(&self, index: usize) -> u16;
        fn write_as_u16(&mut self, index: usize, value: u16);
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
        fn read_as_u16(&self, index: usize) -> u16 {
            assert!(self.len() > index + 1);

            if cfg!(not(prefer_native_ops = "false")) && cfg!(target_endian = "little") {
                unsafe { self.as_ptr().byte_add(index).cast::<u16>().read() }
            } else {
                let lower = self[index] as u16;
                let upper = self[index + 1] as u16;

                (lower) | (upper << 8)
            }
        }
        fn write_as_u16(&mut self, index: usize, value: u16) {
            assert!(self.len() > index + 1);

            if cfg!(not(prefer_native_ops = "false")) && cfg!(target_endian = "little") {
                unsafe { self.as_mut_ptr().byte_add(index).cast::<u16>().write(value) }
            } else {
                self[index] = (value & 0x00FF) as u8;
                self[index + 1] = ((value & 0xFF00) >> 8) as u8;
            }
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
    #[cfg(target_endian = "big")]
    println!("This is a BigEndian system");

    #[cfg(target_endian = "little")]
    println!("This is a Little endian system");

    let mut stdout = &mut std::io::stdout().lock();
    match mode {
        CliMode::Disasm => {
            let content_iter = content.iter().map(|it| *it);
            let instrucitons = process_binary(content_iter);
            write_instructions(instrucitons, &mut stdout);
        }
        CliMode::Exec => {
            use crate::simulator::execute;
            execute(content, Some(stdout));
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

            write_instructions(process_binary(file.iter().map(|it| *it)), &mut res);

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
            "./input/listing_0046_add_sub_cmp",
            "./input/listing_0047_challenge_flags",
            "./input/listing_0048_ip_register",
            "./input/listing_0049_conditional_jumps",
            "./input/listing_0050_challenge_jumps",
            "./input/listing_0051_memory_mov",
            "./input/listing_0052_memory_add_loop",
            "./input/listing_0053_add_loop_challenge",
        ];
        let force_recompilation = std::env::var("RECOMPILE").map_or(false, |it| it == "true");

        for file_path in files {
            compile_asm_if_not(file_path, force_recompilation);
            let file = read_file(file_path);
            let mut res: Vec<u8> = Vec::new();
            println!("evaluating '{}'", file_path);

            execute(file, Some(&mut res));
            insta::assert_snapshot!(str::from_utf8(res.as_slice()).unwrap());
        }
    }
}
