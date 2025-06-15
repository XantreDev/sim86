use crate::{
    decoder, estimation::Architecture, format::Formattable, Instruction, InstructionFlags, IsWide,
    OpCode, Operand, Register,
};
use bitflags::bitflags;
use std::{io::Write, ops::Not};

pub struct SimulatorConfig {
    pub cycle_estimation_mode: Option<Architecture>,
}
impl Default for SimulatorConfig {
    fn default() -> Self {
        SimulatorConfig {
            cycle_estimation_mode: None,
        }
    }
}

pub struct Simulator {
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
    memory: [u8; 65552],

    // at this point instructions is stored separetely
    instructions: Vec<u8>,
    ip: u16,

    flags: Flags,

    config: SimulatorConfig,
}

const START_OF_MEMORY: usize = 8;

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
        (ArithmeticAction::Add, Sign::Positive, Sign::Positive, Sign::Negative) => Flags::Overflow,
        (ArithmeticAction::Add, Sign::Negative, Sign::Negative, Sign::Positive) => Flags::Overflow,
        // positive sum of negative operands
        (ArithmeticAction::Sub, Sign::Negative, Sign::Positive, Sign::Positive) => Flags::Overflow,
        (ArithmeticAction::Sub, Sign::Positive, Sign::Negative, Sign::Negative) => Flags::Overflow,
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

impl Simulator {
    pub fn from(instructions: Vec<u8>, config: SimulatorConfig) -> Simulator {
        Simulator {
            memory: core::array::from_fn(|_| 0u8),
            instructions,
            ip: 0,
            flags: Flags::empty(),
            config: config,
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
                if (self.flags.contains(Flags::Sign) != self.flags.contains(Flags::Overflow)) =>
            {
                displacement
            }
            OpCode::Jle
                if (self.flags.contains(Flags::Sign) != self.flags.contains(Flags::Overflow)
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
                if (self.flags.contains(Flags::Sign) == self.flags.contains(Flags::Overflow)) =>
            {
                displacement
            }
            OpCode::Jnle
                if (self.flags.contains(Flags::Sign) == self.flags.contains(Flags::Overflow)
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

        self.memory.write_as_u16(Register::Cx.to_index(), next_cx);
        // it doesn't affect the flags
        // self.flags = flags;

        return match instruction.op_code {
            OpCode::Loop if (next_cx != 0) => displacement,
            OpCode::Loopz if (next_cx != 0 && flags.contains(Flags::Zero)) => displacement,
            OpCode::Loopnz if (next_cx != 0 && flags.not().contains(Flags::Zero)) => displacement,
            _ => None,
        };
    }

    fn address_of_operand(&self, operand: &Operand) -> Option<usize> {
        match operand {
            Operand::EAC(reg1, reg2, displ) => {
                let a = reg1.map(|it| self.get_reg(&it)).unwrap_or(0);
                let b = reg2.map(|it| self.get_reg(&it)).unwrap_or(0);
                let c = displ.unwrap_or(0);

                let address = (a as i32) + (b as i32) + (c as i32);

                return Some((address as usize) + (START_OF_MEMORY as usize));
            }
            Operand::Reference(reference) => {
                Some((*reference as usize) + (START_OF_MEMORY as usize))
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
                break;
            };
            let movs = iter.movs;
            match (&instr.op_code, instr.left_operand) {
                (OpCode::Mov, Operand::Register(_)) => {
                    self.mov_register(&instr, out_opt);
                }
                (OpCode::Mov, Operand::EAC(_, _, _) | Operand::Reference(_)) => {
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
                break;
            }
        }

        out_opt.if_some_ref_mut(|out| {
            writeln!(out, "\nFinal registers: ").unwrap();
            self.print_registers(out);
        });
    }

    fn get_reg(&self, reg: &Register) -> u16 {
        if reg.is_wide() {
            self.memory.read_as_u16(reg.to_index())
        } else {
            self.memory[reg.to_index()] as u16
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
            Operand::EAC(_, _, _) => {
                let address = self.address_of_operand(&right).expect("should be address");

                if left_reg.is_wide() {
                    self.memory.read_as_u16(address)
                } else {
                    self.memory[address] as u16
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
            self.memory.write_as_u16(left_reg.to_index(), next_value);
        } else {
            self.memory[left_reg.to_index()] = next_value as u8;
        }
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
                    let value = self.memory.read_as_u16(from_reg.to_index());
                    self.memory.write_as_u16(to_reg.to_index(), value);
                }
                Operand::Immediate(data) => {
                    self.memory.write_as_u16(to_reg.to_index(), *data as u16);
                }
                Operand::EAC(_, _, _) => {
                    let Some(address) = self.address_of_operand(&operand) else {
                        panic!("invariant");
                    };

                    let value = self.memory.read_as_u16(address as usize);
                    self.memory.write_as_u16(to_reg.to_index(), value);
                }
                _ => panic!("unexpected operand {}", operand.format()),
            }
        } else {
            match &operand {
                Operand::Register(from_reg) => {
                    #[cfg(debug_assertions)]
                    assert!(!from_reg.is_wide());

                    self.memory[to_reg.to_index()] = self.memory[from_reg.to_index()];
                }
                Operand::Immediate(data) => {
                    #[cfg(debug_assertions)]
                    assert!(*data <= 0x00FFi16);

                    self.memory[to_reg.to_index()] = (*data) as u8;
                }
                Operand::EAC(_, _, _) => {
                    let Some(address) = self.address_of_operand(&operand) else {
                        panic!("invariant");
                    };

                    let value = self.memory[address as usize];
                    self.memory[to_reg.to_index()] = value;
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

    fn process_store<T: Write>(&mut self, instruction: &Instruction, out_opt: &mut Option<&mut T>) {
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
                    let value = self.memory.read_as_u16(reg.to_index());

                    self.memory.write_as_u16(address, value);
                } else {
                    self.memory[address] = self.memory[reg.to_index()];
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
                    self.memory.write_as_u16(address, value as u16);
                } else {
                    self.memory[address] = value as u8;
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

    pub fn dump(&self) -> &[u8] {
        &self.memory[START_OF_MEMORY..self.memory.len()]
    }
}

pub fn execute<T: Write>(
    instructions: Vec<u8>,
    mut out: Option<&mut T>,
    config: SimulatorConfig,
) -> Simulator {
    let mut machine = Simulator::from(instructions, config);
    machine.run(&mut out);

    machine
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

    // little endian encoding
    fn to_index(&self) -> usize {
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
