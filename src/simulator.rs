use crate::{
    decoder,
    estimation::{estimate_cycles_of, Architecture, CycleEstimation},
    format::Formattable,
    Instruction, InstructionFlags, IsWide, OpCode, Operand, Register,
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

impl OpCode {
    fn as_arithmetic_op(&self) -> Option<ArithmeticOp> {
        match self {
            OpCode::Add => Some(ArithmeticOp::Add),
            OpCode::Sub => Some(ArithmeticOp::Sub),
            OpCode::Cmp => Some(ArithmeticOp::Cmp),
            _ => None,
        }
    }
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

#[derive(Debug)]
struct ExecutionRegMutation {
    reg: Register,
    before: u16,
    after: u16,
}

impl Into<ExecutionMutation> for ExecutionRegMutation {
    fn into(self) -> ExecutionMutation {
        ExecutionMutation::Register(self)
    }
}

impl Into<Option<ExecutionMutation>> for ExecutionRegMutation {
    fn into(self) -> Option<ExecutionMutation> {
        ExecutionMutation::Register(self).into()
    }
}

#[derive(Debug)]
struct ExecutionStoreMutation {
    after: u16,
    before: u16,
    target: Address,
}

#[derive(Debug)]
enum ExecutionArithmeticTarget {
    Register(ExecutionRegMutation),
    Store(ExecutionStoreMutation),
}
#[derive(Debug)]
struct ExecutionArithmeticMutation {
    target: Option<ExecutionArithmeticTarget>,
    prev_flags: Flags,
    next_flags: Flags,
}

impl Into<ExecutionMutation> for ExecutionArithmeticMutation {
    fn into(self) -> ExecutionMutation {
        ExecutionMutation::Arithmetic(self)
    }
}

impl Into<Option<ExecutionMutation>> for ExecutionArithmeticMutation {
    fn into(self) -> Option<ExecutionMutation> {
        ExecutionMutation::Arithmetic(self).into()
    }
}

impl Into<ExecutionMutation> for ExecutionStoreMutation {
    fn into(self) -> ExecutionMutation {
        ExecutionMutation::Store(self)
    }
}

impl Into<Option<ExecutionMutation>> for ExecutionStoreMutation {
    fn into(self) -> Option<ExecutionMutation> {
        ExecutionMutation::Store(self).into()
    }
}

impl ExecutionArithmeticTarget {
    fn as_register_unsafe(&self) -> &ExecutionRegMutation {
        match self {
            ExecutionArithmeticTarget::Register(reg) => reg,
            _ => panic!("unexpected {:?}", self),
        }
    }
    fn as_store_unsafe(&self) -> &ExecutionStoreMutation {
        match self {
            ExecutionArithmeticTarget::Store(store) => store,
            _ => panic!("unexpected {:?}", self),
        }
    }
}

#[derive(Debug)]
enum ExecutionMutation {
    Register(ExecutionRegMutation),
    Store(ExecutionStoreMutation),
    Arithmetic(ExecutionArithmeticMutation),
}

impl ExecutionMutation {
    fn as_register_unsafe(&self) -> &ExecutionRegMutation {
        match self {
            ExecutionMutation::Register(reg) => reg,
            _ => panic!("unexpected {:?}", self),
        }
    }
    fn as_store_unsafe(&self) -> &ExecutionStoreMutation {
        match self {
            ExecutionMutation::Store(store) => store,
            _ => panic!("unexpected {:?}", self),
        }
    }
    fn as_arithmetic_unsafe(&self) -> &ExecutionArithmeticMutation {
        match self {
            ExecutionMutation::Arithmetic(arithm) => arithm,
            _ => panic!("unexpected {:?}", self),
        }
    }
}

struct ExecutionMeta {
    resolved_memory_operand: Option<Address>,
    mutation: Option<ExecutionMutation>,
    jump: Option<i8>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Address {
    value: usize,
}
impl Address {
    fn of_memory(value: u16) -> Address {
        Address {
            value: (value as usize) + (START_OF_MEMORY as usize),
        }
    }

    fn is_memory(&self) -> bool {
        self.value >= START_OF_MEMORY
    }

    fn as_memory_offset(&self) -> usize {
        self.value
    }

    fn as_logical_memory_offset_unsafe(&self) -> u16 {
        assert!(self.is_memory(), "is only defined for memory addresses");

        (self.value - START_OF_MEMORY) as u16
    }

    fn of_register(reg: &Register) -> Address {
        let index = match reg {
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
        };

        Address { value: index }
    }
}

struct InstructionSpecificPrintOutput {
    permutation: Option<String>,
    flags_mutation: Option<String>,
}

fn produce_estimation(
    instr: &Instruction,
    clocks_passed: u32,
    arch: &Architecture,
    estimation: CycleEstimation,
    meta: &ExecutionMeta,
) -> (u32, String) {
    use crate::estimation::TransferAddress;
    use std::fmt::Write;
    let is_word = instr.flags.contains(InstructionFlags::Wide);

    let transfer_tax = if estimation.transfers > 0 {
        let operand = meta
            .resolved_memory_operand
            .expect("if has transfers memory operand must be resolved");
        estimation.estimate_trasfer_tax(
            if is_word {
                TransferAddress::Word(operand.as_logical_memory_offset_unsafe())
            } else {
                TransferAddress::Byte
            },
            arch,
        )
    } else {
        0
    };

    let total_clocks = (estimation.base + estimation.eac) as u32 + transfer_tax;

    let mut res = String::with_capacity(
        // "Clocks: +17 = 107 (8 + 9ea)"
        /* prefix */
        9  + /* estimation */ 2 + /* spacing */ 3  + /* clock */ 6 + /* spacing */ 2 + /* base */ 1 + /* spacing */ 3 + /* eac */ 4 + /* spacing */ 3 + /* transfers */ 3 + 3,
    );
    let next_clocks = clocks_passed + total_clocks;
    write!(res, "clocks: +{} = {}", total_clocks, next_clocks).expect("write is ok");

    let needs_details = estimation.eac > 0 || estimation.transfers > 0;
    if needs_details {
        res.push_str(" (");
    }

    if needs_details && estimation.base > 0 {
        res.push_str(estimation.base.to_string().as_str());
    }
    if needs_details && estimation.eac > 0 {
        if res.len() > 0 {
            res.push_str(" + ");
        }
        write!(res, "{}ea", estimation.eac).expect("write is ok");
    }
    if needs_details && transfer_tax > 0 {
        if res.len() > 0 {
            res.push_str(" + ");
        };

        write!(res, "{}t", transfer_tax).expect("write is ok");
    };

    if needs_details {
        res.push_str(")");
    }

    return (next_clocks, res);
}

impl Simulator {
    pub fn from(instructions: Vec<u8>) -> Simulator {
        Simulator {
            memory: core::array::from_fn(|_| 0u8),
            instructions,
            ip: 0,
            flags: Flags::empty(),
        }
    }

    fn process_jump(&self, instruction: &Instruction) -> ExecutionMeta {
        let displacement = instruction.left_operand.as_displacement();
        assert!(displacement.is_some());
        let jump = match instruction.op_code {
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
        };

        ExecutionMeta {
            resolved_memory_operand: None,
            mutation: None,
            jump: jump,
        }
    }

    fn process_loop(&mut self, instruction: &Instruction) -> ExecutionMeta {
        let displacement = instruction.left_operand.as_displacement();
        assert!(displacement.is_some());

        let prev_cx = self.get_reg(&Register::Cx);
        let (next_cx, flags) = execute_arithmetic_op(prev_cx, 1, &ArithmeticOp::Sub, true);

        self.memory
            .write_as_u16(Address::of_register(&Register::Cx), next_cx);
        // it doesn't affect the flags
        // self.flags = flags;

        let jump = match instruction.op_code {
            OpCode::Loop if (next_cx != 0) => displacement,
            OpCode::Loopz if (next_cx != 0 && flags.contains(Flags::Zero)) => displacement,
            OpCode::Loopnz if (next_cx != 0 && flags.not().contains(Flags::Zero)) => displacement,
            _ => None,
        };
        ExecutionMeta {
            resolved_memory_operand: None,
            mutation: Some(
                ExecutionRegMutation {
                    reg: Register::Cx,
                    before: prev_cx,
                    after: next_cx,
                }
                .into(),
            ),
            jump: jump,
        }
    }

    fn address_of_operand(&self, operand: &Operand) -> Option<Address> {
        match operand {
            Operand::EAC(reg1, reg2, displ) => {
                let a = reg1.map(|it| self.get_reg(&it)).unwrap_or(0);
                let b = reg2.map(|it| self.get_reg(&it)).unwrap_or(0);
                let c = displ.unwrap_or(0);

                // there are a lot of edge cases
                let address = ((a as i32) + (b as i32) + (c as i32)) as u16;

                Some(Address::of_memory(address))
            }
            Operand::Reference(reference) => Some(Address::of_memory(*reference)),
            _ => None,
        }
    }

    fn run<T: Write>(&mut self, out_opt: &mut Option<&mut T>, config: &SimulatorConfig) {
        let mut clocks: u32 = 0;
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
            let cycle_estimation_mode = config.cycle_estimation_mode.as_ref();

            let (meta, print_info) = match (&instr.op_code, instr.left_operand) {
                (OpCode::Mov, Operand::Register(_)) => {
                    let meta = self.mov_register(&instr);

                    let _mutation = meta.mutation.as_ref().expect("mov mutates");
                    let mutation = _mutation.as_register_unsafe();

                    let print_out = InstructionSpecificPrintOutput {
                        flags_mutation: None,
                        permutation: Some(format!(
                            "{} {:#06x} -> {:#06x}",
                            mutation.reg.format(),
                            mutation.before,
                            mutation.after,
                        )),
                    };

                    (meta, print_out)
                }
                (OpCode::Mov, Operand::EAC(_, _, _) | Operand::Reference(_)) => {
                    let meta = self.process_store(&instr);
                    let _mutation = meta.mutation.as_ref().expect("store mutates");
                    let mutation = _mutation.as_store_unsafe();
                    let print_out = InstructionSpecificPrintOutput {
                        flags_mutation: None,
                        permutation: Some(format!(
                            "{:#06x} -> {:#06x}",
                            mutation.after,
                            mutation.target.as_logical_memory_offset_unsafe()
                        )),
                    };

                    (meta, print_out)
                }
                (OpCode::Cmp | OpCode::Add | OpCode::Sub, Operand::EAC(_, _, _)) => {
                    let meta = self.arithmetic_store(&instr);
                    let _mutation = meta.mutation.as_ref().expect("store mutates");
                    let mutation = _mutation.as_arithmetic_unsafe();

                    let print_out = InstructionSpecificPrintOutput {
                        permutation: mutation.target.as_ref().map(|it| {
                            let base = it.as_store_unsafe();
                            format!(
                                "{:#06x} -> ({:#06x})",
                                base.before,
                                base.target.as_logical_memory_offset_unsafe()
                            )
                        }),
                        flags_mutation: print_flags(&mutation.prev_flags, &mutation.next_flags)
                            .into(),
                    };
                    (meta, print_out)
                }
                (OpCode::Cmp | OpCode::Add | OpCode::Sub, Operand::Register(_)) => {
                    let meta = self.arithmetic_register(&instr);

                    let _mutation = meta.mutation.as_ref().expect("arithm mutates");
                    let mutation = _mutation.as_arithmetic_unsafe();
                    let print_out = InstructionSpecificPrintOutput {
                        flags_mutation: Some(print_flags(
                            &mutation.prev_flags,
                            &mutation.next_flags,
                        )),
                        permutation: mutation.target.as_ref().map(|base| {
                            let base = base.as_register_unsafe();
                            format!(
                                "{} {:#06x} -> {:#06x}",
                                base.reg.format(),
                                base.before,
                                base.after
                            )
                        }),
                    };

                    (meta, print_out)
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
                    let meta = self.process_jump(&instr);
                    match meta.jump {
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

                    (
                        meta,
                        InstructionSpecificPrintOutput {
                            flags_mutation: format!(
                                "ip {:#06x} -> {:#06x}",
                                self.ip,
                                self.ip + movs
                            )
                            .into(),
                            permutation: None,
                        },
                    )
                }
                (OpCode::Loop | OpCode::Loopz | OpCode::Loopnz, _) => {
                    let meta = self.process_loop(&instr);
                    let _mutation = meta.mutation.as_ref().expect("loop mutates");
                    let mutation = _mutation.as_register_unsafe();

                    match meta.jump {
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

                    let print_out = InstructionSpecificPrintOutput {
                        permutation: format!(
                            "{} {:#06x} -> {:#06x}",
                            mutation.reg.format(),
                            mutation.before,
                            mutation.after,
                        )
                        .into(),
                        flags_mutation: format!("ip {:#06x} -> {:#06x}", self.ip, self.ip + movs)
                            .into(),
                    };
                    (meta, print_out)
                }
                _ => panic!("unsupported intstruction {:?}", instr),
            };

            out_opt.if_some_ref_mut(|out| {
                write!(out, "{} ;", instr.format()).unwrap();

                let estimation = cycle_estimation_mode.map(|arch| {
                    let (next_clocks, estimation_text) =
                        produce_estimation(&instr, clocks, arch, estimate_cycles_of(&instr), &meta);
                    clocks = next_clocks;

                    estimation_text
                });

                let mut should_divide = false;
                match estimation {
                    Some(estimatio_str) => {
                        if !should_divide {
                            write!(out, " ").unwrap();
                        }
                        write!(out, "{}", estimatio_str).unwrap()
                    }
                    _ => {}
                }

                match print_info.permutation {
                    Some(permutation) => {
                        if should_divide {
                            write!(out, " | ").unwrap();
                        } else {
                            write!(out, " ").unwrap();
                        }

                        should_divide = true;
                        write!(out, "{}", permutation).unwrap();
                    }
                    None => {}
                };
                match print_info.flags_mutation {
                    Some(flags) => {
                        if should_divide {
                            write!(out, " | ").unwrap();
                        } else {
                            write!(out, " ").unwrap();
                        }

                        write!(out, "{}", flags).unwrap();
                    }
                    None => {}
                };

                writeln!(out, "").unwrap();
            });

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
        let address = Address::of_register(reg);
        if reg.is_wide() {
            self.memory.read_as_u16(address)
        } else {
            self.memory.read_as_u8(address) as u16
        }
    }

    fn arithmetic_store(&mut self, instruction: &Instruction) -> ExecutionMeta {
        let Some(arithm_op) = instruction.op_code.as_arithmetic_op() else {
            panic!("wrong opcode {:?}", instruction.op_code)
        };
        let right_value = match instruction.right_operand {
            Operand::Register(reg) => self.get_reg(&reg),
            Operand::Immediate(imm) => imm as u16,
            _ => panic!("expected reg or immediate"),
        };

        let is_wide = instruction.flags.contains(InstructionFlags::Wide);
        assert!(matches!(instruction.left_operand, Operand::EAC(_, _, _)));

        let address = self
            .address_of_operand(&instruction.left_operand)
            .expect("should be address");

        let (resolved_left_value_address, left_value) = match instruction.left_operand {
            Operand::EAC(_, _, _) => {
                let value = if is_wide {
                    self.memory.read_as_u16(address)
                } else {
                    self.memory.read_as_u8(address) as u16
                };
                (address, value)
            }
            _ => panic!("unwxpected left in {:?}", instruction),
        };

        let prev_flags = self.flags;
        let (next_value, next_flags) =
            execute_arithmetic_op(left_value, right_value, &arithm_op, is_wide);

        self.flags = next_flags;
        if is_wide {
            self.memory
                .write_as_u16(resolved_left_value_address, next_value);
        } else {
            self.memory
                .write_as_u8(resolved_left_value_address, next_value as u8);
        }

        ExecutionMeta {
            resolved_memory_operand: resolved_left_value_address.into(),
            jump: None,
            mutation: ExecutionArithmeticMutation {
                prev_flags: prev_flags,
                next_flags: next_flags,
                target: match arithm_op {
                    ArithmeticOp::Cmp => None,
                    ArithmeticOp::Sub | ArithmeticOp::Add => {
                        Some(ExecutionArithmeticTarget::Store(ExecutionStoreMutation {
                            target: address,
                            before: left_value,
                            after: next_value,
                        }))
                    }
                },
            }
            .into(),
        }
    }

    fn arithmetic_register(&mut self, instruction: &Instruction) -> ExecutionMeta {
        let Some(arithm_op) = instruction.op_code.as_arithmetic_op() else {
            panic!("wrong opcode {:?}", instruction.op_code)
        };
        let left_reg = instruction
            .left_operand
            .as_register()
            .expect("it's arithmetic op for register operands");

        assert_eq!(
            instruction.flags.contains(InstructionFlags::Wide),
            left_reg.is_wide()
        );
        let right = instruction.right_operand;

        let is_wide = left_reg.is_wide();

        let left_word_reg = left_reg.to_word();
        let prev_value = self.get_reg(&left_reg.to_word());

        let left_value = self.get_reg(&left_reg);
        let (resolved_right_memory_address, right_value) = match right {
            Operand::Immediate(value) => (None, value as u16),
            Operand::Register(right_reg) => (None, self.get_reg(&right_reg)),
            Operand::EAC(_, _, _) => {
                let address = self.address_of_operand(&right).expect("should be address");

                let value = if left_reg.is_wide() {
                    self.memory.read_as_u16(address)
                } else {
                    self.memory.read_as_u8(address) as u16
                };
                (Some(address), value)
            }
            _ => panic!("invariant arithm r_val {:#?}", right),
        };

        let prev_flags = self.flags;
        let (next_value, next_flags) =
            execute_arithmetic_op(left_value, right_value, &arithm_op, is_wide);

        self.flags = next_flags;
        if left_reg.is_wide() {
            self.memory
                .write_as_u16(Address::of_register(&left_reg), next_value);
        } else {
            self.memory
                .write_as_u8(Address::of_register(&left_reg), next_value as u8);
        }

        ExecutionMeta {
            resolved_memory_operand: resolved_right_memory_address,
            jump: None,
            mutation: ExecutionArithmeticMutation {
                prev_flags: prev_flags,
                next_flags: next_flags,
                target: match arithm_op {
                    ArithmeticOp::Cmp => None,
                    ArithmeticOp::Sub | ArithmeticOp::Add => {
                        Some(ExecutionArithmeticTarget::Register(ExecutionRegMutation {
                            reg: left_word_reg,
                            before: prev_value,
                            after: next_value,
                        }))
                    }
                },
            }
            .into(),
        }
    }

    fn mov_register(&mut self, instruction: &Instruction) -> ExecutionMeta {
        let Operand::Register(to_reg) = &instruction.left_operand else {
            panic!("unexpected instruction {}", instruction.format());
        };
        let to_reg_address = Address::of_register(to_reg);
        let operand = instruction.right_operand;
        let flags = instruction.flags;

        let is_wide = to_reg.is_wide();
        assert_eq!(flags.contains(InstructionFlags::Wide), is_wide);

        let word_reg = to_reg.to_word();
        let prev_value = self.get_reg(&word_reg);

        let resolved_memory_operand = if is_wide {
            match &operand {
                Operand::Register(from_reg) => {
                    let value = self.memory.read_as_u16(Address::of_register(from_reg));
                    self.memory.write_as_u16(to_reg_address, value);
                    None
                }
                Operand::Immediate(data) => {
                    self.memory.write_as_u16(to_reg_address, *data as u16);
                    None
                }
                Operand::EAC(_, _, _) => {
                    let Some(address) = self.address_of_operand(&operand) else {
                        panic!("invariant");
                    };

                    let value = self.memory.read_as_u16(address);
                    self.memory.write_as_u16(to_reg_address, value);

                    Some(address)
                }
                _ => panic!("unexpected operand {}", operand.format()),
            }
        } else {
            match &operand {
                Operand::Register(from_reg) => {
                    #[cfg(debug_assertions)]
                    assert!(!from_reg.is_wide());

                    let value = self.memory.read_as_u8(Address::of_register(from_reg));
                    self.memory.write_as_u8(to_reg_address, value);

                    None
                }
                Operand::Immediate(data) => {
                    #[cfg(debug_assertions)]
                    assert!(*data <= 0x00FFi16);

                    self.memory.write_as_u8(to_reg_address, (*data) as u8);
                    None
                }
                Operand::EAC(_, _, _) => {
                    let Some(address) = self.address_of_operand(&operand) else {
                        panic!("invariant");
                    };

                    let value = self.memory.read_as_u8(address);
                    self.memory.write_as_u8(Address::of_register(to_reg), value);

                    Some(address)
                }
                _ => panic!("unexpected operand {}", operand.format()),
            }
        };

        ExecutionMeta {
            resolved_memory_operand: resolved_memory_operand,
            mutation: Some(
                ExecutionRegMutation {
                    reg: word_reg,
                    before: prev_value,
                    after: self.get_reg(&word_reg),
                }
                .into(),
            ),
            jump: None,
        }
    }

    fn process_store(&mut self, instruction: &Instruction) -> ExecutionMeta {
        let address = self
            .address_of_operand(&instruction.left_operand)
            .expect("address must exist");

        let prev_value = self.memory.read_as_u16(address);

        let mutation_value = match instruction.right_operand {
            Operand::Register(reg) => {
                let value = self.get_reg(&reg);
                if reg.is_wide() {
                    self.memory.write_as_u16(address, value);
                } else {
                    self.memory.write_as_u8(address, value as u8);
                }
                value
            }
            Operand::Immediate(value) => {
                if instruction.flags.contains(InstructionFlags::Wide) {
                    self.memory.write_as_u16(address, value as u16);
                } else {
                    self.memory.write_as_u8(address, value as u8);
                }

                value as u16
            }
            _ => panic!("invariant {:#?}", instruction),
        };

        ExecutionMeta {
            resolved_memory_operand: Some(address),
            jump: None,
            mutation: ExecutionStoreMutation {
                after: mutation_value,
                before: prev_value,
                target: address,
            }
            .into(),
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
    let mut machine = Simulator::from(instructions);
    machine.run(&mut out, &config);

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

trait SimulatorMemory {
    fn as_u16_ref(&self) -> &[u16];
    fn as_u16_ref_mut(&mut self) -> &mut [u16];
    fn read_as_u8(&self, index: Address) -> u8;
    fn read_as_u16(&self, index: Address) -> u16;
    fn write_as_u8(&mut self, index: Address, value: u8);
    fn write_as_u16(&mut self, index: Address, value: u16);
}

impl SimulatorMemory for [u8] {
    fn as_u16_ref(&self) -> &[u16] {
        let size = size_of_val(self);

        unsafe { std::slice::from_raw_parts(self.as_ptr().cast::<u16>(), size) }
    }

    fn as_u16_ref_mut(&mut self) -> &mut [u16] {
        let size = size_of_val(self);

        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr().cast::<u16>(), size) }
    }
    fn read_as_u16(&self, index: Address) -> u16 {
        let index = index.as_memory_offset();
        assert!(self.len() > index + 1);

        if cfg!(not(prefer_native_ops = "false")) && cfg!(target_endian = "little") {
            unsafe { self.as_ptr().byte_add(index).cast::<u16>().read() }
        } else {
            let lower = self[index] as u16;
            let upper = self[index + 1] as u16;

            (lower) | (upper << 8)
        }
    }
    fn read_as_u8(&self, index: Address) -> u8 {
        let index = index.as_memory_offset();

        self[index]
    }
    fn write_as_u8(&mut self, index: Address, value: u8) {
        let index = index.as_memory_offset();

        self[index] = value;
    }
    fn write_as_u16(&mut self, index: Address, value: u16) {
        let index = index.as_memory_offset();
        assert!(self.len() > index + 1);

        if cfg!(not(prefer_native_ops = "false")) && cfg!(target_endian = "little") {
            unsafe { self.as_mut_ptr().byte_add(index).cast::<u16>().write(value) }
        } else {
            self[index] = (value & 0x00FF) as u8;
            self[index + 1] = ((value & 0xFF00) >> 8) as u8;
        }
    }
}
