use super::structs::*;

pub enum Architecture {
    X8086,
    X8088,
}

fn estimate_eac(operand: &Operand) -> u8 {
    let Operand::EAC(reg1, reg2, displ) = operand else {
        panic!("unexpected {:?}", operand)
    };

    // segment overwrites is not supported
    // MOV AX, ES:[BX+SI]
    match (reg1, reg2, displ) {
        (None, None, Some(_)) => 6,
        (
            Some(
                // base or index
                Register::Bx
                | Register::Bl
                | Register::Bh
                | Register::Bp
                | Register::Si
                | Register::Di,
            ),
            None,
            None,
        ) => 5,
        (
            Some(
                // base or index
                Register::Bx
                // is it even possible to have parital registers here?
                | Register::Bl
                | Register::Bh
                | Register::Bp
                | Register::Si
                | Register::Di,
            ),
            None,
            Some(_),
        ) => 9,
        (
            Some(
                Register::Bp
            ),
            Some(
                Register::Di
            ),
            None
        ) | (
            Some(
                Register::Bx
            ),
            Some(
                Register::Si
            ),
            None
        ) => 7,
        (
            Some(
                Register::Bp
            ),
            Some(
                Register::Si
            ),
            None
        ) | (
            Some(
                Register::Bx
            ),
            Some(
                Register::Di
            ),
            None
        ) => 8,
        (
            Some(
                Register::Bp
            ),
            Some(
                Register::Di
            ),
            Some(_)
        ) | (
            Some(
                Register::Bx
            ),
            Some(Register::Si),
            Some(_)
        ) => 11,
        (
            Some(Register::Bp),
            Some(Register::Si),
            Some(_)
        ) | (
            Some(Register::Bx),
            Some(Register::Di),
            Some(_)
        ) => 12,
        _ => panic!("invariant EAC")
    }
}

enum TransferAddress {
    Word(u16),
    Byte(u16),
}

impl CycleEstimation {
    #[inline(always)]
    pub fn estimate_trasfer_tax(
        &self,
        address: TransferAddress,
        architecture: &Architecture,
    ) -> u32 {
        let x8088_odd_operand_transfer_cost = 4u32;
        let cost = match (architecture, address) {
            (Architecture::X8088, TransferAddress::Word(address)) if address % 2 != 0 => {
                x8088_odd_operand_transfer_cost
            }
            _ => 0,
        };

        cost * self.transfers as u32
    }
}

struct CycleEstimation {
    base: u8,
    transfers: u8,
    eac: u8,
}

impl CycleEstimation {
    fn base(base: u8) -> CycleEstimation {
        CycleEstimation {
            eac: 0,
            base,
            transfers: 0,
        }
    }
    fn with_transfers(cycles: u8, transfers: u8) -> CycleEstimation {
        CycleEstimation {
            eac: 0,
            base: cycles,
            transfers: transfers,
        }
    }
    fn with_eac(base_cycles: u8, transfers: u8, operand: &Operand) -> CycleEstimation {
        CycleEstimation {
            base: base_cycles,
            eac: estimate_eac(operand),
            transfers: transfers,
        }
    }
}

pub fn estimate_cycles_of(instr: &Instruction) -> CycleEstimation {
    match instr.op_code {
        OpCode::Jb
        | OpCode::Je
        | OpCode::Jl
        | OpCode::Jo
        | OpCode::Jp
        | OpCode::Js
        | OpCode::Jbe
        | OpCode::Jle
        | OpCode::Jnb
        | OpCode::Jne
        | OpCode::Jnl
        | OpCode::Jno
        | OpCode::Jnp
        | OpCode::Jns
        | OpCode::Jcxz
        | OpCode::Jnbe
        | OpCode::Jnle
        | OpCode::Loop
        | OpCode::Loopz
        | OpCode::Loopnz => panic!("TODO unknown estimation"),
        OpCode::Add | OpCode::Sub
            if instr.left_operand.is_reg() && instr.right_operand.is_reg() =>
        {
            CycleEstimation::base(3)
        }
        OpCode::Add | OpCode::Sub
            if instr.left_operand.is_reg() && instr.right_operand.is_eac() =>
        {
            CycleEstimation::with_eac(9, 1, &instr.right_operand)
        }
        OpCode::Add | OpCode::Sub
            if instr.left_operand.is_eac() && instr.right_operand.is_reg() =>
        {
            CycleEstimation::with_eac(16, 2, &instr.left_operand)
        }
        OpCode::Add | OpCode::Sub
            if instr.left_operand.is_acc_reg() && instr.right_operand.is_immediate() =>
        {
            CycleEstimation::base(4)
        }
        OpCode::Add | OpCode::Sub
            if instr.left_operand.is_reg() && instr.right_operand.is_immediate() =>
        {
            CycleEstimation::base(4)
        }
        OpCode::Add | OpCode::Sub
            if instr.left_operand.is_eac() && instr.right_operand.is_immediate() =>
        {
            CycleEstimation::with_eac(17, 2, &instr.left_operand)
        }
        OpCode::Add | OpCode::Sub => panic!("invalid instruction {:?}", instr),
        OpCode::Cmp if instr.left_operand.is_reg() && instr.right_operand.is_reg() => {
            CycleEstimation::base(3)
        }
        OpCode::Cmp if instr.left_operand.is_reg() && instr.right_operand.is_eac() => {
            CycleEstimation::with_eac(9, 1, &instr.right_operand)
        }
        OpCode::Cmp if instr.left_operand.is_eac() && instr.right_operand.is_reg() => {
            CycleEstimation::with_eac(9, 1, &instr.left_operand)
        }
        // acc is also here
        OpCode::Cmp if instr.left_operand.is_reg() && instr.right_operand.is_immediate() => {
            CycleEstimation::base(4)
        }
        OpCode::Cmp if instr.left_operand.is_eac() && instr.right_operand.is_immediate() => {
            CycleEstimation::with_eac(10, 1, &instr.left_operand)
        }
        OpCode::Cmp => panic!("invalid cmp instruction {:?}", instr),
        OpCode::Mov if instr.left_operand.is_mem_ref() && instr.right_operand.is_acc_reg() => {
            CycleEstimation::with_transfers(10, 1)
        }
        OpCode::Mov if instr.left_operand.is_acc_reg() && instr.right_operand.is_mem_ref() => {
            CycleEstimation::with_transfers(10, 1)
        }

        OpCode::Mov if instr.left_operand.is_reg() && instr.right_operand.is_reg() => {
            CycleEstimation::base(2)
        }
        OpCode::Mov if instr.left_operand.is_reg() && instr.right_operand.is_eac() => {
            CycleEstimation::with_eac(8, 1, &instr.right_operand)
        }
        OpCode::Mov if instr.left_operand.is_eac() && instr.right_operand.is_reg() => {
            CycleEstimation::with_eac(9, 1, &instr.left_operand)
        }
        OpCode::Mov if instr.left_operand.is_reg() && instr.right_operand.is_immediate() => {
            CycleEstimation::base(4)
        }
        OpCode::Mov if instr.left_operand.is_eac() && instr.right_operand.is_immediate() => {
            CycleEstimation::with_eac(10, 1, &instr.left_operand)
        }
        OpCode::Mov => panic!("unknown mov instr {:?}", instr),
    }
}
