use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct InstructionFlags: u8 {
        const Lock = 0b0000_0001;
        const Rep = 0b0000_0010;
        const Segment = 0b0000_0100;
        const Wide = 0b0000_1000;
    }
}

#[rustfmt::skip]
#[derive(Debug, Clone, Copy)]
pub enum Register {
  Ax, Al, Ah,
  Bx, Bl, Bh,
  Cx, Cl, Ch,
  Dx, Dl, Dh,

  Si,
  Di,
  Bp,
  Sp,
  // Ip,
  // Cs,
}

pub trait IsWide {
    fn is_wide(&self) -> bool;
}
impl IsWide for Register {
    fn is_wide(&self) -> bool {
        match self {
            Register::Ax
            | Register::Bx
            | Register::Cx
            | Register::Dx
            | Register::Si
            | Register::Di
            | Register::Bp
            | Register::Sp => true,
            //| Register::Ip
            //| Register::Cs
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Operand {
    Register(Register),
    Address(Option<Register>, Option<Register>, Option<i16>),
    JumpDisplacement(i8),
    Empty,
    Immediate(i16),
}

pub enum OpCode {
    Mov,
    Add,
    Sub,
    Cmp,

    // jumps
    Je,
    Jl,
    Jle,
    Jb,
    Jbe,
    Jp,
    Jo,
    Js,
    Jne,
    Jnl,
    Jnle,
    Jnb,
    Jnbe,
    Jnp,
    Jno,
    Jns,
    Loop,
    Loopz,
    Loopnz,
    Jcxz,
}

impl OpCode {
    pub fn is_jump(&self) -> bool {
        use OpCode::*;
        match self {
            Je | Jl | Jle | Jb | Jbe | Jp | Jo | Js | Jne | Jnl | Jnle | Jnb | Jnbe | Jnp | Jno
            | Jns | Loop | Loopz | Loopnz | Jcxz => true,
            _ => false,
        }
    }
}

pub struct Instruction {
    pub op_code: OpCode,
    pub flags: InstructionFlags,
    pub left_operand: Operand,
    pub right_operand: Operand,
}
