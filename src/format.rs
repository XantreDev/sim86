use super::structs::*;

pub trait Formattable {
    fn format(&self) -> String;
}

impl Formattable for Register {
    fn format(&self) -> String {
        match self {
            Register::Ax => "ax",
            Register::Ah => "ah",
            Register::Al => "al",

            Register::Bx => "bx",
            Register::Bh => "bh",
            Register::Bl => "bl",
            Register::Bp => "bp",

            Register::Cx => "cx",
            Register::Cl => "cl",
            Register::Ch => "ch",
            Register::Cs => "cs",

            Register::Dx => "dx",
            Register::Dl => "dl",
            Register::Dh => "dh",
            Register::Di => "di",

            Register::Ip => "ip",
            Register::Si => "si",
            Register::Sp => "sp",
        }
        .to_string()
    }
}
impl Formattable for OpCode {
    fn format(&self) -> String {
        match self {
            OpCode::Add => "add",
            OpCode::Cmp => "cmp",
            OpCode::Mov => "mov",
            OpCode::Sub => "sub",

            OpCode::Je => "je",
            OpCode::Jl => "jl",
            OpCode::Jle => "jle",
            OpCode::Jb => "jb",
            OpCode::Jbe => "jbe",
            OpCode::Jp => "jp",
            OpCode::Jo => "jo",
            OpCode::Js => "js",
            OpCode::Jne => "jne",
            OpCode::Jnl => "jnl",
            OpCode::Jnle => "jnle",
            OpCode::Jnb => "jnb",
            OpCode::Jnbe => "jnbe",
            OpCode::Jnp => "jnp",
            OpCode::Jno => "jno",
            OpCode::Jns => "jns",
            OpCode::Loop => "loop",
            OpCode::Loopz => "loopz",
            OpCode::Loopnz => "loopnz",
            OpCode::Jcxz => "jcxz",
        }
        .to_string()
    }
}

impl Formattable for Operand {
    fn format(&self) -> String {
        fn pad_displacement(immediate: &Option<i16>) -> String {
            match immediate {
                Some(imm) if *imm > 0 => format!("+{}", imm),
                Some(imm) if *imm < 0 => imm.to_string(),
                _ => "".to_string(),
            }
        }
        match self {
            Self::Register(reg) => reg.format(),

            Self::Address(Some(a), Option::None, Option::None) => format!("[{}]", a.format()),
            Self::Address(Option::None, Option::None, Some(displacement)) => {
                format!("[{}]", displacement)
            }
            Self::Address(Some(a), Option::None, immediate) => {
                format!("[{}{}]", a.format(), pad_displacement(immediate))
            }
            Self::Address(Some(a), Some(b), immediate) => {
                format!(
                    "[{}+{}{}]",
                    a.format(),
                    b.format(),
                    pad_displacement(immediate)
                )
            }
            Self::Address(_, _, _) => panic!("unsupported {:?}", self),

            Self::Empty => "".to_string(),
            Self::JumpDisplacement(displacement) => {
                if displacement + 2 > 0 {
                    format!("+{}+0", displacement + 2)
                } else if displacement + 2 == 0 {
                    "+0".to_string()
                } else {
                    format!("{}+0", displacement + 2)
                }
            }
            Self::Immediate(immediate) => immediate.to_string(),
        }
    }
}

impl Formattable for Instruction {
    fn format(&self) -> String {
        if self.op_code.is_jump() {
            assert!(matches!(self.right_operand, Operand::Empty));
            assert!(matches!(self.right_operand, Operand::JumpDisplacement(_)));
            return format!("{} {}", self.op_code.format(), self.left_operand.format());
        }

        format!(
            "{} {}, {}",
            self.op_code.format(),
            self.left_operand.format(),
            self.right_operand.format()
        )
    }
}
