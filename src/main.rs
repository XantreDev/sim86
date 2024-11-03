use bitflags::bitflags;
use std::{io::Write, slice::Iter};

bitflags! {
    struct InstructionFlags: u8 {
        const Lock = 0b0000_0001;
        const Rep = 0b0000_0010;
        const Segment = 0b0000_0100;
        const Wide = 0b0000_1000;
    }
}

const W_TO_REG_NAME: &'static [&'static str; 24] = &[
    "al", "cl", "dl", "bl", "ah", "ch", "dh", "bh", // w = 0
    "ax", "cx", "dx", "bx", "sp", "bp", "si", "di", // w = 1
    "bx+si", "bx+di", "bp+si", "bp+di", "si", "di", "bp", "bx", // rm2registers
];

#[rustfmt::skip]
#[derive(Debug, Clone, Copy)]
enum Register {
    Ax, Al, Ah,
    Bx, Bl, Bh,
    Cx, Cl, Ch,
    Dx, Dl, Dh,

    Si,
    Di,
    Bp,
    Sp,
    Ip,
    Cs,
}
#[derive(Debug, Clone, Copy)]
struct Operand {
    a: Option<Register>,
    b: Option<Register>,
    immidiate: Option<i16>,
}

impl Operand {
    fn is_empty(&self) -> bool {
        match self {
            Operand {
                a: None,
                b: None,
                immidiate: None,
            } => true,
            _ => false,
        }
    }
    fn empty() -> Operand {
        Operand {
            a: None,
            b: None,
            immidiate: None,
        }
    }

    fn register(register: Register) -> Operand {
        Operand {
            a: Some(register),
            b: None,
            immidiate: None,
        }
    }
    fn immidiate(immidiate: i16) -> Operand {
        Operand {
            a: None,
            b: None,
            immidiate: Some(immidiate),
        }
    }
}

enum OpCode {
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

struct Instruction {
    op_code: OpCode,
    left_operand: Operand,
    right_operand: Operand,
}

trait Formattable {
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
        fn pad_immidiate(immediate: Option<i16>) -> String {
            match immediate {
                Some(imm) if imm > 0 => format!("+{}", imm),
                Some(imm) if imm < 0 => imm.to_string(),
                _ => "".to_string(),
            }
        }
        match (&self.a, &self.b, self.immidiate) {
            (Some(a), Some(b), immediate) => {
                format!("{}+{}{}", a.format(), b.format(), pad_immidiate(immediate))
            }
            (Some(a), None, None) => a.format(),
            (None, None, Some(immediate)) => immediate.to_string(),
            _ => panic!("unsupported {:?}", self),
        }
    }
}

fn decode_reg(reg: u8, w: u8) -> &'static Register {
    #[rustfmt::skip]
    const W_TO_REG: &'static[Register] = &[
        Register::Al, Register::Cl, Register::Dl, Register::Bl, Register::Ah, Register::Ch , Register::Dh, Register::Bh, // w = 0
        Register::Ax, Register::Cx, Register::Dx, Register::Bx, Register::Sp, Register::Bp, Register::Si, Register::Di, // w = 1
    ];

    &W_TO_REG[usize::from(w) * 8 + usize::from(reg)]
}

// enum AddressOrReg {
//     Reg(Register),
//     Address(i16),
// }

fn decode_address(content_iter: &mut Iter<u8>, _mod: u8, rm: u8) -> Operand {
    let mut displacement_number: u16 = 0;
    let is_direct_address = _mod == 0b00 && rm == 0b110;
    let displacement_bytes = if is_direct_address { 2 } else { _mod };
    match displacement_bytes {
        2 => {
            displacement_number |= u16::from(*(content_iter.next().unwrap()));
            displacement_number |= u16::from(*(content_iter.next().unwrap())) << 8;
        }
        1 => {
            displacement_number |= u16::from(*(content_iter.next().unwrap()));
        }
        _ => {}
    }

    if is_direct_address {
        return Operand {
            a: None,
            b: None,
            immidiate: Some(displacement_number as i16),
        };
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

    match (displacement_number, displacement_bytes) {
        (0, _) => Operand {
            a: Some(*a_reg),
            b: *b_reg,
            immidiate: None,
        },
        (displacement_number, 1) if displacement_number >= 1 << 7 => Operand {
            a: Some(*a_reg),
            b: *b_reg,
            immidiate: Some(-(displacement_number as i16)),
        },
        (displacement_number, _) => Operand {
            a: Some(*a_reg),
            b: *b_reg,
            immidiate: Some(displacement_number as i16),
        },
    }
}

struct Operands {
    left_operand: Operand,
    right_operand: Operand,
}

fn rm_to_reg(content_iter: &mut Iter<u8>, first: &u8, second: &u8) -> Operands {
    let w = first & 0b1;
    let d = (first >> 1) & 0b1;

    let _mod = (second >> 6) & 0b11;
    let reg = (second >> 3) & 0b111;
    let rm = second & 0b111;

    let address_or_reg = match _mod {
        0b00..=0b10 => decode_address(content_iter, _mod, rm),
        0b11 => Operand {
            a: Some(*decode_reg(rm, w)),
            b: None,
            immidiate: None,
        },
        _ => {
            panic!("invalid mod processing");
        }
    };

    let reg_name = decode_reg(reg, w);
    let (left, right) = if d == 1 {
        (reg_name, address_or_reg)
    } else {
        (address_or_reg, reg_name)
    };

    format!("{}, {}", left, right)
}

struct Im2Rm {
    address: String,
    s: bool,
    w: bool,
    immediate: u16,
}

impl Im2Rm {
    // mov have no s byte. it always 1 in that case
    fn new(content_iter: &mut Iter<u8>, first: &u8, second: &u8, is_arthmetic: bool) -> Operand {
        let s = first >> 1 & 0b1;
        let w = first & 0b1;

        // assert!(second & 0b00111000 == 0);
        let _mod = second >> 6 & 0b11;
        let rm = second & 0b111;

        let address_or_reg = match _mod {
            0b00..=0b10 => decode_address(content_iter, _mod, rm),
            0b11 => Operand {
                a: Some(*decode_reg(rm, w)),
                b: None,
                immidiate: None,
            },
            _ => {
                panic!("invalid mod processing");
            }
        };

        let mut immediate: u16 = 0;
        immediate |= u16::from(*content_iter.next().unwrap());

        if w == 0b1 && (!is_arthmetic || s == 0b0) {
            immediate |= u16::from(*content_iter.next().unwrap()) << 8;
        }

        Im2Rm {
            s: s == 0b1,
            w: w == 0b1,
            immediate,
            address: address_or_reg.into(),
        }
    }

    fn format_as_mov(&self) -> String {
        let word_type = if self.w { "word" } else { "byte" };
        format!("{}, {} {}", self.address, word_type, self.immediate)
    }
    fn format_as_arithmetic(&self) -> String {
        if self.address.starts_with('[') {
            let word_type = if self.s { "word" } else { "byte" };
            format!("{} {}, {}", word_type, self.address, self.immediate)
        } else {
            format!("{}, {}", self.address, self.immediate)
        }
    }
}

fn mov_immediate_to_reg(content_iter: &mut Iter<u8>, first: &u8, second: &u8) -> String {
    assert!(first >> 4 == 0b1011);

    let w = first >> 3 & 0b1;
    let reg = first & 0b111;
    let mut data: u16 = 0;

    data |= u16::from(*second);
    if w == 0b1 {
        data |= u16::from(*content_iter.next().unwrap()) << 8;
    }

    let reg_name = decode_reg(reg, w);

    format!("mov {}, {}", reg_name, data)
}

fn mov_acc(content_iter: &mut Iter<u8>, first: &u8, second: &u8) -> String {
    assert!(first >> 2 == 0b101000);
    let w = first & 0b1;
    let to_acc = (first >> 1 & 0b1) == 0;

    let mut displacement_number: u16 = u16::from(*(second));
    if w == 1 {
        displacement_number |= u16::from(*(content_iter.next().unwrap())) << 8;
    }

    if to_acc {
        format!("mov ax, [{}]", displacement_number)
    } else {
        format!("mov [{}], ax", displacement_number)
    }
}

fn im_to_acc(content_iter: &mut Iter<u8>, first: &u8, second: &u8) -> Operands {
    let w = first & 0b1;

    let mut displacement_number: u16 = u16::from(*(second));
    if w == 1 {
        displacement_number |= u16::from(*(content_iter.next().unwrap())) << 8;
    }

    Operands {
        left_operand: Operand::register(if w == 0 { Register::Al } else { Register::Ax }),
        right_operand: Operand::immidiate(displacement_number as i16),
    }
    // format!("al, {}", displacement_number)
}

// rm - stands for Register or Memory
enum ArithmeticInstructionType {
    Sub,
    Add,
    Cmp,
}
enum ArithmeticInstructionAction {
    RmToEither,
    ImToRm,
    ImToAcc,
}

struct ArithmeticInstruction {
    t: ArithmeticInstructionType,
    action: ArithmeticInstructionAction,
}

impl ArithmeticInstruction {
    fn from(first: &u8, second: &u8) -> Option<ArithmeticInstruction> {
        let is_immediate_to_rm = first >> 2 == 0b10_0000;
        let middle_bits_to_operation = |byte: &u8| match (byte >> 3) & 0b111 {
            0b000 => Some(ArithmeticInstructionType::Add),
            0b101 => Some(ArithmeticInstructionType::Sub),
            0b111 => Some(ArithmeticInstructionType::Cmp),
            _ => None,
        };

        if is_immediate_to_rm {
            return middle_bits_to_operation(second).map(|it| ArithmeticInstruction {
                t: it,
                action: ArithmeticInstructionAction::ImToRm,
            });
        }

        // exluding middle byte
        match first & 0b11_000_111 {
            0b00_000_000..=0b00_000_011 => {
                middle_bits_to_operation(first).map(|it| ArithmeticInstruction {
                    t: it,
                    action: ArithmeticInstructionAction::RmToEither,
                })
            }
            0b00_000_100..=0b00_000_101 => {
                middle_bits_to_operation(first).map(|it| ArithmeticInstruction {
                    t: it,
                    action: ArithmeticInstructionAction::ImToAcc,
                })
            }
            _ => None,
        }
    }

    fn parse_and_format(self, first: &u8, second: &u8, content_iter: &mut Iter<u8>) -> String {
        let operation = match &self.t {
            ArithmeticInstructionType::Add => "add",
            ArithmeticInstructionType::Cmp => "cmp",
            ArithmeticInstructionType::Sub => "sub",
        };

        let content = match self.action {
            ArithmeticInstructionAction::ImToAcc => im_to_acc(content_iter, first, second),
            ArithmeticInstructionAction::ImToRm => {
                Im2Rm::new(content_iter, first, second, true).format_as_arithmetic()
            }
            ArithmeticInstructionAction::RmToEither => rm_to_reg(content_iter, first, second),
        };

        format!("{} {}", operation, content)
    }
}

fn parse_and_format_jump(first: &u8, second: &u8) -> Option<String> {
    let Some(jump_code) = (match first {
        0b0111_0100 => Some("je"),
        0b0111_1100 => Some("jl"),
        0b0111_1110 => Some("jle"),
        0b0111_0010 => Some("jb"),
        0b0111_0110 => Some("jbe"),
        0b0111_1010 => Some("jp"),
        0b0111_0000 => Some("jo"),
        0b0111_1000 => Some("js"),
        0b0111_0101 => Some("jne"),
        0b0111_1101 => Some("jnl"),
        0b0111_1111 => Some("jnle"),
        0b0111_0011 => Some("jnb"),
        0b0111_0111 => Some("jnbe"),
        0b0111_1011 => Some("jnp"),
        0b0111_0001 => Some("jno"),
        0b0111_1001 => Some("jns"),
        0b1110_0010 => Some("loop"),
        0b1110_0001 => Some("loopz"),
        0b1110_0000 => Some("loopnz"),
        0b1110_0011 => Some("jcxz"),
        _ => None,
    }) else {
        return None;
    };

    let signed_second = *second as i8;
    Some(format!(
        "{} ${}",
        jump_code,
        if signed_second + 2 > 0 {
            format!("+{}+0", signed_second + 2)
        } else if signed_second + 2 == 0 {
            "+0".to_string()
        } else {
            format!("{}+0", signed_second + 2)
        }
    ))
}

fn process_binary<T: Write>(mut content_iter: Iter<u8>, out: &mut T) {
    writeln!(out, "bits 16\n").unwrap();
    loop {
        let Some(first) = content_iter.next() else {
            return;
        };
        let second = content_iter.next().unwrap();

        writeln!(
            out,
            "{}",
            if let Some(arimthmetic_inst) = ArithmeticInstruction::from(first, second) {
                arimthmetic_inst.parse_and_format(first, second, &mut content_iter)
            } else if let Some(formatted_instr) = parse_and_format_jump(first, second) {
                formatted_instr
            } else {
                match first {
                    0b1100_0110..=0b1100_0111 => format!(
                        "mov {}",
                        Im2Rm::new(&mut content_iter, first, second, false).format_as_mov()
                    ),
                    0b1000_1000..=0b1000_1011 => {
                        format!("mov {}", rm_to_reg(&mut content_iter, first, second))
                    }
                    0b1010_0000..=0b1010_0011 => mov_acc(&mut content_iter, first, second),
                    0b1011_0000..=0b1011_1111 => {
                        mov_immediate_to_reg(&mut content_iter, first, second)
                    }
                    _ => panic!("unknown operand: {:08b}", first),
                }
            }
        )
        .unwrap()
    }
}

fn disassemble<T: Write>(mut content_iter: Iter<u8>, out: &mut T) {
    writeln!(out, "bits 16\n").unwrap();
    loop {
        let Some(first) = content_iter.next() else {
            return;
        };
        let second = content_iter.next().unwrap();

        writeln!(
            out,
            "{}",
            if let Some(arimthmetic_inst) = ArithmeticInstruction::from(first, second) {
                arimthmetic_inst.parse_and_format(first, second, &mut content_iter)
            } else if let Some(formatted_instr) = parse_and_format_jump(first, second) {
                formatted_instr
            } else {
                match first {
                    0b1100_0110..=0b1100_0111 => format!(
                        "mov {}",
                        Im2Rm::new(&mut content_iter, first, second, false).format_as_mov()
                    ),
                    0b1000_1000..=0b1000_1011 => {
                        format!("mov {}", rm_to_reg(&mut content_iter, first, second))
                    }
                    0b1010_0000..=0b1010_0011 => mov_acc(&mut content_iter, first, second),
                    0b1011_0000..=0b1011_1111 => {
                        mov_immediate_to_reg(&mut content_iter, first, second)
                    }
                    _ => panic!("unknown operand: {:08b}", first),
                }
            }
        )
        .unwrap()
    }
}

fn read_file<T: Into<String>>(file_path: T) -> Vec<u8> {
    let Ok(content) = std::fs::read(file_path.into()) else {
        panic!("Failed to read file");
    };

    content
}
fn main() {
    let Some(path) = std::env::args().skip(1).next() else {
        std::println!("Please provide assembly file path");
        return;
    };

    let content = read_file(path);
    let content_iter = content.iter();
    process_binary(content_iter, &mut std::io::stdout().lock());
}

#[cfg(test)]
mod tests {
    use std::{fs::create_dir, fs::File, io::Write, path::Path, process::Command};

    use crate::{process_binary, read_file};

    #[test]
    fn test_files() {
        let files = [
            "./input/listing_0037_single_register_mov",
            "./input/listing_0038_many_register_mov",
            "./input/listing_0039_more_movs",
            "./input/listing_0040_challenge_movs",
            "./input/listing_0041_add_sub_cmp_jnz",
        ];

        for file_path in files {
            if std::env::var("RECOMPILE").map_or(false, |it| it == "true")
                || !Path::new(file_path).exists()
            {
                let asm_path = format!("{}.asm", file_path);
                assert!(Path::new(&asm_path).exists());

                Command::new("nasm").args([asm_path]).output().unwrap();
                assert!(Path::new(&file_path).exists());
            }
            let file = read_file(file_path);
            let mut res: Vec<u8> = Vec::new();
            println!("checking '{}'", file_path);

            process_binary(file.iter(), &mut res);

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
