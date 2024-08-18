use std::{io::Write, slice::Iter};

const W_TO_REG_NAME: &'static [&'static str; 24] = &[
    "al", "cl", "dl", "bl", "ah", "ch", "dh", "bh", // w = 0
    "ax", "cx", "dx", "bx", "sp", "bp", "si", "di", // w = 1
    "bx+si", "bx+di", "bp+si", "bp+di", "si", "di", "bp", "bx", // rm2registers
];

fn decode_reg(reg: u8, w: u8) -> &'static str {
    W_TO_REG_NAME[usize::from(w) * 8 + usize::from(reg)]
}

fn decode_address(content_iter: &mut Iter<u8>, _mod: u8, rm: u8) -> String {
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

    let registers_sum = if is_direct_address {
        None
    } else {
        Some(W_TO_REG_NAME[usize::from(16 + rm)].to_string())
    };

    match registers_sum {
        None => format!("[{}]", displacement_number),
        Some(registers_sum) if displacement_number == 0 => {
            format!("[{}]", registers_sum)
        }
        Some(registers_sum) if displacement_bytes == 1 && displacement_number >= 128 => {
            format!(
                "[{}-{}]",
                registers_sum,
                (1 + !displacement_number) & 0b11111111
            )
        }
        Some(registers_sum) if displacement_bytes == 2 && displacement_number >= 1 << 15 => {
            format!("[{}-{}]", registers_sum, 1 + !displacement_number)
        }
        Some(registers_sum) => {
            format!("[{}+{}]", registers_sum, displacement_number)
        }
    }
}

fn rm_tf_reg(content_iter: &mut Iter<u8>, first: &u8, second: &u8) -> String {
    let w = first & 0b1;
    let d = (first >> 1) & 0b1;

    let _mod = (second >> 6) & 0b11;
    let reg = (second >> 3) & 0b111;
    let rm = second & 0b111;

    match _mod {
        0b00..=0b10 => {
            let address = &decode_address(content_iter, _mod, rm);

            let reg_name = decode_reg(reg, w);
            let left_reg_name = if d == 1 { reg_name } else { address };
            let right_reg_name = if d == 1 { address } else { reg_name };

            format!("{}, {}", left_reg_name, right_reg_name)
        }
        0b11 => {
            let reg_name = decode_reg(reg, w);
            let rm_reg_name = decode_reg(rm, w);

            let left_reg_name = if d == 1 { reg_name } else { rm_reg_name };
            let right_reg_name = if d == 1 { rm_reg_name } else { reg_name };

            format!("{}, {}", left_reg_name, right_reg_name)
        }
        _ => {
            panic!("Unknown mod");
        }
    }
}

fn mov_immediate_to_rm(content_iter: &mut Iter<u8>, first: &u8, second: &u8) -> String {
    let w = first & 0b1;

    assert!(second & 0b00111000 == 0);
    let _mod = second >> 6 & 0b11;
    let rm = second & 0b111;

    let address = decode_address(content_iter, _mod, rm);

    let mut immediate: u16 = 0;
    immediate |= u16::from(*content_iter.next().unwrap());
    if w == 0b1 {
        immediate |= u16::from(*content_iter.next().unwrap()) << 8;
    }

    let word_type = if w == 0b1 { "word" } else { "byte" };

    format!("mov {}, {} {}", address, word_type, immediate)
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
    let to_acc = first >> 1 & 0b1 == 0;

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

fn instruction_of_first_byte(first: &u8) -> &'static str {
    match first {
        0b1100_0110..=0b1100_0111
        | 0b1010_0000..=0b1010_0011
        | 0b1000_1000..=0b1000_1011
        | 0b1011_0000..=0b1011_1111 => "mov",
        0b0000_0000..=0b0000_0011 | 0b1000_000..=0b1000_0011 | 0b0000_0100..=0b0000_0101 => "add",
        _ => panic!("unknown instruction"),
    }
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
            match first {
                0b1000_1000..=0b1000_1011 =>
                    format!("mov {}", rm_tf_reg(&mut content_iter, first, second)),
                0b0000_0000..=0b0000_0011 =>
                    format!("add {}", rm_tf_reg(&mut content_iter, first, second)),
                0b1100_0110..=0b1100_0111 => mov_immediate_to_rm(&mut content_iter, first, second),
                0b1010_0000..=0b1010_0011 => mov_acc(&mut content_iter, first, second),
                0b1011_0000..=0b1011_1111 => mov_immediate_to_reg(&mut content_iter, first, second),
                _ => panic!("unknown operand"),
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
        ];

        for file_path in files {
            if !Path::new(file_path).exists() {
                let asm_path = format!("{}.asm", file_path);
                assert!(Path::new(&asm_path).exists());

                Command::new("nasm").args([asm_path]).output().unwrap();
                assert!(Path::new(&file_path).exists());
            }
            let file = read_file(file_path);
            let mut res: Vec<u8> = Vec::new();

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
