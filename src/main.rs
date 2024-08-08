use std::slice::Iter;

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
        Some(registers_sum) => {
            if displacement_number == 0 {
                format!("[{}]", registers_sum)
            } else {
                format!("[{}+{}]", registers_sum, displacement_number)
            }
        }
    }
}

fn mov_reg_to_reg(content_iter: &mut Iter<u8>, first: &u8, second: &u8) -> String {
    assert_eq!(first >> 2, 0b100010);
    let w = first & 0b1;
    let d = (first >> 1) & 0b1;

    let _mod = (second >> 6) & 0b11;
    let reg = (second >> 3) & 0b111;
    let rm = second & 0b111;

    match _mod {
        0b00 | 0b01 | 0b10 => {
            let address = &decode_address(content_iter, _mod, rm);

            let reg_name = decode_reg(reg, w);
            let left_reg_name = if d == 1 { reg_name } else { address };
            let right_reg_name = if d == 1 { address } else { reg_name };

            format!("mov {}, {}", left_reg_name, right_reg_name)
        }
        0b11 => {
            let reg_name = decode_reg(reg, w);
            let rm_reg_name = decode_reg(rm, w);

            let left_reg_name = if d == 1 { reg_name } else { rm_reg_name };
            let right_reg_name = if d == 1 { rm_reg_name } else { reg_name };

            format!("mov {}, {}", left_reg_name, right_reg_name)
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

fn main() {
    let Some(path) = std::env::args().skip(1).next() else {
        std::println!("Please provide assembly file path");
        return;
    };
    let Ok(content) = std::fs::read(path) else {
        std::println!("Failed to read file");
        return;
    };

    // if content.len() % 2 == 1 {
    //     println!("Content size must be even");
    //     return;
    // }
    let mut content_iter = content.iter();
    println!("bits 16\n");
    loop {
        let Some(first) = content_iter.next() else {
            return;
        };
        let second = content_iter.next().unwrap();

        println!(
            "{}",
            if first >> 1 == 0b1100011 {
                mov_immediate_to_rm(&mut content_iter, first, second)
            } else if first >> 2 == 0b100010 {
                mov_reg_to_reg(&mut content_iter, first, second)
            } else if first >> 4 == 0b1011 {
                mov_immediate_to_reg(&mut content_iter, first, second)
            } else {
                panic!("unknown operand")
            }
        )
    }
}
