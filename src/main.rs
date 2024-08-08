use std::slice::Iter;

const W_TO_REG_NAME: &'static [&'static str; 24] = &[
    "al", "cl", "dl", "bl", "ah", "ch", "dh", "bh", // w = 0
    "ax", "cx", "dx", "bx", "sp", "bp", "si", "di", // w = 1
    "bx+si", "bx+di", "bp+si", "bp+di", "si", "di", "bp", "bx", // rm2registers
];

fn decode_reg(reg: u8, w: u8) -> &'static str {
    W_TO_REG_NAME[usize::from(w) * 8 + usize::from(reg)]
}

fn mov_to_register_or_memory(content_iter: &mut Iter<u8>, first: &u8, second: &u8) -> String {
    assert_eq!(first >> 2, 0b100010);
    let w = first & 0b1;
    let d = (first >> 1) & 0b1;

    let _mod = (second >> 6) & 0b11;
    let reg = (second >> 3) & 0b111;
    let rm = second & 0b111;

    match _mod {
        0b00 | 0b01 | 0b10 => {
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
                ""
            } else {
                W_TO_REG_NAME[usize::from(16 + rm)]
            };
            let address = &(if registers_sum.len() == 0 {
                format!("[{}]", displacement_number)
            } else if displacement_number == 0 {
                format!("[{}]", registers_sum)
            } else {
                format!("[{}+{}]", registers_sum, displacement_number)
            });

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

fn mov_to_register(content_iter: &mut Iter<u8>, first: &u8, second: &u8) -> String {
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
            if first >> 2 == 0b100010 {
                mov_to_register_or_memory(&mut content_iter, first, second)
            } else if first >> 4 == 0b1011 {
                mov_to_register(&mut content_iter, first, second)
            } else {
                panic!("unknown operand")
            }
        )
    }
}
