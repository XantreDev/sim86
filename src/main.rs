const W_TO_REG_NAME: &'static [&'static [&str; 8]; 2] = &[
    &["al", "cl", "dl", "bl", "ah", "ch", "dh", "bh"],
    &["ax", "cx", "dx", "bx", "sp", "bp", "si", "di"],
];

fn main() {
    let Some(path) = std::env::args().skip(1).next() else {
        std::println!("Please provide assembly file path");
        return;
    };
    let Ok(content) = std::fs::read(path) else {
        std::println!("Failed to read file");
        return;
    };

    if content.len() % 2 == 1 {
        println!("Content size must be even");
        return;
    }
    let mut content_iter = content.iter();
    println!("bits 16\n");
    loop {
        let Some(first) = content_iter.next() else {
            return;
        };
        let second = content_iter.next().unwrap();

        assert_eq!(first >> 2, 0b100010);
        let w = first & 0b1;
        let d = (first >> 1) & 0b1;

        assert_eq!((second >> 6) & 0b11, 0b11);
        let reg = (second >> 3) & 0b111;
        let rm = second & 0b111;

        let reg_name = W_TO_REG_NAME[usize::from(w)][usize::from(reg)];
        let rm_reg_name = W_TO_REG_NAME[usize::from(w)][usize::from(rm)];

        let left_reg_name = if d == 1 { reg_name } else { rm_reg_name };
        let right_reg_name = if d == 1 { rm_reg_name } else { reg_name };

        println!("mov {}, {}", left_reg_name, right_reg_name);
    }
}
