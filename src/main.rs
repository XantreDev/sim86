use std::{fs, io::Write, path::Path};

mod decoder;
mod estimation;
mod format;
mod simulator;
mod structs;

use structs::*;

use format::Formattable;

use crate::estimation::Architecture;

fn process_binary<T: Iterator<Item = u8>>(mut content_iter: T) -> Vec<Instruction> {
    use decoder;

    let mut result: Vec<Instruction> = Vec::with_capacity(2048);
    loop {
        let Some(instr) = decoder::decode_instruction(&mut content_iter) else {
            return result;
        };
        result.push(instr);
    }
}

fn write_instructions<T: Write>(instructions: Vec<Instruction>, out: &mut T) {
    writeln!(out, "bits 16\n").unwrap();
    instructions.iter().for_each(|it| {
        out.write((it.format() + "\n").as_bytes())
            .expect("must write");
    });
}

fn read_file<T: Into<String>>(file_path: T) -> Vec<u8> {
    let Ok(content) = std::fs::read(file_path.into()) else {
        panic!("Failed to read file");
    };

    content
}

struct DumpMemoryParams {
    path: String,
}
struct EstimateCyclesMode {
    mode: Architecture,
}

enum CliMode {
    Disasm,
    Exec(Option<DumpMemoryParams>, Option<EstimateCyclesMode>),
}

enum CliOptionState<T: Sized> {
    Unseen,
    SeenBefore,
    Parsed(T),
}

impl<T: Sized> CliOptionState<T> {
    fn is_seen_before(self: &Self) -> bool {
        match self {
            Self::SeenBefore => true,
            _ => false,
        }
    }
}

impl<T: Sized> Default for CliOptionState<T> {
    fn default() -> Self {
        CliOptionState::Unseen
    }
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<String>>();
    let mut positional_args = args.iter().filter(|it| !it.starts_with("--"));
    let mode = match positional_args.next() {
        Some(str) if str == "disasm" => CliMode::Disasm,
        Some(str) if str == "exec" => {
            let mut iter = args.iter();

            let mut dump_cli_flag: CliOptionState<DumpMemoryParams> = CliOptionState::default();

            let mut cycles_estimation_option: CliOptionState<EstimateCyclesMode> =
                CliOptionState::default();

            loop {
                let Some(value) = iter.next() else {
                    break;
                };

                if matches!(dump_cli_flag, CliOptionState::Unseen) && value == "--dump" {
                    dump_cli_flag = CliOptionState::SeenBefore
                } else if matches!(dump_cli_flag, CliOptionState::SeenBefore) {
                    dump_cli_flag = CliOptionState::Parsed(DumpMemoryParams {
                        path: value.clone(),
                    });
                } else if matches!(cycles_estimation_option, CliOptionState::Unseen)
                    && value == "--estimate-cycles"
                {
                    cycles_estimation_option = CliOptionState::SeenBefore;
                } else if matches!(cycles_estimation_option, CliOptionState::SeenBefore)
                    && value == "8086"
                {
                    cycles_estimation_option = CliOptionState::Parsed(EstimateCyclesMode {
                        mode: Architecture::X8086,
                    });
                } else if matches!(cycles_estimation_option, CliOptionState::SeenBefore)
                    && value == "8088"
                {
                    cycles_estimation_option = CliOptionState::Parsed(EstimateCyclesMode {
                        mode: Architecture::X8088,
                    })
                } else if matches!(cycles_estimation_option, CliOptionState::SeenBefore) {
                    panic!("invalid option for cycle estimation: {}", value)
                }
            }
            let dump_params = match dump_cli_flag {
                CliOptionState::Unseen => None,
                CliOptionState::SeenBefore => panic!("no path for dump argument"),
                CliOptionState::Parsed(params) => Some(params),
            };
            let cycles_estimation_param = match cycles_estimation_option {
                CliOptionState::Unseen => None,
                CliOptionState::SeenBefore => {
                    panic!("no value for cycles estimation option (expected '8086' or '8088')")
                }
                CliOptionState::Parsed(params) => Some(params),
            };
            CliMode::Exec(dump_params, cycles_estimation_param)
        }
        Some(str) => {
            std::println!("Unknown mode: {}", str);
            return;
        }
        _ => {
            std::println!("Please at least two args");
            return;
        }
    };
    let Some(path) = positional_args.last() else {
        std::println!("Please provide assembly file path");
        return;
    };

    let content = read_file(path);
    #[cfg(target_endian = "big")]
    println!("This is a BigEndian system");

    #[cfg(target_endian = "little")]
    println!("This is a Little endian system");

    let mut stdout = &mut std::io::stdout().lock();
    match mode {
        CliMode::Disasm => {
            let content_iter = content.iter().map(|it| *it);
            let instrucitons = process_binary(content_iter);
            write_instructions(instrucitons, &mut stdout);
        }
        CliMode::Exec(dump_params, estimation) => {
            use crate::simulator::{execute, SimulatorConfig};
            let memory = execute(
                content,
                Some(stdout),
                SimulatorConfig {
                    cycle_estimation_mode: estimation.map(|it| it.mode),
                },
            );
            match dump_params {
                Some(DumpMemoryParams { path }) => {
                    let mut file = fs::File::create(Path::new(&path)).expect("should open file");
                    let dump = memory.dump();
                    file.write_all(dump).expect("should write succesfully");
                }
                None => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core::str;
    use std::{
        fs,
        io::Write,
        path::Path,
        process::{Command, ExitStatus},
    };

    use crate::{estimation::Architecture, process_binary, read_file, write_instructions};

    fn compile_asm_if_not(path: &str, force_recompile: bool) {
        let is_exist = Path::new(path).exists();
        if is_exist && force_recompile {
            fs::remove_file(Path::new(path)).expect("must delete");
        };
        if force_recompile || !is_exist {
            let asm_path = format!("{}.asm", path);
            assert!(Path::new(&asm_path).exists());

            let execution_result = Command::new("nasm").args([asm_path]).output().unwrap();
            if !execution_result.status.success() {
                let help = str::from_utf8(execution_result.stderr.as_slice())
                    .unwrap_or("<failed to parse stdout>");
                panic!("failed to compile asm:\n {}", help);
            }
            assert!(execution_result.status.success());
            assert!(Path::new(&path).exists());
        }
    }

    #[test]
    fn test_parser() {
        let files = [
            "./input/listing_0037_single_register_mov",
            "./input/listing_0038_many_register_mov",
            "./input/listing_0039_more_movs",
            "./input/listing_0040_challenge_movs",
            "./input/listing_0041_add_sub_cmp_jnz",
            "./input/listing_0043_immediate_movs",
            "./input/listing_0044_register_movs",
            "./input/listing_0044_register_movs_extended",
        ];
        let force_recompilation = std::env::var("RECOMPILE").map_or(false, |it| it == "true");

        for file_path in files {
            compile_asm_if_not(file_path, force_recompilation);
            let file = read_file(file_path);
            let mut res: Vec<u8> = Vec::new();
            println!("checking '{}'", file_path);

            write_instructions(process_binary(file.iter().map(|it| *it)), &mut res);

            if !Path::new("./.test").exists() {
                fs::create_dir("./.test").unwrap();
            }

            fs::File::create("./.test/test.asm")
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

    #[test]
    fn test_execution() {
        let files = [
            "./input/listing_0043_immediate_movs",
            "./input/listing_0044_register_movs",
            "./input/listing_0044_register_movs_extended",
            "./input/listing_0046_add_sub_cmp",
            "./input/listing_0047_challenge_flags",
            "./input/listing_0048_ip_register",
            "./input/listing_0049_conditional_jumps",
            "./input/listing_0050_challenge_jumps",
            "./input/listing_0051_memory_mov",
            "./input/listing_0052_memory_add_loop",
            "./input/listing_0053_add_loop_challenge",
            "./input/listing_0054_draw_rectangle",
            "./input/listing_0055_challenge_rectangle",
        ];
        let force_recompilation = std::env::var("RECOMPILE").map_or(false, |it| it == "true");

        for file_path in files {
            compile_asm_if_not(file_path, force_recompilation);
            let file = read_file(file_path);
            let mut res: Vec<u8> = Vec::new();
            println!("evaluating '{}'", file_path);

            use crate::simulator::{execute, SimulatorConfig};
            execute(file, Some(&mut res), SimulatorConfig::default());
            insta::assert_snapshot!(str::from_utf8(res.as_slice()).unwrap());
        }
    }

    #[test]
    fn test_estimation() {
        let files = [
            "./input/listing_0056_estimating_cycles",
            "./input/listing_0057_challenge_cycles",
        ];
        let force_recompilation = std::env::var("RECOMPILE").map_or(false, |it| it == "true");

        for file_path in files {
            compile_asm_if_not(file_path, force_recompilation);
            let file = read_file(file_path);
            let mut res = Vec::new();

            write!(res, "X8086\n\n").unwrap();
            use crate::simulator::{execute, SimulatorConfig};
            execute(
                file.clone(),
                Some(&mut res),
                SimulatorConfig {
                    cycle_estimation_mode: Some(Architecture::X8086),
                },
            );

            write!(res, "\n\nX8088\n\n").unwrap();

            execute(
                file,
                Some(&mut res),
                SimulatorConfig {
                    cycle_estimation_mode: Some(Architecture::X8088),
                },
            );

            insta::assert_snapshot!(str::from_utf8(res.as_slice()).unwrap());
        }
    }
}
