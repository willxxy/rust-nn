mod examples;

use clap::{Arg, Command};

fn main() {
    let matches = Command::new("Rust NN Examples")
        .version("1.0")
        .author("Your Name <your.email@example.com>")
        .about("Runs different Rust NN examples")
        .arg(Arg::new("example")
            .short('e')
            .long("example")
            .takes_value(true)
            .help("Specifies which example to run: create_tensors, gpu_vs_cpu_add, gpu_vs_cpu_mm"))
        .get_matches();

    if let Some(example) = matches.value_of("example") {
        match example {
            "create" => {
                println!("Created Some Tensors!");
                examples::create_tensors::run();
            },
            "add" => {
                println!("GPU vs CPU Addition!");
                examples::gpu_vs_cpu_add::run();
            },
            "mm" => {
                println!("GPU vs CPU Matrix Multiplication!");
                examples::gpu_vs_cpu_mm::run();
            },
            _ => println!("Unknown example: {}", example),
        }
    } else {
        println!("Let's see some examples...");
        println!("Created Some Tensors!");
        examples::create_tensors::run();
        println!("---------------------------------");
        println!("GPU vs CPU Addition!");
        examples::gpu_vs_cpu_add::run();
        println!("---------------------------------");
        println!("GPU vs CPU Matrix Multiplication!");
        examples::gpu_vs_cpu_mm::run();
    }
}
