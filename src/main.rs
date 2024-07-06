mod examples;

fn main() {
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