# rust-nn

Exploration of deep learning through rust, more specifically through [tracel-ai's burn deep learning framework](https://github.com/tracel-ai/burn).

After some experimentation, found that [tch-rs](https://github.com/LaurentMazare/tch-rs) may be better for now (more stable).
So will switch the focus to that.

Tested on macOS Sonoma version 14.5, Apple M2 Max, Macbook Pro and rustc version 1.79.0.

## Usage
1. `git clone https://github.com/willxxy/rust-nn.git`
2. `cd rust-nn`
3. `cargo run`

Note: Make sure to have [rust installed](https://rustup.rs/).

## Current Examples

All examples are under `./src/examples`

Current examples include:

1. `create_tensors.rs` - Simple tensor creation and addition provided in the [burn book tutorial](https://burn.dev/book/getting-started.html).

2. `gpu_vs_cpu_add.rs` - Testing GPU vs CPU performance times for 1000x1000 matrix addition for 100 iterations.

3. `gpu_vs_cpu_mm.rs` - Testing GPU vs CPU performance times for 1000x1000 matrix multiplication for 100 iterations.

Note: The matrices created for GPU is done through burn and the matrices created for CPU is done through ndarray. Additionally, the .matmul() operation for burn and .dot() operation for ndarray are a bit different so this doesn't make it a completely fair comparison...

## GPU vs CPU Performance

Note that there is no warmup for the GPU.

### Addition

| Device | Duration per Iteration | Duration for 100 Iterations |
|--------|------------------------|-----------------------------|
| **GPU** | 15.457µs              | 1.545708ms                 |
| **CPU** | 26.682145ms            | 2.668214584s               |

### Matrix Multiplication

| Device | Duration per Iteration | Duration for 100 Iterations |
|--------|------------------------|-----------------------------|
| **GPU** | 321.857µs              | 32.18575ms                  |
| **CPU** | 2.09207541s           | 209.207541041s              |


## TODO
[ ] Switch to tch-rs
[ ] Create MNIST tutorial.

