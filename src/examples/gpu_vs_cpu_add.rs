use burn::tensor::Tensor;
use burn::backend::Wgpu;
use burn::tensor::Shape;
use burn::tensor::Distribution;
use ndarray::Array;
use std::time::Instant;
use ndarray_rand::{rand_distr::Uniform, RandomExt};

type BackendGpu = Wgpu;

pub fn run() {
    let device = Default::default();
    
    const MATRIX_SIZE: usize = 1000;
    const ITERATIONS: usize = 100;

    // GPU
    let tensor_1_gpu = Tensor::<BackendGpu, 2>::random(
        Shape::new([MATRIX_SIZE, MATRIX_SIZE]),
        Distribution::Uniform(0.0, 1.0),
        &device
    );

    let tensor_2_gpu = Tensor::<BackendGpu, 2>::random(
        Shape::new([MATRIX_SIZE, MATRIX_SIZE]),
        Distribution::Uniform(0.0, 1.0),
        &device
    );

    let start_gpu = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = tensor_1_gpu.clone().add(tensor_2_gpu.clone());
    }
    let duration_gpu = start_gpu.elapsed();

    println!("GPU duration: {:?} per iteration", duration_gpu / ITERATIONS as u32);
    println!("GPU duration: {:?} for {:?} iterations", duration_gpu, ITERATIONS as u32);

    // CPU
    let array_1_cpu = Array::<f32, _>::random((MATRIX_SIZE, MATRIX_SIZE), Uniform::new(0.0, 1.0));
    let array_2_cpu = Array::<f32, _>::random((MATRIX_SIZE, MATRIX_SIZE), Uniform::new(0.0, 1.0));

    let start_cpu = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = &array_1_cpu + &array_2_cpu;
    }
    let duration_cpu = start_cpu.elapsed();

    println!("CPU duration: {:?} per iteration", duration_cpu / ITERATIONS as u32);
    println!("CPU duration: {:?} for {:?} iterations", duration_cpu, ITERATIONS as u32);
}