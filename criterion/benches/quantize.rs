use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jpeg_encoder::{quantize_block_scalar, AlignedBlock, QuantizationTable, QuantizationTableType};
use std::time::Duration;

#[cfg(feature = "simd")]
use jpeg_encoder::quantize_block_simd;

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }
}

fn create_blocks(count: usize) -> Vec<AlignedBlock> {
    let mut rng = SimpleRng::new(0x1234_5678_9ABC_DEF0);
    let mut blocks = Vec::with_capacity(count);

    for _ in 0..count {
        let mut data = [0i16; 64];
        for value in &mut data {
            *value = ((rng.next_u64() as i32 % 4096) - 2048) as i16;
        }
        blocks.push(AlignedBlock::new(data));
    }

    blocks
}

#[inline(always)]
fn quantize_all_scalar(
    inputs: &[AlignedBlock],
    outputs: &mut [AlignedBlock],
    luma: &QuantizationTable,
    chroma: &QuantizationTable,
) {
    for (i, (input, output)) in inputs.iter().zip(outputs.iter_mut()).enumerate() {
        let table = if i & 1 == 0 { luma } else { chroma };
        quantize_block_scalar(input, output, table);
    }
}

#[cfg(feature = "simd")]
#[inline(always)]
fn quantize_all_simd(
    inputs: &[AlignedBlock],
    outputs: &mut [AlignedBlock],
    luma: &QuantizationTable,
    chroma: &QuantizationTable,
) {
    for (i, (input, output)) in inputs.iter().zip(outputs.iter_mut()).enumerate() {
        let table = if i & 1 == 0 { luma } else { chroma };
        quantize_block_simd(input, output, table);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let inputs = create_blocks(4096);
    let mut outputs = vec![AlignedBlock::default(); inputs.len()];

    let luma = QuantizationTable::new_with_quality(&QuantizationTableType::Default, 80, true);
    let chroma = QuantizationTable::new_with_quality(&QuantizationTableType::Default, 80, false);

    let mut group = c.benchmark_group("quantize");
    group.measurement_time(Duration::from_secs(45));
    group.warm_up_time(Duration::from_secs(10));

    group.bench_function("quantize scalar", |b| {
        b.iter(|| {
            quantize_all_scalar(
                black_box(&inputs),
                black_box(&mut outputs),
                black_box(&luma),
                black_box(&chroma),
            );
            black_box(&outputs);
        });
    });

    #[cfg(feature = "simd")]
    group.bench_function("quantize simd", |b| {
        b.iter(|| {
            quantize_all_simd(
                black_box(&inputs),
                black_box(&mut outputs),
                black_box(&luma),
                black_box(&chroma),
            );
            black_box(&outputs);
        });
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
