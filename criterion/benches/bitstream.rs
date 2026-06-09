use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use jpeg_encoder::{ColorType, DefaultBitStream, Encoder};
#[cfg(feature = "simd")]
use jpeg_encoder::SimdVecBitStream;
use std::time::Duration;


fn encode_default_vec(mut out: Vec<u8>, data: &[u8], width: u16, height: u16) -> Vec<u8> {
    {
        let encoder = Encoder::new(DefaultBitStream::new(&mut out), 90);
        encoder
            .encode(data, width, height, ColorType::Rgb)
            .expect("default vec encode failed");
    }
    out
}

#[cfg(feature = "simd")]
fn encode_simd_vec(out: Vec<u8>, data: &[u8], width: u16, height: u16) -> Vec<u8> {
    let mut stream = SimdVecBitStream::new(out);
    {
        let encoder = Encoder::new(&mut stream, 90);
        encoder
            .encode(data, width, height, ColorType::Rgb)
            .expect("simd vec encode failed");
    }
    stream.data
}

fn encode_default_sink(data: &[u8], width: u16, height: u16) {
    let encoder = Encoder::new(DefaultBitStream::new(std::io::sink()), 90);
    encoder
        .encode(data, width, height, ColorType::Rgb)
        .expect("default sink encode failed");
}

fn criterion_benchmark(c: &mut Criterion) {
    let img = image::open("sample-image.png")
        .expect("failed to open test image")
        .into_rgb8();
    let (width, height) = img.dimensions();
    let width = width as u16;
    let height = height as u16;
    let data = img.as_raw();

    let reference = encode_default_vec(Vec::new(), &data, width, height);
    assert!(!reference.is_empty(), "reference jpeg should not be empty");

    #[cfg(feature = "simd")]
    assert_eq!(
        encode_simd_vec(Vec::new(), &data, width, height),
        reference,
        "simd vec output should match default vec output"
    );

    let preallocated_template = Vec::<u8>::with_capacity(reference.len());

    let mut group = c.benchmark_group("bitstream_encode_image");
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(8));

    #[cfg(feature = "simd")]
    group.bench_function("simd_vec_preallocated", |b| {
        b.iter_batched(
            || preallocated_template.clone(),
            |out| encode_simd_vec(out, &data, width, height),
            BatchSize::SmallInput,
        );
    });

    #[cfg(feature = "simd")]
    group.bench_function("simd_vec_new", |b| {
        b.iter(|| encode_simd_vec(Vec::new(), &data, width, height));
    });

    group.bench_function("default_vec_preallocated", |b| {
        b.iter_batched(
            || preallocated_template.clone(),
            |out| encode_default_vec(out, &data, width, height),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("default_vec_new", |b| {
        b.iter(|| encode_default_vec(Vec::new(), &data, width, height));
    });

    group.bench_function("default_sink", |b| {
        b.iter(|| encode_default_sink(&data, width, height));
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
