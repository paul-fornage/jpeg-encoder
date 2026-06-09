use criterion::{criterion_group, criterion_main, Criterion};
use imgref::ImgVec;

#[cfg(feature = "simd")]
use jpeg_encoder::{get_block_simd, RgbImageSimd, SimdOperations};
use jpeg_encoder::{get_block_linear, DefaultBitStream, DefaultOperations, ImageBuffer, QuantizationTable, QuantizationTableType, RgbImage};


use std::time::Duration;

fn bench_misc(c: &mut Criterion) {
    let img = image::open("sample-image.png")
        .expect("failed to open test image")
        .into_rgb8();
    let (width, height) = img.dimensions();
    let buf: Vec<u8> = img
        .pixels()
        .map(|p| p.0[0] )
        .collect();
    let red_img_vec = ImgVec::<u8>::new(buf.clone(), width as usize, height as usize);
    let rgb_img_vec = ImgVec::<u8>::new(img.to_vec(), width as usize, height as usize);

    let x_blocks = (width / 8) as usize;
    let y_blocks = (height / 8) as usize;

    let mut group = c.benchmark_group("bench_misc");

    let q_tables = [
        QuantizationTable::new_with_quality(&QuantizationTableType::Default, 90, true),
        QuantizationTable::new_with_quality(&QuantizationTableType::Default, 90, false),
    ];

    let image_buffer = RgbImage(rgb_img_vec.buf(), width as u16, height as u16);

    let mut encoder = jpeg_encoder::Encoder::new(DefaultBitStream::new(std::io::sink()), 90);
    encoder.set_sampling_factor(jpeg_encoder::SamplingFactor::F_1_1);

    encoder.init_components(image_buffer.get_jpeg_color_type());

    group.warm_up_time(Duration::from_secs(8));
    group.measurement_time(Duration::from_secs(16));

    #[cfg(feature = "simd")]
    group.bench_function("encode_blocks_simd", |b| {
        b.iter(|| {
            encoder.encode_blocks::<_, SimdOperations>(&image_buffer, &q_tables);
        });
    });

    group.bench_function("encode_blocks_linear", |b| {
        b.iter(|| {
            encoder.encode_blocks::<_, DefaultOperations>(&image_buffer, &q_tables);
        });
    });


    group.bench_function("get_blocks_linear", |b| {
        b.iter(|| {
            for y_block in 0..y_blocks-1 {
                for x_block in 0..x_blocks-1 {
                    get_block_linear(
                        red_img_vec.buf(),
                        x_block * 8,
                        y_block * 8,
                        1,
                        1,
                        width as usize
                    );
                }
            }
        });
    });

    #[cfg(feature = "simd")]
    group.bench_function("get_blocks_simd", |b| {
        b.iter(|| {
            for y_block in 0..y_blocks-1 {
                for x_block in 0..x_blocks-1 {
                    get_block_simd(
                        red_img_vec.buf(),
                        x_block * 8,
                        y_block * 8,
                        1,
                        1,
                        width as usize
                    );
                }
            }
        });
    });

    group.finish();
}

criterion_group!(misc, bench_misc);
criterion_main!(misc);
