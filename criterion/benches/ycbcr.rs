use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(feature = "simd")]
use jpeg_encoder::{RgbImageSimd, RgbaImageSimd};

use jpeg_encoder::{ImageBuffer, RgbImage, RgbaImage};
use std::time::Duration;


fn criterion_benchmark(c: &mut Criterion) {

    let img_rgb8 = image::open("sample-image.png")
        .expect("failed to open test image").into_rgb8();

    let img_rgba8: image::RgbaImage = image::DynamicImage::from(img_rgb8.clone()).into_rgba8();

    let (width, height) = img_rgb8.dimensions();
    let width = width as u16;
    let height = height as u16;

    let res1 = Vec::with_capacity(width as usize);
    let res2 = Vec::with_capacity(width as usize);
    let res3 = Vec::with_capacity(width as usize);
    let res4 = Vec::with_capacity(width as usize);

    let mut res = [res1, res2, res3, res4];

    let mut group = c.benchmark_group("ycbcr");
    group.measurement_time(Duration::from_secs(20));
    group.warm_up_time(Duration::from_secs(5));

    group.bench_function("default rgb to ycbcr", |b| {
        let image_buffer = RgbImage(img_rgb8.as_raw(), width, height);

        b.iter(|| {
            for y in 0..height {
                image_buffer.fill_buffers(y, &mut res);
            }
            black_box(&res);
            res[0].clear();
            res[1].clear();
            res[2].clear();
        });
    });

    #[cfg(feature = "simd")]
    group.bench_function("simd rgb to ycbcr", |b| {
        let image_buffer = RgbImageSimd(img_rgb8.as_raw(), width, height);

        b.iter(|| {
            for y in 0..height {
                image_buffer.fill_buffers(y, &mut res);
            }
            black_box(&res);
            res[0].clear();
            res[1].clear();
            res[2].clear();
        });
    });

    group.bench_function("default rgba to ycbcr", |b| {
        let image_buffer = RgbaImage(img_rgba8.as_raw(), width, height);

        b.iter(|| {
            for y in 0..height {
                image_buffer.fill_buffers(y, &mut res);
            }
            black_box(&res);
            res[0].clear();
            res[1].clear();
            res[2].clear();
            res[3].clear();
        });
    });

    #[cfg(feature = "simd")]
    group.bench_function("simd rgba to ycbcr", |b| {
        let image_buffer = RgbaImageSimd(img_rgba8.as_raw(), width, height);

        b.iter(|| {
            for y in 0..height {
                image_buffer.fill_buffers(y, &mut res);
            }
            black_box(&res);
            res[0].clear();
            res[1].clear();
            res[2].clear();
            res[3].clear();
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
