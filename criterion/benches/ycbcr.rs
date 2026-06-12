use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(feature = "simd")]
use jpeg_encoder::{RgbImageSimd, RgbaImageSimd};

use jpeg_encoder::{fill_buffer_padded, ImageBuffer, RgbImage, RgbaImage};
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
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(3));

    let image_buffer = RgbImage(img_rgb8.as_raw(), width, height);
    group.bench_function("default rgb to ycbcr", |b| {
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
    let image_buffer = RgbImageSimd(img_rgb8.as_raw(), width, height);
    #[cfg(feature = "simd")]
    group.bench_function("simd rgb to ycbcr", |b| {
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

    let image_buffer = RgbaImage(img_rgba8.as_raw(), width, height);
    group.bench_function("default rgba to ycbcr", |b| {
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
    let image_buffer = RgbaImageSimd(img_rgba8.as_raw(), width, height);
    #[cfg(feature = "simd")]
    group.bench_function("simd rgba to ycbcr", |b| {
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

    fn get_padded_dims(x: u16, y: u16, rounding: u16) -> (usize, usize) {
        (x.next_multiple_of(rounding) as usize, y.next_multiple_of(rounding) as usize)
    }
    let image_buffer = RgbImage(img_rgb8.as_raw(), width, height);
    let (padded_x, padded_y) = get_padded_dims(width, height, 8);
    group.bench_function("default rgb to ycbcr padded 8", |b| {
        b.iter(|| {
            fill_buffer_padded(&image_buffer, padded_x, padded_y, &mut res);
            black_box(&res);
            res[0].clear();
            res[1].clear();
            res[2].clear();
        });
    });

    #[cfg(feature = "simd")]
    let image_buffer = RgbImageSimd(img_rgb8.as_raw(), width, height);
    #[cfg(feature = "simd")]
    let (padded_x, padded_y) = get_padded_dims(width, height, 8);
    #[cfg(feature = "simd")]
    group.bench_function("simd rgb to ycbcr padded 8", |b| {
        b.iter(|| {
            fill_buffer_padded(&image_buffer, padded_x, padded_y, &mut res);
            black_box(&res);
            res[0].clear();
            res[1].clear();
            res[2].clear();
        });
    });

    let image_buffer = RgbaImage(img_rgba8.as_raw(), width, height);
    let (padded_x, padded_y) = get_padded_dims(width, height, 8);
    group.bench_function("default rgba to ycbcr padded 8", |b| {
        b.iter(|| {
            fill_buffer_padded(&image_buffer, padded_x, padded_y, &mut res);
            black_box(&res);
            res[0].clear();
            res[1].clear();
            res[2].clear();
            res[3].clear();
        });
    });

    #[cfg(feature = "simd")]
    let image_buffer = RgbaImageSimd(img_rgba8.as_raw(), width, height);
    #[cfg(feature = "simd")]
    let (padded_x, padded_y) = get_padded_dims(width, height, 8);
    #[cfg(feature = "simd")]
    group.bench_function("simd rgba to ycbcr padded 8", |b| {
        b.iter(|| {
            fill_buffer_padded(&image_buffer, padded_x, padded_y, &mut res);
            black_box(&res);
            res[0].clear();
            res[1].clear();
            res[2].clear();
            res[3].clear();
        });
    });

    let image_buffer = RgbImage(img_rgb8.as_raw(), width, height);
    let (padded_x, padded_y) = get_padded_dims(width, height, 16);
    group.bench_function("default rgb to ycbcr padded 16", |b| {
        b.iter(|| {
            fill_buffer_padded(&image_buffer, padded_x, padded_y, &mut res);
            black_box(&res);
            res[0].clear();
            res[1].clear();
            res[2].clear();
        });
    });

    #[cfg(feature = "simd")]
    let image_buffer = RgbImageSimd(img_rgb8.as_raw(), width, height);
    #[cfg(feature = "simd")]
    let (padded_x, padded_y) = get_padded_dims(width, height, 16);
    #[cfg(feature = "simd")]
    group.bench_function("simd rgb to ycbcr padded 16", |b| {
        b.iter(|| {
            fill_buffer_padded(&image_buffer, padded_x, padded_y, &mut res);
            black_box(&res);
            res[0].clear();
            res[1].clear();
            res[2].clear();
        });
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);


/*
Dev machine (AVX2)
ycbcr/default rgb to ycbcr
                        time:   [607.44 µs 612.37 µs 618.48 µs]
ycbcr/simd rgb to ycbcr
                        time:   [758.66 µs 768.65 µs 779.46 µs]
ycbcr/default rgba to ycbcr
                        time:   [655.43 µs 666.44 µs 678.60 µs]
ycbcr/simd rgba to ycbcr
                        time:   [530.95 µs 536.73 µs 544.57 µs]
ycbcr/default rgb to ycbcr padded 8
                        time:   [624.41 µs 632.94 µs 642.08 µs]
ycbcr/simd rgb to ycbcr padded 8
                        time:   [742.15 µs 745.52 µs 749.85 µs]
ycbcr/default rgba to ycbcr padded 8
                        time:   [638.37 µs 641.13 µs 644.14 µs]
ycbcr/simd rgba to ycbcr padded 8
                        time:   [535.26 µs 536.54 µs 537.90 µs]
ycbcr/default rgb to ycbcr padded 16
                        time:   [630.29 µs 636.80 µs 643.40 µs]
ycbcr/simd rgb to ycbcr padded 16
                        time:   [740.93 µs 741.71 µs 742.59 µs]

RPI 5:

ycbcr/default rgb to ycbcr
                        time:   [1.0033 ms 1.0061 ms 1.0091 ms]
ycbcr/simd rgb to ycbcr
                        time:   [959.78 µs 961.46 µs 963.22 µs]
ycbcr/default rgba to ycbcr
                        time:   [1.4812 ms 1.4847 ms 1.4884 ms]
ycbcr/simd rgba to ycbcr
                        time:   [1.0122 ms 1.0150 ms 1.0182 ms]
ycbcr/default rgb to ycbcr padded 8
                        time:   [1.1016 ms 1.1040 ms 1.1069 ms]
ycbcr/simd rgb to ycbcr padded 8
                        time:   [1.0477 ms 1.0502 ms 1.0532 ms]
ycbcr/default rgba to ycbcr padded 8
                        time:   [1.4960 ms 1.4992 ms 1.5027 ms]
ycbcr/simd rgba to ycbcr padded 8
                        time:   [1.1655 ms 1.1693 ms 1.1734 ms]
ycbcr/default rgb to ycbcr padded 16
                        time:   [1.1094 ms 1.1115 ms 1.1140 ms]
ycbcr/simd rgb to ycbcr padded 16
                        time:   [1.0601 ms 1.0627 ms 1.0658 ms]


 */