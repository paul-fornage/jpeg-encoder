use criterion::{criterion_group, criterion_main, Criterion};
use imgref::ImgVec;
use jpeg_encoder::{get_block_linear, get_block_simd};
use std::time::Duration;

fn bench_misc(c: &mut Criterion) {
    let img = image::open("sample-image.png")
        .expect("failed to open test image")
        .into_rgb8();
    let (width, height) = img.dimensions();
    let buf = img
        .pixels()
        .map(|p| p.0[0] )
        .collect();
    let red_img_vec = ImgVec::<u8>::new(buf, width as usize, height as usize);

    let x_blocks = (width / 8) as usize;
    let y_blocks = (height / 8) as usize;

    let mut group = c.benchmark_group("bench_misc");

    group.warm_up_time(Duration::from_secs(8));
    group.measurement_time(Duration::from_secs(16));

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