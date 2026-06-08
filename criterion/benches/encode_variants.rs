use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jpeg_encoder::{AlignedBlock, DefaultOperations, EncodingError, JfifWrite, Operations, QuantizationTable, QuantizationTableType, SamplingFactor};
use std::time::Duration;
use image::codecs::jpeg;
use imgref::ImgVec;
use rgb::{Rgb, RGB8};
#[cfg(feature = "simd")]
use jpeg_encoder::SimdOperations;


pub fn encode_jpeg(img: &ImgVec<RGB8>, quality: u8, sampling: SamplingFactor, progressive: bool) -> Result<Vec<u8>, EncodingError> {
    struct LocalWriter<'a> {
        buf: &'a mut Vec<u8>,
    }

    impl<'a> JfifWrite for LocalWriter<'a> {
        fn write_all(&mut self, buf: &[u8]) -> Result<(), EncodingError> {
            self.buf.extend_from_slice(buf);
            Ok(())
        }
    }

    let mut buf = Vec::with_capacity(img.buf().len());
    let wrapper = LocalWriter { buf: &mut buf };
    let mut encoder = jpeg_encoder::Encoder::new(wrapper, quality);
    encoder.set_sampling_factor(sampling);
    encoder.set_progressive(progressive);
    encoder.encode(
        bytemuck::cast_slice(img.buf()),
        img.width() as u16,
        img.height() as u16,
        jpeg_encoder::ColorType::Rgb,
    )?;

    Ok(buf)
}



fn criterion_benchmark(c: &mut Criterion) {
    let img = image::open("sample-image.png")
        .expect("failed to open test image")
        .into_rgb8();
    let (width, height) = img.dimensions();
    let img = ImgVec::<RGB8>::new(
        img.pixels().map(|pixel| pixel.0.into()).collect(), width as usize, height as usize
    );

    let mut group = c.benchmark_group("encode_variants");
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(10));

    group.bench_function("quality_100-f_1_1", |b| {
        b.iter(|| {
            encode_jpeg(
                &img,
                100,
                SamplingFactor::F_1_1,
                false
            )
        });
    });

    group.bench_function("quality_90-f_1_1", |b| {
        b.iter(|| {
            encode_jpeg(
                &img,
                90,
                SamplingFactor::F_1_1,
                false
            )
        });
    });

    group.bench_function("quality_75-f_1_1", |b| {
        b.iter(|| {
            encode_jpeg(
                &img,
                75,
                SamplingFactor::F_1_1,
                false
            )
        });
    });

    group.bench_function("quality_100-f_1_1_progressive", |b| {
        b.iter(|| {
            encode_jpeg(
                &img,
                100,
                SamplingFactor::F_1_1,
                true
            )
        });
    });

    group.bench_function("quality_90-f_1_1_progressive", |b| {
        b.iter(|| {
            encode_jpeg(
                &img,
                90,
                SamplingFactor::F_1_1,
                true
            )
        });
    });

    group.bench_function("quality_75-f_1_1_progressive", |b| {
        b.iter(|| {
            encode_jpeg(
                &img,
                75,
                SamplingFactor::F_1_1,
                true
            )
        });
    });

    group.bench_function("quality_100-f_2_2", |b| {
        b.iter(|| {
            encode_jpeg(
                &img,
                100,
                SamplingFactor::F_2_2,
                false
            )
        });
    });

    group.bench_function("quality_90-f_2_2", |b| {
        b.iter(|| {
            encode_jpeg(
                &img,
                90,
                SamplingFactor::F_2_2,
                false
            )
        });
    });

    group.bench_function("quality_75-f_2_2", |b| {
        b.iter(|| {
            encode_jpeg(
                &img,
                75,
                SamplingFactor::F_2_2,
                false
            )
        });
    });

    group.bench_function("quality_90-f_2_2_progressive", |b| {
        b.iter(|| {
            encode_jpeg(
                &img,
                90,
                SamplingFactor::F_2_2,
                true
            )
        });
    });

    group.bench_function("quality_100-f_4_2", |b| {
        b.iter(|| {
            encode_jpeg(
                &img,
                100,
                SamplingFactor::F_4_2,
                false
            )
        });
    });

    group.bench_function("quality_90-f_4_2", |b| {
        b.iter(|| {
            encode_jpeg(
                &img,
                90,
                SamplingFactor::F_4_2,
                false
            )
        });
    });

    group.bench_function("quality_75-f_4_2", |b| {
        b.iter(|| {
            encode_jpeg(
                &img,
                75,
                SamplingFactor::F_4_2,
                false
            )
        });
    });

    group.bench_function("quality_90-f_4_2_progressive", |b| {
        b.iter(|| {
            encode_jpeg(
                &img,
                90,
                SamplingFactor::F_4_2,
                true
            )
        });
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
