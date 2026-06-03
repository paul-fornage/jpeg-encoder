use criterion::{criterion_group, criterion_main, Criterion};
use imgref::ImgVec;
use rgb::{Rgb, RGB8};
use jpeg_encoder::{EncodingError, JfifWrite};

pub fn encode_jpeg(img: &ImgVec<RGB8>, quality: u8) -> Result<Vec<u8>, EncodingError> {
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
    encoder.set_sampling_factor(jpeg_encoder::SamplingFactor::F_1_1);
    encoder.encode(
        bytemuck::cast_slice(img.buf()),
        img.width() as u16,
        img.height() as u16,
        jpeg_encoder::ColorType::Rgb,
    )?;

    Ok(buf)
}


fn bench_jpeg(c: &mut Criterion) {
    let img = image::open("sample-image.png")
        .expect("failed to open test image")
        .into_rgb8();
    let (width, height) = img.dimensions();
    let buf = img
        .pixels()
        .map(|p| Rgb {
            r: p.0[0],
            g: p.0[1],
            b: p.0[2],
        })
        .collect();
    let img_vec = ImgVec::<RGB8>::new(buf, width as usize, height as usize);

    let mut group = c.benchmark_group("jpeg");

    group.bench_function("encode_jpeg 99", |b| {
        b.iter(|| encode_jpeg(&img_vec, 99));
    });

    group.bench_function("encode_jpeg 95", |b| {
        b.iter(|| encode_jpeg(&img_vec, 95));
    });

    group.bench_function("encode_jpeg 90", |b| {
        b.iter(|| encode_jpeg(&img_vec, 90));
    });

    group.bench_function("encode_jpeg 85", |b| {
        b.iter(|| encode_jpeg(&img_vec, 85));
    });

    group.bench_function("encode_jpeg 70", |b| {
        b.iter(|| encode_jpeg(&img_vec, 70));
    });

    group.bench_function("encode_jpeg 55", |b| {
        b.iter(|| encode_jpeg(&img_vec, 55));
    });

    group.finish();
}

criterion_group!(jpeg, bench_jpeg);
criterion_main!(jpeg);
