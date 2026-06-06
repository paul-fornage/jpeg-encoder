use criterion::{black_box, criterion_group, criterion_main, Criterion};
use imgref::ImgVec;
use jpeg_encoder::{EncodingError, JfifWrite};
use rgb::{Rgb, RGB8};
use std::io::Cursor;
use std::time::Duration;

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

pub fn encode_jpeg_image(img: &ImgVec<RGB8>, quality: u8) -> Result<Vec<u8>, image::ImageError> {
    let mut encoded_jpeg = Vec::with_capacity(img.buf().len());
    image::codecs::jpeg::JpegEncoder::new_with_quality(Cursor::new(&mut encoded_jpeg), quality)
        .encode(
            bytemuck::cast_slice(img.buf()),
            img.width() as u32,
            img.height() as u32,
            image::ExtendedColorType::Rgb8,
        )?;

    Ok(encoded_jpeg)
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

    group.warm_up_time(Duration::from_secs(8));
    group.measurement_time(Duration::from_secs(16));

    group.bench_function("encode_jpeg 100", |b| {
        b.iter(|| encode_jpeg(black_box(&img_vec), 100));
    });

    group.bench_function("encode_jpeg_image 100", |b| {
        b.iter(|| encode_jpeg_image(black_box(&img_vec), 100));
    });

    group.bench_function("encode_jpeg 99", |b| {
        b.iter(|| encode_jpeg(black_box(&img_vec), 99));
    });

    group.bench_function("encode_jpeg_image 99", |b| {
        b.iter(|| encode_jpeg_image(black_box(&img_vec), 99));
    });

    group.bench_function("encode_jpeg 95", |b| {
        b.iter(|| encode_jpeg(black_box(&img_vec), 95));
    });

    group.bench_function("encode_jpeg_image 95", |b| {
        b.iter(|| encode_jpeg_image(black_box(&img_vec), 95));
    });

    group.bench_function("encode_jpeg 90", |b| {
        b.iter(|| encode_jpeg(black_box(&img_vec), 90));
    });

    group.bench_function("encode_jpeg_image 90", |b| {
        b.iter(|| encode_jpeg_image(black_box(&img_vec), 90));
    });

    group.bench_function("encode_jpeg 85", |b| {
        b.iter(|| encode_jpeg(black_box(&img_vec), 85));
    });

    group.bench_function("encode_jpeg_image 85", |b| {
        b.iter(|| encode_jpeg_image(black_box(&img_vec), 85));
    });

    group.bench_function("encode_jpeg 70", |b| {
        b.iter(|| encode_jpeg(black_box(&img_vec), 70));
    });

    group.bench_function("encode_jpeg_image 70", |b| {
        b.iter(|| encode_jpeg_image(black_box(&img_vec), 70));
    });

    group.bench_function("encode_jpeg 55", |b| {
        b.iter(|| encode_jpeg(black_box(&img_vec), 55));
    });

    group.bench_function("encode_jpeg_image 55", |b| {
        b.iter(|| encode_jpeg_image(black_box(&img_vec), 55));
    });

    group.finish();
}

criterion_group!(jpeg, bench_jpeg);
criterion_main!(jpeg);

/*
RPI 5 results:

jpeg/encode_jpeg 100    time:   [17.186 ms 17.189 ms 17.192 ms]

jpeg/encode_jpeg_image 100
                        time:   [14.616 ms 14.616 ms 14.617 ms]

jpeg/encode_jpeg 99     time:   [14.758 ms 14.759 ms 14.760 ms]

jpeg/encode_jpeg_image 99
                        time:   [14.611 ms 14.612 ms 14.613 ms]

jpeg/encode_jpeg 95     time:   [12.351 ms 12.351 ms 12.352 ms]

jpeg/encode_jpeg_image 95
                        time:   [12.932 ms 12.933 ms 12.934 ms]

jpeg/encode_jpeg 90     time:   [11.740 ms 11.741 ms 11.742 ms]

jpeg/encode_jpeg_image 90
                        time:   [11.894 ms 11.896 ms 11.897 ms]

jpeg/encode_jpeg 85     time:   [11.513 ms 11.514 ms 11.515 ms]

jpeg/encode_jpeg_image 85
                        time:   [11.535 ms 11.536 ms 11.537 ms]

jpeg/encode_jpeg 70     time:   [11.123 ms 11.124 ms 11.125 ms]

jpeg/encode_jpeg_image 70
                        time:   [10.979 ms 10.980 ms 10.982 ms]

jpeg/encode_jpeg 55     time:   [10.954 ms 10.955 ms 10.956 ms]

jpeg/encode_jpeg_image 55
                        time:   [10.706 ms 10.707 ms 10.707 ms]

Found 2 outliers among 100 measurements (2.00%)
  2 (2.00%) high mild

7950X3D results: !! (No simd)

jpeg/encode_jpeg 100    time:   [26.369 ms 26.759 ms 27.219 ms]

jpeg/encode_jpeg_image 100
                        time:   [8.8544 ms 8.8668 ms 8.8815 ms]

jpeg/encode_jpeg 99     time:   [29.321 ms 31.198 ms 33.167 ms]

jpeg/encode_jpeg_image 99
                        time:   [8.8488 ms 8.8574 ms 8.8666 ms]

jpeg/encode_jpeg 95     time:   [21.220 ms 21.268 ms 21.322 ms]

jpeg/encode_jpeg_image 95
                        time:   [8.0361 ms 8.0444 ms 8.0549 ms]

jpeg/encode_jpeg 90     time:   [20.439 ms 20.483 ms 20.531 ms]

jpeg/encode_jpeg_image 90
                        time:   [7.4702 ms 7.4755 ms 7.4821 ms]

jpeg/encode_jpeg 85     time:   [20.099 ms 20.145 ms 20.195 ms]

jpeg/encode_jpeg_image 85
                        time:   [7.2979 ms 7.3030 ms 7.3086 ms]

jpeg/encode_jpeg 70     time:   [19.556 ms 19.592 ms 19.632 ms]

jpeg/encode_jpeg_image 70
                        time:   [7.0255 ms 7.0341 ms 7.0439 ms]

jpeg/encode_jpeg 55     time:   [19.430 ms 19.498 ms 19.580 ms]

jpeg/encode_jpeg_image 55
                        time:   [6.8735 ms 6.8832 ms 6.8945 ms]

*/
