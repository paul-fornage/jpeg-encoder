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

7950X3D results: !! (79591de5) same as 87d7b53c WAY WORSE ON d5600505!!! NO SIMD!

jpeg/encode_jpeg 100    time:   [8.0174 ms 8.0235 ms 8.0301 ms]
jpeg/encode_jpeg_image 100
                        time:   [9.1579 ms 9.1672 ms 9.1772 ms]

jpeg/encode_jpeg 99     time:   [6.9098 ms 6.9138 ms 6.9181 ms]
jpeg/encode_jpeg_image 99
                        time:   [9.1303 ms 9.1392 ms 9.1485 ms]

jpeg/encode_jpeg 95     time:   [5.7708 ms 5.7752 ms 5.7803 ms]
jpeg/encode_jpeg_image 95
                        time:   [8.3462 ms 8.3537 ms 8.3617 ms]

jpeg/encode_jpeg 90     time:   [5.5052 ms 5.5108 ms 5.5163 ms]
jpeg/encode_jpeg_image 90
                        time:   [7.8111 ms 7.8218 ms 7.8336 ms]

jpeg/encode_jpeg 85     time:   [5.4037 ms 5.4066 ms 5.4097 ms]
jpeg/encode_jpeg_image 85
                        time:   [7.6454 ms 7.6518 ms 7.6591 ms]

jpeg/encode_jpeg 70     time:   [5.2347 ms 5.2443 ms 5.2553 ms]
jpeg/encode_jpeg_image 70
                        time:   [7.3613 ms 7.3688 ms 7.3766 ms]

jpeg/encode_jpeg 55     time:   [5.1618 ms 5.1684 ms 5.1752 ms]
jpeg/encode_jpeg_image 55
                        time:   [7.2556 ms 7.2660 ms 7.2776 ms]


*/
