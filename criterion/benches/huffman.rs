use criterion::{criterion_group, criterion_main, Criterion};

use jpeg_encoder::{DefaultBlockQuantizer, DefaultFDCT, HuffmanSampleDataSet};
#[cfg(feature = "simd")]
use jpeg_encoder::SimdHuffmanEncoder;

use jpeg_encoder::{DefaultHuffmanEncoder, HuffmanEncoder, JfifWriter, RgbImage};
use std::time::Duration;

fn criterion_benchmark(c: &mut Criterion) {

    let mut writer: JfifWriter<Vec<u8>> = JfifWriter::new(Vec::new());

    let img = image::open("sample-image.png")
        .expect("failed to open test image")
        .into_rgb8();
    let (width, height) = img.dimensions();
    let image_buffer = RgbImage(img.as_raw(), width as u16, height as u16);

    let sample_set = HuffmanSampleDataSet::from_image::<_, DefaultFDCT, DefaultBlockQuantizer>(&image_buffer);
    
    let tables = sample_set.huffman_tables;
    let samples = sample_set.samples;

    let mut group = c.benchmark_group("huffman");
    group.measurement_time(Duration::from_secs(45));
    group.warm_up_time(Duration::from_secs(10));

    group.bench_function("huffman-default", |b| {
        b.iter(|| {
            for sample in &samples {
                DefaultHuffmanEncoder::write_block(&mut writer, &sample.block, sample.last_dc, &tables[sample.dc_huffman_table as usize].0, &tables[sample.ac_huffman_table as usize].1).unwrap()
            }
        });
    });

    #[cfg(feature = "simd")]
    group.bench_function("huffman-simd", |b| {
        b.iter(|| {
            for sample in &samples {
                SimdHuffmanEncoder::write_block(&mut writer, &sample.block, sample.last_dc, &tables[sample.dc_huffman_table as usize].0, &tables[sample.ac_huffman_table as usize].1).unwrap()
            }
        });
    });


    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
