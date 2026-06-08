use criterion::{criterion_group, criterion_main, Criterion};

use jpeg_encoder::{encode_blocks, init_components, AlignedBlock, Component, EncodingError, HuffmanTable, ImageBuffer, JfifWrite, JfifWriter, QuantizationTable, QuantizationTableType, RgbImage, SamplingFactor};
use std::time::Duration;
use jpeg_encoder::huffman_sample_data::HuffmanSampleDataSet;

fn criterion_benchmark(c: &mut Criterion) {

    let mut writer: JfifWriter<Vec<u8>> = JfifWriter::new(Vec::new());

    let img = image::open("sample-image.png")
        .expect("failed to open test image")
        .into_rgb8();
    let (width, height) = img.dimensions();
    let image_buffer = RgbImage(img.as_raw(), width as u16, height as u16);

    let sample_set = HuffmanSampleDataSet::from_image(&image_buffer);
    
    let tables = sample_set.huffman_tables;
    let samples = sample_set.samples;

    let mut group = c.benchmark_group("huffman");
    group.measurement_time(Duration::from_secs(45));
    group.warm_up_time(Duration::from_secs(10));

    group.bench_function("huffman", |b| {
        b.iter(|| {
            for sample in &samples {
                writer.write_block(&sample.block, sample.last_dc, &tables[sample.dc_huffman_table as usize].0, &tables[sample.ac_huffman_table as usize].1).unwrap()
            }
        });
    });


    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
