use criterion::{criterion_group, criterion_main, Criterion};

use jpeg_encoder::{EncodingError, JfifWrite, JfifWriter};
use std::time::Duration;


fn criterion_benchmark(c: &mut Criterion) {

    struct LocalWriter<'a> {
        buf: &'a mut Vec<u8>,
    }

    impl<'a> JfifWrite for LocalWriter<'a> {
        fn write_all(&mut self, buf: &[u8]) -> Result<(), EncodingError> {
            self.buf.extend_from_slice(buf);
            Ok(())
        }
    }

    let mut buf = Vec::new();
    let local_writer = LocalWriter { buf: &mut buf };
    let mut writer: JfifWriter<LocalWriter> = JfifWriter::new(local_writer);

    let samples_string = std::fs::read_to_string("huffman_data_set.json").unwrap();
    let sample_set: jpeg_encoder::huffman_sample_data::HuffmanSampleDataSet = serde_json::from_str(&samples_string).unwrap();
    
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
