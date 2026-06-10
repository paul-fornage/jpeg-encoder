use crate::encoder::AlignedBlock;
use crate::huffman::HuffmanTable;
use crate::writer::JfifWriter;
use crate::{EncodingError, JfifWrite};
use std::simd::cmp::SimdPartialEq;
use std::simd::Simd;
use crate::huffman::encoder::{DefaultHuffmanEncoder, HuffmanEncoder};

pub struct SimdHuffmanEncoder;

impl SimdHuffmanEncoder{
    pub fn write_ac_block_finish_first_n<const N: usize, W: JfifWrite>(
        writer: &mut JfifWriter<W>,
        block: &AlignedBlock,
        ac_table: &HuffmanTable,
    ) -> Result<u8, EncodingError> {
        let mut zero_run = 0;
        assert!(
            N <= 16,
            "N must be less than or equal to 16. This function does not handle 0 rollovers"
        );
        for &value in &block.data[1..N] {
            if value == 0 {
                zero_run += 1;
            } else {
                DefaultHuffmanEncoder::write_val_with_preceding_zeros(writer, value, zero_run, ac_table)?;
                zero_run = 0;
            }
        }
        Ok(zero_run)
    }


    pub fn write_ac_block_simd<W: JfifWrite>(
        writer: &mut JfifWriter<W>,
        block: &AlignedBlock,
        start: usize,
        end: usize,
        ac_table: &HuffmanTable,
    ) -> Result<(), EncodingError> {
        debug_assert_eq!(start, 1, "SIMD AC writer currently expects start=1");
        debug_assert_eq!(end, 64, "SIMD AC writer currently expects end=64");

        const BLOCK_SIZE: usize = 64;
        const SIMD_BIT_WIDTH: usize = 256;
        const SIMD_I16_WIDTH: usize = SIMD_BIT_WIDTH / 16;
        type SimdI16 = Simd<i16, SIMD_I16_WIDTH>;

        const INITIAL_LINEAR: usize = 16;
        let mut zero_run = Self::write_ac_block_finish_first_n::<INITIAL_LINEAR, W>(writer, block, ac_table)?;
        const REMAINING_ELEMENTS: usize = BLOCK_SIZE - INITIAL_LINEAR;
        assert_eq!(
            REMAINING_ELEMENTS % SIMD_I16_WIDTH,
            0,
            "Remaining elements must be divisible by SIMD_I16_WIDTH for vectorized processing"
        );
        let chunks = block.data[INITIAL_LINEAR..BLOCK_SIZE].chunks_exact(SIMD_I16_WIDTH);
        'chunk_loop: for chunk in chunks.into_iter() {
            // std::eprintln!("chunk: {:?}", chunk);
            let simd_values = SimdI16::from_slice(chunk);
            let non_zero = simd_values.simd_ne(SimdI16::splat(0));
            let mut checked_vals: u8 = 0;
            assert_eq!(
                SIMD_I16_WIDTH,
                size_of::<u16>() * 8,
                "SIMD_I16_WIDTH must match size of u16 for bitmask conversion"
            );
            let mut non_zeros_bitmask = (non_zero.to_bitmask() as u16).reverse_bits();
            if non_zeros_bitmask == 0 {
                zero_run += 16;
                continue 'chunk_loop;
            }

            while non_zeros_bitmask != 0 {
                let leading_zeros = non_zeros_bitmask.leading_zeros() as u8;
                zero_run += leading_zeros - checked_vals;
                while zero_run >= 16 {
                    DefaultHuffmanEncoder::huffman_encode(writer, 0xF0, ac_table)?;
                    zero_run -= 16;
                }
                let next_val = chunk[leading_zeros as usize];
                DefaultHuffmanEncoder::write_val_with_preceding_zeros(writer, next_val, zero_run, ac_table)?;
                zero_run = 0;
                checked_vals = leading_zeros + 1;
                non_zeros_bitmask &= !((1 << 15) >> leading_zeros);
            }
            zero_run = 16 - checked_vals;
        }

        if zero_run > 0 {
            DefaultHuffmanEncoder::huffman_encode(writer, 0x00, ac_table)?;
        }

        Ok(())
    }
}

impl HuffmanEncoder for SimdHuffmanEncoder {
    fn write_block<W: JfifWrite>(
        writer: &mut JfifWriter<W>,
        block: &AlignedBlock,
        prev_dc: i16,
        dc_table: &HuffmanTable,
        ac_table: &HuffmanTable,
    ) -> Result<(), EncodingError> {
        DefaultHuffmanEncoder::write_dc(writer, block.data[0], prev_dc, dc_table)?;
        Self::write_ac_block_simd(writer, block, 1, 64, ac_table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::huffman::HuffmanTable;
    use crate::huffman::huffman_sample_data::{HuffmanSampleData, HuffmanSampleDataSet};
    use crate::tests::create_test_img_rgb;
    use crate::writer::get_code;
    use crate::{RgbImage};
    use crate::fdct::DefaultFDCT;
    use alloc::vec::Vec;
    use core::array;
    use crate::quantization::DefaultBlockQuantizer;

    const START: usize = 1;
    const END: usize = 64;
    const PREFIX_BITS: (u32, u8) = (0b101, 3);
    const FLUSH_BITS: (u32, u8) = (0xA5, 8);
    const MIN_DC_COEFFICIENT: i16 = -1024;
    const MAX_DC_COEFFICIENT: i16 = 1023;
    const MIN_AC_COEFFICIENT: i16 = -1023;
    const MAX_AC_COEFFICIENT: i16 = 1023;

    struct TestRng {
        state: u64,
    }

    impl TestRng {
        fn new(seed: u64) -> Self {
            let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            if state == 0 {
                state = 0xD1B5_4A32_D192_ED03;
            }
            Self { state }
        }

        fn next_u32(&mut self) -> u32 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            (x >> 32) as u32
        }

        fn next_bool(&mut self) -> bool {
            self.next_u32() & 1 != 0
        }

        fn next_range_i16(&mut self, min: i16, max: i16) -> i16 {
            debug_assert!(min <= max);
            let span = (i32::from(max) - i32::from(min) + 1) as u32;
            (i32::from(min) + (self.next_u32() % span) as i32) as i16
        }
    }

    struct LocalWriter<'a> {
        buf: &'a mut Vec<u8>,
    }

    impl<'a> JfifWrite for LocalWriter<'a> {
        fn write_all(&mut self, buf: &[u8]) -> Result<(), EncodingError> {
            self.buf.extend_from_slice(buf);
            Ok(())
        }
    }

    fn write_og(block: &AlignedBlock, table: &HuffmanTable) -> Result<Vec<u8>, EncodingError> {
        let mut out = Vec::new();
        {
            let local_writer = LocalWriter { buf: &mut out };
            let mut writer = JfifWriter::new(local_writer);

            writer.write_bits(PREFIX_BITS.0, PREFIX_BITS.1)?;
            DefaultHuffmanEncoder::write_ac_block(&mut writer, block, START, END, table)?;
            writer.write_bits(FLUSH_BITS.0, FLUSH_BITS.1)?;
            writer.flush_bit_buffer()?;
        }
        Ok(out)
    }

    fn write_simd(block: &AlignedBlock, table: &HuffmanTable) -> Result<Vec<u8>, EncodingError> {
        let mut out = Vec::new();
        {
            let local_writer = LocalWriter { buf: &mut out };
            let mut writer = JfifWriter::new(local_writer);

            writer.write_bits(PREFIX_BITS.0, PREFIX_BITS.1)?;
            SimdHuffmanEncoder::write_ac_block_simd(&mut writer, block, START, END, table)?;
            writer.write_bits(FLUSH_BITS.0, FLUSH_BITS.1)?;
            writer.flush_bit_buffer()?;
        }
        Ok(out)
    }

    #[test]
    fn simd_ac_writer_matches_original_for_test_image() {

        let (data, width, height) = create_test_img_rgb();
        let image_buffer = RgbImage(&data, width, height);
        let sample_set = HuffmanSampleDataSet::from_image::<_, DefaultFDCT, DefaultBlockQuantizer>(&image_buffer);

        for (idx, sample) in sample_set.samples.iter().enumerate() {
            let ac_table = &sample_set.huffman_tables[sample.ac_huffman_table as usize].1;
            let block = sample.block;
            let og = write_og(&block, ac_table).unwrap();
            let simd = write_simd(&block, ac_table).unwrap();

            assert_eq!(og, simd, "Mismatch at sample {idx} with block: {block:?}",);
        }
    }

    fn random_ac_value(rng: &mut TestRng) -> i16 {
        // Keep AC values in category 0..10 (baseline default AC tables).
        if (rng.next_u32() & 0x0F) < 10 {
            return 0;
        }

        let size = (rng.next_u32() % 10 + 1) as u8;
        let min_magnitude = 1_i16 << (size - 1);
        let max_magnitude = (1_i16 << size) - 1;
        let magnitude = rng.next_range_i16(min_magnitude, max_magnitude);

        if rng.next_bool() {
            magnitude
        } else {
            -magnitude
        }
    }

    fn test_block(seed: u64, is_chroma: bool) -> HuffmanSampleData {
        let mut rng = TestRng::new(seed);
        let mut block = AlignedBlock {
            data: array::from_fn(|idx| {
                if idx == 0 {
                    rng.next_range_i16(MIN_DC_COEFFICIENT, MAX_DC_COEFFICIENT)
                } else {
                    random_ac_value(&mut rng)
                }
            }),
        };
        let mut last_dc = rng.next_range_i16(MIN_DC_COEFFICIENT, MAX_DC_COEFFICIENT);

        // Force boundary DC differences periodically to hit size=11.
        match seed & 0b11 {
            0 => {
                block.data[0] = MIN_DC_COEFFICIENT;
                last_dc = MAX_DC_COEFFICIENT;
            }
            1 => {
                block.data[0] = MAX_DC_COEFFICIENT;
                last_dc = MIN_DC_COEFFICIENT;
            }
            _ => {}
        }

        HuffmanSampleData {
            block,
            last_dc,
            dc_huffman_table: if is_chroma { 1 } else { 0 },
            ac_huffman_table: if is_chroma { 1 } else { 0 },
        }
    }

    #[test]
    fn random_test_blocks_stay_within_huffman_limits() {
        for seed in 0..4096 {
            let sample = test_block(seed, (seed & 1) != 0);

            assert!(
                (MIN_DC_COEFFICIENT..=MAX_DC_COEFFICIENT).contains(&sample.block.data[0]),
                "DC coefficient out of range at seed {seed}: {}",
                sample.block.data[0]
            );
            assert!(
                (MIN_DC_COEFFICIENT..=MAX_DC_COEFFICIENT).contains(&sample.last_dc),
                "last_dc out of range at seed {seed}: {}",
                sample.last_dc
            );

            let dc_diff = sample.block.data[0] - sample.last_dc;
            let dc_size = get_code(dc_diff).0;
            assert!(
                dc_size <= 11,
                "DC diff category too large at seed {seed}: diff={dc_diff}, size={dc_size}"
            );

            for (idx, &value) in sample.block.data[1..].iter().enumerate() {
                assert!(
                    (MIN_AC_COEFFICIENT..=MAX_AC_COEFFICIENT).contains(&value),
                    "AC value out of range at seed {seed}, index {}: {value}",
                    idx + 1
                );

                if value != 0 {
                    let ac_size = get_code(value).0;
                    assert!(
                        ac_size <= 10,
                        "AC category too large at seed {seed}, index {}: value={value}, size={ac_size}",
                        idx + 1
                    );
                }
            }

            assert!(sample.dc_huffman_table <= 1);
            assert!(sample.ac_huffman_table <= 1);
        }
    }

    #[test]
    fn simd_ac_writer_matches_original_for_random_samples() {
        let huffman_tables = [
            (
                HuffmanTable::default_luma_dc(),
                HuffmanTable::default_luma_ac(),
            ),
            (
                HuffmanTable::default_chroma_dc(),
                HuffmanTable::default_chroma_ac(),
            ),
        ];

        for seed in 0..20_000 {
            let sample = test_block(seed, (seed & 1) != 0);
            let ac_table = &huffman_tables[sample.ac_huffman_table as usize].1;

            let og = write_og(&sample.block, ac_table).unwrap();
            let simd = write_simd(&sample.block, ac_table).unwrap();

            assert_eq!(
                og, simd,
                "Mismatch at random seed {seed} with block: {:?}",
                sample.block
            );
        }
    }
}
