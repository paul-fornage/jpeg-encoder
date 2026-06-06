use std::eprintln;
use std::simd::cmp::SimdPartialEq;
use std::simd::Simd;
use crate::encoder::AlignedBlock;
use crate::{EncodingError, JfifWrite};
use crate::huffman::HuffmanTable;
use crate::writer::{get_code, JfifWriter};
impl<W: JfifWrite> JfifWriter<W> {

    pub fn write_ac_block_og(
        &mut self,
        block: &AlignedBlock,
        start: usize,
        end: usize,
        ac_table: &HuffmanTable,
    ) -> Result<(), EncodingError> {
        let mut zero_run = 0;

        for &value in &block.data[start..end] {
            if value == 0 {
                zero_run += 1;
            } else {
                while zero_run > 15 {
                    // eprintln!("[original]: huffman_encode(0xF0, ac_table)");
                    self.huffman_encode(0xF0, ac_table)?;
                    zero_run -= 16;
                }
                // eprintln!("[original]: write_val_with_preceding_zeros({value}, {zero_run}, ac_table)");
                self.write_val_with_preceding_zeros(value, zero_run, ac_table)?;

                zero_run = 0;
            }
        }

        if zero_run > 0 {
            // eprintln!("[original]: huffman_encode(0x00, ac_table)");
            self.huffman_encode(0x00, ac_table)?;
        }

        Ok(())
    }

    pub fn write_ac_block_finish_first_n<const N: usize>(
        &mut self,
        block: &AlignedBlock,
        ac_table: &HuffmanTable,
    ) -> Result<u8, EncodingError> {
        let mut zero_run = 0;
        assert!(N <= 16, "N must be less than or equal to 16. This function does not handle 0 rollovers");
        for &value in &block.data[1..N] {
            if value == 0 {
                zero_run += 1;
            } else {
                // eprintln!("[new]: write_val_with_preceding_zeros({value}, {zero_run}, ac_table)");
                self.write_val_with_preceding_zeros(value, zero_run, ac_table)?;
                zero_run = 0;
            }
        }
        Ok(zero_run)
    }

    #[inline]
    pub fn write_val_with_preceding_zeros(&mut self, value: i16, preceding_zeros: u8, ac_table: &HuffmanTable) -> Result<(), EncodingError> {
        let (size, value) = get_code(value);
        let symbol = (preceding_zeros << 4) | size;
        self.huffman_encode_value(size, symbol, value, ac_table)
    }

    pub fn write_ac_block_tweaked(
        &mut self,
        block: &AlignedBlock,
        start: usize,
        end: usize,
        ac_table: &HuffmanTable,
    ) -> Result<(), EncodingError> {

        const BLOCK_SIZE: usize = 64;
        const SIMD_BIT_WIDTH: usize = 256;
        const SIMD_I16_WIDTH: usize = SIMD_BIT_WIDTH / 16;
        type SimdI16 = Simd<i16, SIMD_I16_WIDTH>;

        const INITIAL_LINEAR: usize = 16;
        let mut zero_run = self.write_ac_block_finish_first_n::<INITIAL_LINEAR>(block, ac_table)?;
        const REMAINING_ELEMENTS: usize = BLOCK_SIZE-INITIAL_LINEAR;
        assert_eq!(REMAINING_ELEMENTS % SIMD_I16_WIDTH, 0,
                   "Remaining elements must be divisible by SIMD_I16_WIDTH for vectorized processing");
        let chunks = block.data[INITIAL_LINEAR..BLOCK_SIZE].chunks_exact(SIMD_I16_WIDTH);
        'chunk_loop: for chunk in chunks.into_iter(){
            // eprintln!("chunk: {:?}", chunk);
            let simd_values = SimdI16::from_slice(chunk);
            let non_zero = simd_values.simd_ne(SimdI16::splat(0));
            let mut checked_vals: u8 = 0;
            assert_eq!(SIMD_I16_WIDTH, size_of::<u16>()*8, "SIMD_I16_WIDTH must match size of u16 for bitmask conversion");
            let mut non_zeros_bitmask = (non_zero.to_bitmask() as u16).reverse_bits();
            if non_zeros_bitmask == 0 {
                zero_run += 16;
                continue 'chunk_loop;
            }

            'vals_loop: while non_zeros_bitmask != 0 {
                // eprintln!("start loop non_zeros_bitmask: {non_zeros_bitmask:016b}");
                let leading_zeros = non_zeros_bitmask.leading_zeros() as u8;
                // eprintln!("leading_zeros: {leading_zeros}");
                zero_run += leading_zeros - checked_vals;
                // eprintln!("zero_run: {zero_run}");
                if zero_run >= 16 {
                    eprintln!("[new]: huffman_encode(0xF0, ac_table)");
                    self.huffman_encode(0xF0, ac_table)?;
                    zero_run -= 16;
                }
                let next_val = chunk[leading_zeros as usize];
                // eprintln!("next_val: {next_val}");
                // eprintln!("[new]: write_val_with_preceding_zeros({next_val}, {zero_run}, ac_table)");
                self.write_val_with_preceding_zeros(next_val, zero_run, ac_table)?;
                zero_run = 0;
                checked_vals = leading_zeros+1;
                // eprintln!("checked_vals: {checked_vals}");
                non_zeros_bitmask &= !((1<<15) >> leading_zeros);
                // eprintln!("leading_zeros: {leading_zeros}, new non_zeros_bitmask: {non_zeros_bitmask:016b}");
            }
            zero_run = 16 - checked_vals;
            // eprintln!("exit non_zeros_bitmask: {:016b}", non_zeros_bitmask);
        }

        if zero_run > 0 {
            // eprintln!("[new]: huffman_encode(0x00, ac_table)");
            self.huffman_encode(0x00, ac_table)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::huffman_sample_data::HuffmanSampleDataSet;
    use alloc::vec::Vec;

    const START: usize = 1;
    const END: usize = 64;
    const PREFIX_BITS: (u32, u8) = (0b101, 3);
    const FLUSH_BITS: (u32, u8) = (0xA5, 8);

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
            writer.write_ac_block_og(block, START, END, table)?;
            writer.write_bits(FLUSH_BITS.0, FLUSH_BITS.1)?;
            writer.flush_bit_buffer()?;
        }
        Ok(out)
    }

    fn write_tweaked(block: &AlignedBlock, table: &HuffmanTable) -> Result<Vec<u8>, EncodingError> {
        let mut out = Vec::new();
        {
            let local_writer = LocalWriter { buf: &mut out };
            let mut writer = JfifWriter::new(local_writer);

            writer.write_bits(PREFIX_BITS.0, PREFIX_BITS.1)?;
            writer.write_ac_block_tweaked(block, START, END, table)?;
            writer.write_bits(FLUSH_BITS.0, FLUSH_BITS.1)?;
            writer.flush_bit_buffer()?;
        }
        Ok(out)
    }

    #[test]
    fn simd_tweaked_ac_writer_matches_original_for_captured_samples() {
        let samples_string = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/criterion/huffman_data_set.json"
        ));
        let sample_set: HuffmanSampleDataSet = serde_json::from_str(samples_string).unwrap();

        for (idx, sample) in sample_set.samples.iter().enumerate() {
            let ac_table = &sample_set.huffman_tables[sample.ac_huffman_table as usize].1;
            let block = sample.block;
            let og = write_og(&block, ac_table).unwrap();
            let tweaked = write_tweaked(&block, ac_table).unwrap();

            assert_eq!(
                og, tweaked,
                "Mismatch at sample {idx} with block: {block:?}",
            );
        }
    }
}
