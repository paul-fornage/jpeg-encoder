use std::simd::cmp::SimdPartialEq;
use std::simd::Simd;
use crate::{AlignedBlock, EncodingError, JfifWrite};
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
                    self.huffman_encode(0xF0, ac_table)?;
                    zero_run -= 16;
                }

                let (size, value) = get_code(value);
                let symbol = (zero_run << 4) | size;

                self.huffman_encode_value(size, symbol, value, ac_table)?;

                zero_run = 0;
            }
        }

        if zero_run > 0 {
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
                let (size, value) = get_code(value);
                let symbol = (zero_run << 4) | size;
                self.huffman_encode_value(size, symbol, value, ac_table)?;
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
            let simd_values = SimdI16::from_slice(chunk);
            let non_zero = simd_values.simd_ne(SimdI16::splat(0));
            let mut checked_vals: u8 = 0;
            assert_eq!(SIMD_I16_WIDTH, size_of::<u16>(), "SIMD_I16_WIDTH must match size of u16 for bitmask conversion");
            let mut non_zeros_bitmask = non_zero.to_bitmask() as u16;
            if non_zeros_bitmask == 0 {
                self.huffman_encode(0xF0, ac_table)?;
                continue 'chunk_loop;
            }
            'vals_loop: while non_zeros_bitmask != 0 {
                let leading_zeros = non_zeros_bitmask.leading_zeros() as u8 - checked_vals;
                zero_run += leading_zeros;
                if zero_run >= 16 {
                    self.huffman_encode(0xF0, ac_table)?;
                    zero_run -= 16;
                }
                let next_val = chunk[leading_zeros as usize];
                self.write_val_with_preceding_zeros(next_val, zero_run, ac_table)?;
                zero_run = 0;
                checked_vals = leading_zeros;
                non_zeros_bitmask &= !(1 << leading_zeros);
            }
        }

        if zero_run > 0 {
            self.huffman_encode(0x00, ac_table)?;
        }

        Ok(())
    }
}
