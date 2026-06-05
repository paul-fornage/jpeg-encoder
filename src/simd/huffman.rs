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

        const INITIAL_LINEAR: usize = 16;
        let mut zero_run = self.write_ac_block_finish_first_n::<INITIAL_LINEAR>(block, ac_table)?;
        let mut iter = &block.data[INITIAL_LINEAR..BLOCK_SIZE].iter();



        todo!()
    }
}
