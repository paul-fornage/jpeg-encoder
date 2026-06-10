use crate::{EncodingError};
use crate::writer::{get_code, JfifWrite, JfifWriter};
use crate::encoder::AlignedBlock;
use crate::huffman::HuffmanTable;


pub trait HuffmanEncoder {
    fn write_block<W: JfifWrite>(
        writer: &mut JfifWriter<W>,
        block: &AlignedBlock,
        prev_dc: i16,
        dc_table: &HuffmanTable,
        ac_table: &HuffmanTable,
    ) -> Result<(), EncodingError>;
}

pub struct DefaultHuffmanEncoder;

impl HuffmanEncoder for DefaultHuffmanEncoder{
    fn write_block<W: JfifWrite>(
        writer: &mut JfifWriter<W>,
        block: &AlignedBlock,
        prev_dc: i16,
        dc_table: &HuffmanTable,
        ac_table: &HuffmanTable
    ) -> Result<(), EncodingError> {
        Self::write_dc(writer, block.data[0], prev_dc, dc_table)?;
        Self::write_ac_block(writer, block, 1, 64, ac_table)
    }
}

impl DefaultHuffmanEncoder {
    #[inline]
    pub fn huffman_encode_value<W: JfifWrite>(
        writer: &mut JfifWriter<W>,
        size: u8,
        symbol: u8,
        value: u16,
        table: &HuffmanTable,
    ) -> Result<(), EncodingError> {
        let &(num_bits, code) = table.get_for_value(symbol);

        let mut temp = value as u32;
        temp |= (code as u32) << size;
        let size = size + num_bits;

        writer.write_bits(temp, size)
    }

    pub fn write_dc<W: JfifWrite>(
        writer: &mut JfifWriter<W>,
        value: i16,
        prev_dc: i16,
        dc_table: &HuffmanTable,
    ) -> Result<(), EncodingError> {
        let diff = value - prev_dc;
        let (size, value) = get_code(diff);

        Self::huffman_encode_value(writer, size, size, value, dc_table)?;

        Ok(())
    }

    pub fn write_ac_block<W: JfifWrite>(
        writer: &mut JfifWriter<W>,
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
                    Self::huffman_encode(writer, 0xF0, ac_table)?;
                    zero_run -= 16;
                }

                Self::write_val_with_preceding_zeros(writer, value, zero_run as u8, ac_table)?;

                zero_run = 0;
            }
        }

        if zero_run > 0 {
            Self::huffman_encode(writer, 0x00, ac_table)?;
        }

        Ok(())
    }

    #[inline]
    pub fn write_val_with_preceding_zeros<W: JfifWrite>(
        writer: &mut JfifWriter<W>,
        value: i16,
        preceding_zeros: u8,
        ac_table: &HuffmanTable,
    ) -> Result<(), EncodingError> {
        let (size, value) = get_code(value);
        let symbol = (preceding_zeros << 4) | size;
        Self::huffman_encode_value(writer, size, symbol, value, ac_table)
    }

    #[inline]
    pub fn huffman_encode<W: JfifWrite>(writer: &mut JfifWriter<W>, val: u8, table: &HuffmanTable) -> Result<(), EncodingError> {
        let &(size, code) = table.get_for_value(val);
        writer.write_bits(code as u32, size)
    }
}