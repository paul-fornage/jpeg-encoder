
use std::vec::Vec;
use crate::{BitStream, EncodingError};

pub struct SimdVecBitstream {
    pub data: Vec<u8>,
    pub len: usize,
}

impl BitStream for SimdVecBitstream {
    fn write(&mut self, buf: &[u8]) -> Result<(), EncodingError> {
        todo!()
    }

    fn finalize_bit_buffer(&mut self) -> Result<(), EncodingError> {
        todo!()
    }

    fn flush_bit_buffer(&mut self) -> Result<(), EncodingError> {
        todo!()
    }

    fn write_bits(&mut self, value: u32, size: u8) -> Result<(), EncodingError> {
        todo!()
    }
}