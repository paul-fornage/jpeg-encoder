mod fdct;
mod huffman;
mod quantize;
mod ycbcr;

use crate::encoder::{AlignedBlock, Operations};
use crate::quantization::QuantizationTable;
pub use fdct::fdct_simd;
pub use quantize::quantize_block_simd;
pub use ycbcr::*;

pub struct SimdOperations;

impl Operations for SimdOperations {
    #[inline(always)]
    fn fdct(data: &mut AlignedBlock) {
        fdct_simd(data);
    }

    #[inline(always)]
    fn quantize_block(block: &AlignedBlock, q_block: &mut AlignedBlock, table: &QuantizationTable) {
        quantize_block_simd(block, q_block, table);
    }
}
