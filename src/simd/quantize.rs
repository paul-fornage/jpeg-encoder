use std::simd::Simd;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdInt;
use std::simd::prelude::Select;

use crate::encoder::AlignedBlock;
use crate::quantization::{BlockQuantizer, QuantizationTable};
use crate::writer::ZIGZAG;

const BLOCK_SIZE: usize = 64;
const SHIFT: i32 = 2 * 8 - 1;
const SIMD_BIT_WIDTH: usize = 256;
const SIMD_I32_WIDTH: usize = SIMD_BIT_WIDTH / 32;

type SimdI32 = Simd<i32, SIMD_I32_WIDTH>;

const SIMD_PER_QUANT: usize = BLOCK_SIZE / SIMD_I32_WIDTH;

pub struct SimdBlockQuantizer;

impl BlockQuantizer for SimdBlockQuantizer {
    #[inline(always)]
    fn quantize_block(
        block: &AlignedBlock,
        q_block: &mut AlignedBlock,
        table: &QuantizationTable,
    ) {
        let reciprocals = table.reciprocals();
        let corrections = table.corrections();

        let data: &mut [i16; BLOCK_SIZE] = &mut q_block.data;
        let chunks = data.chunks_exact_mut(SIMD_I32_WIDTH);
        assert_eq!(
            SIMD_PER_QUANT * SIMD_I32_WIDTH,
            BLOCK_SIZE,
            "Block data must be a multiple of SIMD_I32_WIDTH"
        );
        assert_eq!(
            chunks.len(),
            SIMD_PER_QUANT,
            "Block data must be a multiple of SIMD_I32_WIDTH"
        );

        for (chunk_idx, out_chunk) in chunks.enumerate() {
            let base = chunk_idx * SIMD_I32_WIDTH;

            let values = SimdI32::from_array(core::array::from_fn(|lane| {
                let z = ZIGZAG[base + lane] as usize;
                i32::from(block.data[z])
            }));

            let reciprocal = SimdI32::from_slice(reciprocals[base..base + SIMD_I32_WIDTH].into());

            let correction = SimdI32::from_slice(corrections[base..base + SIMD_I32_WIDTH].into());

            let mut product = (values.abs() + correction) * reciprocal;
            product >>= SimdI32::splat(SHIFT);

            let result = values.simd_lt(SimdI32::splat(0)).select(-product, product);

            result.cast::<i16>().copy_to_slice(out_chunk);
        }
    }
}



// before dev machine: quantize/quantize simd  time:   [128.01 µs 128.21 µs 128.43 µs]
// after dev machine: quantize/quantize simd  time:   [109.01 µs 109.14 µs 109.31 µs]

#[cfg(test)]
mod tests {
    use super::SimdBlockQuantizer;
    use crate::encoder::{AlignedBlock};
    use crate::quantization::{BlockQuantizer, DefaultBlockQuantizer, QuantizationTable, QuantizationTableType};


    const INPUT: [i16; 64] = [
        21, 28, 11, 24, -45, -37, -55, -103, 38, -8, 31, 17, -19, 49, 15, -76, 22, -48, -36, -31,
        -23, 35, -23, -72, 13, -30, -45, -42, -44, -15, -20, -44, 13, -30, -45, -42, -44, -15, -20,
        -44, 13, -30, -45, -42, -44, -15, -20, -44, 13, -30, -45, -42, -44, -15, -20, -44, 13, -30,
        -45, -42, -44, -15, -20, -44,
    ];

    #[test]
    fn simd_quantize_matches_scalar_quantize_block() {
        let block = AlignedBlock::new(INPUT);
        let table = QuantizationTable::new_with_quality(&QuantizationTableType::Default, 80, true);

        let mut scalar = AlignedBlock::default();
        let mut simd = AlignedBlock::default();

        DefaultBlockQuantizer::quantize_block(
            &block,
            &mut scalar,
            &table,
        );
        SimdBlockQuantizer::quantize_block(
            &block,
            &mut simd,
            &table,
        );

        assert_eq!(simd.data, scalar.data);
    }

    #[test]
    fn simd_operations_quantize_override_matches_scalar() {
        let block = AlignedBlock::new(INPUT);
        let table = QuantizationTable::new_with_quality(&QuantizationTableType::Default, 55, false);

        let mut scalar = AlignedBlock::default();
        let mut simd = AlignedBlock::default();

        DefaultBlockQuantizer::quantize_block(
            &block,
            &mut scalar,
            &table,
        );
        SimdBlockQuantizer::quantize_block(
            &block,
            &mut simd,
            &table,
        );

        assert_eq!(simd.data, scalar.data);
    }
}
