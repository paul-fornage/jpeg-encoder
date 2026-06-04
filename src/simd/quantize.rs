use std::simd::Simd;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdInt;
use std::simd::prelude::Select;

use crate::encoder::AlignedBlock;
use crate::quantization::QuantizationTable;
use crate::writer::ZIGZAG;

const SHIFT: i32 = 2 * 8 - 1;
const SIMD_BIT_WIDTH: usize = 256;
const SIMD_I32_WIDTH: usize = SIMD_BIT_WIDTH / 32;

type SimdI32 = Simd<i32, SIMD_I32_WIDTH>;

#[inline(always)]
const fn zigzag_index(index: usize) -> usize {
    ZIGZAG[index] as usize
}

const SIMD_PER_QUANT: usize = 64 / SIMD_I32_WIDTH;


#[inline(always)]
pub fn quantize_block_simd(
    block: &AlignedBlock,
    q_block: &mut AlignedBlock,
    table: &QuantizationTable,
) {
    let reciprocals = table.reciprocals();
    let corrections = table.corrections();

    let mut chunks = q_block.data.chunks_exact_mut(SIMD_I32_WIDTH);
    let mut processed = 0;

    for (chunk_idx, out_chunk) in chunks.by_ref().enumerate() {
        let base = chunk_idx * SIMD_I32_WIDTH;

        let values = SimdI32::from_array(core::array::from_fn(|lane| {
            let z = zigzag_index(base + lane);
            i32::from(block.data[z])
        }));

        let reciprocal = SimdI32::from_array(core::array::from_fn(|lane| {
            let z = zigzag_index(base + lane);
            reciprocals[z]
        }));

        let correction = SimdI32::from_array(core::array::from_fn(|lane| {
            let z = zigzag_index(base + lane);
            corrections[z]
        }));

        let mut product = (values.abs() + correction) * reciprocal;
        product >>= SimdI32::splat(SHIFT);

        let result = values.simd_lt(SimdI32::splat(0)).select(-product, product);

        result.cast::<i16>().copy_to_slice(out_chunk);

        processed += SIMD_I32_WIDTH;
    }

    for (i, out) in chunks.into_remainder().iter_mut().enumerate() {
        let z = zigzag_index(processed + i);
        *out = table.quantize(block.data[z], z);
    }
}

#[cfg(test)]
mod tests {
    use super::quantize_block_simd;
    use crate::encoder::{AlignedBlock, Operations};
    use crate::quantization::{QuantizationTable, QuantizationTableType};
    use crate::simd::SimdOperations;

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

        <crate::encoder::DefaultOperations as Operations>::quantize_block(
            &block,
            &mut scalar,
            &table,
        );
        quantize_block_simd(&block, &mut simd, &table);

        assert_eq!(simd.data, scalar.data);
    }

    #[test]
    fn simd_operations_quantize_override_matches_scalar() {
        let block = AlignedBlock::new(INPUT);
        let table = QuantizationTable::new_with_quality(&QuantizationTableType::Default, 55, false);

        let mut scalar = AlignedBlock::default();
        let mut simd = AlignedBlock::default();

        <crate::encoder::DefaultOperations as Operations>::quantize_block(
            &block,
            &mut scalar,
            &table,
        );
        <SimdOperations as Operations>::quantize_block(&block, &mut simd, &table);

        assert_eq!(simd.data, scalar.data);
    }
}
