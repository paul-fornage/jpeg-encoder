mod fdct;
mod ycbcr;

use crate::encoder::{AlignedBlock, Operations};
pub use fdct::fdct_simd;
pub use ycbcr::*;

pub(crate) struct SimdOperations;

impl Operations for SimdOperations {
    #[inline(always)]
    fn fdct(data: &mut AlignedBlock) {
        fdct_simd(data);
    }
}
