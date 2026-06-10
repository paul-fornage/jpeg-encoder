use std::simd::Simd;

use crate::encoder::AlignedBlock;
use crate::fdct::FDCT;

const LANES: usize = 8;

const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

const FIX_0_298631336: i32 = 2446;
const FIX_0_390180644: i32 = 3196;
const FIX_0_541196100: i32 = 4433;
const FIX_0_765366865: i32 = 6270;
const FIX_0_899976223: i32 = 7373;
const FIX_1_175875602: i32 = 9633;
const FIX_1_501321110: i32 = 12299;
const FIX_1_847759065: i32 = 15137;
const FIX_1_961570560: i32 = 16069;
const FIX_2_053119869: i32 = 16819;
const FIX_2_562915447: i32 = 20995;
const FIX_3_072711026: i32 = 25172;

#[inline(always)]
fn descale(x: Simd<i32, LANES>, n: i32) -> Simd<i32, LANES> {
    (x + Simd::<i32, LANES>::splat(1 << (n - 1))) >> Simd::<i32, LANES>::splat(n)
}

#[inline(always)]
fn load_col(data: &[i16; 64], x: usize) -> Simd<i32, LANES> {
    Simd::<i32, LANES>::from_array(core::array::from_fn(|y| i32::from(data[y * 8 + x])))
}

#[inline(always)]
fn load_row(data: &[i32; 64], y: usize) -> Simd<i32, LANES> {
    Simd::<i32, LANES>::from_array(core::array::from_fn(|x| data[y * 8 + x]))
}

#[inline(always)]
fn fdct_pass(first_pass: bool, input: [Simd<i32, LANES>; 8]) -> [Simd<i32, LANES>; 8] {
    let [i0, i1, i2, i3, i4, i5, i6, i7] = input;

    let tmp0 = i0 + i7;
    let tmp7 = i0 - i7;
    let tmp1 = i1 + i6;
    let tmp6 = i1 - i6;
    let tmp2 = i2 + i5;
    let tmp5 = i2 - i5;
    let tmp3 = i3 + i4;
    let tmp4 = i3 - i4;

    let tmp10 = tmp0 + tmp3;
    let tmp13 = tmp0 - tmp3;
    let tmp11 = tmp1 + tmp2;
    let tmp12 = tmp1 - tmp2;

    let out0 = if first_pass {
        (tmp10 + tmp11) << Simd::<i32, LANES>::splat(PASS1_BITS)
    } else {
        descale(tmp10 + tmp11, PASS1_BITS)
    };

    let out4 = if first_pass {
        (tmp10 - tmp11) << Simd::<i32, LANES>::splat(PASS1_BITS)
    } else {
        descale(tmp10 - tmp11, PASS1_BITS)
    };

    let z1 = (tmp12 + tmp13) * Simd::<i32, LANES>::splat(FIX_0_541196100);
    let descale_bits = if first_pass {
        CONST_BITS - PASS1_BITS
    } else {
        CONST_BITS + PASS1_BITS
    };

    let out2 = descale(
        z1 + tmp13 * Simd::<i32, LANES>::splat(FIX_0_765366865),
        descale_bits,
    );
    let out6 = descale(
        z1 + tmp12 * Simd::<i32, LANES>::splat(-FIX_1_847759065),
        descale_bits,
    );

    let z1 = tmp4 + tmp7;
    let z2 = tmp5 + tmp6;
    let z3 = tmp4 + tmp6;
    let z4 = tmp5 + tmp7;
    let z5 = (z3 + z4) * Simd::<i32, LANES>::splat(FIX_1_175875602);

    let tmp4 = tmp4 * Simd::<i32, LANES>::splat(FIX_0_298631336);
    let tmp5 = tmp5 * Simd::<i32, LANES>::splat(FIX_2_053119869);
    let tmp6 = tmp6 * Simd::<i32, LANES>::splat(FIX_3_072711026);
    let tmp7 = tmp7 * Simd::<i32, LANES>::splat(FIX_1_501321110);
    let z1 = z1 * Simd::<i32, LANES>::splat(-FIX_0_899976223);
    let z2 = z2 * Simd::<i32, LANES>::splat(-FIX_2_562915447);
    let z3 = z3 * Simd::<i32, LANES>::splat(-FIX_1_961570560) + z5;
    let z4 = z4 * Simd::<i32, LANES>::splat(-FIX_0_390180644) + z5;

    let out7 = descale(tmp4 + z1 + z3, descale_bits);
    let out5 = descale(tmp5 + z2 + z4, descale_bits);
    let out3 = descale(tmp6 + z2 + z3, descale_bits);
    let out1 = descale(tmp7 + z1 + z4, descale_bits);

    [out0, out1, out2, out3, out4, out5, out6, out7]
}

pub struct SimdFDCT;

impl FDCT for SimdFDCT{
    fn fdct(data: &mut AlignedBlock) {
        let first_pass = fdct_pass(
            true,
            [
                load_col(&data.data, 0),
                load_col(&data.data, 1),
                load_col(&data.data, 2),
                load_col(&data.data, 3),
                load_col(&data.data, 4),
                load_col(&data.data, 5),
                load_col(&data.data, 6),
                load_col(&data.data, 7),
            ],
        );

        let mut data2 = [0i32; 64];
        for (x, values) in first_pass.into_iter().enumerate() {
            for (y, value) in values.to_array().into_iter().enumerate() {
                data2[y * 8 + x] = value;
            }
        }

        let second_pass = fdct_pass(
            false,
            [
                load_row(&data2, 0),
                load_row(&data2, 1),
                load_row(&data2, 2),
                load_row(&data2, 3),
                load_row(&data2, 4),
                load_row(&data2, 5),
                load_row(&data2, 6),
                load_row(&data2, 7),
            ],
        );

        for (y, values) in second_pass.into_iter().enumerate() {
            for (x, value) in values.to_array().into_iter().enumerate() {
                data.data[y * 8 + x] = value as i16;
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::SimdFDCT;
    use crate::encoder::AlignedBlock;
    use crate::fdct::{DefaultFDCT, FDCT};

    const INPUT1: [i16; 64] = [
        -70, -71, -70, -68, -67, -67, -67, -67, -72, -73, -72, -70, -69, -69, -68, -69, -75, -76,
        -74, -73, -73, -72, -71, -70, -77, -78, -77, -75, -76, -75, -73, -71, -78, -77, -77, -76,
        -79, -77, -76, -75, -78, -78, -77, -77, -77, -77, -78, -77, -79, -79, -78, -78, -78, -78,
        -79, -78, -80, -79, -78, -78, -81, -80, -78, -76,
    ];

    const INPUT2: [i16; 64] = [
        21, 28, 11, 24, -45, -37, -55, -103, 38, -8, 31, 17, -19, 49, 15, -76, 22, -48, -36, -31,
        -23, 35, -23, -72, 13, -30, -45, -42, -44, -15, -20, -44, 13, -30, -45, -42, -44, -15, -20,
        -44, 13, -30, -45, -42, -44, -15, -20, -44, 13, -30, -45, -42, -44, -15, -20, -44, 13, -30,
        -45, -42, -44, -15, -20, -44,
    ];

    #[test]
    fn simd_fdct_matches_scalar() {
        for input in [INPUT1, INPUT2] {
            let mut scalar = AlignedBlock::new(input);
            let mut simd = AlignedBlock::new(input);

            DefaultFDCT::fdct(&mut scalar);
            SimdFDCT::fdct(&mut simd);

            assert_eq!(simd.data, scalar.data);
        }
    }
}
