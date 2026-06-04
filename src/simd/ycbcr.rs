use alloc::vec::Vec;
use std::simd::num::{SimdInt, SimdUint};
use std::simd::{Simd, simd_swizzle};

use crate::{ImageBuffer, JpegColorType, rgb_to_ycbcr};

const SIMD_BIT_WIDTH: usize = 256;
const SIMD_BYTE_WIDTH: usize = SIMD_BIT_WIDTH / 8;
const SIMD_I32_WIDTH: usize = SIMD_BIT_WIDTH / 32;

pub type SimdI32 = Simd<i32, SIMD_I32_WIDTH>;
const CONVERT_RGB_BYTE_WIDTH: usize = SIMD_I32_WIDTH;
pub type SimdColor = Simd<u8, CONVERT_RGB_BYTE_WIDTH>;

pub struct SimdRgb {
    pub r: SimdColor,
    pub g: SimdColor,
    pub b: SimdColor,
}

pub struct SimdYCbCr {
    pub y: SimdColor,
    pub cb: SimdColor,
    pub cr: SimdColor,
}

const fn gen_swizzler<const N: usize>(chunk_idx: usize, chunk_size: usize) -> [usize; N] {
    assert!(
        chunk_idx < chunk_size,
        "chunk_idx must be less than chunk_size"
    );
    let mut out = [0; N];
    let mut i = 0;
    while i < N {
        out[i] = (i * chunk_size) + chunk_idx;
        i += 1;
    }
    out
}
const SWIZZLER_0_3: [usize; SIMD_I32_WIDTH] = gen_swizzler(0, 3);
const SWIZZLER_1_3: [usize; SIMD_I32_WIDTH] = gen_swizzler(1, 3);
const SWIZZLER_2_3: [usize; SIMD_I32_WIDTH] = gen_swizzler(2, 3);
const SWIZZLER_0_4: [usize; SIMD_I32_WIDTH] = gen_swizzler(0, 4);
const SWIZZLER_1_4: [usize; SIMD_I32_WIDTH] = gen_swizzler(1, 4);
const SWIZZLER_2_4: [usize; SIMD_I32_WIDTH] = gen_swizzler(2, 4);

#[inline(always)]
fn load_rgb(data: &[u8]) -> SimdRgb {
    let vals = Simd::<u8, { 3 * SIMD_I32_WIDTH }>::from_slice(data);
    SimdRgb {
        r: simd_swizzle!(vals, SWIZZLER_0_3),
        g: simd_swizzle!(vals, SWIZZLER_1_3),
        b: simd_swizzle!(vals, SWIZZLER_2_3),
    }
}

#[inline(always)]
fn load_bgr(data: &[u8]) -> SimdRgb {
    let vals = Simd::<u8, { 3 * SIMD_I32_WIDTH }>::from_slice(data);
    SimdRgb {
        r: simd_swizzle!(vals, SWIZZLER_2_3),
        g: simd_swizzle!(vals, SWIZZLER_1_3),
        b: simd_swizzle!(vals, SWIZZLER_0_3),
    }
}
#[inline(always)]
fn load_rgba(data: &[u8]) -> SimdRgb {
    let vals = Simd::<u8, { 4 * SIMD_I32_WIDTH }>::from_slice(data);
    SimdRgb {
        r: simd_swizzle!(vals, SWIZZLER_0_4),
        g: simd_swizzle!(vals, SWIZZLER_1_4),
        b: simd_swizzle!(vals, SWIZZLER_2_4),
    }
}

#[inline(always)]
fn load_bgra(data: &[u8]) -> SimdRgb {
    let vals = Simd::<u8, { 4 * SIMD_I32_WIDTH }>::from_slice(data);
    SimdRgb {
        r: simd_swizzle!(vals, SWIZZLER_2_4),
        g: simd_swizzle!(vals, SWIZZLER_1_4),
        b: simd_swizzle!(vals, SWIZZLER_0_4),
    }
}

#[inline(always)]
fn convert_rgb(rgb: SimdRgb) -> SimdYCbCr {
    const ROUNDING: SimdI32 = SimdI32::from_array([0x7FFF; SIMD_I32_WIDTH]);
    let r = rgb.r.cast::<i32>();
    let g = rgb.g.cast::<i32>();
    let b = rgb.b.cast::<i32>();

    let y = r * SimdI32::splat(19595) + g * SimdI32::splat(38470) + b * SimdI32::splat(7471);
    let cb = r * SimdI32::splat(-11059) - g * SimdI32::splat(21709)
        + b * SimdI32::splat(32768)
        + SimdI32::splat(128 << 16);
    let cr = r * SimdI32::splat(32768) - g * SimdI32::splat(27439) - b * SimdI32::splat(5329)
        + SimdI32::splat(128 << 16);

    SimdYCbCr {
        y: ((y + ROUNDING) >> SimdI32::splat(16)).cast::<u8>(),
        cb: ((cb + ROUNDING) >> SimdI32::splat(16)).cast::<u8>(),
        cr: ((cr + ROUNDING) >> SimdI32::splat(16)).cast::<u8>(),
    }
}

#[inline(always)]
fn extend_from_simd(buffer: &mut Vec<u8>, values: SimdColor) {
    buffer.extend_from_slice(&values.to_array());
}

fn fill_buffers_simd<const NUM_COLORS: usize, const R: usize, const G: usize, const B: usize, F>(
    data: &[u8],
    width: u16,
    y: u16,
    buffers: &mut [Vec<u8>; 4],
    load_channels: F,
) where
    F: Fn(&[u8]) -> SimdRgb,
{
    let width = usize::from(width);
    let start = usize::from(y) * width * NUM_COLORS;
    let line = &data[start..start + width * NUM_COLORS];

    let [y_buffer, cb_buffer, cr_buffer, _] = buffers;
    y_buffer.reserve(width);
    cb_buffer.reserve(width);
    cr_buffer.reserve(width);

    let chunks = line.chunks_exact(NUM_COLORS * SIMD_I32_WIDTH);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let rgb = load_channels(chunk);
        let SimdYCbCr { y, cb, cr } = convert_rgb(rgb);

        extend_from_simd(y_buffer, y);
        extend_from_simd(cb_buffer, cb);
        extend_from_simd(cr_buffer, cr);
    }

    for pixel in remainder.chunks_exact(NUM_COLORS) {
        let (y, cb, cr) = rgb_to_ycbcr(pixel[R], pixel[G], pixel[B]);

        y_buffer.push(y);
        cb_buffer.push(cb);
        cr_buffer.push(cr);
    }
}

macro_rules! ycbcr_image_simd {
    ($name:ident, $num_colors:expr, $r:expr, $g:expr, $b:expr, $load_channels:ident) => {
        pub struct $name<'a>(pub &'a [u8], pub u16, pub u16);

        impl<'a> ImageBuffer for $name<'a> {
            fn get_jpeg_color_type(&self) -> JpegColorType {
                JpegColorType::Ycbcr
            }

            fn width(&self) -> u16 {
                self.1
            }

            fn height(&self) -> u16 {
                self.2
            }

            #[inline(always)]
            fn fill_buffers(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
                fill_buffers_simd::<$num_colors, $r, $g, $b, _>(
                    self.0,
                    self.width(),
                    y,
                    buffers,
                    $load_channels,
                );
            }
        }
    };
}

ycbcr_image_simd!(RgbImageSimd, 3, 0, 1, 2, load_rgb);
ycbcr_image_simd!(RgbaImageSimd, 4, 0, 1, 2, load_rgba);
ycbcr_image_simd!(BgrImageSimd, 3, 2, 1, 0, load_bgr);
ycbcr_image_simd!(BgraImageSimd, 4, 2, 1, 0, load_bgra);

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    struct SimpleRng {
        state: u64,
    }

    impl SimpleRng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.state
        }

        fn next_byte(&mut self) -> u8 {
            (self.next_u64() & 0xFF) as u8
        }

        fn random_bytes(&mut self, len: usize) -> Vec<u8> {
            (0..len).map(|_| self.next_byte()).collect()
        }
    }

    fn assert_matches_scalar<I: ImageBuffer>(
        input: &[u8],
        image: I,
        bytes_per_pixel: usize,
        offsets: [usize; 3],
    ) {
        let scalar_result: Vec<[u8; 3]> = input
            .chunks_exact(bytes_per_pixel)
            .map(|chunk| {
                let [r, g, b] = offsets.map(|offset| chunk[offset]);
                let (y, cb, cr) = rgb_to_ycbcr(r, g, b);
                [y, cb, cr]
            })
            .collect();

        let mut buffers = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        image.fill_buffers(0, &mut buffers);

        for buffer in &buffers[..3] {
            assert_eq!(buffer.len(), input.len() / bytes_per_pixel);
        }

        for (i, pixel) in scalar_result.iter().copied().enumerate() {
            let simd_pixel = [buffers[0][i], buffers[1][i], buffers[2][i]];
            assert_eq!(pixel, simd_pixel, "mismatch at index {i}");
        }
    }

    #[test]
    fn simd_matches_scalar_rgb_formats() {
        let mut rng = SimpleRng::new(42);
        let width = 512 + 3;
        let height = 1;

        let rgb = rng.random_bytes(width * height * 3);
        assert_matches_scalar(
            &rgb,
            RgbImageSimd(&rgb, width.try_into().unwrap(), height.try_into().unwrap()),
            3,
            [0, 1, 2],
        );

        let rgba = rng.random_bytes(width * height * 4);
        assert_matches_scalar(
            &rgba,
            RgbaImageSimd(&rgba, width.try_into().unwrap(), height.try_into().unwrap()),
            4,
            [0, 1, 2],
        );

        let bgr = rng.random_bytes(width * height * 3);
        assert_matches_scalar(
            &bgr,
            BgrImageSimd(&bgr, width.try_into().unwrap(), height.try_into().unwrap()),
            3,
            [2, 1, 0],
        );

        let bgra = rng.random_bytes(width * height * 4);
        assert_matches_scalar(
            &bgra,
            BgraImageSimd(&bgra, width.try_into().unwrap(), height.try_into().unwrap()),
            4,
            [2, 1, 0],
        );
    }
}
