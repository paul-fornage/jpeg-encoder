use std::vec::Vec;
use crate::{BitStream, EncodingError};

const BUFFER_SIZE: usize = core::mem::size_of::<usize>() * 8;

pub struct SimdVecBitStream<'a> {
    pub data: &'a mut Vec<u8>,
    pub len: usize,
    bit_buffer: usize,
    free_bits: i8,
}



impl<'a> SimdVecBitStream<'a> {
    pub fn new(data: &'a mut Vec<u8>) -> Self {
        let len = data.len();
        Self {
            data,
            len,
            bit_buffer: 0,
            free_bits: BUFFER_SIZE as i8,
        }
    }

    #[inline(always)]
    fn push_raw_byte(&mut self, byte: u8) {
        self.data.push(byte);
        self.len += 1;
    }

    #[inline(always)]
    fn flush_byte_from_bit_buffer(&mut self, free_bits: i8) -> Result<(), EncodingError> {
        let value = (self.bit_buffer >> (BUFFER_SIZE as i8 - 8 - free_bits)) & 0xFF;
        self.push_raw_byte(value as u8);

        if value == 0xFF {
            self.push_raw_byte(0x00);
        }

        Ok(())
    }

    #[inline(always)]
    #[allow(overflowing_literals)]
    fn write_bit_buffer(&mut self) -> Result<(), EncodingError> {
        if (self.bit_buffer
            & 0x8080808080808080
            & !(self.bit_buffer.wrapping_add(0x0101010101010101)))
            != 0
        {
            for i in 0..(BUFFER_SIZE / 8) {
                self.flush_byte_from_bit_buffer((i * 8) as i8)?;
            }
        } else {
            let bytes = self.bit_buffer.to_be_bytes();
            self.data.extend_from_slice(&bytes);
            self.len += bytes.len();
        }

        Ok(())
    }

    fn flush_bit_buffer(&mut self) -> Result<(), EncodingError> {
        while self.free_bits <= (BUFFER_SIZE as i8 - 8) {
            self.flush_byte_from_bit_buffer(self.free_bits)?;
            self.free_bits += 8;
        }

        Ok(())
    }
}

impl<'a> BitStream for SimdVecBitStream<'a> {
    fn write(&mut self, buf: &[u8]) -> Result<(), EncodingError> {
        self.data.extend_from_slice(buf);
        self.len += buf.len();
        Ok(())
    }

    fn finalize_bit_buffer(&mut self) -> Result<(), EncodingError> {
        self.write_bits(0x7F, 7)?;
        self.flush_bit_buffer()?;
        self.bit_buffer = 0;
        self.free_bits = BUFFER_SIZE as i8;

        Ok(())
    }

    fn write_bits(&mut self, value: u32, size: u8) -> Result<(), EncodingError> {
        let size = size as i8;
        let value = value as usize;
        let free_bits = self.free_bits - size;

        if free_bits < 0 {
            self.bit_buffer = (self.bit_buffer << (size + free_bits)) | (value >> -free_bits);
            self.write_bit_buffer()?;
            self.bit_buffer = value;
            self.free_bits = free_bits + BUFFER_SIZE as i8;
        } else {
            self.free_bits = free_bits;
            self.bit_buffer = (self.bit_buffer << size) | value;
        }

        Ok(())
    }
}

// #![feature(funnel_shifts)]?
// extract_bits?

#[cfg(test)]
mod tests {
    use super::SimdVecBitStream;
    use crate::{BitStream, ColorType, DefaultBitStream, Encoder, SamplingFactor};
    use alloc::vec::Vec;

    fn create_test_img_rgb() -> (Vec<u8>, u16, u16) {
        let width = 258;
        let height = 128;
        let mut data = Vec::with_capacity(width * height * 3);

        for y in 0..height {
            for x in 0..width {
                let x = x.min(255);
                data.push(x as u8);
                data.push((y * 2) as u8);
                data.push(((x + y * 2) / 2) as u8);
            }
        }

        (data, width as u16, height as u16)
    }

    #[derive(Clone, Copy)]
    struct EncodeConfig {
        quality: u8,
        sampling: Option<SamplingFactor>,
        progressive: bool,
        optimized_huffman: bool,
        restart_interval: Option<u16>,
    }

    fn encode_with_default(data: &[u8], width: u16, height: u16, cfg: EncodeConfig) -> Vec<u8> {
        let mut out = Vec::new();
        let mut encoder = Encoder::new(DefaultBitStream::new(&mut out), cfg.quality);
        if let Some(sampling) = cfg.sampling {
            encoder.set_sampling_factor(sampling);
        }
        if cfg.progressive {
            encoder.set_progressive(true);
        }
        encoder.set_optimized_huffman_tables(cfg.optimized_huffman);
        if let Some(restart_interval) = cfg.restart_interval {
            encoder.set_restart_interval(restart_interval);
        }
        encoder.encode(data, width, height, ColorType::Rgb).unwrap();
        out
    }

    fn encode_with_simd_stream(
        data: &[u8],
        width: u16,
        height: u16,
        cfg: EncodeConfig,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        let mut encoder = Encoder::new(SimdVecBitStream::new(&mut out), cfg.quality);
        if let Some(sampling) = cfg.sampling {
            encoder.set_sampling_factor(sampling);
        }
        if cfg.progressive {
            encoder.set_progressive(true);
        }
        encoder.set_optimized_huffman_tables(cfg.optimized_huffman);
        if let Some(restart_interval) = cfg.restart_interval {
            encoder.set_restart_interval(restart_interval);
        }
        encoder.encode(data, width, height, ColorType::Rgb).unwrap();
        out
    }

    #[test]
    fn simd_vec_bitstream_matches_default_bitstream_full_pipeline_rgb() {
        let (data, width, height) = create_test_img_rgb();
        let cases = [
            EncodeConfig {
                quality: 100,
                sampling: Some(SamplingFactor::F_1_1),
                progressive: false,
                optimized_huffman: false,
                restart_interval: None,
            },
            EncodeConfig {
                quality: 80,
                sampling: None,
                progressive: false,
                optimized_huffman: false,
                restart_interval: None,
            },
            EncodeConfig {
                quality: 100,
                sampling: Some(SamplingFactor::F_4_1),
                progressive: false,
                optimized_huffman: false,
                restart_interval: None,
            },
            EncodeConfig {
                quality: 100,
                sampling: Some(SamplingFactor::F_2_1),
                progressive: true,
                optimized_huffman: false,
                restart_interval: None,
            },
            EncodeConfig {
                quality: 100,
                sampling: Some(SamplingFactor::F_2_1),
                progressive: true,
                optimized_huffman: true,
                restart_interval: None,
            },
            EncodeConfig {
                quality: 100,
                sampling: Some(SamplingFactor::F_2_2),
                progressive: false,
                optimized_huffman: false,
                restart_interval: Some(32),
            },
            EncodeConfig {
                quality: 85,
                sampling: Some(SamplingFactor::F_2_2),
                progressive: true,
                optimized_huffman: false,
                restart_interval: Some(32),
            },
        ];

        for cfg in cases {
            let expected = encode_with_default(&data[..], width, height, cfg);
            let actual = encode_with_simd_stream(&data[..], width, height, cfg);
            assert!(actual == expected, "bitstream mismatch for test case");
        }
    }

    #[test]
    fn simd_vec_bitstream_matches_default_for_bit_ops() {
        let mut expected = Vec::new();
        let mut default = DefaultBitStream::new(&mut expected);
        default.write(&[0x12, 0x34]).unwrap();
        default.write_bits(0xFF, 8).unwrap();
        default.write_bits(0b101, 3).unwrap();
        default.write_bits(0xA5, 8).unwrap();
        default.finalize_bit_buffer().unwrap();
        default.write(&[0xFF, 0xD9]).unwrap();
        default.write_bits(0x7F, 7).unwrap();
        default.finalize_bit_buffer().unwrap();

        let mut simd_result = Vec::new();
        let mut simd = SimdVecBitStream::new(&mut simd_result);
        simd.write(&[0x12, 0x34]).unwrap();
        simd.write_bits(0xFF, 8).unwrap();
        simd.write_bits(0b101, 3).unwrap();
        simd.write_bits(0xA5, 8).unwrap();
        simd.finalize_bit_buffer().unwrap();
        simd.write(&[0xFF, 0xD9]).unwrap();
        simd.write_bits(0x7F, 7).unwrap();
        simd.finalize_bit_buffer().unwrap();

        assert!(simd_result == expected);
    }
}
