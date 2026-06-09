
use std::vec::Vec;
use crate::{BitStream, EncodingError};

pub struct SimdVecBitStream {
    pub data: Vec<u8>,
    pub len: usize,
    pending_byte: u8,
    pending_bits: u8,
    has_unfinalized_bits: bool,
}

impl SimdVecBitStream {
    pub fn new(data: Vec<u8>) -> Self {
        let len = data.len();
        Self {
            data,
            len,
            pending_byte: 0,
            pending_bits: 0,
            has_unfinalized_bits: false,
        }
    }

    #[inline(always)]
    fn push_raw_byte(&mut self, byte: u8) {
        self.data.push(byte);
        self.len += 1;
    }

    #[inline(always)]
    fn push_stuffed_byte(&mut self, byte: u8) {
        self.push_raw_byte(byte);
        if byte == 0xFF {
            self.push_raw_byte(0x00);
        }
    }

    #[inline(always)]
    fn flush_pending_byte(&mut self) {
        debug_assert_eq!(self.pending_bits, 8);
        self.push_stuffed_byte(self.pending_byte);
        self.pending_byte = 0;
        self.pending_bits = 0;
    }
}

impl BitStream for SimdVecBitStream {

    /// Write bytes to the bitstream.
    /// It is the callers job to make sure they have called `finalize_bit_buffer`
    ///  after doing bitwise operations and before calling this.
    fn write(&mut self, buf: &[u8]) -> Result<(), EncodingError> {
        debug_assert!(
            !self.has_unfinalized_bits,
            "write called after write_bits without an intervening finalize_bit_buffer"
        );
        debug_assert_eq!(
            self.pending_bits, 0,
            "write called while partial bit-buffer data is still pending"
        );

        self.data.extend_from_slice(buf);
        self.len += buf.len();
        Ok(())
    }

    /// Flush the bit buffer
    fn finalize_bit_buffer(&mut self) -> Result<(), EncodingError> {
        self.write_bits(0x7F, 7)?;

        // Match DefaultBitStream behavior: discard partial bits after flushing
        // complete bytes produced by the 0x7F terminator write.
        self.pending_byte = 0;
        self.pending_bits = 0;
        self.has_unfinalized_bits = false;
        Ok(())
    }

    /// Write bits to the bitstream. Sub-byte index is preserved across calls.
    fn write_bits(&mut self, value: u32, size: u8) -> Result<(), EncodingError> {
        if size == 0 {
            return Ok(());
        }

        debug_assert!(size <= 32, "write_bits size must be <= 32, got {size}");
        self.has_unfinalized_bits = true;

        let mut remaining = size;

        while remaining > 0 {
            if self.pending_bits == 0 && remaining >= 8 {
                let shift = u32::from(remaining - 8);
                let byte = ((value >> shift) & 0xFF) as u8;
                self.push_stuffed_byte(byte);
                remaining -= 8;
                continue;
            }

            let free = 8 - self.pending_bits;
            let take = remaining.min(free);
            let shift = u32::from(remaining - take);
            let mask = ((1u32 << u32::from(take)) - 1) as u8;
            let bits = ((value >> shift) as u8) & mask;

            self.pending_byte = (self.pending_byte << take) | bits;
            self.pending_bits += take;
            remaining -= take;

            if self.pending_bits == 8 {
                self.flush_pending_byte();
            }
        }

        Ok(())
    }
}

impl BitStream for &mut SimdVecBitStream {
    #[inline(always)]
    fn write(&mut self, buf: &[u8]) -> Result<(), EncodingError> {
        (**self).write(buf)
    }

    #[inline(always)]
    fn finalize_bit_buffer(&mut self) -> Result<(), EncodingError> {
        (**self).finalize_bit_buffer()
    }

    #[inline(always)]
    fn write_bits(&mut self, value: u32, size: u8) -> Result<(), EncodingError> {
        (**self).write_bits(value, size)
    }
}

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
        let mut stream = SimdVecBitStream::new(Vec::new());
        {
            let mut encoder = Encoder::new(&mut stream, cfg.quality);
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
        }
        stream.data
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
            let expected = encode_with_default(&data, width, height, cfg);
            let actual = encode_with_simd_stream(&data, width, height, cfg);
            assert_eq!(actual, expected, "bitstream mismatch for test case");
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

        let mut simd = SimdVecBitStream::new(Vec::new());
        simd.write(&[0x12, 0x34]).unwrap();
        simd.write_bits(0xFF, 8).unwrap();
        simd.write_bits(0b101, 3).unwrap();
        simd.write_bits(0xA5, 8).unwrap();
        simd.finalize_bit_buffer().unwrap();
        simd.write(&[0xFF, 0xD9]).unwrap();
        simd.write_bits(0x7F, 7).unwrap();
        simd.finalize_bit_buffer().unwrap();

        assert_eq!(simd.data, expected);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(
        expected = "write called after write_bits without an intervening finalize_bit_buffer"
    )]
    fn write_panics_in_debug_when_finalize_was_not_called() {
        let mut simd = SimdVecBitStream::new(Vec::new());
        simd.write_bits(0b101, 3).unwrap();
        let _ = simd.write(&[0x00]);
    }
}
