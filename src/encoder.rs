use crate::fdct::{DefaultFDCT, FDCT};
use crate::quantization::{DefaultBlockQuantizer, BlockQuantizer};
use crate::huffman::encoder::{DefaultHuffmanEncoder, HuffmanEncoder};
use crate::quantized_block_iter::encode_blocks_iter;
use crate::huffman::{CodingClass, HuffmanTable};
use crate::image_buffer::*;
use crate::marker::Marker;
use crate::quantization::{QuantizationTable, QuantizationTableType};
use crate::writer::{JfifWrite, JfifWriter};
use crate::{EncodingError, PixelDensity};

use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{Debug, Formatter};
#[cfg(feature = "std")]
use std::io::BufWriter;

#[cfg(feature = "std")]
use std::fs::File;

#[cfg(feature = "std")]
use std::path::Path;

#[cfg(feature = "simd")]
use crate::simd::{BgrImageSimd, BgraImageSimd, RgbaImageSimd, SimdBlockQuantizer, SimdHuffmanEncoder};
#[cfg(feature = "simd")]
use std::simd::{num::SimdUint, Simd};



/// # Color types used in encoding
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum JpegColorType {
    /// One component grayscale colorspace
    Luma,

    /// Three component YCbCr colorspace
    Ycbcr,

    /// 4 Component CMYK colorspace
    Cmyk,

    /// 4 Component YCbCrK colorspace
    Ycck,
}

#[derive(Copy, Clone)]
#[repr(C, align(32))]
pub struct AlignedBlock {
    pub data: [i16; 64],
}

impl Debug for AlignedBlock {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "AlignedBlock {{\n\t{:?}\n\t{:?}\n\t{:?}\n\t{:?}\n\t{:?}\n\t{:?}\n\t{:?}\n\t{:?}\n}}",
            &self.data[0..8],
            &self.data[8..16],
            &self.data[16..24],
            &self.data[24..32],
            &self.data[32..40],
            &self.data[40..48],
            &self.data[48..56],
            &self.data[56..64]
        )
    }
}

impl AlignedBlock {
    pub const fn new(data: [i16; 64]) -> Self {
        AlignedBlock { data }
    }
}

impl Default for AlignedBlock {
    fn default() -> Self {
        AlignedBlock { data: [0i16; 64] }
    }
}

impl JpegColorType {
    pub(crate) fn get_num_components(self) -> usize {
        use JpegColorType::*;

        match self {
            Luma => 1,
            Ycbcr => 3,
            Cmyk | Ycck => 4,
        }
    }
}

/// # Color types for input images
///
/// Available color input formats for [Encoder::encode]. Other types can be used
/// by implementing an [ImageBuffer](crate::ImageBuffer).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ColorType {
    /// Grayscale with 1 byte per pixel
    Luma,

    /// RGB with 3 bytes per pixel
    Rgb,

    /// Red, Green, Blue with 4 bytes per pixel. The alpha channel will be ignored during encoding.
    Rgba,

    /// RGB with 3 bytes per pixel
    Bgr,

    /// RGBA with 4 bytes per pixel. The alpha channel will be ignored during encoding.
    Bgra,

    /// YCbCr with 3 bytes per pixel.
    Ycbcr,

    /// CMYK with 4 bytes per pixel.
    Cmyk,

    /// CMYK with 4 bytes per pixel. Encoded as YCCK (YCbCrK)
    CmykAsYcck,

    /// YCCK (YCbCrK) with 4 bytes per pixel.
    Ycck,
}

impl ColorType {
    pub(crate) fn get_bytes_per_pixel(self) -> usize {
        use ColorType::*;

        match self {
            Luma => 1,
            Rgb | Bgr | Ycbcr => 3,
            Rgba | Bgra | Cmyk | CmykAsYcck | Ycck => 4,
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// # Sampling factors for chroma subsampling
///
/// ## Warning
/// Sampling factor of 4 are not supported by all decoders or applications
#[allow(non_camel_case_types)]
pub enum SamplingFactor {
    F_1_1 = 1 << 4 | 1,
    F_2_1 = 2 << 4 | 1,
    F_1_2 = 1 << 4 | 2,
    F_2_2 = 2 << 4 | 2,
    F_4_1 = 4 << 4 | 1,
    F_4_2 = 4 << 4 | 2,
    F_1_4 = 1 << 4 | 4,
    F_2_4 = 2 << 4 | 4,

    /// Alias for F_1_1
    R_4_4_4 = 0x80 | 1 << 4 | 1,

    /// Alias for F_1_2
    R_4_4_0 = 0x80 | 1 << 4 | 2,

    /// Alias for F_1_4
    R_4_4_1 = 0x80 | 1 << 4 | 4,

    /// Alias for F_2_1
    R_4_2_2 = 0x80 | 2 << 4 | 1,

    /// Alias for F_2_2
    R_4_2_0 = 0x80 | 2 << 4 | 2,

    /// Alias for F_2_4
    R_4_2_1 = 0x80 | 2 << 4 | 4,

    /// Alias for F_4_1
    R_4_1_1 = 0x80 | 4 << 4 | 1,

    /// Alias for F_4_2
    R_4_1_0 = 0x80 | 4 << 4 | 2,
}

impl SamplingFactor {
    /// Get variant for supplied factors or None if not supported
    pub fn from_factors(horizontal: u8, vertical: u8) -> Option<SamplingFactor> {
        use SamplingFactor::*;

        match (horizontal, vertical) {
            (1, 1) => Some(F_1_1),
            (1, 2) => Some(F_1_2),
            (1, 4) => Some(F_1_4),
            (2, 1) => Some(F_2_1),
            (2, 2) => Some(F_2_2),
            (2, 4) => Some(F_2_4),
            (4, 1) => Some(F_4_1),
            (4, 2) => Some(F_4_2),
            _ => None,
        }
    }

    pub(crate) fn get_sampling_factors(self) -> (u8, u8) {
        let value = self as u8;
        ((value >> 4) & 0x07, value & 0xf)
    }

    pub(crate) fn supports_interleaved(self) -> bool {
        use SamplingFactor::*;

        // Interleaved mode is only supported with h/v sampling factors of 1 or 2.
        // Sampling factors of 4 needs sequential encoding
        matches!(
            self,
            F_1_1 | F_2_1 | F_1_2 | F_2_2 | R_4_4_4 | R_4_4_0 | R_4_2_2 | R_4_2_0
        )
    }
}

pub struct Component {
    pub id: u8,
    pub quantization_table: u8,
    pub dc_huffman_table: u8,
    pub ac_huffman_table: u8,
    pub horizontal_sampling_factor: u8,
    pub vertical_sampling_factor: u8,
}

macro_rules! add_component {
    ($components:expr, $id:expr, $dest:expr, $h_sample:expr, $v_sample:expr) => {
        $components.push(Component {
            id: $id,
            quantization_table: $dest,
            dc_huffman_table: $dest,
            ac_huffman_table: $dest,
            horizontal_sampling_factor: $h_sample,
            vertical_sampling_factor: $v_sample,
        });
    };
}

/// # The JPEG encoder
pub struct Encoder<W: JfifWrite> {
    writer: JfifWriter<W>,
    density: PixelDensity,
    quality: u8,

    components: Vec<Component>,
    quantization_tables: [QuantizationTableType; 2],
    huffman_tables: [(HuffmanTable, HuffmanTable); 2],

    sampling_factor: SamplingFactor,

    progressive_scans: Option<u8>,

    restart_interval: Option<u16>,

    optimize_huffman_table: bool,

    app_segments: Vec<(u8, Vec<u8>)>,
}

impl<W: JfifWrite> Encoder<W> {
    /// Create a new encoder with the given quality
    ///
    /// The quality must be between 1 and 100 where 100 is the highest image quality.<br>
    /// By default, quality settings below 90 use a chroma subsampling (2x2 / 4:2:0) which can
    /// be changed with [set_sampling_factor](Encoder::set_sampling_factor)
    pub fn new(w: W, quality: u8) -> Encoder<W> {
        let huffman_tables = [
            (
                HuffmanTable::default_luma_dc(),
                HuffmanTable::default_luma_ac(),
            ),
            (
                HuffmanTable::default_chroma_dc(),
                HuffmanTable::default_chroma_ac(),
            ),
        ];

        let quantization_tables = [
            QuantizationTableType::Default,
            QuantizationTableType::Default,
        ];

        let sampling_factor = if quality < 90 {
            SamplingFactor::F_2_2
        } else {
            SamplingFactor::F_1_1
        };

        Encoder {
            writer: JfifWriter::new(w),
            density: PixelDensity::default(),
            quality,
            components: vec![],
            quantization_tables,
            huffman_tables,
            sampling_factor,
            progressive_scans: None,
            restart_interval: None,
            optimize_huffman_table: false,
            app_segments: Vec::new(),
        }
    }

    /// Set pixel density for the image
    ///
    /// By default, this value is None which is equal to "1 pixel per pixel".
    pub fn set_density(&mut self, density: PixelDensity) {
        self.density = density;
    }

    /// Return pixel density
    pub fn density(&self) -> PixelDensity {
        self.density
    }

    /// Set chroma subsampling factor
    pub fn set_sampling_factor(&mut self, sampling: SamplingFactor) {
        self.sampling_factor = sampling;
    }

    /// Get chroma subsampling factor
    pub fn sampling_factor(&self) -> SamplingFactor {
        self.sampling_factor
    }

    /// Set quantization tables for luma and chroma components
    pub fn set_quantization_tables(
        &mut self,
        luma: QuantizationTableType,
        chroma: QuantizationTableType,
    ) {
        self.quantization_tables = [luma, chroma];
    }

    /// Get configured quantization tables
    pub fn quantization_tables(&self) -> &[QuantizationTableType; 2] {
        &self.quantization_tables
    }

    /// Controls if progressive encoding is used.
    ///
    /// By default, progressive encoding uses 4 scans.<br>
    /// Use [set_progressive_scans](Encoder::set_progressive_scans) to use a different number of scans
    pub fn set_progressive(&mut self, progressive: bool) {
        self.progressive_scans = if progressive { Some(4) } else { None };
    }

    /// Set number of scans per component for progressive encoding
    ///
    /// Number of scans must be between 2 and 64.
    /// There is at least one scan for the DC coefficients and one for the remaining 63 AC coefficients.
    ///
    /// # Panics
    /// If number of scans is not within valid range
    pub fn set_progressive_scans(&mut self, scans: u8) {
        assert!(
            (2..=64).contains(&scans),
            "Invalid number of scans: {}",
            scans
        );
        self.progressive_scans = Some(scans);
    }

    /// Return number of progressive scans if progressive encoding is enabled
    pub fn progressive_scans(&self) -> Option<u8> {
        self.progressive_scans
    }

    /// Set restart interval
    ///
    /// Set numbers of MCUs between restart markers.
    pub fn set_restart_interval(&mut self, interval: u16) {
        self.restart_interval = if interval == 0 { None } else { Some(interval) };
    }

    /// Return the restart interval
    pub fn restart_interval(&self) -> Option<u16> {
        self.restart_interval
    }

    /// Set if optimized huffman table should be created
    ///
    /// Optimized tables result in slightly smaller file sizes but decrease encoding performance.
    pub fn set_optimized_huffman_tables(&mut self, optimize_huffman_table: bool) {
        self.optimize_huffman_table = optimize_huffman_table;
    }

    /// Returns if optimized huffman table should be generated
    pub fn optimized_huffman_tables(&self) -> bool {
        self.optimize_huffman_table
    }

    /// Appends a custom app segment to the JFIF file
    ///
    /// Segment numbers need to be in the range between 1 and 15<br>
    /// The maximum allowed data length is 2^16 - 2 bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the segment number is invalid or data exceeds the allowed size
    pub fn add_app_segment(&mut self, segment_nr: u8, data: Vec<u8>) -> Result<(), EncodingError> {
        if segment_nr == 0 || segment_nr > 15 {
            Err(EncodingError::InvalidAppSegment(segment_nr))
        } else if data.len() > 65533 {
            Err(EncodingError::AppSegmentTooLarge(data.len()))
        } else {
            self.app_segments.push((segment_nr, data));
            Ok(())
        }
    }

    /// Add an ICC profile
    ///
    /// The maximum allowed data length is 16,707,345 bytes.
    ///
    /// # Errors
    ///
    /// Returns an Error if the data exceeds the maximum size for the ICC profile
    pub fn add_icc_profile(&mut self, data: &[u8]) -> Result<(), EncodingError> {
        // Based on https://www.color.org/ICC_Minor_Revision_for_Web.pdf
        // B.4  Embedding ICC profiles in JFIF files

        const MARKER: &[u8; 12] = b"ICC_PROFILE\0";
        const MAX_CHUNK_LENGTH: usize = 65535 - 2 - 12 - 2;

        let num_chunks = data.len().div_ceil(MAX_CHUNK_LENGTH);

        // Sequence number is stored as a byte and starts with 1
        if num_chunks >= 255 {
            return Err(EncodingError::IccTooLarge(data.len()));
        }

        for (i, data) in data.chunks(MAX_CHUNK_LENGTH).enumerate() {
            let mut chunk_data = Vec::with_capacity(MAX_CHUNK_LENGTH);
            chunk_data.extend_from_slice(MARKER);
            chunk_data.push(i as u8 + 1);
            chunk_data.push(num_chunks as u8);
            chunk_data.extend_from_slice(data);

            self.add_app_segment(2, chunk_data)?;
        }

        Ok(())
    }

    /// Embeds Exif metadata into the image
    ///
    /// The maximum allowed data length is 65,528 bytes.
    ///
    /// # Errors
    ///
    /// Returns an Error if the data exceeds the maximum size for the Exif metadata
    pub fn add_exif_metadata(&mut self, data: &[u8]) -> Result<(), EncodingError> {
        // E x i f \0 \0
        /// The header for an EXIF APP1 segment
        const EXIF_HEADER: [u8; 6] = [0x45, 0x78, 0x69, 0x66, 0x00, 0x00];

        let mut formatted = EXIF_HEADER.to_vec();
        formatted.extend_from_slice(data);

        self.add_app_segment(1, formatted)
    }

    /// Encode an image
    ///
    /// Data format and length must conform to specified width, height and color type.
    pub fn encode(
        self, // TODO: Consumes self? Can't reuse buffers???????
        data: &[u8],
        width: u16,
        height: u16,
        color_type: ColorType,
    ) -> Result<(), EncodingError> {
        let required_data_len = width as usize * height as usize * color_type.get_bytes_per_pixel();

        if data.len() < required_data_len {
            return Err(EncodingError::BadImageData {
                length: data.len(),
                required: required_data_len,
            });
        }

        #[cfg(feature = "simd")]
        {
            use crate::simd::*;

            match color_type {
                ColorType::Luma => {
                    self.encode_image_internal::<_, SimdFDCT, SimdBlockQuantizer, SimdHuffmanEncoder>(GrayImage(data, width, height))
                }
                ColorType::Rgb => {
                    self.encode_image_internal::<_, SimdFDCT, SimdBlockQuantizer, SimdHuffmanEncoder>(RgbImageSimd(data, width, height))
                },
                ColorType::Rgba => {
                    self.encode_image_internal::<_, SimdFDCT, SimdBlockQuantizer, SimdHuffmanEncoder>(RgbaImageSimd(data, width, height))
                },
                ColorType::Bgr => {
                    self.encode_image_internal::<_, SimdFDCT, SimdBlockQuantizer, SimdHuffmanEncoder>(BgrImageSimd(data, width, height))
                },
                ColorType::Bgra => {
                    self.encode_image_internal::<_, SimdFDCT, SimdBlockQuantizer, SimdHuffmanEncoder>(BgraImageSimd(data, width, height))
                },
                ColorType::Ycbcr => {
                    self.encode_image_internal::<_, SimdFDCT, SimdBlockQuantizer, SimdHuffmanEncoder>(YCbCrImage(data, width, height))
                }
                ColorType::Cmyk => {
                    self.encode_image_internal::<_, SimdFDCT, SimdBlockQuantizer, SimdHuffmanEncoder>(CmykImage(data, width, height))
                }
                ColorType::CmykAsYcck => {
                    self.encode_image_internal::<_, SimdFDCT, SimdBlockQuantizer, SimdHuffmanEncoder>(CmykAsYcckImage(data, width, height))
                },
                ColorType::Ycck => {
                    self.encode_image_internal::<_, SimdFDCT, SimdBlockQuantizer, SimdHuffmanEncoder>(YcckImage(data, width, height))
                }
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            match color_type {
                ColorType::Luma => {
                    self.encode_image_internal::<_, DefaultFDCT, DefaultBlockQuantizer, DefaultHuffmanEncoder>(GrayImage(data, width, height))
                }
                ColorType::Rgb => {
                    self.encode_image_internal::<_, DefaultFDCT, DefaultBlockQuantizer, DefaultHuffmanEncoder>(RgbImage(data, width, height))
                },
                ColorType::Rgba => {
                    self.encode_image_internal::<_, DefaultFDCT, DefaultBlockQuantizer, DefaultHuffmanEncoder>(RgbaImage(data, width, height))
                },
                ColorType::Bgr => {
                    self.encode_image_internal::<_, DefaultFDCT, DefaultBlockQuantizer, DefaultHuffmanEncoder>(BgrImage(data, width, height))
                },
                ColorType::Bgra => {
                    self.encode_image_internal::<_, DefaultFDCT, DefaultBlockQuantizer, DefaultHuffmanEncoder>(BgraImage(data, width, height))
                },
                ColorType::Ycbcr => {
                    self.encode_image_internal::<_, DefaultFDCT, DefaultBlockQuantizer, DefaultHuffmanEncoder>(YCbCrImage(data, width, height))
                }
                ColorType::Cmyk => {
                    self.encode_image_internal::<_, DefaultFDCT, DefaultBlockQuantizer, DefaultHuffmanEncoder>(CmykImage(data, width, height))
                }
                ColorType::CmykAsYcck => {
                    self.encode_image_internal::<_, DefaultFDCT, DefaultBlockQuantizer, DefaultHuffmanEncoder>(CmykAsYcckImage(data, width, height))
                },
                ColorType::Ycck => {
                    self.encode_image_internal::<_, DefaultFDCT, DefaultBlockQuantizer, DefaultHuffmanEncoder>(YcckImage(data, width, height))
                }
            }
        }
    }

    fn encode_image_internal<I: ImageBuffer, F: FDCT, Q: BlockQuantizer, H: HuffmanEncoder>(
        mut self,
        image: I,
    ) -> Result<(), EncodingError> {
        if image.width() == 0 || image.height() == 0 {
            return Err(EncodingError::ZeroImageDimensions {
                width: image.width(),
                height: image.height(),
            });
        }

        let q_tables = [
            QuantizationTable::new_with_quality(&self.quantization_tables[0], self.quality, true),
            QuantizationTable::new_with_quality(&self.quantization_tables[1], self.quality, false),
        ];

        let jpeg_color_type = image.get_jpeg_color_type();
        self.init_components(jpeg_color_type);

        self.writer.write_marker(Marker::SOI)?;

        self.writer.write_header(&self.density)?;

        if jpeg_color_type == JpegColorType::Cmyk {
            //Set ColorTransform info to "Unknown"
            let app_14 = b"Adobe\0\0\0\0\0\0\0";
            self.writer
                .write_segment(Marker::APP(14), app_14.as_ref())?;
        } else if jpeg_color_type == JpegColorType::Ycck {
            //Set ColorTransform info to YCCK
            let app_14 = b"Adobe\0\0\0\0\0\0\x02";
            self.writer
                .write_segment(Marker::APP(14), app_14.as_ref())?;
        }

        for (nr, data) in &self.app_segments {
            self.writer.write_segment(Marker::APP(*nr), data)?;
        }

        if let Some(scans) = self.progressive_scans {
            self.encode_image_progressive::<_, F, Q>(image, scans, &q_tables)?;
        } else if self.optimize_huffman_table || !self.sampling_factor.supports_interleaved() {
            self.encode_image_sequential::<_, F, Q, H>(image, &q_tables)?;
        } else {
            self.encode_image_interleaved::<_, F, Q, H>(image, &q_tables)?;
        }

        self.writer.write_marker(Marker::EOI)?;

        Ok(())
    }

    pub fn init_components(&mut self, color: JpegColorType) {
        init_components(&mut self.components, self.sampling_factor, color);
    }


    fn write_frame_header<I: ImageBuffer>(
        &mut self,
        image: &I,
        q_tables: &[QuantizationTable; 2],
    ) -> Result<(), EncodingError> {
        self.writer.write_frame_header(
            image.width(),
            image.height(),
            &self.components,
            self.progressive_scans.is_some(),
        )?;

        self.writer.write_quantization_segment(0, &q_tables[0])?;
        self.writer.write_quantization_segment(1, &q_tables[1])?;

        self.writer
            .write_huffman_segment(CodingClass::Dc, 0, &self.huffman_tables[0].0)?;

        self.writer
            .write_huffman_segment(CodingClass::Ac, 0, &self.huffman_tables[0].1)?;

        if image.get_jpeg_color_type().get_num_components() >= 3 {
            self.writer
                .write_huffman_segment(CodingClass::Dc, 1, &self.huffman_tables[1].0)?;

            self.writer
                .write_huffman_segment(CodingClass::Ac, 1, &self.huffman_tables[1].1)?;
        }

        if let Some(restart_interval) = self.restart_interval {
            self.writer.write_dri(restart_interval)?;
        }

        Ok(())
    }



    /// Encode all components with one scan
    ///
    /// This is only valid for sampling factors of 1 and 2
    fn encode_image_interleaved<I: ImageBuffer, F: FDCT, Q: BlockQuantizer, H: HuffmanEncoder>(
        &mut self,
        image: I,
        q_tables: &[QuantizationTable; 2],
    ) -> Result<(), EncodingError> {
        self.write_frame_header(&image, q_tables)?;
        self.writer
            .write_scan_header(&self.components.iter().collect::<Vec<_>>(), None)?;

        let (max_h_sampling, max_v_sampling) = get_max_sampling_size(&self.components);

        let width = image.width();
        let height = image.height();

        let num_cols = usize::from(width).div_ceil(8 * max_h_sampling);
        let num_rows = usize::from(height).div_ceil(8 * max_v_sampling);

        let buffer_width = num_cols * 8 * max_h_sampling;
        let buffer_size = buffer_width * 8 * max_v_sampling;

        // contains sets of 8 rows broken down by component.
        // when subsampling factor for a components is greater than 1,
        //  this actually contains the next 8*subsampling
        let mut row: [Vec<u8>; 4] = allocate_component_vecs(&self.components, buffer_size);

        let mut prev_dc = [0i16; 4];

        let restart_interval = self.restart_interval.unwrap_or(0);
        let mut restarts = 0;
        let mut restarts_to_go = restart_interval;

        for block_y_index in 0..num_rows {
            for r in &mut row {
                r.clear();
            }

            for inter_block_y in 0..(8 * max_v_sampling) {
                let global_y = inter_block_y + block_y_index * 8 * max_v_sampling;

                let global_y = (global_y.min(height as usize - 1)) as u16;

                // fill the rows
                image.fill_buffers(global_y, &mut row);

                // pad out the rows
                for _ in usize::from(width)..buffer_width {
                    for channel in &mut row {
                        if !channel.is_empty() {
                            channel.push(channel[channel.len() - 1]);
                        }
                    }
                }
            }

            for block_x in 0..num_cols {
                if restart_interval > 0 && restarts_to_go == 0 {
                    self.writer.finalize_bit_buffer()?;
                    self.writer
                        .write_marker(Marker::RST((restarts % 8) as u8))?;

                    prev_dc[0] = 0;
                    prev_dc[1] = 0;
                    prev_dc[2] = 0;
                    prev_dc[3] = 0;
                }

                for (i, component) in self.components.iter().enumerate() {
                    for v_offset in 0..component.vertical_sampling_factor as usize {
                        for h_offset in 0..component.horizontal_sampling_factor as usize {
                            let mut block = get_block(
                                &row[i],
                                block_x * 8 * max_h_sampling + (h_offset * 8),
                                v_offset * 8,
                                max_h_sampling / component.horizontal_sampling_factor as usize,
                                max_v_sampling / component.vertical_sampling_factor as usize,
                                buffer_width,
                            );

                            F::fdct(&mut block);

                            let mut q_block = AlignedBlock::default();

                            Q::quantize_block(
                                &block,
                                &mut q_block,
                                &q_tables[component.quantization_table as usize],
                            );
                            H::write_block(
                                &mut self.writer,
                                &q_block,
                                prev_dc[i],
                                &self.huffman_tables[component.dc_huffman_table as usize].0,
                                &self.huffman_tables[component.ac_huffman_table as usize].1,
                            )?;

                            prev_dc[i] = q_block.data[0];
                        }
                    }
                }

                if restart_interval > 0 {
                    if restarts_to_go == 0 {
                        restarts_to_go = restart_interval;
                        restarts += 1;
                        restarts &= 7;
                    }
                    restarts_to_go -= 1;
                }
            }
        }

        self.writer.finalize_bit_buffer()?;

        Ok(())
    }

    /// Encode components with one scan per component
    fn encode_image_sequential<I: ImageBuffer, F: FDCT, Q: BlockQuantizer, H: HuffmanEncoder>(
        &mut self,
        image: I,
        q_tables: &[QuantizationTable; 2],
    ) -> Result<(), EncodingError> {
        let blocks = self.encode_blocks::<_, F, Q>(&image, q_tables);

        if self.optimize_huffman_table {
            self.optimize_huffman_table(&blocks);
        }

        self.write_frame_header(&image, q_tables)?;

        for (i, component) in self.components.iter().enumerate() {
            let restart_interval = self.restart_interval.unwrap_or(0);
            let mut restarts = 0;
            let mut restarts_to_go = restart_interval;

            self.writer.write_scan_header(&[component], None)?;

            let mut prev_dc = 0;

            for block in &blocks[i] {
                if restart_interval > 0 && restarts_to_go == 0 {
                    self.writer.finalize_bit_buffer()?;
                    self.writer
                        .write_marker(Marker::RST((restarts % 8) as u8))?;

                    prev_dc = 0;
                }

                H::write_block(
                    &mut self.writer,
                    block,
                    prev_dc,
                    &self.huffman_tables[component.dc_huffman_table as usize].0,
                    &self.huffman_tables[component.ac_huffman_table as usize].1,
                )?;

                prev_dc = block.data[0];

                if restart_interval > 0 {
                    if restarts_to_go == 0 {
                        restarts_to_go = restart_interval;
                        restarts += 1;
                        restarts &= 7;
                    }
                    restarts_to_go -= 1;
                }
            }

            self.writer.finalize_bit_buffer()?;
        }

        Ok(())
    }

    /// Encode image in progressive mode
    ///
    /// This only support spectral selection for now
    fn encode_image_progressive<I: ImageBuffer, F: FDCT, Q: BlockQuantizer>(
        &mut self,
        image: I,
        scans: u8,
        q_tables: &[QuantizationTable; 2],
    ) -> Result<(), EncodingError> {
        let blocks = self.encode_blocks::<_, F, Q>(&image, q_tables);

        if self.optimize_huffman_table {
            self.optimize_huffman_table(&blocks);
        }

        self.write_frame_header(&image, q_tables)?;

        // Phase 1: DC Scan
        //          Only the DC coefficients can be transfer in the first component scans
        for (i, component) in self.components.iter().enumerate() {
            self.writer.write_scan_header(&[component], Some((0, 0)))?;

            let restart_interval = self.restart_interval.unwrap_or(0);
            let mut restarts = 0;
            let mut restarts_to_go = restart_interval;

            let mut prev_dc = 0;

            for block in &blocks[i] {
                if restart_interval > 0 && restarts_to_go == 0 {
                    self.writer.finalize_bit_buffer()?;
                    self.writer
                        .write_marker(Marker::RST((restarts % 8) as u8))?;

                    prev_dc = 0;
                }

                DefaultHuffmanEncoder::write_dc(
                    &mut self.writer,
                    block.data[0],
                    prev_dc,
                    &self.huffman_tables[component.dc_huffman_table as usize].0,
                )?;

                prev_dc = block.data[0];

                if restart_interval > 0 {
                    if restarts_to_go == 0 {
                        restarts_to_go = restart_interval;
                        restarts += 1;
                        restarts &= 7;
                    }
                    restarts_to_go -= 1;
                }
            }

            self.writer.finalize_bit_buffer()?;
        }

        // Phase 2: AC scans
        let scans = scans as usize - 1;

        let values_per_scan = 64 / scans;

        for scan in 0..scans {
            let start = (scan * values_per_scan).max(1);
            let end = if scan == scans - 1 {
                // ensure last scan is always transfers the remaining coefficients
                64
            } else {
                (scan + 1) * values_per_scan
            };

            for (i, component) in self.components.iter().enumerate() {
                let restart_interval = self.restart_interval.unwrap_or(0);
                let mut restarts = 0;
                let mut restarts_to_go = restart_interval;

                self.writer
                    .write_scan_header(&[component], Some((start as u8, end as u8 - 1)))?;

                for block in &blocks[i] {
                    if restart_interval > 0 && restarts_to_go == 0 {
                        self.writer.finalize_bit_buffer()?;
                        self.writer
                            .write_marker(Marker::RST((restarts % 8) as u8))?;
                    }

                    DefaultHuffmanEncoder::write_ac_block(
                        &mut self.writer,
                        block,
                        start,
                        end,
                        &self.huffman_tables[component.ac_huffman_table as usize].1,
                    )?;

                    if restart_interval > 0 {
                        if restarts_to_go == 0 {
                            restarts_to_go = restart_interval;
                            restarts += 1;
                            restarts &= 7;
                        }
                        restarts_to_go -= 1;
                    }
                }

                self.writer.finalize_bit_buffer()?;
            }
        }

        Ok(())
    }

    pub fn encode_blocks<I: ImageBuffer, F: FDCT, Q: BlockQuantizer>(
        &mut self,
        image: &I,
        q_tables: &[QuantizationTable; 2],
    ) -> [Vec<AlignedBlock>; 4] {
        let width = image.width();
        let height = image.height();
        let (max_h_sampling, max_v_sampling) = get_max_sampling_size(&self.components);

        let num_cols = usize::from(width).div_ceil(8 * max_h_sampling) * max_h_sampling;
        let num_rows = usize::from(height).div_ceil(8 * max_v_sampling) * max_v_sampling;

        debug_assert!(num_cols > 0);
        debug_assert!(num_rows > 0);

        let buffer_width = num_cols * 8;
        let buffer_size = num_cols * num_rows * 64;

        let mut row: [Vec<u8>; 4] = allocate_component_vecs(&self.components, buffer_size);

        for y in 0..num_rows * 8 {
            let y = (y.min(usize::from(height) - 1)) as u16;

            image.fill_buffers(y, &mut row);

            for _ in usize::from(width)..num_cols * 8 {
                for channel in &mut row {
                    if !channel.is_empty() {
                        channel.push(channel[channel.len() - 1]);
                    }
                }
            }
        }

        let num_cols = usize::from(width).div_ceil(8);
        let num_rows = usize::from(height).div_ceil(8);

        debug_assert!(num_cols > 0);
        debug_assert!(num_rows > 0);

        let mut blocks: [Vec<AlignedBlock>; 4] = allocate_component_vecs(&self.components, buffer_size / 64);

        for (i, component) in self.components.iter().enumerate() {
            let h_scale = max_h_sampling / component.horizontal_sampling_factor as usize;
            let v_scale = max_v_sampling / component.vertical_sampling_factor as usize;

            let cols = num_cols.div_ceil(h_scale);
            let rows = num_rows.div_ceil(v_scale);

            debug_assert!(cols > 0);
            debug_assert!(rows > 0);

            for block_y in 0..rows {
                for block_x in 0..cols {
                    let mut block = get_block(
                        &row[i],
                        block_x * 8 * h_scale,
                        block_y * 8 * v_scale,
                        h_scale,
                        v_scale,
                        buffer_width,
                    );

                    F::fdct(&mut block);

                    let mut q_block = AlignedBlock::default();

                    Q::quantize_block(
                        &block,
                        &mut q_block,
                        &q_tables[component.quantization_table as usize],
                    );

                    blocks[i].push(q_block);
                }
            }
        }
        blocks
    }


    pub fn encode_blocks_iter<I: ImageBuffer, F: FDCT, Q: BlockQuantizer>(
        &mut self,
        image: &I,
        q_tables: &[QuantizationTable; 2],
    ) -> [Vec<AlignedBlock>; 4] {
        let iters = encode_blocks_iter::<I, F, Q>(image, q_tables, &self.components);
        let mut fellas = iters.into_iter().map(|iter| {
            iter.collect::<Vec<AlignedBlock>>()
        }).collect::<heapless::Vec<Vec<AlignedBlock>, 4>>();
        while fellas.len() < 4 {
            fellas.push(Vec::new()).expect("Expect to be able to push to list with capacity 4 while length is less than 4");
        }
        fellas.into_array().expect("Expect to be able to convert heapless Vec to fixed array of length 4 after filling with empty vectors")
    }


    // Create new huffman tables optimized for this image
    fn optimize_huffman_table(&mut self, blocks: &[Vec<AlignedBlock>; 4]) {
        // TODO: Find out if it's possible to reuse some code from the writer

        let max_tables = self.components.len().min(2) as u8;

        for table in 0..max_tables {
            let mut dc_freq = [0u32; 257];
            dc_freq[256] = 1;
            let mut ac_freq = [0u32; 257];
            ac_freq[256] = 1;

            let mut had_ac = false;
            let mut had_dc = false;

            for (i, component) in self.components.iter().enumerate() {
                if component.dc_huffman_table == table {
                    had_dc = true;

                    let mut prev_dc = 0;

                    debug_assert!(!blocks[i].is_empty());

                    for block in &blocks[i] {
                        let value = block.data[0];
                        let diff = value - prev_dc;
                        let num_bits = get_num_bits(diff);

                        dc_freq[num_bits as usize] += 1;

                        prev_dc = value;
                    }
                }

                if component.ac_huffman_table == table {
                    had_ac = true;

                    if let Some(scans) = self.progressive_scans {
                        let scans = scans as usize - 1;

                        let values_per_scan = 64 / scans;

                        for scan in 0..scans {
                            let start = (scan * values_per_scan).max(1);
                            let end = if scan == scans - 1 {
                                // Due to rounding we might need to transfer more than values_per_scan values in the last scan
                                64
                            } else {
                                (scan + 1) * values_per_scan
                            };

                            debug_assert!(!blocks[i].is_empty());

                            for block in &blocks[i] {
                                let mut zero_run = 0;

                                for &value in &block.data[start..end] {
                                    if value == 0 {
                                        zero_run += 1;
                                    } else {
                                        while zero_run > 15 {
                                            ac_freq[0xF0] += 1;
                                            zero_run -= 16;
                                        }
                                        let num_bits = get_num_bits(value);
                                        let symbol = (zero_run << 4) | num_bits;

                                        ac_freq[symbol as usize] += 1;

                                        zero_run = 0;
                                    }
                                }

                                if zero_run > 0 {
                                    ac_freq[0] += 1;
                                }
                            }
                        }
                    } else {
                        for block in &blocks[i] {
                            let mut zero_run = 0;

                            for &value in &block.data[1..] {
                                if value == 0 {
                                    zero_run += 1;
                                } else {
                                    while zero_run > 15 {
                                        ac_freq[0xF0] += 1;
                                        zero_run -= 16;
                                    }
                                    let num_bits = get_num_bits(value);
                                    let symbol = (zero_run << 4) | num_bits;

                                    ac_freq[symbol as usize] += 1;

                                    zero_run = 0;
                                }
                            }

                            if zero_run > 0 {
                                ac_freq[0] += 1;
                            }
                        }
                    }
                }
            }

            assert!(had_dc, "Missing DC data for table {}", table);
            assert!(had_ac, "Missing AC data for table {}", table);

            self.huffman_tables[table as usize] = (
                HuffmanTable::new_optimized(dc_freq),
                HuffmanTable::new_optimized(ac_freq),
            );
        }
    }
}

#[cfg(feature = "std")]
impl Encoder<BufWriter<File>> {
    /// Create a new decoder that writes into a file
    ///
    /// See [new](Encoder::new) for further information.
    ///
    /// # Errors
    ///
    /// Returns an `IoError(std::io::Error)` if the file can't be created
    pub fn new_file<P: AsRef<Path>>(
        path: P,
        quality: u8,
    ) -> Result<Encoder<BufWriter<File>>, EncodingError> {
        let file = File::create(path)?;
        let buf = BufWriter::new(file);
        Ok(Self::new(buf, quality))
    }
}

pub fn get_max_sampling_size(components: &[Component]) -> (usize, usize) {
    let max_h_sampling = components.iter().fold(1, |value, component| {
        value.max(component.horizontal_sampling_factor)
    });

    let max_v_sampling = components.iter().fold(1, |value, component| {
        value.max(component.vertical_sampling_factor)
    });

    (usize::from(max_h_sampling), usize::from(max_v_sampling))
}

pub(crate) fn allocate_component_vecs<T>(components: &[Component], buffer_size: usize) -> [Vec<T>; 4] {
    // To simplify the code and to give the compiler more infos to optimize stuff we always initialize 4 components
    // Resource overhead should be minimal because an empty Vec doesn't allocate

    match components.len() {
        1 => [
            Vec::with_capacity(buffer_size),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ],
        3 => [
            Vec::with_capacity(buffer_size),
            Vec::with_capacity(buffer_size),
            Vec::with_capacity(buffer_size),
            Vec::new(),
        ],
        4 => [
            Vec::with_capacity(buffer_size),
            Vec::with_capacity(buffer_size),
            Vec::with_capacity(buffer_size),
            Vec::with_capacity(buffer_size),
        ],
        len => unreachable!("Unsupported component length: {}", len),
    }
}

pub fn init_components(
    components: &mut Vec<Component>,
    sampling_factor: SamplingFactor,
    color: JpegColorType
) {
    let (horizontal_sampling_factor, vertical_sampling_factor) =
        sampling_factor.get_sampling_factors();
    match color {
        JpegColorType::Luma => {
            add_component!(components, 0, 0, 1, 1);
        }
        JpegColorType::Ycbcr => {
            add_component!(
                    components,
                    0,
                    0,
                    horizontal_sampling_factor,
                    vertical_sampling_factor
                );
            add_component!(components, 1, 1, 1, 1);
            add_component!(components, 2, 1, 1, 1);
        }
        JpegColorType::Cmyk => {
            add_component!(components, 0, 1, 1, 1);
            add_component!(components, 1, 1, 1, 1);
            add_component!(components, 2, 1, 1, 1);
            add_component!(
                    components,
                    3,
                    0,
                    horizontal_sampling_factor,
                    vertical_sampling_factor
                );
        }
        JpegColorType::Ycck => {
            add_component!(
                    components,
                    0,
                    0,
                    horizontal_sampling_factor,
                    vertical_sampling_factor
                );
            add_component!(components, 1, 1, 1, 1);
            add_component!(components, 2, 1, 1, 1);
            add_component!(
                    components,
                    3,
                    0,
                    horizontal_sampling_factor,
                    vertical_sampling_factor
                );
        }
    }
}

pub fn get_block(
    data: &[u8],
    start_x: usize,
    start_y: usize,
    col_stride: usize,
    row_stride: usize,
    width: usize,
) -> AlignedBlock {
    #[cfg(feature = "simd")]
    {
        if col_stride == 1 {
            get_block_simd(data, start_x, start_y, col_stride, row_stride, width)
        } else {
            get_block_linear(data, start_x, start_y, col_stride, row_stride, width)
        }
    }
    #[cfg(not(feature = "simd"))]
    {
        get_block_linear(data, start_x, start_y, col_stride, row_stride, width)
    }
}

pub fn get_block_linear(
    data: &[u8],
    start_x: usize,
    start_y: usize,
    col_stride: usize,
    row_stride: usize,
    width: usize,
) -> AlignedBlock {
    let mut block = [0i16; 64];

    for y in 0..8 {
        for x in 0..8 {
            let ix = start_x + (x * col_stride);
            let iy = start_y + (y * row_stride);

            block[y * 8 + x] = (data[iy * width + ix] as i16) - 128;
        }
    }

    AlignedBlock::new(block)
}

#[cfg(feature = "simd")]
pub fn get_block_simd(
    data: &[u8],
    start_x: usize,
    start_y: usize,
    col_stride: usize,
    row_stride: usize,
    width: usize,
) -> AlignedBlock {
    debug_assert!(col_stride == 1, "cannot do horizontal sub sampling with SIMD. (can but no gains. Use `get_block` instead)");
    let mut block = [0i16; 64];
    for y in 0..8 {
        let iy = start_y + (y * row_stride);
        let row_bytes = Simd::<u8, 8>::from_slice(&data[iy * width + start_x..]);
        let row: Simd::<i16, 8> = row_bytes.cast() - Simd::<i16, 8>::splat(128);
        row.copy_to_slice(&mut block[y * 8..(y + 1) * 8])
    }

    AlignedBlock::new(block)
}

fn get_num_bits(mut value: i16) -> u8 {
    if value < 0 {
        value = -value;
    }

    let mut num_bits = 0;

    while value > 0 {
        num_bits += 1;
        value >>= 1;
    }

    num_bits
}




#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;
    use super::*;
    use crate::encoder::get_num_bits;
    use crate::image_buffer::RgbImage;
    use crate::quantization::{DefaultBlockQuantizer, QuantizationTable, QuantizationTableType};
    use crate::writer::get_code;
    use crate::{Encoder, SamplingFactor};

    #[test]
    fn test_get_num_bits() {
        let min_max = 2i16.pow(13);

        for value in -min_max..=min_max {
            let num_bits1 = get_num_bits(value);
            let (num_bits2, _) = get_code(value);

            assert_eq!(
                num_bits1, num_bits2,
                "Difference in num bits for value {}: {} vs {}",
                value, num_bits1, num_bits2
            );
        }
    }

    #[test]
    fn sampling_factors() {
        assert_eq!(SamplingFactor::F_1_1.get_sampling_factors(), (1, 1));
        assert_eq!(SamplingFactor::F_2_1.get_sampling_factors(), (2, 1));
        assert_eq!(SamplingFactor::F_1_2.get_sampling_factors(), (1, 2));
        assert_eq!(SamplingFactor::F_2_2.get_sampling_factors(), (2, 2));
        assert_eq!(SamplingFactor::F_4_1.get_sampling_factors(), (4, 1));
        assert_eq!(SamplingFactor::F_4_2.get_sampling_factors(), (4, 2));
        assert_eq!(SamplingFactor::F_1_4.get_sampling_factors(), (1, 4));
        assert_eq!(SamplingFactor::F_2_4.get_sampling_factors(), (2, 4));

        assert_eq!(SamplingFactor::R_4_4_4.get_sampling_factors(), (1, 1));
        assert_eq!(SamplingFactor::R_4_4_0.get_sampling_factors(), (1, 2));
        assert_eq!(SamplingFactor::R_4_4_1.get_sampling_factors(), (1, 4));
        assert_eq!(SamplingFactor::R_4_2_2.get_sampling_factors(), (2, 1));
        assert_eq!(SamplingFactor::R_4_2_0.get_sampling_factors(), (2, 2));
        assert_eq!(SamplingFactor::R_4_2_1.get_sampling_factors(), (2, 4));
        assert_eq!(SamplingFactor::R_4_1_1.get_sampling_factors(), (4, 1));
        assert_eq!(SamplingFactor::R_4_1_0.get_sampling_factors(), (4, 2));
    }

    #[test]
    fn test_set_progressive() {
        let mut encoder = Encoder::new(vec![], 100);
        encoder.set_progressive(true);
        assert_eq!(encoder.progressive_scans(), Some(4));

        encoder.set_progressive(false);
        assert_eq!(encoder.progressive_scans(), None);
    }

    #[test]
    fn test_encode_blocks_iter_matches_encode_blocks() {
        let width = 587u16;
        let height = 181u16;
        let mut data = Vec::with_capacity(width as usize * height as usize * 3);

        for y in 0..height {
            for x in 0..width {
                data.push((x * 7 + y * 3) as u8);
                data.push((x * 5 + y * 11) as u8);
                data.push((x * 13 + y * 17) as u8);
            }
        }

        let image = RgbImage(&data, width, height);
        let q_tables = [
            QuantizationTable::new_with_quality(&QuantizationTableType::Default, 90, true),
            QuantizationTable::new_with_quality(&QuantizationTableType::Default, 90, false),
        ];

        for sampling_factor in [SamplingFactor::F_1_1, SamplingFactor::F_2_2] {
            let mut encoder_old = Encoder::new(vec![], 90);
            encoder_old.set_sampling_factor(sampling_factor);
            encoder_old.init_components(super::JpegColorType::Ycbcr);

            let old_blocks = encoder_old.encode_blocks::<_, DefaultFDCT, DefaultBlockQuantizer>(&image, &q_tables);

            let mut encoder_new = Encoder::new(vec![], 90);
            encoder_new.set_sampling_factor(sampling_factor);
            encoder_new.init_components(super::JpegColorType::Ycbcr);

            let iter_blocks = encoder_new.encode_blocks_iter::<_, DefaultFDCT, DefaultBlockQuantizer>(&image, &q_tables);

            old_blocks
                .iter()
                .zip(iter_blocks.iter())
                .enumerate().for_each(|(component_index, (old_component, new_component))|
            {
                assert_eq!(
                    old_component.len(),
                    new_component.len(),
                    "different number of blocks for sampling {:?}, component {}",
                    sampling_factor,
                    component_index
                );

                for (block_index, (old_block, new_block)) in
                    old_component.iter().zip(new_component.iter()).enumerate()
                {
                    assert_eq!(
                        old_block.data,
                        new_block.data,
                        "different block for sampling {:?}, component {}, block {}",
                        sampling_factor,
                        component_index,
                        block_index
                    );
                }
            })
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_get_block_linear_match_get_block_optimized() {
        use crate::encoder::{get_block_linear, get_block_simd};

        let source_data = (0..=u16::MAX).map(|p|(p & 0x00FF) as u8).collect::<vec::Vec<_>>();

        let normal = get_block_linear(&source_data, 0, 0, 1, 1, 256);
        let optimized = get_block_simd(&source_data, 0, 0, 1, 1, 256);
        assert_eq!(normal.data, optimized.data);

        // Test case: Non-zero starting offsets (x and y)
        let normal = get_block_linear(&source_data, 7, 13, 1, 1, 256);
        let optimized = get_block_simd(&source_data, 7, 13, 1, 1, 256);
        assert_eq!(normal.data, optimized.data);

        // Test case: Increased row_stride for vertically subsampled/spaced data
        let normal = get_block_linear(&source_data, 0, 0, 1, 2, 256);
        let optimized = get_block_simd(&source_data, 0, 0, 1, 2, 256);
        assert_eq!(normal.data, optimized.data);

        // Test case: Unique odd width, along with row_stride and offsets combined
        let normal = get_block_linear(&source_data, 15, 8, 1, 3, 127);
        let optimized = get_block_simd(&source_data, 15, 8, 1, 3, 127);
        assert_eq!(normal.data, optimized.data);

        // Test case: Near the end of the buffer boundaries
        let normal = get_block_linear(&source_data, 248, 248, 1, 1, 256);
        let optimized = get_block_simd(&source_data, 248, 248, 1, 1, 256);
        assert_eq!(normal.data, optimized.data);
    }
}
