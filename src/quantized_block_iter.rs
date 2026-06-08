use std::vec::Vec;
use crate::{AlignedBlock, Encoder, ImageBuffer, QuantizationTable};
use crate::encoder::{get_block, get_max_sampling_size, init_rows, Component, Operations};

pub struct QBlockIter<'a>{
    rows: [Vec<u8>; 4],
    component_iter: std::iter::Enumerate<&'a Component>,
    i: usize
}

impl<'a> QBlockIter<'a> {
    pub fn encode_blocks<I: ImageBuffer, OP: Operations>(
        image: &I,
        q_tables: &[QuantizationTable; 2],
        components: &'a [Component],
    ) -> (Self, usize) {
        let (max_h_sampling, max_v_sampling) = get_max_sampling_size(components);

        let converted_image = convert_full_image(image, components, max_h_sampling, max_v_sampling);

        let ConvertedImage {
            rows,
            max_num_chunk_cols,
            max_num_chunk_rows,
            min_num_chunk_cols,
            min_num_chunk_rows
        } = converted_image;

        let buffer_width = max_num_chunk_cols * 8;

        for (i, component) in components.iter().enumerate() {
            let h_scale = max_h_sampling / component.horizontal_sampling_factor as usize;
            let v_scale = max_v_sampling / component.vertical_sampling_factor as usize;

            let num_cols = min_num_chunk_cols.div_ceil(h_scale);
            let num_rows = min_num_chunk_rows.div_ceil(v_scale);

            debug_assert!(num_cols > 0);
            debug_assert!(num_rows > 0);

            for block_y in 0..num_rows {
                for block_x in 0..num_cols {
                    let mut block = get_block(
                        &rows[i],
                        block_x * 8 * h_scale,
                        block_y * 8 * v_scale,
                        h_scale,
                        v_scale,
                        buffer_width,
                    );

                    OP::fdct(&mut block);

                    let mut q_block = AlignedBlock::default();

                    OP::quantize_block(
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
}

impl Iterator for QBlockIter {
    type Item = (usize, AlignedBlock);

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}


pub struct ConvertedImage{
    pub rows: [Vec<u8>; 4],
    /// This is the number of the total number of chunks needed.
    ///  When downsampling with factor 2, if `min_num_chunk_cols` is odd,
    ///  then we need to pad the image.
    ///  This number includes the padding but is still in 8 pixel blocks
    pub max_num_chunk_cols: usize,
    pub max_num_chunk_rows: usize,
    /// This number is the smallest number of normal 8x8 chunks needed to cover all original pixels.
    pub min_num_chunk_cols: usize,
    pub min_num_chunk_rows: usize,
}

pub fn convert_full_image<I: ImageBuffer, OP: Operations>(
    image: &I,
    components: &[Component],
    max_h_sampling: usize,
    max_v_sampling: usize,
) -> ConvertedImage {
    let pixel_width = image.width();
    let pixel_height = image.height();

    // number of 8 chunks rounded up to the sampling on the channel with most subsampling
    let max_num_chunk_cols = usize::from(pixel_width).div_ceil(8 * max_h_sampling) * max_h_sampling;
    let max_num_chunk_rows = usize::from(pixel_height).div_ceil(8 * max_v_sampling) * max_v_sampling;

    debug_assert!(max_num_chunk_cols > 0);
    debug_assert!(max_num_chunk_rows > 0);

    let buffer_width = max_num_chunk_cols * 8;
    let buffer_size = max_num_chunk_cols * max_num_chunk_rows * 64;

    let mut rows: [Vec<u8>; 4] = init_rows(components, buffer_size);

    for y in 0..max_num_chunk_rows * 8 {
        let y = (y.min(usize::from(pixel_height) - 1)) as u16;

        image.fill_buffers(y, &mut rows);

        for _ in usize::from(pixel_width)..buffer_width {
            for channel in &mut rows {
                if !channel.is_empty() {
                    channel.push(channel[channel.len() - 1]);
                }
            }
        }
    }

    let min_num_chunk_cols = usize::from(pixel_width).div_ceil(8);
    let min_num_chunk_rows = usize::from(pixel_height).div_ceil(8);

    debug_assert!(min_num_chunk_cols > 0);
    debug_assert!(min_num_chunk_rows > 0);
    ConvertedImage{
        rows,
        max_num_chunk_cols,
        max_num_chunk_rows,
        min_num_chunk_cols,
        min_num_chunk_rows,
    }
}