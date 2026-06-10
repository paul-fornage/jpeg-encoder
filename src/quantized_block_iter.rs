use core::fmt::Formatter;
use std::marker::PhantomData;
use std::vec::Vec;
use crate::quantization::{BlockQuantizer, QuantizationTable};
use crate::fdct::FDCT;
use crate::{ImageBuffer};
use crate::encoder::{get_block, get_max_sampling_size, allocate_component_vecs, Component, AlignedBlock};


pub fn encode_blocks_iter<'a, I: ImageBuffer, F: FDCT, Q: BlockQuantizer>(
    image: &I,
    q_tables: &'a [QuantizationTable; 2],
    components: &[Component],
) -> heapless::Vec<QBlockComponentIter<'a, F, Q>, 4> {
    let (max_h_sampling, max_v_sampling) = get_max_sampling_size(components);

    let converted_image = convert_full_image::<I>(image, components, max_h_sampling, max_v_sampling);
    
    let mut out: heapless::Vec<QBlockComponentIter<'a, F, Q>, 4> = heapless::Vec::new();

    let ConvertedImage {
        rows: all_rows,
        buffer_width,
        num_chunk_cols,
        num_chunk_rows
    } = converted_image;


    for (i, (component, rows)) in components.iter().zip(all_rows).enumerate() {
        let h_scale = max_h_sampling / component.horizontal_sampling_factor as usize;
        let v_scale = max_v_sampling / component.vertical_sampling_factor as usize;

        let num_cols = num_chunk_cols.div_ceil(h_scale);
        let num_rows = num_chunk_rows.div_ceil(v_scale);

        debug_assert!(num_cols > 0);
        debug_assert!(num_rows > 0);

        out.push(QBlockComponentIter::<'a, F, Q>{
            rows,
            q_table: &q_tables[component.quantization_table as usize],
            h_scale,
            v_scale,
            buffer_width,
            i,
            num_cols,
            num_rows,
            next_index: 0,
            _fdct_impl: Default::default(),
            _block_quantizer: Default::default(),
        }).unwrap();

    }
    out
}

pub struct QBlockComponentIter<'a, F: FDCT, Q: BlockQuantizer>{
    rows: Vec<u8>,
    q_table: &'a QuantizationTable,
    h_scale: usize,
    v_scale: usize,
    buffer_width: usize,
    i: usize,
    num_cols: usize,
    num_rows: usize,
    next_index: usize,
    _fdct_impl: PhantomData<F>,
    _block_quantizer: PhantomData<Q>,
}

impl<'a, F: FDCT, Q: BlockQuantizer> QBlockComponentIter<'a, F, Q>{
    pub fn eval(&mut self, block_x: usize, block_y: usize) -> AlignedBlock {
        let mut block = get_block(
            &self.rows,
            block_x * 8 * self.h_scale,
            block_y * 8 * self.v_scale,
            self.h_scale,
            self.v_scale,
            self.buffer_width,
        );

        F::fdct(&mut block);
        let mut q_block = AlignedBlock::default();
        Q::quantize_block(
            &block,
            &mut q_block,
            &self.q_table,
        );

        q_block
    }

    pub fn new(rows: Vec<u8>, q_table: &'a QuantizationTable, h_scale: usize, v_scale: usize,
               buffer_width: usize, i: usize, num_cols: usize, num_rows: usize
    ) -> Self {
        QBlockComponentIter::<'a, F, Q>{
            rows,
            q_table,
            h_scale,
            v_scale,
            buffer_width,
            i,
            num_cols,
            num_rows,
            next_index: 0,
            _fdct_impl: Default::default(),
            _block_quantizer: Default::default(),
        }
    }
}

impl<'a, F: FDCT, Q: BlockQuantizer> core::fmt::Debug for QBlockComponentIter<'a, F, Q>{
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "QBlockComponentIter {{\
            h_scale: {},\
            v_scale: {},\
            buffer_width: {},\
            i: {},\
            num_cols: {},\
            num_rows: {}\
        }}", self.h_scale, self.v_scale, self.buffer_width, self.i, self.num_cols, self.num_rows)
    }
}

impl<'a, F: FDCT, Q: BlockQuantizer> Iterator for QBlockComponentIter<'a, F, Q> {
    type Item = AlignedBlock;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.num_cols * self.num_rows {
            return None;
        }

        let index = self.next_index;
        self.next_index += 1;

        let block_x = index % self.num_cols;
        let block_y = index / self.num_cols;

        Some(self.eval(block_x, block_y))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len();
        (remaining, Some(remaining))
    }
}

impl<'a, F: FDCT, Q: BlockQuantizer> ExactSizeIterator for QBlockComponentIter<'a, F, Q> {
    fn len(&self) -> usize {
        (self.num_cols * self.num_rows) - self.next_index
    }
}


pub struct ConvertedImage{
    pub rows: [Vec<u8>; 4],
    buffer_width: usize,
    num_chunk_cols: usize,
    num_chunk_rows: usize,
}

pub fn convert_full_image<I: ImageBuffer>(
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

    let mut rows: [Vec<u8>; 4] = allocate_component_vecs(components, buffer_size);

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

    let num_chunk_cols = usize::from(pixel_width).div_ceil(8);
    let num_chunk_rows = usize::from(pixel_height).div_ceil(8);

    debug_assert!(num_chunk_cols > 0);
    debug_assert!(num_chunk_rows > 0);
    ConvertedImage{
        rows,
        buffer_width,
        num_chunk_cols,
        num_chunk_rows
    }
}


