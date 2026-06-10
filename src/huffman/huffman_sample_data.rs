use crate::huffman::HuffmanTable;
use crate::{init_components, AlignedBlock, Component, ImageBuffer, QuantizationTable, QuantizationTableType, SamplingFactor};
use alloc::vec::Vec;
use core::fmt::{Debug, Formatter};
use crate::quantization::BlockQuantizer;
use crate::quantized_block_iter::encode_blocks_iter;
use crate::fdct::FDCT;

#[derive(PartialEq, Debug)]
pub struct HuffmanSampleDataSet {
    pub huffman_tables: [(HuffmanTable, HuffmanTable); 2],
    pub samples: Vec<HuffmanSampleData>,
}

#[derive(PartialEq, Debug)]
pub struct HuffmanSampleData {
    pub block: AlignedBlock,
    pub last_dc: i16,
    pub dc_huffman_table: u8,
    pub ac_huffman_table: u8,
}


// #[cfg(test)]
impl PartialEq for AlignedBlock {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}


// #[cfg(test)]
impl PartialEq for HuffmanTable {
    fn eq(&self, other: &Self) -> bool {
        self.lookup_table == other.lookup_table
            && self.length == other.length
            && self.values == other.values
    }
}

impl Clone for HuffmanTable {
    fn clone(&self) -> Self {
        Self {
            lookup_table: self.lookup_table.clone(),
            length: self.length.clone(),
            values: self.values.clone(),
        }
    }
}

// #[cfg(test)]
impl Debug for HuffmanTable {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "HuffmanTable {{ lookup_table: {:?}, length: {:?}, values: {:?} }}",
            self.lookup_table, self.length, self.values
        )
    }
}

impl HuffmanSampleDataSet{
    pub fn from_image<I: ImageBuffer, F: FDCT, Q: BlockQuantizer>(image: &I) -> Self {

        let q_tables = [
            QuantizationTable::new_with_quality(&QuantizationTableType::Default, 90, true),
            QuantizationTable::new_with_quality(&QuantizationTableType::Default, 90, false),
        ];


        let mut components: Vec<Component> = Vec::new();
        init_components(&mut components, SamplingFactor::F_1_1, image.get_jpeg_color_type());

        let q_block_iters = encode_blocks_iter::<I, F, Q>(&image, &q_tables, &components);

        let huffman_tables = [(
            HuffmanTable::default_luma_dc(),
            HuffmanTable::default_luma_ac(),
        ), (
            HuffmanTable::default_chroma_dc(),
            HuffmanTable::default_chroma_ac(),
        )];

        let mut samples: Vec<HuffmanSampleData> = Vec::new();
        for (q_block_buffer, component) in q_block_iters.into_iter().zip(components) {
            let mut last_dc = 0;
            for q_block in q_block_buffer{
                samples.push(HuffmanSampleData {
                    block: q_block,
                    last_dc,
                    dc_huffman_table: component.dc_huffman_table,
                    ac_huffman_table: component.ac_huffman_table,
                });
                last_dc = q_block.data[0];
            }
        }

        HuffmanSampleDataSet { huffman_tables, samples }
    }
}