use jpeg_encoder::{AlignedBlock, HuffmanTable};

pub struct HuffmanSampleDataSet {
    huffman_tables: [(HuffmanTable, HuffmanTable); 2],
    samples: Vec<HuffmanSampleData>,
}

pub struct HuffmanSampleData{
    block: AlignedBlock,
    last_dc: i16,
    dc_huffman_table: u8,
    ac_huffman_table: u8,
}

