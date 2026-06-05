use jpeg_encoder::{AlignedBlock, HuffmanTable};

pub struct HuffmanSampleDataSet{
    huffman_tables: [(HuffmanTable, HuffmanTable); 2],
    samples: Vec<HuffmanSampleData>,
}

pub struct HuffmanSampleData{
    block: AlignedBlock,
    last_dc: i16,
    table_id: u8,
}

