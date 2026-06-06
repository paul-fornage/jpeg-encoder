use crate::AlignedBlock;
use crate::huffman::HuffmanTable;
use alloc::vec::Vec;
use core::fmt::{Debug, Formatter};
use serde;

#[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
pub struct HuffmanSampleDataSet {
    pub huffman_tables: [(HuffmanTable, HuffmanTable); 2],
    pub samples: Vec<HuffmanSampleData>,
}

#[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
pub struct HuffmanSampleData {
    pub block: AlignedBlock,
    pub last_dc: i16,
    pub dc_huffman_table: u8,
    pub ac_huffman_table: u8,
}

// #[cfg(test)]
impl serde::Serialize for AlignedBlock {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeTuple;
        let mut seq = serializer.serialize_tuple(64)?;
        for item in self.data {
            seq.serialize_element(&item)?;
        }
        seq.end()
    }
}

// #[cfg(test)]
impl<'de> serde::Deserialize<'de> for AlignedBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct AlignedBlockVisitor;

        impl<'de> serde::de::Visitor<'de> for AlignedBlockVisitor {
            type Value = AlignedBlock;

            fn expecting(&self, f: &mut Formatter) -> core::fmt::Result {
                write!(f, "a tuple of 64 i16 values")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut data = [0i16; 64];
                for i in 0..64 {
                    data[i] = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                }
                Ok(AlignedBlock { data })
            }
        }

        deserializer.deserialize_tuple(64, AlignedBlockVisitor)
    }
}

// #[cfg(test)]
impl PartialEq for AlignedBlock {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

// #[cfg(test)]
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

/*
pub struct HuffmanTable {
    lookup_table: [(u8, u16); 256],
    length: [u8; 16],
    values: Vec<u8>,
}
*/
// #[cfg(test)]
impl serde::Serialize for HuffmanTable {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeTuple;
        let mut seq = serializer.serialize_tuple(3)?;
        seq.serialize_element(&LookupTable(&self.lookup_table))?;
        seq.serialize_element(&Length(&self.length))?;
        seq.serialize_element(&self.values)?;
        seq.end()
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

// #[cfg(test)]
impl<'de> serde::Deserialize<'de> for HuffmanTable {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct HuffmanTableVisitor;

        impl<'de> serde::de::Visitor<'de> for HuffmanTableVisitor {
            type Value = HuffmanTable;

            fn expecting(&self, f: &mut Formatter) -> core::fmt::Result {
                write!(f, "a tuple of (lookup_table, length, values)")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let lookup_table: LookupTableOwned = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let length: LengthOwned = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let values: Vec<u8> = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
                Ok(HuffmanTable {
                    lookup_table: lookup_table.0,
                    length: length.0,
                    values,
                })
            }
        }

        deserializer.deserialize_tuple(3, HuffmanTableVisitor)
    }
}

// #[cfg(test)]
struct LookupTable<'a>(&'a [(u8, u16); 256]);

// #[cfg(test)]
impl<'a> serde::Serialize for LookupTable<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeTuple;
        let mut seq = serializer.serialize_tuple(256)?;
        for item in self.0 {
            seq.serialize_element(item)?;
        }
        seq.end()
    }
}

// #[cfg(test)]
struct LookupTableOwned([(u8, u16); 256]);

// #[cfg(test)]
impl<'de> serde::Deserialize<'de> for LookupTableOwned {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct LookupTableVisitor;

        impl<'de> serde::de::Visitor<'de> for LookupTableVisitor {
            type Value = LookupTableOwned;

            fn expecting(&self, f: &mut Formatter) -> core::fmt::Result {
                write!(f, "a tuple of 256 (u8, u16) values")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut data = [(0u8, 0u16); 256];
                for i in 0..256 {
                    data[i] = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                }
                Ok(LookupTableOwned(data))
            }
        }

        deserializer.deserialize_tuple(256, LookupTableVisitor)
    }
}

// #[cfg(test)]
struct Length<'a>(&'a [u8; 16]);

// #[cfg(test)]
impl<'a> serde::Serialize for Length<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeTuple;
        let mut seq = serializer.serialize_tuple(16)?;
        for item in self.0 {
            seq.serialize_element(item)?;
        }
        seq.end()
    }
}

// #[cfg(test)]
struct LengthOwned([u8; 16]);

// #[cfg(test)]
impl<'de> serde::Deserialize<'de> for LengthOwned {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct LengthVisitor;

        impl<'de> serde::de::Visitor<'de> for LengthVisitor {
            type Value = LengthOwned;

            fn expecting(&self, f: &mut Formatter) -> core::fmt::Result {
                write!(f, "a tuple of 16 u8 values")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut data = [0u8; 16];
                for i in 0..16 {
                    data[i] = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                }
                Ok(LengthOwned(data))
            }
        }

        deserializer.deserialize_tuple(16, LengthVisitor)
    }
}

#[cfg(test)]
#[cfg(feature = "generate-huffman-data")]
mod tests {
    use super::*;
    use std::{array, println};
    use crate::{ColorType, Encoder, SamplingFactor};
    use crate::tests::create_test_img_rgb;

    fn sample_data_test_gen(seed: u64) -> HuffmanSampleData {
        HuffmanSampleData {
            block: AlignedBlock {
                data: array::from_fn(|i| {
                    ((seed.overflowing_add(i as u64).0 % (u16::MAX as u64)) as u16).cast_signed()
                }),
            },
            last_dc: ((seed % (u16::MAX as u64)) as u16).cast_signed(),
            dc_huffman_table: (seed & 0b1) as u8,
            ac_huffman_table: (seed & 0b1) as u8,
        }
    }

    #[test]
    fn test_serde_huffman_samples() {
        let samples = HuffmanSampleDataSet {
            huffman_tables: [
                (
                    HuffmanTable::default_luma_dc(),
                    HuffmanTable::default_luma_ac(),
                ),
                (
                    HuffmanTable::default_chroma_dc(),
                    HuffmanTable::default_chroma_ac(),
                ),
            ],
            samples: (0..5).map(|i| sample_data_test_gen(i)).collect(),
        };

        let ir = serde_json::to_string(&samples).unwrap();

        println!("{}", ir);

        let recovered_samples: HuffmanSampleDataSet = serde_json::from_str(&ir).unwrap();

        assert_eq!(samples, recovered_samples);
    }

    #[test]
    fn generate_real_image_sample_huffman_data() {
        let img = image::open("criterion/sample-image.png")
            .expect("failed to open test image")
            .into_rgb8();
        let (width, height) = img.dimensions();
        let mut buf = Vec::with_capacity(img.pixels().len());
        let mut encoder = crate::Encoder::new(&mut buf, 85);
        encoder.set_sampling_factor(crate::SamplingFactor::F_1_1);

        encoder
            .encode(
                img.as_raw(),
                width as u16,
                height as u16,
                crate::ColorType::Rgb,
            )
            .unwrap();
    }

    #[test]
    fn generate_test_image_sample_huffman_data() {
        let (data, width, height) = create_test_img_rgb();

        let mut result = Vec::new();
        let mut encoder = Encoder::new(&mut result, 100);
        encoder.set_sampling_factor(SamplingFactor::F_2_1);
        encoder
            .encode(&data, width, height, ColorType::Rgb)
            .unwrap();
    }
}
