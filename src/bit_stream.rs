use crate::EncodingError;

const BUFFER_SIZE: usize = core::mem::size_of::<usize>() * 8;

/// A no_std alternative for `std::io::Write`
///
/// An implementation of a subset of `std::io::Write` necessary to use the encoder without `std`.
/// This trait is implemented for `std::io::Write` if the `std` feature is enabled.
pub trait JfifWrite {
    /// Writes the whole buffer. The behavior must be identical to std::io::Write::write_all
    /// # Errors
    ///
    /// Return an error if the data can't be written
    fn write_all(&mut self, buf: &[u8]) -> Result<(), EncodingError>;
}

#[cfg(not(feature = "std"))]
impl<W: JfifWrite + ?Sized> JfifWrite for &mut W {
    fn write_all(&mut self, buf: &[u8]) -> Result<(), EncodingError> {
        (**self).write_all(buf)
    }
}

#[cfg(not(feature = "std"))]
impl JfifWrite for alloc::vec::Vec<u8> {
    fn write_all(&mut self, buf: &[u8]) -> Result<(), EncodingError> {
        self.extend_from_slice(buf);
        Ok(())
    }
}

#[cfg(feature = "std")]
impl<W: std::io::Write + ?Sized> JfifWrite for W {
    #[inline(always)]
    fn write_all(&mut self, buf: &[u8]) -> Result<(), EncodingError> {
        self.write_all(buf)?;
        Ok(())
    }
}

pub struct DefaultBitStream<W: JfifWrite> {
    w: W,
    bit_buffer: usize,
    free_bits: i8,
}

impl<W: JfifWrite> DefaultBitStream<W>{
    pub fn new(w: W) -> Self {
        DefaultBitStream {
            w,
            bit_buffer: 0,
            free_bits: BUFFER_SIZE as i8,
        }
    }

    #[inline(always)]
    fn flush_byte_from_bit_buffer(&mut self, free_bits: i8) -> Result<(), EncodingError> {
        let value = (self.bit_buffer >> (BUFFER_SIZE as i8 - 8 - free_bits)) & 0xFF;

        self.write_u8(value as u8)?;

        if value == 0xFF {
            self.write_u8(0x00)?;
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
            Ok(())
        } else {
            self.w.write_all(&self.bit_buffer.to_be_bytes())
        }
    }
}

impl<W: JfifWrite> BitStream for DefaultBitStream<W> {

    #[inline(always)]
    fn write(&mut self, buf: &[u8]) -> Result<(), EncodingError> {
        self.w.write_all(buf)
    }

    fn finalize_bit_buffer(&mut self) -> Result<(), EncodingError> {
        self.write_bits(0x7F, 7)?;
        self.flush_bit_buffer()?;
        self.bit_buffer = 0;
        self.free_bits = BUFFER_SIZE as i8;

        Ok(())
    }

    fn flush_bit_buffer(&mut self) -> Result<(), EncodingError> {
        while self.free_bits <= (BUFFER_SIZE as i8 - 8) {
            self.flush_byte_from_bit_buffer(self.free_bits)?;
            self.free_bits += 8;
        }

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


pub trait BitStream {
    fn write(&mut self, buf: &[u8]) -> Result<(), EncodingError>;
    fn write_u8(&mut self, value: u8) -> Result<(), EncodingError>{
        self.write(&[value])
    }
    fn write_u16(&mut self, value: u16) -> Result<(), EncodingError>{
        self.write(&value.to_be_bytes())
    }
    fn finalize_bit_buffer(&mut self) -> Result<(), EncodingError>;
    fn flush_bit_buffer(&mut self) -> Result<(), EncodingError>;
    // fn flush_byte_from_bit_buffer(&mut self, free_bits: i8) -> Result<(), EncodingError>;
    // fn write_bit_buffer(&mut self) -> Result<(), EncodingError>;
    fn write_bits(&mut self, value: u32, size: u8) -> Result<(), EncodingError>;
}