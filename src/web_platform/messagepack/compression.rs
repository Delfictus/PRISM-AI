/// Additional compression utilities for MessagePack data
use flate2::Compression;
use flate2::write::{GzEncoder, ZlibEncoder};
use flate2::read::{GzDecoder, ZlibDecoder};
use std::io::{Write, Read};

/// Compression level
#[derive(Debug, Clone, Copy)]
pub enum CompressionLevel {
    None,
    Fast,
    Default,
    Best,
}

impl CompressionLevel {
    fn to_flate2(&self) -> Compression {
        match self {
            CompressionLevel::None => Compression::none(),
            CompressionLevel::Fast => Compression::fast(),
            CompressionLevel::Default => Compression::default(),
            CompressionLevel::Best => Compression::best(),
        }
    }
}

/// Compression utilities
pub struct CompressionCodec;

impl CompressionCodec {
    /// Compress data with Gzip
    pub fn gzip_compress(data: &[u8], level: CompressionLevel) -> Result<Vec<u8>, String> {
        let mut encoder = GzEncoder::new(Vec::new(), level.to_flate2());
        encoder.write_all(data)
            .map_err(|e| format!("Compression failed: {}", e))?;
        encoder.finish()
            .map_err(|e| format!("Compression finalization failed: {}", e))
    }

    /// Decompress Gzip data
    pub fn gzip_decompress(data: &[u8]) -> Result<Vec<u8>, String> {
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| format!("Decompression failed: {}", e))?;
        Ok(decompressed)
    }

    /// Compress data with Zlib
    pub fn zlib_compress(data: &[u8], level: CompressionLevel) -> Result<Vec<u8>, String> {
        let mut encoder = ZlibEncoder::new(Vec::new(), level.to_flate2());
        encoder.write_all(data)
            .map_err(|e| format!("Compression failed: {}", e))?;
        encoder.finish()
            .map_err(|e| format!("Compression finalization failed: {}", e))
    }

    /// Decompress Zlib data
    pub fn zlib_decompress(data: &[u8]) -> Result<Vec<u8>, String> {
        let mut decoder = ZlibDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| format!("Decompression failed: {}", e))?;
        Ok(decompressed)
    }

    /// Calculate compression ratio
    pub fn compression_ratio(original_size: usize, compressed_size: usize) -> f64 {
        if original_size == 0 {
            return 0.0;
        }
        (1.0 - (compressed_size as f64 / original_size as f64)) * 100.0
    }
}

/// Compression statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompressionStats {
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub compression_time_ms: f64,
}

impl CompressionStats {
    pub fn new(original_size: usize, compressed_size: usize, compression_time_ms: f64) -> Self {
        let compression_ratio = CompressionCodec::compression_ratio(original_size, compressed_size);
        Self {
            original_size,
            compressed_size,
            compression_ratio,
            compression_time_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gzip_compression() {
        let data = b"Hello, World! This is a test string for compression.".repeat(100);

        let compressed = CompressionCodec::gzip_compress(&data, CompressionLevel::Default).unwrap();
        let decompressed = CompressionCodec::gzip_decompress(&compressed).unwrap();

        assert_eq!(data.to_vec(), decompressed);
        assert!(compressed.len() < data.len());

        let ratio = CompressionCodec::compression_ratio(data.len(), compressed.len());
        println!("Gzip compression ratio: {:.2}%", ratio);
        assert!(ratio > 50.0); // Should achieve significant compression on repeated data
    }

    #[test]
    fn test_zlib_compression() {
        let data = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit.".repeat(50);

        let compressed = CompressionCodec::zlib_compress(&data, CompressionLevel::Default).unwrap();
        let decompressed = CompressionCodec::zlib_decompress(&compressed).unwrap();

        assert_eq!(data.to_vec(), decompressed);
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn test_compression_levels() {
        let data = b"Test data for compression level comparison.".repeat(100);

        let fast = CompressionCodec::gzip_compress(&data, CompressionLevel::Fast).unwrap();
        let default = CompressionCodec::gzip_compress(&data, CompressionLevel::Default).unwrap();
        let best = CompressionCodec::gzip_compress(&data, CompressionLevel::Best).unwrap();

        // Best compression should give smallest size
        assert!(best.len() <= default.len());
        assert!(default.len() <= fast.len());
    }

    #[test]
    fn test_compression_stats() {
        let stats = CompressionStats::new(1000, 250, 1.5);

        assert_eq!(stats.original_size, 1000);
        assert_eq!(stats.compressed_size, 250);
        assert_eq!(stats.compression_ratio, 75.0);
        assert_eq!(stats.compression_time_ms, 1.5);
    }
}
