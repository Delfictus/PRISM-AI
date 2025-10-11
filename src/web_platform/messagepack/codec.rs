/// MessagePack codec for efficient serialization
use serde::{Deserialize, Serialize};

/// MessagePack encoding/decoding utilities
pub struct MessagePackCodec;

impl MessagePackCodec {
    /// Encode data to MessagePack binary format
    pub fn encode<T: Serialize>(data: &T) -> Result<Vec<u8>, CodecError> {
        rmp_serde::to_vec(data)
            .map_err(|e| CodecError::EncodingFailed(e.to_string()))
    }

    /// Decode MessagePack binary to data
    pub fn decode<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<T, CodecError> {
        rmp_serde::from_slice(bytes)
            .map_err(|e| CodecError::DecodingFailed(e.to_string()))
    }

    /// Encode with named fields (more compact for structs)
    pub fn encode_named<T: Serialize>(data: &T) -> Result<Vec<u8>, CodecError> {
        rmp_serde::to_vec_named(data)
            .map_err(|e| CodecError::EncodingFailed(e.to_string()))
    }

    /// Get size reduction percentage compared to JSON
    pub fn compression_ratio<T: Serialize>(data: &T) -> Result<f64, CodecError> {
        let json_size = serde_json::to_string(data)
            .map_err(|e| CodecError::EncodingFailed(e.to_string()))?
            .len();

        let msgpack_size = Self::encode(data)?.len();

        let reduction = 1.0 - (msgpack_size as f64 / json_size as f64);
        Ok(reduction * 100.0)
    }
}

/// Codec errors
#[derive(Debug, Clone)]
pub enum CodecError {
    EncodingFailed(String),
    DecodingFailed(String),
}

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodecError::EncodingFailed(msg) => write!(f, "Encoding failed: {}", msg),
            CodecError::DecodingFailed(msg) => write!(f, "Decoding failed: {}", msg),
        }
    }
}

impl std::error::Error for CodecError {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestData {
        id: u32,
        name: String,
        value: f64,
        tags: Vec<String>,
    }

    #[test]
    fn test_encode_decode() {
        let data = TestData {
            id: 42,
            name: "test".to_string(),
            value: 3.14,
            tags: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        };

        let encoded = MessagePackCodec::encode(&data).unwrap();
        let decoded: TestData = MessagePackCodec::decode(&encoded).unwrap();

        assert_eq!(data, decoded);
    }

    #[test]
    fn test_compression_ratio() {
        let data = TestData {
            id: 42,
            name: "test_compression".to_string(),
            value: 3.14159,
            tags: vec!["tag1".to_string(), "tag2".to_string(), "tag3".to_string()],
        };

        let ratio = MessagePackCodec::compression_ratio(&data).unwrap();
        println!("Compression ratio: {:.2}%", ratio);

        // MessagePack should be more compact than JSON
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_large_structure() {
        let data: Vec<TestData> = (0..1000)
            .map(|i| TestData {
                id: i,
                name: format!("test_{}", i),
                value: i as f64 * 3.14,
                tags: vec![
                    format!("tag_{}", i),
                    format!("tag_{}", i + 1),
                    format!("tag_{}", i + 2),
                ],
            })
            .collect();

        let encoded = MessagePackCodec::encode(&data).unwrap();
        let decoded: Vec<TestData> = MessagePackCodec::decode(&encoded).unwrap();

        assert_eq!(data.len(), decoded.len());
        assert_eq!(data[0], decoded[0]);
        assert_eq!(data[999], decoded[999]);
    }
}
