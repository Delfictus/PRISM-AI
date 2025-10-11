/// WebSocket handler with MessagePack support
use actix::{Actor, StreamHandler};
use actix_web_actors::ws;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::codec::MessagePackCodec;
use super::compression::{CompressionCodec, CompressionLevel};

/// WebSocket actor that supports MessagePack encoding
pub struct MessagePackWebSocket {
    /// Heartbeat interval
    hb_interval: Duration,
    /// Last heartbeat time
    last_hb: Instant,
    /// Use compression
    use_compression: bool,
    /// Compression level
    compression_level: CompressionLevel,
}

impl MessagePackWebSocket {
    /// Create new MessagePack WebSocket handler
    pub fn new(use_compression: bool) -> Self {
        Self {
            hb_interval: Duration::from_secs(5),
            last_hb: Instant::now(),
            use_compression,
            compression_level: CompressionLevel::Fast,
        }
    }

    /// Send MessagePack encoded message
    pub fn send_msgpack<T: Serialize>(&self, ctx: &mut ws::WebsocketContext<Self>, data: &T) {
        match MessagePackCodec::encode(data) {
            Ok(mut bytes) => {
                // Apply compression if enabled
                if self.use_compression {
                    match CompressionCodec::gzip_compress(&bytes, self.compression_level) {
                        Ok(compressed) => {
                            bytes = compressed;
                        }
                        Err(e) => {
                            eprintln!("Compression error: {}", e);
                        }
                    }
                }

                ctx.binary(bytes);
            }
            Err(e) => {
                eprintln!("MessagePack encoding error: {}", e);
            }
        }
    }

    /// Heartbeat to keep connection alive
    fn hb(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(self.hb_interval, |act, ctx| {
            if Instant::now().duration_since(act.last_hb) > Duration::from_secs(30) {
                println!("⚠️  WebSocket heartbeat timeout - disconnecting");
                ctx.stop();
                return;
            }
            ctx.ping(b"");
        });
    }
}

impl Actor for MessagePackWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        println!("🔌 MessagePack WebSocket connected (compression: {})",
                 self.use_compression);
        self.hb(ctx);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        println!("🔌 MessagePack WebSocket disconnected");
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for MessagePackWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                self.last_hb = Instant::now();
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                self.last_hb = Instant::now();
            }
            Ok(ws::Message::Binary(bytes)) => {
                // Handle incoming MessagePack data
                self.last_hb = Instant::now();

                // Decompress if needed
                let data = if self.use_compression {
                    match CompressionCodec::gzip_decompress(&bytes) {
                        Ok(decompressed) => decompressed,
                        Err(e) => {
                            eprintln!("Decompression error: {}", e);
                            return;
                        }
                    }
                } else {
                    bytes.to_vec()
                };

                // Example: decode as generic JSON value
                match MessagePackCodec::decode::<serde_json::Value>(&data) {
                    Ok(value) => {
                        println!("📦 Received MessagePack data: {:?}", value);
                    }
                    Err(e) => {
                        eprintln!("MessagePack decoding error: {}", e);
                    }
                }
            }
            Ok(ws::Message::Text(text)) => {
                // Fall back to JSON if client sends text
                println!("📝 Received text (JSON): {}", text);
            }
            Ok(ws::Message::Close(reason)) => {
                println!("🔌 WebSocket close: {:?}", reason);
                ctx.stop();
            }
            Err(e) => {
                eprintln!("⚠️  WebSocket error: {}", e);
                ctx.stop();
            }
            _ => {}
        }
    }
}

/// Message envelope for WebSocket communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageEnvelope<T> {
    pub message_type: String,
    pub timestamp: u64,
    pub payload: T,
}

impl<T> MessageEnvelope<T> {
    pub fn new(message_type: impl Into<String>, payload: T) -> Self {
        Self {
            message_type: message_type.into(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            payload,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestPayload {
        value: i32,
        message: String,
    }

    #[test]
    fn test_message_envelope() {
        let payload = TestPayload {
            value: 42,
            message: "test".to_string(),
        };

        let envelope = MessageEnvelope::new("test_message", payload.clone());

        assert_eq!(envelope.message_type, "test_message");
        assert_eq!(envelope.payload, payload);
        assert!(envelope.timestamp > 0);
    }

    #[test]
    fn test_envelope_serialization() {
        let payload = TestPayload {
            value: 100,
            message: "serialization test".to_string(),
        };

        let envelope = MessageEnvelope::new("test", payload);

        // Encode to MessagePack
        let bytes = MessagePackCodec::encode(&envelope).unwrap();

        // Decode back
        let decoded: MessageEnvelope<TestPayload> = MessagePackCodec::decode(&bytes).unwrap();

        assert_eq!(decoded.message_type, "test");
        assert_eq!(decoded.payload.value, 100);
    }

    #[test]
    fn test_compressed_envelope() {
        let payload = TestPayload {
            value: 999,
            message: "compression test with a longer message".to_string(),
        };

        let envelope = MessageEnvelope::new("test", payload);

        // Encode to MessagePack
        let msgpack = MessagePackCodec::encode(&envelope).unwrap();

        // Compress
        let compressed = CompressionCodec::gzip_compress(&msgpack, CompressionLevel::Default).unwrap();

        // Decompress
        let decompressed = CompressionCodec::gzip_decompress(&compressed).unwrap();

        // Decode
        let decoded: MessageEnvelope<TestPayload> = MessagePackCodec::decode(&decompressed).unwrap();

        assert_eq!(decoded.payload.value, 999);
    }
}
