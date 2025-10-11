/// MessagePack compression for WebSocket communication
///
/// Binary protocol for efficient data transmission
/// Week 3 Enhancement: MessagePack Compression

pub mod codec;
pub mod websocket_handler;
pub mod compression;

pub use codec::*;
pub use websocket_handler::*;
pub use compression::*;
