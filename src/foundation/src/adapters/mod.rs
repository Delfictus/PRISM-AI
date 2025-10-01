//! Data source adapters for various data types

pub mod market_data;
pub mod sensor_data;
pub mod synthetic;

pub use market_data::AlpacaMarketDataSource;
pub use sensor_data::OpticalSensorArray;
pub use synthetic::SyntheticDataSource;
