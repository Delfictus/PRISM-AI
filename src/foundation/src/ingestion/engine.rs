//! Ingestion engine for managing multiple data sources

use super::buffer::CircularBuffer;
use super::types::{DataPoint, DataSource, SourceInfo};
use anyhow::Result;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};

/// Statistics for ingestion performance monitoring
#[derive(Debug, Clone)]
pub struct IngestionStats {
    /// Total data points ingested
    pub total_points: usize,
    /// Total bytes ingested
    pub total_bytes: usize,
    /// Last update timestamp
    pub last_update: Instant,
    /// Average ingestion rate (points/sec)
    pub average_rate_hz: f64,
    /// Number of active sources
    pub active_sources: usize,
    /// Number of errors encountered
    pub error_count: usize,
}

impl Default for IngestionStats {
    fn default() -> Self {
        Self {
            total_points: 0,
            total_bytes: 0,
            last_update: Instant::now(),
            average_rate_hz: 0.0,
            active_sources: 0,
            error_count: 0,
        }
    }
}

/// Main ingestion engine managing multiple data sources
pub struct IngestionEngine {
    /// Historical buffer for data points
    buffer: CircularBuffer<DataPoint>,
    /// Channel sender for new data
    tx: mpsc::Sender<DataPoint>,
    /// Channel receiver for new data
    rx: Option<mpsc::Receiver<DataPoint>>,
    /// Performance statistics
    stats: Arc<RwLock<IngestionStats>>,
    /// Start time for rate calculation
    start_time: Instant,
}

impl IngestionEngine {
    /// Create a new ingestion engine
    ///
    /// # Arguments
    /// * `channel_size` - Size of the async channel buffer
    /// * `history_size` - Size of the historical circular buffer
    pub fn new(channel_size: usize, history_size: usize) -> Self {
        let (tx, rx) = mpsc::channel(channel_size);

        Self {
            buffer: CircularBuffer::new(history_size),
            tx,
            rx: Some(rx),
            stats: Arc::new(RwLock::new(IngestionStats::default())),
            start_time: Instant::now(),
        }
    }

    /// Start ingesting from a data source
    ///
    /// Spawns an async task to continuously read from the source
    pub async fn start_source(&mut self, mut source: Box<dyn DataSource>) -> Result<()> {
        // Connect to source
        source.connect().await?;

        let source_info = source.get_source_info();
        log::info!("Starting ingestion from: {}", source_info.name);

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.active_sources += 1;
        }

        // Spawn ingestion task
        let tx = self.tx.clone();
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            if let Err(e) = Self::ingest_from_source(source, tx, stats).await {
                log::error!("Ingestion task failed: {}", e);
            }
        });

        Ok(())
    }

    /// Internal task to ingest from a single source
    async fn ingest_from_source(
        mut source: Box<dyn DataSource>,
        tx: mpsc::Sender<DataPoint>,
        stats: Arc<RwLock<IngestionStats>>,
    ) -> Result<()> {
        let source_info = source.get_source_info();

        loop {
            match source.read_batch().await {
                Ok(batch) => {
                    let batch_size = batch.len();

                    if batch_size == 0 {
                        // No data, wait a bit
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        continue;
                    }

                    // Send each point through the channel
                    for point in batch {
                        if tx.send(point).await.is_err() {
                            log::warn!("Ingestion channel closed for {}", source_info.name);
                            return Ok(());
                        }
                    }

                    // Update statistics
                    let mut s = stats.write().await;
                    s.total_points += batch_size;
                    s.last_update = Instant::now();
                }
                Err(e) => {
                    log::error!("Error reading from {}: {}", source_info.name, e);

                    // Update error count
                    let mut s = stats.write().await;
                    s.error_count += 1;

                    // Wait before retrying
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }

    /// Get a batch of data points
    ///
    /// Blocks until `size` points are available or `timeout` is reached
    pub async fn get_batch(&mut self, size: usize, timeout: Duration) -> Result<Vec<DataPoint>> {
        let rx = self
            .rx
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Receiver already taken"))?;

        let mut batch = Vec::with_capacity(size);
        let deadline = tokio::time::Instant::now() + timeout;

        while batch.len() < size {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(point)) => {
                    // Add to buffer
                    self.buffer.push(point.clone()).await;
                    batch.push(point);
                }
                Ok(None) => {
                    // Channel closed
                    break;
                }
                Err(_) => {
                    // Timeout
                    break;
                }
            }
        }

        Ok(batch)
    }

    /// Get a single data point (blocking with timeout)
    pub async fn get_point(&mut self, timeout: Duration) -> Result<DataPoint> {
        let rx = self
            .rx
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Receiver already taken"))?;

        match tokio::time::timeout(timeout, rx.recv()).await {
            Ok(Some(point)) => {
                self.buffer.push(point.clone()).await;
                Ok(point)
            }
            Ok(None) => Err(anyhow::anyhow!("Channel closed")),
            Err(_) => Err(anyhow::anyhow!("Timeout waiting for data")),
        }
    }

    /// Get recent historical data
    pub async fn get_history(&self, n: usize) -> Vec<DataPoint> {
        self.buffer.get_recent(n).await
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> IngestionStats {
        let mut stats = self.stats.read().await.clone();

        // Calculate average rate
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            stats.average_rate_hz = stats.total_points as f64 / elapsed;
        }

        stats
    }

    /// Get buffer size
    pub async fn buffer_size(&self) -> usize {
        self.buffer.len().await
    }

    /// Clear historical buffer
    pub async fn clear_buffer(&self) {
        self.buffer.clear().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingestion::types::{DataPoint, DataSource, SourceInfo};

    struct MockSource {
        counter: usize,
        batch_size: usize,
    }

    #[async_trait::async_trait]
    impl DataSource for MockSource {
        async fn connect(&mut self) -> Result<()> {
            Ok(())
        }

        async fn read_batch(&mut self) -> Result<Vec<DataPoint>> {
            let mut batch = Vec::new();
            for _ in 0..self.batch_size {
                batch.push(DataPoint::new(
                    chrono::Utc::now().timestamp_millis(),
                    vec![self.counter as f64],
                ));
                self.counter += 1;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(batch)
        }

        async fn disconnect(&mut self) -> Result<()> {
            Ok(())
        }

        fn get_source_info(&self) -> SourceInfo {
            SourceInfo {
                name: "MockSource".to_string(),
                data_type: "test".to_string(),
                sampling_rate_hz: 100.0,
                dimensions: 1,
            }
        }
    }

    #[tokio::test]
    async fn test_ingestion_engine_basic() {
        let mut engine = IngestionEngine::new(100, 1000);

        let source = Box::new(MockSource {
            counter: 0,
            batch_size: 5,
        });

        engine.start_source(source).await.unwrap();

        // Wait for some data
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Get a batch
        let batch = engine
            .get_batch(10, Duration::from_millis(500))
            .await
            .unwrap();

        assert!(!batch.is_empty());
        assert!(batch.len() <= 10);
    }

    #[tokio::test]
    async fn test_ingestion_stats() {
        let mut engine = IngestionEngine::new(100, 1000);

        let source = Box::new(MockSource {
            counter: 0,
            batch_size: 10,
        });

        engine.start_source(source).await.unwrap();

        // Wait for data
        tokio::time::sleep(Duration::from_millis(200)).await;

        let stats = engine.get_stats().await;
        assert!(stats.total_points > 0);
        assert_eq!(stats.active_sources, 1);
    }
}
