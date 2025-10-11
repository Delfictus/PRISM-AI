/// GPU metrics collector with caching and aggregation
use super::nvml_wrapper::*;
use super::types::*;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};

/// GPU metrics collector with caching
pub struct GpuMetricsCollector {
    nvml: Arc<NvmlContext>,
    cache: Arc<RwLock<MetricsCache>>,
    cache_duration: Duration,
}

/// Metrics cache
struct MetricsCache {
    metrics: Vec<GpuMetrics>,
    last_update: Instant,
}

impl GpuMetricsCollector {
    /// Create new collector
    pub fn new() -> Result<Self, GpuError> {
        let nvml = NvmlContext::init()?;

        Ok(Self {
            nvml: Arc::new(nvml),
            cache: Arc::new(RwLock::new(MetricsCache {
                metrics: Vec::new(),
                last_update: Instant::now() - Duration::from_secs(3600),
            })),
            cache_duration: Duration::from_secs(1), // Cache for 1 second
        })
    }

    /// Create with custom cache duration
    pub fn with_cache_duration(cache_duration: Duration) -> Result<Self, GpuError> {
        let mut collector = Self::new()?;
        collector.cache_duration = cache_duration;
        Ok(collector)
    }

    /// Get device count
    pub fn device_count(&self) -> u32 {
        self.nvml.device_count()
    }

    /// Check if using simulated data
    pub fn is_simulated(&self) -> bool {
        self.nvml.is_simulated()
    }

    /// Get metrics for all devices
    pub async fn collect_all(&self) -> Result<Vec<GpuMetrics>, GpuError> {
        // Check cache
        {
            let cache = self.cache.read().await;
            if cache.last_update.elapsed() < self.cache_duration && !cache.metrics.is_empty() {
                return Ok(cache.metrics.clone());
            }
        }

        // Collect new metrics
        let mut all_metrics = Vec::new();
        for i in 0..self.nvml.device_count() {
            let device = self.nvml.device(i)?;
            let metrics = device.metrics()?;
            all_metrics.push(metrics);
        }

        // Update cache
        {
            let mut cache = self.cache.write().await;
            cache.metrics = all_metrics.clone();
            cache.last_update = Instant::now();
        }

        Ok(all_metrics)
    }

    /// Get metrics for specific device
    pub async fn collect_device(&self, device_id: u32) -> Result<GpuMetrics, GpuError> {
        let device = self.nvml.device(device_id)?;
        device.metrics()
    }

    /// Get device info for all devices
    pub fn get_all_device_info(&self) -> Result<Vec<GpuDeviceInfo>, GpuError> {
        let mut info = Vec::new();
        for i in 0..self.nvml.device_count() {
            let device = self.nvml.device(i)?;
            info.push(device.info()?);
        }
        Ok(info)
    }

    /// Get system-wide GPU statistics
    pub async fn system_stats(&self) -> Result<SystemGpuStats, GpuError> {
        let metrics = self.collect_all().await?;

        let total_memory_mb: u64 = metrics.iter().map(|m| m.memory.total_mb).sum();
        let used_memory_mb: u64 = metrics.iter().map(|m| m.memory.used_mb).sum();

        let avg_gpu_util = metrics.iter().map(|m| m.utilization.gpu).sum::<f64>()
            / metrics.len() as f64;

        let avg_memory_util = metrics.iter().map(|m| m.utilization.memory).sum::<f64>()
            / metrics.len() as f64;

        let total_power_watts = metrics.iter().map(|m| m.power.current_watts).sum();

        let max_temperature = metrics.iter()
            .map(|m| m.temperature.gpu)
            .fold(f64::NEG_INFINITY, f64::max);

        Ok(SystemGpuStats {
            device_count: metrics.len(),
            total_memory_mb,
            used_memory_mb,
            avg_gpu_utilization: avg_gpu_util,
            avg_memory_utilization: avg_memory_util,
            total_power_watts,
            max_temperature,
        })
    }

    /// Get driver and CUDA versions
    pub fn versions(&self) -> Result<(String, String), GpuError> {
        let driver = self.nvml.driver_version()?;
        let cuda = self.nvml.cuda_version()?;
        Ok((driver, cuda))
    }
}

/// System-wide GPU statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemGpuStats {
    pub device_count: usize,
    pub total_memory_mb: u64,
    pub used_memory_mb: u64,
    pub avg_gpu_utilization: f64,
    pub avg_memory_utilization: f64,
    pub total_power_watts: f64,
    pub max_temperature: f64,
}

impl Default for GpuMetricsCollector {
    fn default() -> Self {
        Self::new().expect("Failed to initialize GPU metrics collector")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_collector_initialization() {
        let collector = GpuMetricsCollector::new();
        assert!(collector.is_ok());

        let collector = collector.unwrap();
        assert!(collector.device_count() > 0);
    }

    #[tokio::test]
    async fn test_collect_all() {
        let collector = GpuMetricsCollector::new().unwrap();
        let metrics = collector.collect_all().await.unwrap();

        assert!(!metrics.is_empty());
        assert_eq!(metrics.len(), collector.device_count() as usize);
    }

    #[tokio::test]
    async fn test_collect_device() {
        let collector = GpuMetricsCollector::new().unwrap();
        let metrics = collector.collect_device(0).await.unwrap();

        assert_eq!(metrics.device_id, 0);
        assert!(metrics.temperature.gpu > 0.0);
    }

    #[tokio::test]
    async fn test_system_stats() {
        let collector = GpuMetricsCollector::new().unwrap();
        let stats = collector.system_stats().await.unwrap();

        assert!(stats.device_count > 0);
        assert!(stats.total_memory_mb > 0);
        assert!(stats.avg_gpu_utilization >= 0.0 && stats.avg_gpu_utilization <= 100.0);
    }

    #[tokio::test]
    async fn test_cache() {
        let collector = GpuMetricsCollector::with_cache_duration(Duration::from_secs(5)).unwrap();

        // First call
        let start = Instant::now();
        let _metrics1 = collector.collect_all().await.unwrap();
        let first_duration = start.elapsed();

        // Second call (should be cached)
        let start = Instant::now();
        let _metrics2 = collector.collect_all().await.unwrap();
        let second_duration = start.elapsed();

        // Cached call should be faster
        assert!(second_duration < first_duration);
    }

    #[test]
    fn test_versions() {
        let collector = GpuMetricsCollector::new().unwrap();
        let (driver, cuda) = collector.versions().unwrap();

        assert!(!driver.is_empty());
        assert!(!cuda.is_empty());
    }
}
