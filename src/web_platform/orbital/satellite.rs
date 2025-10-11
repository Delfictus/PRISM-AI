/// High-level satellite tracking system
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

use super::tle::TLE;
use super::sgp4::SGP4Propagator;
use super::coordinates::GeodeticCoordinates;

/// Satellite tracker - manages multiple satellites with SGP4 propagation
pub struct SatelliteTracker {
    /// Satellites indexed by NORAD ID
    satellites: Arc<RwLock<HashMap<u32, TrackedSatellite>>>,
}

/// A tracked satellite with its propagator
struct TrackedSatellite {
    tle: TLE,
    propagator: SGP4Propagator,
    name: String,
}

/// Satellite state at a specific time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatelliteState {
    pub norad_id: u32,
    pub name: String,
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
    pub velocity: f64,
    pub timestamp: u64,
}

impl SatelliteTracker {
    /// Create new satellite tracker
    pub fn new() -> Self {
        Self {
            satellites: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add satellite to tracker
    pub async fn add_satellite(&self, tle: TLE) -> Result<(), String> {
        let propagator = SGP4Propagator::new(tle.clone())?;

        let tracked = TrackedSatellite {
            name: tle.name.clone(),
            tle: tle.clone(),
            propagator,
        };

        let mut satellites = self.satellites.write().await;
        satellites.insert(tle.elements.norad_id, tracked);

        println!("🛰️  Added satellite: {} (NORAD {})", tle.name, tle.elements.norad_id);
        Ok(())
    }

    /// Remove satellite from tracker
    pub async fn remove_satellite(&self, norad_id: u32) -> Result<(), String> {
        let mut satellites = self.satellites.write().await;
        satellites.remove(&norad_id)
            .ok_or_else(|| format!("Satellite {} not found", norad_id))?;

        Ok(())
    }

    /// Get current state for a specific satellite
    pub async fn get_state(&self, norad_id: u32, unix_timestamp: u64) -> Result<SatelliteState, String> {
        let satellites = self.satellites.read().await;

        let tracked = satellites.get(&norad_id)
            .ok_or_else(|| format!("Satellite {} not found", norad_id))?;

        let (geo, velocity) = tracked.propagator.propagate_geodetic(unix_timestamp)?;

        Ok(SatelliteState {
            norad_id,
            name: tracked.name.clone(),
            latitude: geo.latitude,
            longitude: geo.longitude,
            altitude: geo.altitude,
            velocity,
            timestamp: unix_timestamp,
        })
    }

    /// Get states for all satellites
    pub async fn get_all_states(&self, unix_timestamp: u64) -> Vec<SatelliteState> {
        let satellites = self.satellites.read().await;

        let mut states = Vec::new();
        for (norad_id, tracked) in satellites.iter() {
            if let Ok((geo, velocity)) = tracked.propagator.propagate_geodetic(unix_timestamp) {
                states.push(SatelliteState {
                    norad_id: *norad_id,
                    name: tracked.name.clone(),
                    latitude: geo.latitude,
                    longitude: geo.longitude,
                    altitude: geo.altitude,
                    velocity,
                    timestamp: unix_timestamp,
                });
            }
        }

        states
    }

    /// Get satellite count
    pub async fn satellite_count(&self) -> usize {
        let satellites = self.satellites.read().await;
        satellites.len()
    }

    /// List all satellite names and NORAD IDs
    pub async fn list_satellites(&self) -> Vec<(u32, String)> {
        let satellites = self.satellites.read().await;
        satellites.iter()
            .map(|(norad_id, tracked)| (*norad_id, tracked.name.clone()))
            .collect()
    }

    /// Update TLE for existing satellite
    pub async fn update_tle(&self, tle: TLE) -> Result<(), String> {
        let propagator = SGP4Propagator::new(tle.clone())?;

        let tracked = TrackedSatellite {
            name: tle.name.clone(),
            tle: tle.clone(),
            propagator,
        };

        let mut satellites = self.satellites.write().await;
        satellites.insert(tle.elements.norad_id, tracked);

        println!("🔄 Updated TLE for: {} (NORAD {})", tle.name, tle.elements.norad_id);
        Ok(())
    }
}

impl Default for SatelliteTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Preload example satellite constellation
pub async fn create_example_constellation() -> Result<SatelliteTracker, String> {
    let tracker = SatelliteTracker::new();

    // Transport layer satellites (LEO constellation)
    for i in 0..12 {
        let tle = TLE::parse(
            format!("TRANSPORT-SAT-{:02}", i + 1),
            format!("1 {:05}U 24001A   24001.50000000  .00000100  00000-0  10000-4 0  9999",
                    40000 + i),
            format!("2 {:05}  51.6000 {:7.4} 0001000 {:7.4} {:7.4} 15.50000000100000",
                    40000 + i,
                    i as f64 * 30.0,           // Spread RAAN
                    i as f64 * 30.0,           // Spread arg perigee
                    i as f64 * 30.0),          // Spread mean anomaly
        )?;

        tracker.add_satellite(tle).await?;
    }

    // Tracking layer satellites (MEO constellation)
    for i in 0..6 {
        let tle = TLE::parse(
            format!("TRACKING-SAT-{:02}", i + 1),
            format!("1 {:05}U 24001B   24001.50000000  .00000050  00000-0  50000-5 0  9999",
                    50000 + i),
            format!("2 {:05}  63.4000 {:7.4} 0002000 {:7.4} {:7.4} 12.50000000100000",
                    50000 + i,
                    i as f64 * 60.0,           // Spread RAAN
                    i as f64 * 60.0,           // Spread arg perigee
                    i as f64 * 60.0),          // Spread mean anomaly
        )?;

        tracker.add_satellite(tle).await?;
    }

    println!("✅ Created example constellation with {} satellites", tracker.satellite_count().await);
    Ok(tracker)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::web_platform::orbital::tle::example_tle_iss;

    #[tokio::test]
    async fn test_satellite_tracker() {
        let tracker = SatelliteTracker::new();
        let tle = example_tle_iss();

        tracker.add_satellite(tle).await.unwrap();

        assert_eq!(tracker.satellite_count().await, 1);

        let unix_time = 1704153600; // 2024-01-02
        let state = tracker.get_state(25544, unix_time).await.unwrap();

        assert_eq!(state.norad_id, 25544);
        assert!(state.latitude.abs() <= 52.0);
        assert!(state.altitude > 400.0 && state.altitude < 450.0);
    }

    #[tokio::test]
    async fn test_get_all_states() {
        let tracker = create_example_constellation().await.unwrap();

        let unix_time = 1704153600;
        let states = tracker.get_all_states(unix_time).await;

        assert_eq!(states.len(), 18); // 12 transport + 6 tracking
        assert!(states.iter().all(|s| s.altitude > 0.0));
    }

    #[tokio::test]
    async fn test_update_tle() {
        let tracker = SatelliteTracker::new();
        let tle1 = example_tle_iss();

        tracker.add_satellite(tle1.clone()).await.unwrap();

        // Update with newer TLE
        let mut tle2 = tle1.clone();
        tle2.elements.epoch_day += 1.0;

        tracker.update_tle(tle2).await.unwrap();

        assert_eq!(tracker.satellite_count().await, 1); // Should still be 1, not 2
    }
}
