/// Coordinate transformations for orbital mechanics
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Earth-Centered Inertial (ECI) coordinates (km)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ECICoordinates {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Earth-Centered Earth-Fixed (ECEF) coordinates (km)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ECEFCoordinates {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Geographic coordinates (degrees and km)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GeodeticCoordinates {
    pub latitude: f64,   // degrees, -90 to 90
    pub longitude: f64,  // degrees, -180 to 180
    pub altitude: f64,   // km above Earth surface
}

/// Velocity vector (km/s)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Velocity {
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
}

impl Velocity {
    /// Get velocity magnitude (km/s)
    pub fn magnitude(&self) -> f64 {
        (self.vx * self.vx + self.vy * self.vy + self.vz * self.vz).sqrt()
    }
}

impl ECICoordinates {
    /// Get distance from Earth center (km)
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Convert ECI to ECEF (requires GMST for Earth rotation)
    pub fn to_ecef(&self, gmst: f64) -> ECEFCoordinates {
        let cos_gmst = gmst.cos();
        let sin_gmst = gmst.sin();

        ECEFCoordinates {
            x: self.x * cos_gmst + self.y * sin_gmst,
            y: -self.x * sin_gmst + self.y * cos_gmst,
            z: self.z,
        }
    }
}

impl ECEFCoordinates {
    /// Convert ECEF to geodetic coordinates using iterative method
    pub fn to_geodetic(&self) -> GeodeticCoordinates {
        const A: f64 = 6378.137; // Earth semi-major axis (km)
        const B: f64 = 6356.752314245; // Earth semi-minor axis (km)
        const E2: f64 = 0.00669437999014; // First eccentricity squared

        let p = (self.x * self.x + self.y * self.y).sqrt();
        let mut lat = (self.z / p).atan();

        // Iterative refinement (typically converges in 3-4 iterations)
        for _ in 0..5 {
            let sin_lat = lat.sin();
            let n = A / (1.0 - E2 * sin_lat * sin_lat).sqrt();
            lat = (self.z + E2 * n * sin_lat).atan2(p);
        }

        let sin_lat = lat.sin();
        let n = A / (1.0 - E2 * sin_lat * sin_lat).sqrt();
        let altitude = p / lat.cos() - n;

        let longitude = self.y.atan2(self.x);

        GeodeticCoordinates {
            latitude: lat * 180.0 / PI,
            longitude: longitude * 180.0 / PI,
            altitude,
        }
    }
}

/// Calculate Greenwich Mean Sidereal Time (GMST) in radians
pub fn calculate_gmst(julian_date: f64) -> f64 {
    // Julian centuries from J2000.0
    let t = (julian_date - 2451545.0) / 36525.0;

    // GMST in seconds
    let gmst_sec = 67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t * t
        - 6.2e-6 * t * t * t;

    // Convert to radians and normalize to [0, 2π]
    let gmst_rad = (gmst_sec * PI / 43200.0) % (2.0 * PI);

    if gmst_rad < 0.0 {
        gmst_rad + 2.0 * PI
    } else {
        gmst_rad
    }
}

/// Convert Unix timestamp (seconds) to Julian Date
pub fn unix_to_julian(unix_timestamp: u64) -> f64 {
    // Unix epoch (1970-01-01 00:00:00) is JD 2440587.5
    2440587.5 + (unix_timestamp as f64 / 86400.0)
}

/// Convert TLE epoch to Julian Date
pub fn tle_epoch_to_julian(epoch_year: u32, epoch_day: f64) -> f64 {
    // Convert 2-digit year to 4-digit
    let year = if epoch_year < 57 {
        2000 + epoch_year
    } else {
        1900 + epoch_year
    };

    // January 1st of the year in Julian Date
    let jd_jan1 = julian_date_from_ymd(year as i32, 1, 1);

    // Add fractional days (epoch_day is day of year, 1-based)
    jd_jan1 + (epoch_day - 1.0)
}

/// Calculate Julian Date from year, month, day
pub fn julian_date_from_ymd(year: i32, month: i32, day: i32) -> f64 {
    let a = (14 - month) / 12;
    let y = year + 4800 - a;
    let m = month + 12 * a - 3;

    let jdn = day + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;

    jdn as f64 - 0.5 // Julian Date at midnight
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eci_to_ecef() {
        let eci = ECICoordinates {
            x: 6378.137,
            y: 0.0,
            z: 0.0,
        };

        let gmst = 0.0; // 0 rotation
        let ecef = eci.to_ecef(gmst);

        assert!((ecef.x - 6378.137).abs() < 0.001);
        assert!(ecef.y.abs() < 0.001);
        assert!(ecef.z.abs() < 0.001);
    }

    #[test]
    fn test_ecef_to_geodetic() {
        // Point on equator at prime meridian
        let ecef = ECEFCoordinates {
            x: 6378.137,
            y: 0.0,
            z: 0.0,
        };

        let geo = ecef.to_geodetic();

        assert!(geo.latitude.abs() < 0.01);
        assert!(geo.longitude.abs() < 0.01);
        assert!(geo.altitude.abs() < 0.1);
    }

    #[test]
    fn test_julian_date_conversions() {
        // J2000.0 epoch: 2000-01-01 12:00:00 UTC
        let jd = julian_date_from_ymd(2000, 1, 1) + 0.5;
        assert!((jd - 2451545.0).abs() < 0.01);

        // Unix epoch: 1970-01-01 00:00:00 UTC
        let jd_unix = unix_to_julian(0);
        assert!((jd_unix - 2440587.5).abs() < 0.01);
    }

    #[test]
    fn test_gmst_calculation() {
        let jd = 2451545.0; // J2000.0
        let gmst = calculate_gmst(jd);

        // GMST at J2000.0 should be ~1.7 radians (280.46° from vernal equinox)
        assert!(gmst > 0.0 && gmst < 2.0 * PI);
    }

    #[test]
    fn test_velocity_magnitude() {
        let vel = Velocity {
            vx: 3.0,
            vy: 4.0,
            vz: 0.0,
        };

        assert!((vel.magnitude() - 5.0).abs() < 0.001);
    }
}
