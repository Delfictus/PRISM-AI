/// Two-Line Element (TLE) parser for satellite orbital elements
use serde::{Deserialize, Serialize};
use std::fmt;

/// Two-Line Element set
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TLE {
    /// Satellite name
    pub name: String,
    /// Line 1 of TLE
    pub line1: String,
    /// Line 2 of TLE
    pub line2: String,
    /// Parsed orbital elements
    pub elements: OrbitalElements,
}

/// Orbital elements extracted from TLE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitalElements {
    /// NORAD catalog number
    pub norad_id: u32,
    /// Epoch year (2-digit)
    pub epoch_year: u32,
    /// Epoch day of year (with fractional day)
    pub epoch_day: f64,
    /// Mean motion (revolutions per day)
    pub mean_motion: f64,
    /// Eccentricity
    pub eccentricity: f64,
    /// Inclination (degrees)
    pub inclination: f64,
    /// Right ascension of ascending node (degrees)
    pub raan: f64,
    /// Argument of perigee (degrees)
    pub arg_perigee: f64,
    /// Mean anomaly (degrees)
    pub mean_anomaly: f64,
    /// B-star drag term
    pub bstar: f64,
}

impl TLE {
    /// Parse TLE from three lines (name, line1, line2)
    pub fn parse(name: impl Into<String>, line1: impl Into<String>, line2: impl Into<String>) -> Result<Self, TLEError> {
        let name = name.into();
        let line1_str = line1.into();
        let line2_str = line2.into();

        // Validate line lengths
        if line1_str.len() < 69 {
            return Err(TLEError::InvalidFormat("Line 1 too short".to_string()));
        }
        if line2_str.len() < 69 {
            return Err(TLEError::InvalidFormat("Line 2 too short".to_string()));
        }

        // Parse Line 1
        let norad_id = line1_str[2..7].trim().parse::<u32>()
            .map_err(|e| TLEError::ParseError(format!("NORAD ID: {}", e)))?;

        let epoch_year = line1_str[18..20].trim().parse::<u32>()
            .map_err(|e| TLEError::ParseError(format!("Epoch year: {}", e)))?;

        let epoch_day = line1_str[20..32].trim().parse::<f64>()
            .map_err(|e| TLEError::ParseError(format!("Epoch day: {}", e)))?;

        let bstar_str = line1_str[53..61].trim().replace("−", "-"); // Handle minus sign
        let bstar = parse_exponential(&bstar_str)
            .map_err(|e| TLEError::ParseError(format!("B-star: {}", e)))?;

        // Parse Line 2
        let inclination = line2_str[8..16].trim().parse::<f64>()
            .map_err(|e| TLEError::ParseError(format!("Inclination: {}", e)))?;

        let raan = line2_str[17..25].trim().parse::<f64>()
            .map_err(|e| TLEError::ParseError(format!("RAAN: {}", e)))?;

        let eccentricity_str = format!("0.{}", line2_str[26..33].trim());
        let eccentricity = eccentricity_str.parse::<f64>()
            .map_err(|e| TLEError::ParseError(format!("Eccentricity: {}", e)))?;

        let arg_perigee = line2_str[34..42].trim().parse::<f64>()
            .map_err(|e| TLEError::ParseError(format!("Arg perigee: {}", e)))?;

        let mean_anomaly = line2_str[43..51].trim().parse::<f64>()
            .map_err(|e| TLEError::ParseError(format!("Mean anomaly: {}", e)))?;

        let mean_motion = line2_str[52..63].trim().parse::<f64>()
            .map_err(|e| TLEError::ParseError(format!("Mean motion: {}", e)))?;

        let elements = OrbitalElements {
            norad_id,
            epoch_year,
            epoch_day,
            mean_motion,
            eccentricity,
            inclination,
            raan,
            arg_perigee,
            mean_anomaly,
            bstar,
        };

        Ok(TLE {
            name,
            line1: line1_str,
            line2: line2_str,
            elements,
        })
    }

    /// Get semi-major axis (km)
    pub fn semi_major_axis(&self) -> f64 {
        // a = (μ / n²)^(1/3)
        // where μ = 398600.4418 km³/s² (Earth's gravitational parameter)
        // n = mean motion in radians per minute
        const MU: f64 = 398600.4418; // km³/s²
        const MINUTES_PER_DAY: f64 = 1440.0;

        let n = self.elements.mean_motion * 2.0 * std::f64::consts::PI / MINUTES_PER_DAY; // rad/min
        let n_per_sec = n / 60.0; // rad/sec

        (MU / (n_per_sec * n_per_sec)).powf(1.0 / 3.0)
    }

    /// Get orbital period (minutes)
    pub fn period(&self) -> f64 {
        1440.0 / self.elements.mean_motion
    }

    /// Get altitude at perigee (km)
    pub fn altitude_perigee(&self) -> f64 {
        const EARTH_RADIUS: f64 = 6371.0; // km
        let a = self.semi_major_axis();
        let e = self.elements.eccentricity;
        a * (1.0 - e) - EARTH_RADIUS
    }

    /// Get altitude at apogee (km)
    pub fn altitude_apogee(&self) -> f64 {
        const EARTH_RADIUS: f64 = 6371.0; // km
        let a = self.semi_major_axis();
        let e = self.elements.eccentricity;
        a * (1.0 + e) - EARTH_RADIUS
    }
}

/// Parse exponential notation in TLE format (e.g., "-12345-3" = -0.12345e-3)
fn parse_exponential(s: &str) -> Result<f64, String> {
    if s.is_empty() {
        return Ok(0.0);
    }

    // TLE format: first char is sign, last 2 chars are exponent
    if s.len() < 6 {
        return Err(format!("Exponential string too short: {}", s));
    }

    let sign = if s.starts_with('-') { -1.0 } else { 1.0 };
    let mantissa_str = &s[1..s.len()-2];
    let exponent_str = &s[s.len()-2..];

    let mantissa: f64 = format!("0.{}", mantissa_str).parse()
        .map_err(|e| format!("Failed to parse mantissa: {}", e))?;

    let exponent: i32 = exponent_str.parse()
        .map_err(|e| format!("Failed to parse exponent: {}", e))?;

    Ok(sign * mantissa * 10_f64.powi(exponent))
}

/// TLE parsing errors
#[derive(Debug, Clone)]
pub enum TLEError {
    InvalidFormat(String),
    ParseError(String),
    ChecksumError(String),
}

impl fmt::Display for TLEError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TLEError::InvalidFormat(msg) => write!(f, "Invalid TLE format: {}", msg),
            TLEError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            TLEError::ChecksumError(msg) => write!(f, "Checksum error: {}", msg),
        }
    }
}

impl std::error::Error for TLEError {}

/// Example TLE data for testing
pub fn example_tle_iss() -> TLE {
    TLE::parse(
        "ISS (ZARYA)",
        "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9999",
        "2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.72125391428000",
    ).expect("Failed to parse ISS TLE")
}

pub fn example_tle_starlink() -> TLE {
    TLE::parse(
        "STARLINK-1007",
        "1 44713U 19074A   24001.50000000  .00001234  00000-0  12345-4 0  9999",
        "2 44713  53.0000  90.0000 0001000  90.0000 270.0000 15.06000000100000",
    ).expect("Failed to parse Starlink TLE")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tle_parsing() {
        let tle = example_tle_iss();
        assert_eq!(tle.elements.norad_id, 25544);
        assert_eq!(tle.elements.epoch_year, 24);
        assert!((tle.elements.inclination - 51.64).abs() < 0.01);
    }

    #[test]
    fn test_orbital_calculations() {
        let tle = example_tle_iss();
        let period = tle.period();
        assert!(period > 90.0 && period < 95.0); // ISS period ~92 minutes

        let altitude = (tle.altitude_perigee() + tle.altitude_apogee()) / 2.0;
        assert!(altitude > 400.0 && altitude < 450.0); // ISS altitude ~420 km
    }

    #[test]
    fn test_exponential_parsing() {
        assert!((parse_exponential("12345-3").unwrap() - 0.12345e-3).abs() < 1e-10);
        assert!((parse_exponential("-12345-3").unwrap() + 0.12345e-3).abs() < 1e-10);
    }
}
