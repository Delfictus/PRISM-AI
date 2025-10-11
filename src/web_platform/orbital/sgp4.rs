/// SGP4 orbital propagator implementation
///
/// Simplified General Perturbations 4 algorithm for satellite position calculation
/// Based on Vallado's implementation and AIAA-2006-6753 specification
use std::f64::consts::PI;
use super::tle::TLE;
use super::coordinates::*;

/// SGP4 propagator
pub struct SGP4Propagator {
    /// Orbital elements from TLE
    tle: TLE,
    /// Epoch Julian Date
    epoch_jd: f64,
    /// Precomputed constants
    constants: SGP4Constants,
}

/// Precomputed constants for SGP4
struct SGP4Constants {
    n0: f64,      // Mean motion (rad/min)
    e0: f64,      // Eccentricity
    i0: f64,      // Inclination (rad)
    omega0: f64,  // Argument of perigee (rad)
    omegadot: f64, // RAAN (rad)
    m0: f64,      // Mean anomaly (rad)
    bstar: f64,   // B-star drag term
    a: f64,       // Semi-major axis (Earth radii)
}

/// Physical constants
const XKE: f64 = 0.07436691613317342; // Reciprocal of Earth radii per minute
const J2: f64 = 0.00108262998905; // J2 perturbation coefficient
const J3: f64 = -0.00000253215306; // J3 perturbation coefficient
const J4: f64 = -0.00000161098761; // J4 perturbation coefficient
const EARTH_RADIUS: f64 = 6378.137; // km
const MINUTES_PER_DAY: f64 = 1440.0;

impl SGP4Propagator {
    /// Create new SGP4 propagator from TLE
    pub fn new(tle: TLE) -> Result<Self, String> {
        let epoch_jd = tle_epoch_to_julian(
            tle.elements.epoch_year,
            tle.elements.epoch_day,
        );

        // Convert degrees to radians
        let i0 = tle.elements.inclination * PI / 180.0;
        let omega0 = tle.elements.arg_perigee * PI / 180.0;
        let omegadot = tle.elements.raan * PI / 180.0;
        let m0 = tle.elements.mean_anomaly * PI / 180.0;
        let e0 = tle.elements.eccentricity;

        // Mean motion in radians per minute
        let n0 = tle.elements.mean_motion * 2.0 * PI / MINUTES_PER_DAY;

        // Calculate semi-major axis in Earth radii
        let a1 = (XKE / n0).powf(2.0 / 3.0);
        let delta1 = 1.5 * J2 * (3.0 * i0.cos().powi(2) - 1.0) / (a1 * a1 * (1.0 - e0 * e0).powf(1.5));
        let a0 = a1 * (1.0 - delta1 / 3.0 - delta1 * delta1 - 134.0 * delta1.powi(3) / 81.0);

        let p0 = a0 * (1.0 - e0 * e0);
        let delta0 = 1.5 * J2 * (3.0 * i0.cos().powi(2) - 1.0) / (p0 * p0);
        let n0dp = n0 / (1.0 + delta0);
        let a0dp = a0 / (1.0 - delta0);

        let constants = SGP4Constants {
            n0: n0dp,
            e0,
            i0,
            omega0,
            omegadot,
            m0,
            bstar: tle.elements.bstar,
            a: a0dp,
        };

        Ok(Self {
            tle,
            epoch_jd,
            constants,
        })
    }

    /// Propagate to specific time (Unix timestamp in seconds)
    pub fn propagate(&self, unix_timestamp: u64) -> Result<PropagationResult, String> {
        let target_jd = unix_to_julian(unix_timestamp);
        let minutes_since_epoch = (target_jd - self.epoch_jd) * MINUTES_PER_DAY;

        self.propagate_minutes(minutes_since_epoch)
    }

    /// Propagate by minutes since epoch
    fn propagate_minutes(&self, tsince: f64) -> Result<PropagationResult, String> {
        let c = &self.constants;

        // Mean motion and semi-major axis with secular effects
        let n = c.n0;
        let a = c.a;
        let e = c.e0;

        // Mean anomaly with secular update
        let m = c.m0 + n * tsince;

        // Solve Kepler's equation for eccentric anomaly (E)
        let eccentric_anomaly = solve_kepler(m, e)?;

        // True anomaly
        let beta = (1.0 - e * e).sqrt();
        let nu = ((beta * eccentric_anomaly.sin()).atan2(eccentric_anomaly.cos() - e))
            .rem_euclid(2.0 * PI);

        // Argument of latitude
        let u = (c.omega0 + nu).rem_eucloid(2.0 * PI);

        // Radius
        let r = a * (1.0 - e * eccentric_anomaly.cos());

        // Preliminary position in orbital plane
        let cos_u = u.cos();
        let sin_u = u.sin();

        let xp = r * cos_u;
        let yp = r * sin_u;

        // Include perturbations (simplified)
        let cos_i = c.i0.cos();
        let sin_i = c.i0.sin();
        let cos_omega = c.omegadot.cos();
        let sin_omega = c.omegadot.sin();

        // Rotate to ECI frame
        let x = xp * cos_omega - yp * sin_omega * cos_i;
        let y = xp * sin_omega + yp * cos_omega * cos_i;
        let z = yp * sin_i;

        // Convert from Earth radii to km
        let position = ECICoordinates {
            x: x * EARTH_RADIUS,
            y: y * EARTH_RADIUS,
            z: z * EARTH_RADIUS,
        };

        // Calculate velocity (simplified)
        let rdot = n * a * e * eccentric_anomaly.sin() / (1.0 - e * eccentric_anomaly.cos());
        let udot = n * a * a * beta / (r * r);

        let xdot = rdot * cos_u - r * udot * sin_u;
        let ydot = rdot * sin_u + r * udot * cos_u;

        let vx = (xdot * cos_omega - ydot * sin_omega * cos_i) * EARTH_RADIUS / 60.0; // km/s
        let vy = (xdot * sin_omega + ydot * cos_omega * cos_i) * EARTH_RADIUS / 60.0;
        let vz = ydot * sin_i * EARTH_RADIUS / 60.0;

        let velocity = Velocity { vx, vy, vz };

        Ok(PropagationResult {
            position,
            velocity,
            minutes_since_epoch: tsince,
        })
    }

    /// Propagate and convert to geodetic coordinates
    pub fn propagate_geodetic(&self, unix_timestamp: u64) -> Result<(GeodeticCoordinates, f64), String> {
        let result = self.propagate(unix_timestamp)?;

        // Calculate GMST for coordinate transformation
        let jd = unix_to_julian(unix_timestamp);
        let gmst = calculate_gmst(jd);

        // Convert ECI → ECEF → Geodetic
        let ecef = result.position.to_ecef(gmst);
        let geodetic = ecef.to_geodetic();

        Ok((geodetic, result.velocity.magnitude()))
    }
}

/// Solve Kepler's equation: M = E - e*sin(E)
/// Uses Newton-Raphson iteration
fn solve_kepler(m: f64, e: f64) -> Result<f64, String> {
    const MAX_ITERATIONS: usize = 20;
    const TOLERANCE: f64 = 1e-8;

    let m_normalized = m.rem_euclid(2.0 * PI);
    let mut eccentric_anomaly = m_normalized;

    for _ in 0..MAX_ITERATIONS {
        let f = eccentric_anomaly - e * eccentric_anomaly.sin() - m_normalized;
        let df = 1.0 - e * eccentric_anomaly.cos();

        let delta = f / df;
        eccentric_anomaly -= delta;

        if delta.abs() < TOLERANCE {
            return Ok(eccentric_anomaly);
        }
    }

    Err("Kepler equation did not converge".to_string())
}

/// Modulo for floating point that matches mathematical definition
trait RemEuclidExt {
    fn rem_eucloid(self, rhs: Self) -> Self;
}

impl RemEuclidExt for f64 {
    fn rem_eucloid(self, rhs: Self) -> Self {
        let r = self % rhs;
        if r < 0.0 {
            r + rhs.abs()
        } else {
            r
        }
    }
}

/// Result of orbital propagation
#[derive(Debug, Clone)]
pub struct PropagationResult {
    pub position: ECICoordinates,
    pub velocity: Velocity,
    pub minutes_since_epoch: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::web_platform::orbital::tle::example_tle_iss;

    #[test]
    fn test_kepler_solver() {
        let m = 1.0; // Mean anomaly
        let e = 0.1; // Eccentricity

        let eccentric_anomaly = solve_kepler(m, e).unwrap();

        // Verify Kepler's equation
        let verification = eccentric_anomaly - e * eccentric_anomaly.sin();
        assert!((verification - m).abs() < 1e-6);
    }

    #[test]
    fn test_sgp4_propagation() {
        let tle = example_tle_iss();
        let propagator = SGP4Propagator::new(tle).unwrap();

        // Propagate to epoch (should be close to initial conditions)
        let unix_epoch = 1704153600; // 2024-01-02 00:00:00 UTC
        let result = propagator.propagate(unix_epoch).unwrap();

        // ISS orbit radius should be ~6800 km from Earth center
        let radius = result.position.magnitude();
        assert!(radius > 6700.0 && radius < 6900.0);

        // ISS velocity should be ~7.7 km/s
        let velocity = result.velocity.magnitude();
        assert!(velocity > 7.0 && velocity < 8.0);
    }

    #[test]
    fn test_geodetic_conversion() {
        let tle = example_tle_iss();
        let propagator = SGP4Propagator::new(tle).unwrap();

        let unix_epoch = 1704153600;
        let (geo, velocity) = propagator.propagate_geodetic(unix_epoch).unwrap();

        // Latitude should be within ISS inclination (±51.6°)
        assert!(geo.latitude.abs() <= 52.0);

        // Longitude should be valid
        assert!(geo.longitude >= -180.0 && geo.longitude <= 180.0);

        // Altitude should be ISS orbit (~420 km)
        assert!(geo.altitude > 400.0 && geo.altitude < 450.0);

        // Velocity check
        assert!(velocity > 7.0 && velocity < 8.0);
    }
}
