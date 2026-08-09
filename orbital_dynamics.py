"""orbit_dynamics.py

Pure orbital and geometric calculations for a circular, unperturbed
two-body sun-synchronous orbit: satellite position, sun direction,
ground-station visibility and eclipse state as functions of time.

"""

import numpy as np

# physical constants
MU_EARTH = 398600.4418           # km^3/s^2
R_EARTH = 6378.137                # km
J2 = 1.08262668e-3
OMEGA_EARTH = 7.2921150e-5        # rad/s
SIDEREAL_YEAR_S = 365.2422 * 86400.0


def sun_sync_inclination(altitude_km: float) -> float:
    """Inclination [rad] required for a circular SSO at the given altitude"""
    a = R_EARTH + altitude_km
    n = np.sqrt(MU_EARTH / a ** 3)
    raan_dot_target = 2 * np.pi / SIDEREAL_YEAR_S
    cos_i = raan_dot_target / (-1.5 * n * J2 * (R_EARTH / a) ** 2)
    return np.arccos(cos_i)


def orbital_period_s(altitude_km: float) -> float:
    a = R_EARTH + altitude_km
    return 2 * np.pi * np.sqrt(a ** 3 / MU_EARTH)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


class OrbitPropagator:
    """Propagates satellite position and evaluates eclipse or ground-
    station geometry as a function of time since epoch. t=0 is defined
    at the ascending node, with the sun's right ascension fixed to 0
    as an arbitrary reference epoch (valid because the RAAN of an SSO
    precesses at exactly the rate needed to track the sun, so the
    relative sun-orbit geometry does not change with time)x."""

    def __init__(self, orbit_params, gs_params):
        self.a = R_EARTH + orbit_params.altitude_km
        self.i = sun_sync_inclination(orbit_params.altitude_km)
        self.n = np.sqrt(MU_EARTH / self.a ** 3)          # mean motion [rad/s]
        self.period_s = 2 * np.pi / self.n

        # RAAN from desired LTAN, relative to sun RA = 0 reference
        self.raan = np.deg2rad((orbit_params.ltan_hours - 12.0) * 15.0)

        self.delta_sun = np.deg2rad(orbit_params.solar_declination_deg)
        self.alpha_sun = 0.0  # reference epoch convention

        self.gs_lat = np.deg2rad(gs_params.latitude_deg)
        self.gs_lon = np.deg2rad(gs_params.longitude_deg)
        self.elev_mask = np.deg2rad(gs_params.elevation_mask_deg)

        self.gs_ecef = R_EARTH * np.array([
            np.cos(self.gs_lat) * np.cos(self.gs_lon),
            np.cos(self.gs_lat) * np.sin(self.gs_lon),
            np.sin(self.gs_lat),
        ])

    def satellite_eci(self, t: float) -> np.ndarray:
        u = self.n * t  # argument of latitude (u0 = 0 at ascending node)
        cu, su = np.cos(u), np.sin(u)
        cO, sO = np.cos(self.raan), np.sin(self.raan)
        ci = np.cos(self.i)

        x = self.a * (cO * cu - sO * su * ci)
        y = self.a * (sO * cu + cO * su * ci)
        z = self.a * su * np.sin(self.i)
        return np.array([x, y, z])

    def nadir_direction(self, t: float) -> np.ndarray:
        r = self.satellite_eci(t)
        return -_unit(r)

    def sun_direction(self, t: float = 0.0) -> np.ndarray:
        """Fixed sun unit vector in ECI under the reference-epoch convention."""
        cd = np.cos(self.delta_sun)
        return np.array([
            cd * np.cos(self.alpha_sun),
            cd * np.sin(self.alpha_sun),
            np.sin(self.delta_sun),
        ])

    def ground_station_eci(self, t: float) -> np.ndarray:
        gst = OMEGA_EARTH * t  # GST0 = 0, same reference-epoch convention
        cg, sg = np.cos(gst), np.sin(gst)
        x, y, z = self.gs_ecef
        return np.array([x * cg - y * sg, x * sg + y * cg, z])

    def ground_station_los_direction(self, t: float) -> np.ndarray:
        """Unit vector from the satellite to the ground station —
        the direction the spacecraft must point to aim its antenna
        at Redu."""
        r_sat = self.satellite_eci(t)
        r_gs = self.ground_station_eci(t)
        return _unit(r_gs - r_sat)

    def elevation_deg(self, t: float) -> float:
        """Elevation of the satellite above the ground station's local
        horizon — determines whether a pass is geometrically possible
        at all, independent of the spacecraft's own attitude."""
        r_sat = self.satellite_eci(t)
        r_gs = self.ground_station_eci(t)
        los = r_sat - r_gs
        zenith_hat = _unit(r_gs)
        elevation = np.arcsin(np.dot(los, zenith_hat) / np.linalg.norm(los))
        return np.rad2deg(elevation)

    def is_ground_contact(self, t: float) -> bool:
        return self.elevation_deg(t) > np.rad2deg(self.elev_mask)

    def is_eclipse(self, t: float) -> bool:
        """Cylindrical Earth-shadow model: eclipsed if on the night
        side AND within one Earth radius of the Earth-Sun line."""
        r = self.satellite_eci(t)
        s_hat = self.sun_direction(t)
        along_sun = np.dot(r, s_hat)
        if along_sun > 0:
            return False  # sunward side, never eclipsed
        perp_dist = np.linalg.norm(r - along_sun * s_hat)
        return perp_dist < R_EARTH

    def get_geometry(self, t: float) -> dict:
        """Gives every geometric signal needed by environment.py for a single timestep"""
        return {
            "nadir_dir": self.nadir_direction(t),
            "sun_dir": self.sun_direction(t),
            "gs_los_dir": self.ground_station_los_direction(t),
            "elevation_deg": self.elevation_deg(t),
            "is_ground_contact": self.is_ground_contact(t),
            "is_eclipse": self.is_eclipse(t),
        }