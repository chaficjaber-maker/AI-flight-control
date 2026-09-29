from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindField:
    """Steady wind in NED (m/s) plus optional gust noise."""

    north_m_s: float = 0.0
    east_m_s: float = 0.0
    gust_std_m_s: float = 0.0

    @staticmethod
    def calm() -> "WindField":
        return WindField()

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        gust_n = float(rng.normal(0.0, self.gust_std_m_s)) if self.gust_std_m_s > 0 else 0.0
        gust_e = float(rng.normal(0.0, self.gust_std_m_s)) if self.gust_std_m_s > 0 else 0.0
        return self.north_m_s + gust_n, self.east_m_s + gust_e


def ground_velocity_ned(
    airspeed_m_s: float,
    heading_rad: float,
    flight_path_angle_rad: float,
    wind_n_m_s: float,
    wind_e_m_s: float,
) -> tuple[float, float, float]:
    """
    Ground velocity from body airspeed and wind.

    Aircraft velocity relative to air mass is along heading; wind advects the air mass.
    """
    vn = airspeed_m_s * np.cos(flight_path_angle_rad)
    ground_n = vn * np.cos(heading_rad) + wind_n_m_s
    ground_e = vn * np.sin(heading_rad) + wind_e_m_s
    ground_h = airspeed_m_s * np.sin(flight_path_angle_rad)
    return float(ground_n), float(ground_e), float(ground_h)


def airspeed_from_ground_and_wind(
    ground_n: float,
    ground_e: float,
    ground_h: float,
    wind_n_m_s: float,
    wind_e_m_s: float,
) -> float:
    rel_n = ground_n - wind_n_m_s
    rel_e = ground_e - wind_e_m_s
    rel_h = ground_h
    return float(np.sqrt(rel_n * rel_n + rel_e * rel_e + rel_h * rel_h))
