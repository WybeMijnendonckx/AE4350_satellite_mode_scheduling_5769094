import numpy as np
import matplotlib.pyplot as plt

from orbital_dynamics import OrbitPropagator, orbital_period_s
from config import OrbitParams, GroundStationParams

orbit_params = OrbitParams()          # defaults: 600 km, LTAN 10.5h, declination 0
gs_params = GroundStationParams()     # defaults: Redu, 10 deg elevation mask

prop = OrbitPropagator(orbit_params, gs_params)
rev = 15
T = orbital_period_s(orbit_params.altitude_km)
dt = 5.0  # seconds
ts = np.arange((rev-1)*T, rev*T, dt)
t_min = ts / 60.0  # x-axis in minutes for readability

eclipse_flags = np.array([prop.is_eclipse(t) for t in ts])
contact_flags = np.array([prop.is_ground_contact(t) for t in ts])
elevations = np.array([prop.elevation_deg(t) for t in ts])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax1.plot(t_min, elevations, color="black", linewidth=1)
ax1.axhline(gs_params.elevation_mask_deg, color="gray", linestyle="--", linewidth=0.8, label="elevation mask")
ax1.fill_between(t_min, -90, 90, where=eclipse_flags, color="navy", alpha=0.2, label="eclipse")
ax1.fill_between(t_min, -90, 90, where=contact_flags, color="orange", alpha=0.4, label="ground contact")
ax1.set_ylabel("Elevation to Redu [deg]")
ax1.set_ylim(-90, 90)
ax1.legend(loc="upper right", fontsize=8)
ax1.set_title(f"Orbit geometry over one period (T = {T/60:.2f} min), {orbit_params.altitude_km:.0f} km SSO")

ax2.fill_between(t_min, 0, 1, where=eclipse_flags, step="mid", color="navy", alpha=0.5, label="eclipse")
ax2.fill_between(t_min, 1, 2, where=contact_flags, step="mid", color="orange", alpha=0.5, label="ground contact")
ax2.set_yticks([0.5, 1.5])
ax2.set_yticklabels(["eclipse", "contact"])
ax2.set_xlabel("Time [min]")
ax2.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("test_results/orbit_geometry_check.png", dpi=150)
plt.show()

print(f"Eclipse fraction this orbit: {eclipse_flags.mean()*100:.1f}%")
print(f"Ground contact this orbit: {contact_flags.sum()*dt/60:.1f} min")