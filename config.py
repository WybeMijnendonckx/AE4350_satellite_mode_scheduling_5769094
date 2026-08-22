"""config.py

All changeable input parameters for the spacecraft mode scheduling RL problem, 
grouped per subsystem. Centralizing these here is better for maintainability.
"""

from dataclasses import dataclass, field


@dataclass
class OrbitParams:
    altitude_km: float = 600.0
    ltan_hours: float = 10.5          # 10:30 LTAN, standard EO-SSO convention
    solar_declination_deg: float = 0.0  # equinox default, sweep for seasonal effect


@dataclass
class GroundStationParams:
    name: str = "Redu"                # ESA Estrack core station at Redu,Belgium
    latitude_deg: float = 50.00
    longitude_deg: float = 5.15
    elevation_mask_deg: float = 10.0
    downlink_rate_mbps: float = 50.0
    antenna_beamwidth_deg: float = 15.0   # half-angle acceptance cone


@dataclass
class SpacecraftParams:
    # energy storage
    battery_capacity_wh: float = 90.0
    battery_safety_floor_frac: float = 0.20   # min allowed SOC (DoD margin)

    # data storage
    buffer_capacity_mb: float = 30000.0
    imaging_data_rate_mbps: float = 5.0      # data generated while imaging

    # power draw per mode or activity in Watts (housekeeping is always-on baseline)
    power_housekeeping_w: float = 5.0
    power_imaging_w: float = 12.0             # additional draw while imaging
    power_downlink_w: float =25.0            # additional draw while transmitting
    power_charging_max_w: float = 25.0        # peak solar input, scaled by cosine of sun angle
    power_slew_w: float = 8.0                 # reaction-wheel draw while actively slewing

    # attitude / pointing
    slew_rate_deg_s: float = 2.0
    instrument_fov_deg: float = 5.0           # half-angle acceptance cone around nadir
    solar_cutoff_angle_deg: float = 90.0      # beyond this angle to sun, zero charging

    # degradation
    battery_health_range: tuple = (0.8, 1.0)         # sampled at reset, multiplies battery_capacity_wh
    instrument_efficiency_range: tuple = (0.8, 1.0)  # sampled at reset, multiplies imaging_data_rate_mbps
    battery_degradation_per_step: float = 0.005      # health lost per timestep spent below safety floor
    battery_health_hard_floor: float = 0.30          # below this, episode terminates (mission loss)


    # observation noise (does not affect true dynamics)
    soc_sensor_noise_std: float = 0.01        # additive, fraction of full capacity
    buffer_sensor_noise_std: float = 0.01     # additive, fraction of full capacity

    # process noise (affects true dynamics)
    solar_input_noise_std: float = 0.05       # multiplicative, fraction of instantaneous charging power

@dataclass
class RewardParams:
    """Downlinked data is the objective to maximize;
    everything else is either a constraint consequence or a small
    shaping term, not a reward source (this will avoid the agent
    learning to store data in the buffer instead of actually returning it to the ground).
    """
    reward_per_mb_downlinked: float = 0.001
    imaging_shaping_bonus: float = 0.00005          # zero by default, increased to encourage imaging
    terminal_penalty_hard_floor: float = -10.0  # on mission-loss termination, change with sensitivity analysis if desired



@dataclass
class RLParams:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20000
    replay_buffer_size: int = 50000
    batch_size: int = 64
    target_update_freq: int = 500
    hidden_layer_sizes: tuple = (64, 64)


@dataclass
class TrainingParams:
    timestep_s: float = 30.0          # simulation timestep
    num_orbits_per_episode: int = 15  # about 1 day
    num_episodes: int = 1000
    seed: int = 42