import numpy as np

# ---------------- CONSTANTS ----------------
TANK_VOLUME_M3 = 100
TANK_HEIGHT_M = 5.0
TANK_DIAMETER_M = np.sqrt(4 * TANK_VOLUME_M3 / (np.pi * TANK_HEIGHT_M))
TANK_CSA = np.pi * (TANK_DIAMETER_M / 2) ** 2

INSULATION_R_WALL = 5.0
INSULATION_R_TOP = 3.0
INSULATION_R_BOTTOM = 6.0

MAX_HEAT_INPUT = 50_000
MIN_DISCHARGE_POWER = 40_000

TIME_STEP = 300
TOTAL_TIME = 24 * 3600

MAX_SAFE_TEMP = 120
MIN_SAFE_TEMP = -50

LAYERS = 20

# ---------------- FLUID PROPERTIES ----------------
def glycol_density(T):
    return 1035 - 0.65 * (T - 20)

def glycol_cp(T):
    return 3800 + 2.5 * T

def glycol_k(T):
    return 0.21 - 0.00015 * T


# ---------------- SIMULATION ----------------
def run_simulation(
    ambient_temp,
    capture_rate,
    tank_mass,
    insulation_eff,
    wind_factor,
    simulation_hours=24
):
    # ---- Geometry ----
    layer_height = TANK_HEIGHT_M / LAYERS
    layer_area = TANK_CSA

    # ---- Time Configuration ----
    total_time_seconds = simulation_hours * 3600
    steps = int(total_time_seconds / TIME_STEP)

    # ---- Initial Stratification ----
    layers = np.linspace(20, 85, LAYERS)
    capture_rate = min(capture_rate, MAX_HEAT_INPUT)

    # ---- Storage ----
    avg_temp_series = []
    energy_series = []
    top_temp_series = []
    bottom_temp_series = []
    mode_series = []
    discharge_power_series = []
    layer_history = []

    mode = "CHARGE"
    alarm_triggered = False

    for _ in range(steps):

        densities = np.array([glycol_density(T) for T in layers])
        cp = np.array([glycol_cp(T) for T in layers])
        k = np.array([glycol_k(T) for T in layers])

        layer_masses = densities * (TANK_VOLUME_M3 / LAYERS)

        # ---- Mode Switching ----
        if layers[-1] >= 95:
            mode = "DISCHARGE"
        elif layers[0] <= 30:
            mode = "CHARGE"

        # ---- Heat Loss ----
        avg_temp = np.mean(layers)
        R_eff = INSULATION_R_WALL * insulation_eff
        U = 1 / R_eff
        Q_loss = U * TANK_CSA * (avg_temp - ambient_temp) * wind_factor * TIME_STEP

        layers -= Q_loss / np.sum(layer_masses * cp)

        # ---- Charging ----
        if mode == "CHARGE":
            heat_in = capture_rate * TIME_STEP * 0.85
            mixing_layers = int(np.clip(capture_rate / 15_000, 1, 6))
            heat_per_layer = heat_in / mixing_layers

            for i in range(LAYERS - mixing_layers, LAYERS):
                layers[i] += heat_per_layer / (layer_masses[i] * cp[i])

        # ---- Discharging ----
        else:
            ΔT = layers[-1] - ambient_temp
            discharge_power = min(900 * ΔT, MIN_DISCHARGE_POWER)
            discharge_energy = discharge_power * TIME_STEP * 0.85

            mixing_layers = int(np.clip(discharge_power / 15_000, 1, 6))
            heat_per_layer = discharge_energy / mixing_layers

            for i in range(LAYERS - mixing_layers, LAYERS):
                layers[i] -= heat_per_layer / (layer_masses[i] * cp[i])

        # ---- Vertical Conduction (Energy Conserving) ----
        for i in range(LAYERS - 1):
            k_eff = 0.5 * (k[i] + k[i + 1])
            dT = layers[i + 1] - layers[i]

            q = k_eff * layer_area * dT / layer_height * TIME_STEP

            layers[i]     += q / (layer_masses[i]     * cp[i])
            layers[i + 1] -= q / (layer_masses[i + 1] * cp[i + 1])

        # ---- Buoyancy-Limited Mixing (NO SORTING) ----
        for i in range(LAYERS - 1):
            if layers[i] > layers[i + 1]:
                ΔT = layers[i] - layers[i + 1]
                mix = 0.15 * ΔT
                layers[i]     -= mix
                layers[i + 1] += mix

        # ---- Safety ----
        if layers[-1] > MAX_SAFE_TEMP or layers[0] < MIN_SAFE_TEMP:
            alarm_triggered = True
            break

        # ---- Store Results ----
        avg_temp_series.append(np.mean(layers))
        energy_series.append(np.sum(layer_masses * cp * (layers - ambient_temp)))
        top_temp_series.append(layers[-1])
        bottom_temp_series.append(layers[0])
        mode_series.append(mode)
        discharge_power_series.append(
            discharge_power if mode == "DISCHARGE" else 0
        )
        layer_history.append(layers.copy())

    return {
        "avg_temp": np.array(avg_temp_series),
        "energy": np.array(energy_series),
        "top_temp": np.array(top_temp_series),
        "bottom_temp": np.array(bottom_temp_series),
        "mode": np.array(mode_series),
        "discharge_power": np.array(discharge_power_series),
        "layers": np.array(layer_history),
        "alarm": alarm_triggered
    }
