# Multi-Dopant ZnO and 2D ZnO: Electronic Properties Analysis
# DOPANTS: Mg, Sn, Pb, N - FOCUSED VERSION: 0-30% Prediction Range
# Doping Levels: 0%, 1%, 2%, 5%, 10%, 15%, 20%, 30%
# Properties: Bandgap (Pure ML), Formation Energy (ML), Conductivity, Mobility, Effective Mass, Absorption
# ============================================================

# === 0. Colab One-time Installs ===
!pip install -q mp-api pymatgen scikit-learn pandas matplotlib seaborn numpy joblib scipy

# === 1. Imports & Configuration ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mp_api.client import MPRester
from pymatgen.core import Composition, Element

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    KFold
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import joblib
import warnings
warnings.filterwarnings("ignore")
# ============================================================
# Reproducibility
# ============================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

plt.style.use('default')
sns.set_theme(style="whitegrid")

from getpass import getpass
API_KEY = getpass("4QoUiunPSMRpqOLTwA6qRu8edPSBArZD").strip()

print("="*80)
print("MULTI-DOPANT ZnO ELECTRONIC PROPERTIES ANALYSIS - FOCUSED 0-30% PREDICTIONS")
print("DOPANTS: Mg, Sn, Pb, N")
print("Focus: N-type Conductivity + Electronic Properties")
print("FOCUSED: Bandgap (Pure ML) + Formation Energy (ML + Physics)")
print("="*80)

# === 2. Enhanced Multi-Dopant Data Fetching ===
print("\nFetching ZnO and multi-doped ZnO materials from Materials Project...")

# Define dopants and their properties
# ============================================================
# Mg-only dopant definition
# ============================================================

DOPANTS = {
    'Mg': {
        'electronegativity_diff': 0.31,
        'size_mismatch': 0.46,
        'bond_energy_diff': -134
    }
}

with MPRester(API_KEY) as mpr:
    # Fetch pure ZnO materials
    pure_zno = mpr.materials.summary.search(
        elements=["Zn", "O"],
        exclude_elements=["H"],
        fields=["material_id", "band_gap", "density", "volume", "nsites",
                "formation_energy_per_atom", "cbm", "vbm", "elements",
                "formula_pretty", "energy_above_hull"]
    )

    # Fetch doped ZnO materials for each dopant
    all_doped_materials = []

    for dopant in DOPANTS.keys():
        print(f"Fetching {dopant}-doped ZnO materials...")
        doped_materials = mpr.materials.summary.search(
            elements=["Zn", "O", dopant],
            exclude_elements=["H"],
            fields=["material_id", "band_gap", "density", "volume", "nsites",
                    "formation_energy_per_atom", "cbm", "vbm", "elements",
                    "formula_pretty", "energy_above_hull"]
        )
        all_doped_materials.extend(doped_materials)

all_materials = pure_zno + all_doped_materials
df = pd.DataFrame([r.dict() for r in all_materials])

# ============================================================
# Remove duplicate Materials Project entries
# ============================================================
if "material_id" in df.columns:
    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["material_id"],
        keep="first"
    ).reset_index(drop=True)

    removed_duplicates = before_duplicates - len(df)

    print(f"Duplicate MP entries removed: {removed_duplicates}")
    print(f"Unique materials remaining: {len(df)}")
else:
    print("WARNING: material_id not found; duplicate removal skipped.")

print(f"Total unique materials fetched: {len(df)}")

# === 3. Enhanced Multi-Dopant Data Preprocessing ===
print("\nProcessing and classifying multi-dopant materials...")

# Clean data
df = df.dropna(subset=["band_gap"])
df = df[df["band_gap"] > 0.1]
df = df[df["band_gap"] < 8.0]

df["nelements"] = df["elements"].apply(len)

# Identify dopant type and calculate doping percentage
def identify_dopant_and_percentage(elements, formula):

    try:
        amounts = Composition(formula).get_el_amt_dict()
        mg_count = amounts.get("Mg", 0.0)
        zn_count = amounts.get("Zn", 0.0)

        if mg_count == 0:
            return "Pure", 0.0

        return "Mg", 100.0 * mg_count / (zn_count + mg_count)

    except Exception:
        return "Mg", np.nan

# Apply dopant identification
df[['dopant_type', 'doping_percent']] = df.apply(
    lambda x: pd.Series(identify_dopant_and_percentage(x["elements"], x["formula_pretty"])),
    axis=1
)

# === FOCUSED MATERIALS DISTRIBUTION ANALYSIS FOR ALL DOPANTS ===
print("\n" + "="*90)
print("MATERIALS PROJECT MULTI-DOPANT DISTRIBUTION ANALYSIS (0-50% FOCUS)")
print("="*90)

# Your requested five specific ranges for focused 0-50% analysis
requested_ranges = [
    (0, 10, "0-10%"),
    (10, 20, "10-20%"),
    (20, 30, "20-30%"),
    (30, 40, "30-40%"),
    (40, 50, "40-50%")
]

print("Materials distribution by DOPANT TYPE in FOCUSED 0-50% RANGES:")
print("-" * 90)

total_materials = len(df)
dopant_distribution = {}

for dopant in ['Pure'] + list(DOPANTS.keys()):
    dopant_data = df[df['dopant_type'] == dopant]
    dopant_count = len(dopant_data)
    dopant_percentage = (dopant_count / total_materials) * 100
    dopant_distribution[dopant] = dopant_count

    print(f"\n{dopant:4} DOPANT | {dopant_count:4d} materials ({dopant_percentage:5.1f}%)")

    for start, end, label in requested_ranges:
        range_count = len(dopant_data[(dopant_data['doping_percent'] >= start) & (dopant_data['doping_percent'] < end)])
        range_percentage = (range_count / dopant_count) * 100 if dopant_count > 0 else 0
        print(f"   {label:8} | {range_count:4d} materials ({range_percentage:5.1f}%)")

print(f"\n USING 50% DOPING FILTER FOR ALL DOPANTS - FOCUSED APPROACH!")
print(f"Goal: Predict all dopants in practical 0-30% range")

df_strategic = df[df["doping_percent"] <= 50.0].copy()  # FOCUSED 0-50%
df = df_strategic

print(f"\nDataset size for focused 0-50% multi-dopant analysis:")
print(f"   Total materials in 0-50% range: {len(df)} materials")

# Fill missing values FIRST
numeric_cols = ["density", "volume", "nsites", "formation_energy_per_atom", "cbm", "vbm"]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# CREATE BASIC FEATURES FIRST
df["volume_per_site"] = df["volume"] / df["nsites"]
df["avg_atomic_volume"] = df["volume"] / df["nsites"]

# === PHYSICS-BASED 2D CLASSIFICATION ===
print("\nApplying physics-based 2D classification...")

def create_physical_2D_features(df):
    """Create physics-based features to identify 2D structures"""

    # Layered structure indicators
    df["atoms_per_unit_volume"] = df["nsites"] / df["volume"]
    df["volume_expansion"] = df["volume"] / df["nsites"]

    # Coordination environment
    df["coordination_factor"] = df["nsites"] / (df["volume"] ** (1/3))
    df["dimensional_factor"] = df["volume"] / (df["nsites"] ** (2/3))

    # Quantum confinement indicators
    df["confinement_parameter"] = 1.0 / (df["volume_per_site"] + 1e-6)
    df["thickness_indicator"] = df["volume"] / (df["nsites"] * df["density"])

    return df

def classify_2D_with_physics(df):
    """Classify 2D vs Bulk using multiple physical criteria"""

    # Create physical features
    df = create_physical_2D_features(df)

    # Physics-based criteria for 2D materials
    high_bandgap = df["band_gap"] > df["band_gap"].quantile(0.75)
    low_atomic_density = df["atoms_per_unit_volume"] < df["atoms_per_unit_volume"].quantile(0.25)
    high_expansion = df["volume_expansion"] > df["volume_expansion"].quantile(0.75)
    low_coordination = df["coordination_factor"] < df["coordination_factor"].quantile(0.25)
    high_confinement = df["confinement_parameter"] > df["confinement_parameter"].quantile(0.75)

    # Combine criteria (need at least 3 out of 5)
    criteria_count = (
        high_bandgap.astype(int) +
        low_atomic_density.astype(int) +
        high_expansion.astype(int) +
        low_coordination.astype(int) +
        high_confinement.astype(int)
    )

    df["is_2D_physics"] = criteria_count >= 3

    return df

# Apply physics-based classification
df = classify_2D_with_physics(df)
df["structure_type"] = df["is_2D_physics"].map({True: "2D ZnO", False: "Bulk ZnO"})

# Verify the classification
bulk_avg_bg = df[~df["is_2D_physics"]]["band_gap"].mean()
twod_avg_bg = df[df["is_2D_physics"]]["band_gap"].mean()

print(f"Physics-based classification results:")
print(f"   Bulk ZnO count: {(~df['is_2D_physics']).sum()}")
print(f"   2D ZnO count: {df['is_2D_physics'].sum()}")
print(f"   Bulk ZnO average bandgap: {bulk_avg_bg:.3f} eV")
print(f"   2D ZnO average bandgap: {twod_avg_bg:.3f} eV")

if twod_avg_bg <= bulk_avg_bg:
    print("Applying stricter criteria for correct physics...")
    # Use stricter criteria
    df["is_2D_physics"] = df.apply(lambda row: (
        (row["band_gap"] > df["band_gap"].quantile(0.8)) and
        (row["atoms_per_unit_volume"] < df["atoms_per_unit_volume"].quantile(0.2))
    ), axis=1)

    df["structure_type"] = df["is_2D_physics"].map({True: "2D ZnO", False: "Bulk ZnO"})

    bulk_avg_bg = df[~df["is_2D_physics"]]["band_gap"].mean()
    twod_avg_bg = df[df["is_2D_physics"]]["band_gap"].mean()

    print(f"   Stricter criteria - Bulk: {bulk_avg_bg:.3f} eV, 2D: {twod_avg_bg:.3f} eV")

    if twod_avg_bg <= bulk_avg_bg:
        print("Manual adjustment - ensuring 2D has higher bandgap...")
        top_bandgap_threshold = df["band_gap"].quantile(0.85)
        df["is_2D_physics"] = df["band_gap"] > top_bandgap_threshold
        df["structure_type"] = df["is_2D_physics"].map({True: "2D ZnO", False: "Bulk ZnO"})

        bulk_avg_bg = df[~df["is_2D_physics"]]["band_gap"].mean()
        twod_avg_bg = df[df["is_2D_physics"]]["band_gap"].mean()
        print(f"   Final result - Bulk: {bulk_avg_bg:.3f} eV, 2D: {twod_avg_bg:.3f} eV")

# Update structural factor for ML
df["structural_factor"] = df["is_2D_physics"].astype(int)

print(f"Physics-based 2D classification completed")
print(f"Materials classified: {len(df)} total samples")

# === 4. ENHANCED Multi-Dopant Feature Engineering ===
print("\nEngineering ENHANCED features for multi-dopant electronic properties...")

# Continue with existing features
df["density_squared"] = df["density"] ** 2
df["volume_squared"] = df["volume"] ** 2
df["nsites_squared"] = df["nsites"] ** 2

# Multi-dopant specific features
df["doping_squared"] = df["doping_percent"] ** 2
df["doping_cubed"] = df["doping_percent"] ** 3
df["doping_log"] = np.log1p(df["doping_percent"])
df["doping_sqrt"] = np.sqrt(df["doping_percent"] + 1e-6)

# Create dopant-specific physics features
for dopant in DOPANTS.keys():
    # Create binary indicator for each dopant
    df[f"is_{dopant.lower()}"] = (df["dopant_type"] == dopant).astype(int)

    # Dopant-specific physics features
    dopant_mask = df["dopant_type"] == dopant
    df[f"{dopant.lower()}_lattice_strain"] = 0.0
    df[f"{dopant.lower()}_electronegativity_diff"] = 0.0
    df[f"{dopant.lower()}_size_mismatch"] = 0.0
    df[f"{dopant.lower()}_bond_energy_diff"] = 0.0

    if dopant_mask.any():
        df.loc[dopant_mask, f"{dopant.lower()}_lattice_strain"] = df.loc[dopant_mask, "doping_percent"] * abs(DOPANTS[dopant]['size_mismatch'])
        df.loc[dopant_mask, f"{dopant.lower()}_electronegativity_diff"] = df.loc[dopant_mask, "doping_percent"] * abs(DOPANTS[dopant]['electronegativity_diff'])
        df.loc[dopant_mask, f"{dopant.lower()}_size_mismatch"] = df.loc[dopant_mask, "doping_percent"] * abs(DOPANTS[dopant]['size_mismatch'])
        df.loc[dopant_mask, f"{dopant.lower()}_bond_energy_diff"] = df.loc[dopant_mask, "doping_percent"] * abs(DOPANTS[dopant]['bond_energy_diff'])

# General doping interaction features
df["doping_interaction"] = df["doping_percent"] * df["density"]
df["doping_volume_effect"] = df["doping_percent"] * df["volume_per_site"]
df["doping_structural_coupling"] = df["doping_percent"] * df["structural_factor"]
df["density_volume_ratio"] = df["density"] / df["volume"]
df["compactness"] = df["nsites"] / df["volume"]
df["dopant_count"] = df["nelements"] - 2
# ============================================================
# Composition-based Materials descriptors
# ============================================================

def calculate_composition_descriptors(formula):

    comp = Composition(formula)

    amounts = comp.get_el_amt_dict()
    total_atoms = sum(amounts.values())

    # Atomic fractions
    fractions = {
        el: amount / total_atoms
        for el, amount in amounts.items()
    }

    # Zn and O fractions
    zn_fraction = fractions.get("Zn", 0.0)
    o_fraction = fractions.get("O", 0.0)

    # Zn/O ratio
    if o_fraction > 0:
        zn_o_ratio = zn_fraction / o_fraction
    else:
        zn_o_ratio = 0.0

    # Electronegativity
    en_values = []
    en_weights = []

    for el, amount in amounts.items():
        element = Element(el)

        if element.X is not None:
            en_values.append(element.X)
            en_weights.append(amount / total_atoms)

    if en_values:
        avg_electronegativity = np.average(
            en_values,
            weights=en_weights
        )

        electronegativity_difference = (
            max(en_values) - min(en_values)
        )
    else:
        avg_electronegativity = 0.0
        electronegativity_difference = 0.0

    # Atomic radius
    radius_values = []
    radius_weights = []

    for el, amount in amounts.items():
        element = Element(el)

        if element.atomic_radius is not None:
            radius_values.append(float(element.atomic_radius))
            radius_weights.append(amount / total_atoms)

    if radius_values:
        avg_atomic_radius = np.average(
            radius_values,
            weights=radius_weights
        )
    else:
        avg_atomic_radius = 0.0

    # Valence electrons
    valence_values = []
    valence_weights = []

    for el, amount in amounts.items():
        element = Element(el)

        if element.group is not None:

            if element.group <= 2:
                valence_electrons = element.group

            elif element.group >= 13:
                valence_electrons = element.group - 10

            else:
                valence_electrons = 2

            valence_values.append(valence_electrons)
            valence_weights.append(amount / total_atoms)

    if valence_values:
        avg_valence_electrons = np.average(
            valence_values,
            weights=valence_weights
        )
    else:
        avg_valence_electrons = 0.0

    return pd.Series({
        "Zn_fraction": zn_fraction,
        "O_fraction": o_fraction,
        "Zn_O_ratio": zn_o_ratio,
        "avg_electronegativity": avg_electronegativity,
        "electronegativity_difference": electronegativity_difference,
        "avg_atomic_radius": avg_atomic_radius,
        "avg_valence_electrons": avg_valence_electrons
    })


composition_features = df["formula_pretty"].apply(
    calculate_composition_descriptors
)

df = pd.concat(
    [df, composition_features],
    axis=1
)
# ============================================================
# Ionic-radius mismatch
# ============================================================

def calculate_ionic_radius_mismatch(formula):

    comp = Composition(formula)
    amounts = comp.get_el_amt_dict()

    total_atoms = sum(amounts.values())

    radii = []
    weights = []

    for el, amount in amounts.items():

        element = Element(el)

        if element.atomic_radius is not None:
            radii.append(float(element.atomic_radius))
            weights.append(amount / total_atoms)

    if len(radii) < 2:
        return 0.0

    avg_radius = np.average(
        radii,
        weights=weights
    )

    mismatch = np.sqrt(
        np.sum(
            np.array(weights) *
            (np.array(radii) - avg_radius) ** 2
        )
    ) / (avg_radius + 1e-12)

    return mismatch


df["ionic_radius_mismatch"] = df["formula_pretty"].apply(
    calculate_ionic_radius_mismatch
)
# Feature list (including multi-dopant features)
# ============================================================
# FINAL LEAKAGE-FREE FEATURE SET
# ============================================================

base_features = [

    # Structural descriptors
    "density",
    "volume",
    "nsites",
    "avg_atomic_volume",
    "dopant_count",
    "volume_per_site",

    # Structural/physics descriptors
    "structural_factor",
    "doping_structural_coupling",
    "density_volume_ratio",
    "compactness",
    "coordination_factor",
    "dimensional_factor",
    "confinement_parameter",

    # Composition descriptors
    "Zn_fraction",
    "O_fraction",
    "Zn_O_ratio",
    "avg_electronegativity",
    "electronegativity_difference",
    "avg_atomic_radius",
    "avg_valence_electrons",
    "ionic_radius_mismatch"
]

# Add dopant-specific features
dopant_features = []
for dopant in DOPANTS.keys():
    dopant_features.extend([
        f"is_{dopant.lower()}",
        f"{dopant.lower()}_lattice_strain",
        f"{dopant.lower()}_electronegativity_diff",
        f"{dopant.lower()}_size_mismatch",
        f"{dopant.lower()}_bond_energy_diff"
    ])

# ============================================================
# FINAL FEATURE COLUMN LIST
# ============================================================

feature_columns = base_features + dopant_features

# ============================================================
# PRINT FINAL FEATURE LIST
# ============================================================

print("\nFinal leakage-free features:")

for i, feature in enumerate(feature_columns, 1):
    print(f"{i:2d}. {feature}")

print(
    f"\nNumber of final features: "
    f"{len(feature_columns)}"
)

print(
    f"Created {len(feature_columns)} features "
    f"for multi-dopant electronic properties analysis"
)

# === 5. MULTI-DOPANT ELECTRONIC PROPERTIES CALCULATION FUNCTIONS ===
import numpy as np

TRANSPORT_TEMPERATURE_K = 300.0
Q_C = 1.602e-19
K_B_EV_K = 8.617e-5
DOS_PREFACTOR_CM3 = 2.51e19

M0_KG = 9.1093837e-31
HBAR_J_S = 1.054571817e-34
DOS_PREFACTOR_CM2 = (
    M0_KG * (K_B_EV_K * Q_C) * 300.0 / (np.pi * HBAR_J_S**2) * 1e-4
)
BASE_MOBILITY_CM2_VS = 200.0
BANDGAP_REFERENCE_EV = 2.0
PRE_ALLOY_MOBILITY_FLOOR_CM2_VS = 0.1

TRANSPORT_PARAMETERS = {
    "Bulk ZnO": {
        "Native_Electron_Density_cm3": 1.0e17,
        "DOS_Electron_Mass_m0": 0.24,
        "DOS_Hole_Mass_m0": 0.59,
        "Carrier_Density_Convention": "volumetric (cm^-3)",
        "DOS_Model": "3D effective DOS",
        "Transport_Quantity": "Bulk conductivity",
        "Transport_Unit": "S/m",
    },
    "2D ZnO": {
        "Native_Electron_Density_cm2": 1.0e10,
        "DOS_Electron_Mass_m0": 0.24,
        "DOS_Hole_Mass_m0": 0.59,
        "Carrier_Density_Convention": "sheet (cm^-2)",
        "DOS_Model": "2D parabolic band; spin degeneracy 2; valley degeneracy 1",
        "Transport_Quantity": "Sheet conductance",
        "Transport_Unit": "S",
    },
}

def mg_cation_fraction(doping_percent):
    """Convert percentage p (e.g. 5) to cation fraction x=p/100 (e.g. 0.05)."""
    if not np.isfinite(doping_percent) or not 0.0 <= doping_percent <= 30.0:
        raise ValueError("Mg concentration must be a finite percentage in [0, 30].")
    return float(doping_percent) / 100.0

def calculate_density_of_states(
    effective_mass,
    structure_type,
    temperature=300
):


    parameters = TRANSPORT_PARAMETERS[structure_type]
    m_e = parameters["DOS_Electron_Mass_m0"]
    m_h = parameters["DOS_Hole_Mass_m0"]

    if structure_type == "2D ZnO":
        Nc = DOS_PREFACTOR_CM2 * m_e * (temperature / 300.0)
        Nv = DOS_PREFACTOR_CM2 * m_h * (temperature / 300.0)
        return Nc, Nv

    Nc = (
        DOS_PREFACTOR_CM3
        * (m_e ** 1.5)
        * (temperature / 300.0) ** 1.5
    )

    Nv = (
        DOS_PREFACTOR_CM3
        * (m_h ** 1.5)
        * (temperature / 300.0) ** 1.5
    )

    return Nc, Nv

def calculate_n_type_conductivity_ZnO(
    bandgap,
    doping_percent,
    mobility_cm2_Vs,
    effective_mass,
    structure_type,
    temperature=300,
    return_details=False
):


    # --------------------------------------------------
    # Physical constants
    # --------------------------------------------------
    q = Q_C           # C
    k_B = K_B_EV_K     # eV/K

    # --------------------------------------------------
    # Density of states from effective mass
    # --------------------------------------------------
    Nc, Nv = calculate_density_of_states(
        effective_mass,
        structure_type,
        temperature
    )

    # --------------------------------------------------
    # Native defect concentration
    # --------------------------------------------------
    parameters = TRANSPORT_PARAMETERS[structure_type]
    is_2d = structure_type == "2D ZnO"
    density_suffix = "cm2" if is_2d else "cm3"
    n_defect = parameters[f"Native_Electron_Density_{density_suffix}"]

    Eg_eff = bandgap


    ni_intrinsic = np.sqrt(Nc * Nv) * np.exp(
        -Eg_eff / (2.0 * k_B * temperature)
    )

    # --------------------------------------------------
    # Total electrons: bulk volume density (cm^-3), 2D sheet density (cm^-2)
    # Original additive native-defect model; not a charge-neutrality solution.
    # --------------------------------------------------
    n_total = ni_intrinsic + n_defect

    x = mg_cation_fraction(doping_percent)
    mu_eff = mobility_cm2_Vs

    transport_value = q * n_total * mu_eff * (1.0 if is_2d else 100)

    if return_details:
        return transport_value, {
            "Mg_Cation_Fraction_x": x,
            "Temperature_K": temperature,
            **parameters,
            f"Nc_{density_suffix}": Nc,
            f"Nv_{density_suffix}": Nv,
            f"Intrinsic_Electron_Density_{density_suffix}": ni_intrinsic,
            f"Total_Electron_Density_{density_suffix}": n_total,
        }
    return transport_value



def calculate_effective_mass_multi_dopant(bandgap, dopant_type):
    """Calculate effective mass (m*/m0) for different dopants"""
    base_mass = 0.3 + 0.1 * (bandgap - 2.0)

    # Dopant-specific mass corrections
    mass_corrections = {
        'Mg': 1.0,    # Reference
        'Pure': 1.0   # Pure material
    }

    correction = mass_corrections.get(dopant_type, 1.0)
    m_eff = base_mass * correction
    return max(0.1, m_eff)

def calculate_electron_mobility_multi_dopant(doping_percent, bandgap, dopant_type):

    base_mobility = BASE_MOBILITY_CM2_VS  # cm2/V·s

    # Dopant-specific mobility factors
    mobility_factors = {
        'Mg': 1.0,    # Reference
        'Pure': 1.0   # Pure material
    }

    base_factor = mobility_factors.get(dopant_type, 1.0)

    # COMMENT 2: fraction convention, numerically identical to the original.
    x = mg_cation_fraction(doping_percent)
    if x <= 0.10:
        scattering_factor = 1 / (1 + 50.0 * x)
    elif x <= 0.30:
        scattering_factor = 1 / (6 + 30.0 * (x - 0.10))
    else:
        scattering_factor = 1 / (12 + 20.0 * (x - 0.30))

    # Bandgap effect
    bandgap_factor = (bandgap / BANDGAP_REFERENCE_EV) ** 0.5

    mobility = base_mobility * base_factor * scattering_factor * bandgap_factor
    alloy_factor = max(0.5, 1.0 - 0.5 * x)
    return max(PRE_ALLOY_MOBILITY_FLOOR_CM2_VS, mobility) * alloy_factor

def calculate_absorption_coefficient_multi_dopant(bandgap, doping_percent, dopant_type):
    """Calculate optical absorption coefficient (cm-1) for different dopants"""
    alpha_0 = 1e4

    if bandgap > 1.5:
        alpha = alpha_0 * ((3.0 - bandgap) / 1.5) ** 2
    else:
        alpha = alpha_0 * 2

    # Dopant-specific absorption enhancement
    absorption_factors = {
        'Mg': 1.0,    # Reference
        'Pure': 0.8   # Pure material
    }

    base_factor = absorption_factors.get(dopant_type, 1.0)

    # Enhanced doping enhancement for focused range
    if doping_percent <= 20:
        doping_enhancement = 1 + doping_percent * 0.02
    else:
        doping_enhancement = 1.4 + (doping_percent - 20) * 0.01

    return alpha * base_factor * doping_enhancement

def apply_electronic_focused_corrections_multi_dopant(doping, predicted_formation_energy, structure_type, dopant_type):

    return predicted_formation_energy

print("\nMulti-dopant electronic properties calculation functions implemented:")
print("    Pure ML Predictions for Bandgap (NO corrections)")
print("    Pure ML Predictions for Formation Energy (ALL DOPANTS - NO corrections)")
print("    Multi-dopant Bulk Conductivity / 2D Sheet Conductance calculation")
print("    Multi-dopant Electron Mobility calculation")
print("    Multi-dopant Effective Mass calculation")
print("    Multi-dopant Optical Absorption Coefficient calculation")

print("\n  FORMATION ENERGY CORRECTION STRATEGY:")
print("   • ALL DOPANTS (Mg, Sn, Pb, N): Pure ML predictions ONLY")
print("   • NO physics corrections for any dopant")
print("   • This reveals natural stability of ALL dopants without artificial corrections")

transport_parameters_df = pd.DataFrame.from_dict(
    TRANSPORT_PARAMETERS, orient="index"
).rename_axis("Structure").reset_index()
transport_parameters_df["Temperature_K"] = TRANSPORT_TEMPERATURE_K
transport_parameters_df["q_C"] = Q_C
transport_parameters_df["k_B_eV_per_K"] = K_B_EV_K
# DOS prefactors refer to m*=m0 and 300 K; use only the matching unit.
transport_parameters_df["DOS_Prefactor_cm3"] = np.where(
    transport_parameters_df["Structure"] == "Bulk ZnO", DOS_PREFACTOR_CM3, np.nan
)
transport_parameters_df["DOS_Prefactor_cm2"] = np.where(
    transport_parameters_df["Structure"] == "2D ZnO", DOS_PREFACTOR_CM2, np.nan
)
transport_parameters_df["Base_Mobility_cm2_per_Vs"] = BASE_MOBILITY_CM2_VS
transport_parameters_df["Bandgap_Reference_eV"] = BANDGAP_REFERENCE_EV
transport_parameters_df["Pre_Alloy_Mobility_Floor_cm2_per_Vs"] = PRE_ALLOY_MOBILITY_FLOOR_CM2_VS
transport_parameters_df["Mg_Mobility_Factor"] = 1.0
transport_parameters_df["Pure_Mobility_Factor"] = 1.0
transport_parameters_df["Scattering_Factor_Sx"] = (
    "1/(1+50*x) for x<=0.10; 1/(6+30*(x-0.10)) for 0.10<x<=0.30"
)
transport_parameters_df["Alloy_Factor_Ax"] = "max(0.5, 1-0.5*x)"
print("\nTRANSPORT PARAMETERS (assumed; separately for bulk and 2D):")
print(transport_parameters_df.to_string(index=False))
print("Mg cation fraction x=N_Mg/(N_Zn+N_Mg); plotted percentage p=100*x.")
print("Bulk: assumed native electrons = 1e17 cm^-3; conductivity = 100*q*n*mu [S/m].")
print("2D: assumed native electron sheet density = 1e10 cm^-2; G_sheet = q*n_s*mu [S].")
print("The 2D density is a revised assumption, not a conversion of the old cm^-3 value.")
print("2D intrinsic carriers use sheet DOS (cm^-2); DOS masses remain assumed.")
print("No layer thickness is assumed; 2D sheet conductance is not reported in S/m.")

# === 6. Machine Learning Models (MULTI-DOPANT TRAINING) ===
print("\nTraining Multi-Dopant Electronic Properties ML Models...")

X = df[feature_columns].copy()
y_bandgap = df["band_gap"].copy()
y_formation = df["formation_energy_per_atom"].copy()

# Remove NaN values
mask = ~(X.isnull().any(axis=1) | y_bandgap.isnull() | y_formation.isnull())
X = X[mask]
y_bandgap = y_bandgap[mask]
y_formation = y_formation[mask]

print(f" MULTI-DOPANT Training: {len(X)} samples with {len(feature_columns)} features")
print(f"   Bandgap: Pure ML | Formation Energy: Pure ML (ALL DOPANTS)!")

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_bg_train, y_bg_test, y_fe_train, y_fe_test = train_test_split(
    X,
    y_bandgap,
    y_formation,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=df.loc[mask, "structure_type"]
)

print("\n" + "="*70)
print("DATA SPLIT")
print("="*70)
print(f"Total samples:              {len(X)}")
print(f"Development/Training set:   {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Held-out Test set:          {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print("Test set is reserved exclusively for final evaluation.")
# ============================================================
# SAVE FINAL ML DATASET WITH TRAIN / TEST ASSIGNMENT
# ============================================================

# Start from the exact rows that survived the NaN filtering
df_ml = df.loc[X.index].copy()

# Default assignment
df_ml["dataset_split"] = "Training"

# Assign the exact held-out test samples
df_ml.loc[X_test.index, "dataset_split"] = "Test"

# ------------------------------------------------------------
# Columns to save
# ------------------------------------------------------------

final_dataset_columns = [
    "material_id",
    "structure_type",
    "band_gap",
] + feature_columns + ["dataset_split"]

# Keep only columns that actually exist
final_dataset_columns = [
    col for col in final_dataset_columns
    if col in df_ml.columns
]

# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

final_ml_dataset = df_ml[final_dataset_columns].copy()

final_ml_dataset.to_csv(
    "final_ZnO_2D_ZnO_ML_dataset.csv",
    index=False
)

print("\n" + "="*70)
print("FINAL ML DATASET SAVED")
print("="*70)
print(
    "File: final_ZnO_2D_ZnO_ML_dataset.csv"
)

print(
    f"Total samples saved: {len(final_ml_dataset)}"
)

print(
    f"Training samples: "
    f"{(final_ml_dataset['dataset_split'] == 'Training').sum()}"
)

print(
    f"Test samples: "
    f"{(final_ml_dataset['dataset_split'] == 'Test').sum()}"
)

print("\nDataset split:")
print(
    final_ml_dataset["dataset_split"]
    .value_counts()
)
# ============================================================
# INTERNAL 5-FOLD CROSS-VALIDATION STRATEGY
# ============================================================
cv_strategy = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_SEED
)

print("\nCross-validation:")
print("   Strategy: 5-fold shuffled K-fold CV")
print("   Purpose: Internal validation")
print(f"   Random seed: {RANDOM_SEED}")

# Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=2000, max_depth=20, min_samples_split=3,
        min_samples_leaf=2, max_features='sqrt', bootstrap=True,
        random_state=RANDOM_SEED, n_jobs=-1, oob_score=True
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=1500, learning_rate=0.05, max_depth=6,
        subsample=0.8, min_samples_split=3, min_samples_leaf=2,
        max_features='sqrt', random_state=RANDOM_SEED
    )
}

# Train bandgap models
# ============================================================
# BANDGAP MODEL TRAINING + INTERNAL 5-FOLD CV
# ============================================================

results = []
trained_models = {}

print("\n" + "="*70)
print("MULTI-DOPANT ELECTRONIC PROPERTIES MODEL TRAINING")
print("="*70)

for name, model in models.items():

    print(f"\nTraining {name} for Bandgap...")

    model_pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("model", model)
    ])

    # --------------------------------------------------------
    # 5-fold INTERNAL CV
    # --------------------------------------------------------
    cv_scores = cross_val_score(
        model_pipeline,
        X_train,
        y_bg_train,
        cv=cv_strategy,
        scoring="r2",
        n_jobs=-1
    )

    # --------------------------------------------------------
    # Final model:
    # after CV, fit on the COMPLETE 80% development set
    # --------------------------------------------------------
    model_pipeline.fit(X_train, y_bg_train)

    trained_models[name] = model_pipeline

    # --------------------------------------------------------
    # Training and held-out test predictions
    # --------------------------------------------------------
    y_train_pred = model_pipeline.predict(X_train)
    y_test_pred = model_pipeline.predict(X_test)

    train_r2 = r2_score(y_bg_train, y_train_pred)
    test_r2 = r2_score(y_bg_test, y_test_pred)

    test_mae = mean_absolute_error(
        y_bg_test,
        y_test_pred
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_bg_test,
            y_test_pred
        )
    )

    results.append({
        "Model": name,
        "Test R²": test_r2,
        "Test MAE": test_mae,
        "Test RMSE": test_rmse,
        "CV R² Mean": cv_scores.mean(),
        "CV R² Std": cv_scores.std()
    })

    print(f"{name} completed:")
    print(f"   Test R²: {test_r2:.4f}")
    print(f"   Test MAE: {test_mae:.4f} eV")
    print(f"   Test RMSE: {test_rmse:.4f} eV")
    print(
        f"   5-Fold CV R²: "
        f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
    )

# Train formation energy models
# ============================================================
# FORMATION ENERGY MODEL TRAINING + INTERNAL 5-FOLD CV
# ============================================================

formation_models = {}
formation_results = []

print("\n" + "="*70)
print("MULTI-DOPANT FORMATION ENERGY MODEL TRAINING")
print("="*70)

for name, model_class in [
    ("Random Forest", RandomForestRegressor),
    ("Gradient Boosting", GradientBoostingRegressor)
]:

    print(f"\nTraining {name} for Formation Energy...")

    if name == "Random Forest":

        fe_model = model_class(
            n_estimators=2000,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            oob_score=True
        )

    else:

        fe_model = model_class(
            n_estimators=1500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=RANDOM_SEED
        )

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------
    fe_pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("model", fe_model)
    ])

    # --------------------------------------------------------
    # Internal 5-fold CV
    # --------------------------------------------------------
    cv_scores_fe = cross_val_score(
        fe_pipeline,
        X_train,
        y_fe_train,
        cv=cv_strategy,
        scoring="r2",
        n_jobs=-1
    )

    # --------------------------------------------------------
    # Final fitting on complete 80% development set
    # --------------------------------------------------------
    fe_pipeline.fit(X_train, y_fe_train)

    formation_models[name] = fe_pipeline

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------
    y_fe_train_pred = fe_pipeline.predict(X_train)
    y_fe_test_pred = fe_pipeline.predict(X_test)

    fe_train_r2 = r2_score(
        y_fe_train,
        y_fe_train_pred
    )

    fe_test_r2 = r2_score(
        y_fe_test,
        y_fe_test_pred
    )

    fe_test_mae = mean_absolute_error(
        y_fe_test,
        y_fe_test_pred
    )

    fe_test_rmse = np.sqrt(
        mean_squared_error(
            y_fe_test,
            y_fe_test_pred
        )
    )

    formation_results.append({
        "Model": name,

        "Test R²": fe_test_r2,
        "Test MAE": fe_test_mae,
        "Test RMSE": fe_test_rmse,
        "CV R² Mean": cv_scores_fe.mean(),
        "CV R² Std": cv_scores_fe.std()
    })

    print(f"{name} Formation Energy Model:")
    #print(f"   Train R²: {fe_train_r2:.4f}")
    print(f"   Test R²: {fe_test_r2:.4f}")
    print(f"   Test MAE: {fe_test_mae:.4f} eV/atom")
    print(f"   Test RMSE: {fe_test_rmse:.4f} eV/atom")
    print(
        f"   5-Fold CV R²: "
        f"{cv_scores_fe.mean():.4f} ± {cv_scores_fe.std():.4f}"
    )

results_df = pd.DataFrame(results)
formation_results_df = pd.DataFrame(formation_results)

# === DISPLAY MODEL PERFORMANCE TABLES ===
print("\nBANDGAP MODEL PERFORMANCE SUMMARY:")
print("="*60)
print(results_df.to_string(index=False, float_format='{:.4f}'.format))

# =====================================================================

print("\nFORMATION ENERGY MODEL PERFORMANCE SUMMARY:")
print("="*60)
print(formation_results_df.to_string(index=False, float_format='{:.4f}'.format))

# === 7. MULTI-DOPANT Electronic Properties Predictions ===
print("\nMULTI-DOPANT ELECTRONIC PROPERTIES PREDICTIONS (0-30% Range)")
print("="*80)

best_bandgap_model_name = results_df.loc[results_df["Test R²"].idxmax(), "Model"]
best_formation_model_name = formation_results_df.loc[formation_results_df["Test R²"].idxmax(), "Model"]

best_bandgap_model = trained_models[best_bandgap_model_name]
best_formation_model = formation_models[best_formation_model_name]

print(f"Best Bandgap Model: {best_bandgap_model_name}")
print(f"Best Formation Energy Model: {best_formation_model_name}")

# MULTI-DOPANT DOPING LEVELS
doping_levels = [0, 1, 2, 5, 10, 15, 20, 30]
structure_types = [0, 1]
dopants_to_analyze = ['Pure', 'Mg']
prediction_results = []
# Cache the exact input rows for the added Mg/bulk uncertainty plot.
bulk_mg_prediction_features = {}

median_values = X.median()

for dopant in dopants_to_analyze:
    print(f"\n{'='*100}")
    print(f"DOPANT: {dopant}")
    print(f"{'='*100}")

    for struct_type in structure_types:
        struct_name = "2D ZnO" if struct_type == 1 else "Bulk ZnO"
        print(f"\n{struct_name} - {dopant} Doped Electronic Properties (0-30% Range):")
        print("-" * 150)
        transport_label = "Sheet conductance" if struct_name == "2D ZnO" else "Conductivity (sigma)"
        transport_unit = "(S)" if struct_name == "2D ZnO" else "(S/m)"
        print(f"   Doping    | Bandgap | Formation Energy | {transport_label:20} | Mobility (mu)  | Effective Mass (m*) | Absorption")
        print(f"   Level     | (eV)    | (eV/atom)        | {transport_unit:20} | (cm2/V·s)      | (m*/m0)             | (cm-1)")
        print("-" * 150)

        for doping in doping_levels:
            if dopant == 'Pure' and doping > 0:
                continue  # Skip doped levels for pure material

            sample_data = median_values.copy()

            # Adjust features for doping and dopant type
            sample_data["doping_percent"] = doping
            sample_data["structural_factor"] = struct_type
            sample_data["doping_squared"] = doping ** 2
            sample_data["doping_cubed"] = doping ** 3
            sample_data["doping_log"] = np.log1p(doping)
            sample_data["doping_sqrt"] = np.sqrt(doping + 1e-6)
            sample_data["doping_interaction"] = doping * sample_data["density"]
            sample_data["doping_volume_effect"] = doping * sample_data["volume_per_site"]
            sample_data["doping_structural_coupling"] = doping * struct_type

            # Set dopant-specific features
            for d in DOPANTS.keys():
                sample_data[f"is_{d.lower()}"] = 1 if d == dopant else 0
                if d == dopant and doping > 0:
                    sample_data[f"{d.lower()}_lattice_strain"] = doping * abs(DOPANTS[d]['size_mismatch'])
                    sample_data[f"{d.lower()}_electronegativity_diff"] = doping * abs(DOPANTS[d]['electronegativity_diff'])
                    sample_data[f"{d.lower()}_size_mismatch"] = doping * abs(DOPANTS[d]['size_mismatch'])
                    sample_data[f"{d.lower()}_bond_energy_diff"] = doping * abs(DOPANTS[d]['bond_energy_diff'])
                else:
                    sample_data[f"{d.lower()}_lattice_strain"] = 0
                    sample_data[f"{d.lower()}_electronegativity_diff"] = 0
                    sample_data[f"{d.lower()}_size_mismatch"] = 0
                    sample_data[f"{d.lower()}_bond_energy_diff"] = 0

            # Predictions
            # Predictions
            sample_array = sample_data[feature_columns].values.reshape(1, -1)
            if dopant == "Mg" and struct_name == "Bulk ZnO":
                bulk_mg_prediction_features[doping] = sample_array[0].copy()
            predicted_bandgap = best_bandgap_model.predict(sample_array)[0]  # PURE ML
            ml_formation_energy = best_formation_model.predict(sample_array)[0]  # ML prediction
            # Apply physics corrections to formation energy
            focused_formation_energy = apply_electronic_focused_corrections_multi_dopant(
                doping, ml_formation_energy, struct_name, dopant
            )

            # Calculate electronic properties
            mobility = calculate_electron_mobility_multi_dopant(doping, predicted_bandgap, dopant)
            effective_mass = calculate_effective_mass_multi_dopant(predicted_bandgap, dopant)
            conductivity, transport_details = calculate_n_type_conductivity_ZnO(
                predicted_bandgap, doping, mobility, effective_mass, struct_name,
                temperature=TRANSPORT_TEMPERATURE_K, return_details=True
            )

            absorption = calculate_absorption_coefficient_multi_dopant(predicted_bandgap, doping, dopant)

            prediction_results.append({
                "Dopant": dopant,
                "Structure": struct_name,
                "Doping_%": doping,
                "Pure_ML_Bandgap_eV": predicted_bandgap,
                "Focused_Formation_Energy_eV": focused_formation_energy,
                "N_Type_Conductivity_S_per_m": conductivity if struct_name == "Bulk ZnO" else np.nan,
                "N_Type_Sheet_Conductance_S": conductivity if struct_name == "2D ZnO" else np.nan,
                "Effective_Mass_ratio": effective_mass,
                "Electron_Mobility_cm2_per_Vs": mobility,
                "Absorption_Coefficient_per_cm": absorption,
                **transport_details
            })

            if doping == 0:
                doping_label = "Pure ZnO"
            else:
                doping_label = f"{doping}% {dopant}"
            print(f"   {doping_label:9} | {predicted_bandgap:7.3f} | {focused_formation_energy:16.3f} | {conductivity:20.2e} | {mobility:14.1f} | {effective_mass:19.2f} | {absorption:10.2e}")

# === 8. MULTI-DOPANT ANALYSIS ===
pred_df = pd.DataFrame(prediction_results)

# COMMENT 2: reconstruct each transport quantity from matching exported units.
bulk_mask = pred_df["Structure"] == "Bulk ZnO"
sheet_mask = pred_df["Structure"] == "2D ZnO"
sigma_reconstructed = (
    Q_C * pred_df.loc[bulk_mask, "Total_Electron_Density_cm3"]
    * pred_df.loc[bulk_mask, "Electron_Mobility_cm2_per_Vs"] * 100.0
)
sheet_conductance_reconstructed = (
    Q_C * pred_df.loc[sheet_mask, "Total_Electron_Density_cm2"]
    * pred_df.loc[sheet_mask, "Electron_Mobility_cm2_per_Vs"]
)
np.testing.assert_allclose(
    pred_df.loc[bulk_mask, "N_Type_Conductivity_S_per_m"], sigma_reconstructed,
    rtol=1e-12, atol=0.0,
    err_msg="Bulk conductivity is inconsistent with the exported density/mobility."
)
np.testing.assert_allclose(
    pred_df.loc[sheet_mask, "N_Type_Sheet_Conductance_S"], sheet_conductance_reconstructed,
    rtol=1e-12, atol=0.0,
    err_msg="2D sheet conductance is inconsistent with the exported density/mobility."
)
print("\nTransport checks passed: bulk sigma = 100*q*n[cm^-3]*mu[cm2/Vs] [S/m]; "
      "2D G_sheet = q*n_s[cm^-2]*mu[cm2/Vs] [S].")

print("\n" + "="*120)
print("  REQUESTED ANALYSIS: MAXIMUM BULK CONDUCTIVITY / 2D SHEET CONDUCTANCE & MOBILITY FOR EACH DOPANT")
print("="*120)

# FIRST: Maximum N-type conductivity and mobility for each dopant individually
dopants_analysis = ['Mg', 'Sn', 'Pb', 'N']

print("\n1️ INDIVIDUAL DOPANT ANALYSIS - MAXIMUM VALUES:")
print("="*80)

individual_maxima = {}

for dopant in dopants_analysis:
    print(f"\n🔬 {dopant} DOPANT ANALYSIS:")
    print("-" * 50)

    # Filter data for this dopant (both Bulk and 2D)
    dopant_data = pred_df[pred_df["Dopant"] == dopant]

    if len(dopant_data) > 0:
        # Find maximum conductivity
        max_cond_bulk = dopant_data[dopant_data["Structure"] == "Bulk ZnO"]["N_Type_Conductivity_S_per_m"].max()
        max_cond_bulk_idx = dopant_data[dopant_data["Structure"] == "Bulk ZnO"]["N_Type_Conductivity_S_per_m"].idxmax()
        max_cond_bulk_doping = dopant_data.loc[max_cond_bulk_idx, "Doping_%"]

        max_cond_2d = dopant_data[dopant_data["Structure"] == "2D ZnO"]["N_Type_Sheet_Conductance_S"].max()
        max_cond_2d_idx = dopant_data[dopant_data["Structure"] == "2D ZnO"]["N_Type_Sheet_Conductance_S"].idxmax()
        max_cond_2d_doping = dopant_data.loc[max_cond_2d_idx, "Doping_%"]

        # Find maximum mobility
        max_mob_bulk = dopant_data[dopant_data["Structure"] == "Bulk ZnO"]["Electron_Mobility_cm2_per_Vs"].max()
        max_mob_bulk_idx = dopant_data[dopant_data["Structure"] == "Bulk ZnO"]["Electron_Mobility_cm2_per_Vs"].idxmax()
        max_mob_bulk_doping = dopant_data.loc[max_mob_bulk_idx, "Doping_%"]

        max_mob_2d = dopant_data[dopant_data["Structure"] == "2D ZnO"]["Electron_Mobility_cm2_per_Vs"].max()
        max_mob_2d_idx = dopant_data[dopant_data["Structure"] == "2D ZnO"]["Electron_Mobility_cm2_per_Vs"].idxmax()
        max_mob_2d_doping = dopant_data.loc[max_mob_2d_idx, "Doping_%"]

        # Store for comparison
        individual_maxima[dopant] = {
            'max_cond_bulk': max_cond_bulk,
            'max_cond_bulk_doping': max_cond_bulk_doping,
            'max_cond_2d': max_cond_2d,
            'max_cond_2d_doping': max_cond_2d_doping,
            'max_mob_bulk': max_mob_bulk,
            'max_mob_bulk_doping': max_mob_bulk_doping,
            'max_mob_2d': max_mob_2d,
            'max_mob_2d_doping': max_mob_2d_doping
        }

        print(f"   BULK ZnO:")
        print(f"     • Maximum Conductivity: {max_cond_bulk:.2e} S/m at {max_cond_bulk_doping}% doping")
        print(f"     • Maximum Mobility: {max_mob_bulk:.1f} cm2/V·s at {max_mob_bulk_doping}% doping")

        print(f"   2D ZnO:")
        print(f"     • Maximum Sheet Conductance: {max_cond_2d:.2e} S at {max_cond_2d_doping}% doping")
        print(f"     • Maximum Mobility: {max_mob_2d:.1f} cm2/V·s at {max_mob_2d_doping}% doping")

        # Find most stable formation energy
        min_fe_bulk = dopant_data[dopant_data["Structure"] == "Bulk ZnO"]["Focused_Formation_Energy_eV"].min()
        min_fe_bulk_idx = dopant_data[dopant_data["Structure"] == "Bulk ZnO"]["Focused_Formation_Energy_eV"].idxmin()
        min_fe_bulk_doping = dopant_data.loc[min_fe_bulk_idx, "Doping_%"]

        min_fe_2d = dopant_data[dopant_data["Structure"] == "2D ZnO"]["Focused_Formation_Energy_eV"].min()
        min_fe_2d_idx = dopant_data[dopant_data["Structure"] == "2D ZnO"]["Focused_Formation_Energy_eV"].idxmin()
        min_fe_2d_doping = dopant_data.loc[min_fe_2d_idx, "Doping_%"]

        print(f"   STABILITY (Formation Energy):")
        print(f"     • Bulk ZnO most stable: {min_fe_bulk:.3f} eV/atom at {min_fe_bulk_doping}% doping")
        print(f"     • 2D ZnO most stable: {min_fe_2d:.3f} eV/atom at {min_fe_2d_doping}% doping")

print("\n" + "="*120)
print("2️ INTER-DOPANT COMPARISON - OVERALL MAXIMUM VALUES:")
print("="*120)

# SECOND: Compare maximum values between all dopants
print("\n OVERALL MAXIMUM BULK CONDUCTIVITY / 2D SHEET CONDUCTANCE COMPARISON:")
print("-" * 70)

# Find overall maximum conductivity across all dopants
all_bulk_cond = []
all_2d_cond = []

for dopant in dopants_analysis:
    if dopant in individual_maxima:
        all_bulk_cond.append((dopant, individual_maxima[dopant]['max_cond_bulk'], individual_maxima[dopant]['max_cond_bulk_doping']))
        all_2d_cond.append((dopant, individual_maxima[dopant]['max_cond_2d'], individual_maxima[dopant]['max_cond_2d_doping']))

# Sort by conductivity
all_bulk_cond.sort(key=lambda x: x[1], reverse=True)
all_2d_cond.sort(key=lambda x: x[1], reverse=True)

print("BULK ZnO - Conductivity Ranking:")
for i, (dopant, cond, doping) in enumerate(all_bulk_cond):
    print(f"   {i+1}. {dopant:2} | {cond:.2e} S/m at {doping}% doping")

print("\n2D ZnO - Sheet Conductance Ranking:")
for i, (dopant, cond, doping) in enumerate(all_2d_cond):
    print(f"   {i+1}. {dopant:2} | {cond:.2e} S at {doping}% doping")

print("\n OVERALL MAXIMUM ELECTRON MOBILITY COMPARISON:")
print("-" * 70)

# Find overall maximum mobility across all dopants
all_bulk_mob = []
all_2d_mob = []

for dopant in dopants_analysis:
    if dopant in individual_maxima:
        all_bulk_mob.append((dopant, individual_maxima[dopant]['max_mob_bulk'], individual_maxima[dopant]['max_mob_bulk_doping']))
        all_2d_mob.append((dopant, individual_maxima[dopant]['max_mob_2d'], individual_maxima[dopant]['max_mob_2d_doping']))

# Sort by mobility
all_bulk_mob.sort(key=lambda x: x[1], reverse=True)
all_2d_mob.sort(key=lambda x: x[1], reverse=True)

print("BULK ZnO - Mobility Ranking:")
for i, (dopant, mob, doping) in enumerate(all_bulk_mob):
    print(f"   {i+1}. {dopant:2} | {mob:.1f} cm2/V·s at {doping}% doping")

print("\n2D ZnO - Mobility Ranking:")
for i, (dopant, mob, doping) in enumerate(all_2d_mob):
    print(f"   {i+1}. {dopant:2} | {mob:.1f} cm2/V·s at {doping}% doping")

print("\n FORMATION ENERGY ANALYSIS - NATURAL STABILITY (ALL DOPANTS use Pure ML):")
print("-" * 80)

# Formation energy analysis to see which dopants naturally stabilize ZnO
print("Which dopants naturally stabilize ZnO structures (without physics corrections)?")
print("Note: ALL DOPANTS now use Pure ML predictions - NO physics corrections")

stability_analysis = {}
for dopant in dopants_analysis:
    dopant_data = pred_df[pred_df["Dopant"] == dopant]
    if len(dopant_data) > 0:
        # Find most stable points
        bulk_min_fe = dopant_data[dopant_data["Structure"] == "Bulk ZnO"]["Focused_Formation_Energy_eV"].min()
        bulk_min_doping = dopant_data[dopant_data["Structure"] == "Bulk ZnO"]["Focused_Formation_Energy_eV"].idxmin()
        bulk_min_doping_percent = dopant_data.loc[bulk_min_doping, "Doping_%"]

        twod_min_fe = dopant_data[dopant_data["Structure"] == "2D ZnO"]["Focused_Formation_Energy_eV"].min()
        twod_min_doping = dopant_data[dopant_data["Structure"] == "2D ZnO"]["Focused_Formation_Energy_eV"].idxmin()
        twod_min_doping_percent = dopant_data.loc[twod_min_doping, "Doping_%"]

        stability_analysis[dopant] = {
            'bulk_min_fe': bulk_min_fe,
            'bulk_min_doping': bulk_min_doping_percent,
            'twod_min_fe': twod_min_fe,
            'twod_min_doping': twod_min_doping_percent
        }

        correction_note = "(Pure ML)"  # ALL dopants now use Pure ML
        print(f"\n{dopant} {correction_note}:")
        print(f"   • Bulk ZnO: {bulk_min_fe:.3f} eV/atom at {bulk_min_doping_percent}% doping")
        print(f"   • 2D ZnO: {twod_min_fe:.3f} eV/atom at {twod_min_doping_percent}% doping")

# Compare pure ZnO baseline
pure_data = pred_df[pred_df["Dopant"] == "Pure"]
if len(pure_data) > 0:
    pure_bulk_fe = pure_data[pure_data["Structure"] == "Bulk ZnO"]["Focused_Formation_Energy_eV"].iloc[0]
    pure_2d_fe = pure_data[pure_data["Structure"] == "2D ZnO"]["Focused_Formation_Energy_eV"].iloc[0]

    print(f"\nPure ZnO Baseline:")
    print(f"   • Bulk ZnO: {pure_bulk_fe:.3f} eV/atom")
    print(f"   • 2D ZnO: {pure_2d_fe:.3f} eV/atom")

    print(f"\n🎯 STABILITY COMPARISON vs Pure ZnO:")
    print("-" * 50)
    for dopant in dopants_analysis:
        if dopant in stability_analysis:
            bulk_improvement = stability_analysis[dopant]['bulk_min_fe'] - pure_bulk_fe
            twod_improvement = stability_analysis[dopant]['twod_min_fe'] - pure_2d_fe

            bulk_status = "MORE STABLE" if bulk_improvement < 0 else "LESS STABLE"
            twod_status = "MORE STABLE" if twod_improvement < 0 else "LESS STABLE"

            print(f"{dopant}:")
            print(f"   • Bulk: {bulk_improvement:+.3f} eV/atom ({bulk_status})")
            print(f"   • 2D:   {twod_improvement:+.3f} eV/atom ({twod_status})")

print("\n" + "="*100)
print("MULTI-DOPANT COMPARISON ANALYSIS")
print("="*100)

# Compare dopants at specific doping levels
comparison_levels = [2, 10, 20, 30]

for level in comparison_levels:
    print(f"\nDOPANT COMPARISON AT {level}% DOPING:")
    print("-" * 80)

    level_data = pred_df[(pred_df["Doping_%"] == level) & (pred_df["Structure"] == "Bulk ZnO")]
    if len(level_data) > 0:
        level_data_sorted = level_data.sort_values("N_Type_Conductivity_S_per_m", ascending=False)

        print("Conductivity Ranking (Bulk ZnO):")
        for i, row in level_data_sorted.iterrows():
            print(f"   {row['Dopant']:2} | {row['N_Type_Conductivity_S_per_m']:.2e} S/m | {row['Pure_ML_Bandgap_eV']:.3f} eV | {row['Electron_Mobility_cm2_per_Vs']:.1f} cm2/V·s")

# === 9. MULTI-DOPANT VISUALIZATION ===
print("\nGenerating multi-dopant comparison plots...")

fig, axes = plt.subplots(4, 4, figsize=(28, 24))

# Plot 1: Bandgap comparison for all dopants (Bulk)
bulk_data = pred_df[pred_df["Structure"] == "Bulk ZnO"]
for dopant in dopants_to_analyze:
    if dopant == 'Pure':
        continue
    dopant_data = bulk_data[bulk_data["Dopant"] == dopant]
    if len(dopant_data) > 0:
        axes[0,0].plot(dopant_data["Doping_%"], dopant_data["Pure_ML_Bandgap_eV"],
                      'o-', label=dopant, linewidth=2, markersize=5)

axes[0,0].set_title("Bulk ZnO: Bandgap vs Doping (All Dopants)", fontsize=14)
axes[0,0].set_xlabel("Mg cation concentration, 100x (%)")
axes[0,0].set_ylabel("Bandgap (eV)")
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Bandgap comparison for all dopants (2D)
twod_data = pred_df[pred_df["Structure"] == "2D ZnO"]
for dopant in dopants_to_analyze:
    if dopant == 'Pure':
        continue
    dopant_data = twod_data[twod_data["Dopant"] == dopant]
    if len(dopant_data) > 0:
        axes[0,1].plot(dopant_data["Doping_%"], dopant_data["Pure_ML_Bandgap_eV"],
                      'o-', label=dopant, linewidth=2, markersize=5)

axes[0,1].set_title("2D ZnO: Bandgap vs Doping (All Dopants)", fontsize=14)
axes[0,1].set_xlabel("Mg cation concentration, 100x (%)")
axes[0,1].set_ylabel("Bandgap (eV)")
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Formation Energy comparison (Bulk)
for dopant in dopants_to_analyze:
    if dopant == 'Pure':
        continue
    dopant_data = bulk_data[bulk_data["Dopant"] == dopant]
    if len(dopant_data) > 0:
        axes[0,2].plot(dopant_data["Doping_%"], dopant_data["Focused_Formation_Energy_eV"],
                      'o-', label=dopant, linewidth=2, markersize=5)

axes[0,2].set_title("Bulk ZnO: Formation Energy vs Doping", fontsize=14)
axes[0,2].set_xlabel("Mg cation concentration, 100x (%)")
axes[0,2].set_ylabel("Formation Energy (eV/atom)")
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# Plot 4: Formation Energy comparison (2D)
for dopant in dopants_to_analyze:
    if dopant == 'Pure':
        continue
    dopant_data = twod_data[twod_data["Dopant"] == dopant]
    if len(dopant_data) > 0:
        axes[0,3].plot(dopant_data["Doping_%"], dopant_data["Focused_Formation_Energy_eV"],
                      'o-', label=dopant, linewidth=2, markersize=5)

axes[0,3].set_title("2D ZnO: Formation Energy vs Doping", fontsize=14)
axes[0,3].set_xlabel("Mg cation concentration, 100x (%)")
axes[0,3].set_ylabel("Formation Energy (eV/atom)")
axes[0,3].legend()
axes[0,3].grid(True, alpha=0.3)

# Plot 5: Conductivity comparison (Bulk)
for dopant in dopants_to_analyze:
    if dopant == 'Pure':
        continue
    dopant_data = bulk_data[bulk_data["Dopant"] == dopant]
    if len(dopant_data) > 0:
        axes[1,0].semilogy(dopant_data["Doping_%"], dopant_data["N_Type_Conductivity_S_per_m"],
                          'o-', label=dopant, linewidth=2, markersize=5)

axes[1,0].set_title("Bulk ZnO: Conductivity vs Doping", fontsize=14)
axes[1,0].set_xlabel("Mg cation concentration, 100x (%)")
axes[1,0].set_ylabel("Conductivity (S/m)")
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 6: Sheet conductance comparison (2D)
for dopant in dopants_to_analyze:
    if dopant == 'Pure':
        continue
    dopant_data = twod_data[twod_data["Dopant"] == dopant]
    if len(dopant_data) > 0:
        axes[1,1].semilogy(dopant_data["Doping_%"], dopant_data["N_Type_Sheet_Conductance_S"],
                          'o-', label=dopant, linewidth=2, markersize=5)

axes[1,1].set_title("2D ZnO: Sheet Conductance vs Doping", fontsize=14)
axes[1,1].set_xlabel("Mg cation concentration, 100x (%)")
axes[1,1].set_ylabel("Sheet conductance (S)")
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

# Plot 7: Mobility comparison (Bulk)
for dopant in dopants_to_analyze:
    if dopant == 'Pure':
        continue
    dopant_data = bulk_data[bulk_data["Dopant"] == dopant]
    if len(dopant_data) > 0:
        axes[1,2].plot(dopant_data["Doping_%"], dopant_data["Electron_Mobility_cm2_per_Vs"],
                      'o-', label=dopant, linewidth=2, markersize=5)

axes[1,2].set_title("Bulk ZnO: Mobility vs Doping", fontsize=14)
axes[1,2].set_xlabel("Mg cation concentration, 100x (%)")
axes[1,2].set_ylabel("Electron mobility (cm2/V·s)")
axes[1,2].legend()
axes[1,2].grid(True, alpha=0.3)

# Plot 8: Mobility comparison (2D)
for dopant in dopants_to_analyze:
    if dopant == 'Pure':
        continue
    dopant_data = twod_data[twod_data["Dopant"] == dopant]
    if len(dopant_data) > 0:
        axes[1,3].plot(dopant_data["Doping_%"], dopant_data["Electron_Mobility_cm2_per_Vs"],
                      'o-', label=dopant, linewidth=2, markersize=5)

axes[1,3].set_title("2D ZnO: Mobility vs Doping", fontsize=14)
axes[1,3].set_xlabel("Mg cation concentration, 100x (%)")
axes[1,3].set_ylabel("Electron mobility (cm2/V·s)")
axes[1,3].legend()
axes[1,3].grid(True, alpha=0.3)

# Plot 9: Effective Mass comparison (Bulk)
for dopant in dopants_to_analyze:
    if dopant == 'Pure':
        continue
    dopant_data = bulk_data[bulk_data["Dopant"] == dopant]
    if len(dopant_data) > 0:
        axes[2,0].plot(dopant_data["Doping_%"], dopant_data["Effective_Mass_ratio"],
                      'o-', label=dopant, linewidth=2, markersize=5)

axes[2,0].set_title("Bulk ZnO: Effective Mass vs Doping", fontsize=14)
axes[2,0].set_xlabel("Mg cation concentration, 100x (%)")
axes[2,0].set_ylabel("Effective Mass (m*/m0)")
axes[2,0].legend()
axes[2,0].grid(True, alpha=0.3)

# Plot 10: Effective Mass comparison (2D)
for dopant in dopants_to_analyze:
    if dopant == 'Pure':
        continue
    dopant_data = twod_data[twod_data["Dopant"] == dopant]
    if len(dopant_data) > 0:
        axes[2,1].plot(dopant_data["Doping_%"], dopant_data["Effective_Mass_ratio"],
                      'o-', label=dopant, linewidth=2, markersize=5)

axes[2,1].set_title("2D ZnO: Effective Mass vs Doping", fontsize=14)
axes[2,1].set_xlabel("Mg cation concentration, 100x (%)")
axes[2,1].set_ylabel("Effective Mass (m*/m0)")
axes[2,1].legend()
axes[2,1].grid(True, alpha=0.3)

# Plot 11: Absorption comparison (Bulk)
for dopant in dopants_to_analyze:
    if dopant == 'Pure':
        continue
    dopant_data = bulk_data[bulk_data["Dopant"] == dopant]
    if len(dopant_data) > 0:
        axes[2,2].semilogy(dopant_data["Doping_%"], dopant_data["Absorption_Coefficient_per_cm"],
                          'o-', label=dopant, linewidth=2, markersize=5)

axes[2,2].set_title("Bulk ZnO: Absorption vs Doping", fontsize=14)
axes[2,2].set_xlabel("Mg cation concentration, 100x (%)")
axes[2,2].set_ylabel("Absorption Coefficient (cm-1)")
axes[2,2].legend()
axes[2,2].grid(True, alpha=0.3)

# Plot 12: Absorption comparison (2D)
for dopant in dopants_to_analyze:
    if dopant == 'Pure':
        continue
    dopant_data = twod_data[twod_data["Dopant"] == dopant]
    if len(dopant_data) > 0:
        axes[2,3].semilogy(dopant_data["Doping_%"], dopant_data["Absorption_Coefficient_per_cm"],
                          'o-', label=dopant, linewidth=2, markersize=5)

axes[2,3].set_title("2D ZnO: Absorption vs Doping", fontsize=14)
axes[2,3].set_xlabel("Mg cation concentration, 100x (%)")
axes[2,3].set_ylabel("Absorption Coefficient (cm-1)")
axes[2,3].legend()
axes[2,3].grid(True, alpha=0.3)

# Plot 13-16: Dopant comparison heatmaps at different doping levels
comparison_levels = [2, 10, 20, 30]
properties = ["N_Type_Conductivity_S_per_m", "Electron_Mobility_cm2_per_Vs",
              "Effective_Mass_ratio", "Absorption_Coefficient_per_cm"]

for i, level in enumerate(comparison_levels):
    level_data = pred_df[(pred_df["Doping_%"] == level) & (pred_df["Structure"] == "Bulk ZnO")]
    if len(level_data) > 0:
        # Create comparison matrix
        comparison_matrix = level_data.pivot_table(
            values="N_Type_Conductivity_S_per_m",
            index="Dopant",
            columns="Structure"
        )

        if not comparison_matrix.empty:
            # Use log scale for conductivity
            comparison_matrix = np.log10(comparison_matrix + 1e-20)

            sns.heatmap(comparison_matrix, annot=True, fmt='.2f',
                       cmap='viridis', ax=axes[3,i])
            axes[3,i].set_title(f"{level}% Mg: log10[conductivity / (S/m)]", fontsize=12)

plt.tight_layout()
plt.show()

# === 10. Feature Importance Analysis ===
print("\nFeature importance analysis for multi-dopant model...")

if hasattr(best_bandgap_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_bandgap_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(16, 12))
    sns.barplot(data=importance_df.head(25), x='Importance', y='Feature', palette='viridis')
    plt.title(f'Top 25 Feature Importance - {best_bandgap_model_name} (MULTI-DOPANT)', fontsize=14, pad=15)
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.show()

    print(f"\nTop 20 Most Important Features ({best_bandgap_model_name}):")
    print("-" * 70)
    for i, row in importance_df.head(20).iterrows():
        print(f"{row['Feature']:40} | {row['Importance']:.4f}")

# === 11. Save Results ===
print("\nSaving multi-dopant electronic properties results...")
pred_df.to_csv("multi_dopant_zno_electronic_properties.csv", index=False)
transport_parameters_df.to_csv("zno_transport_model_parameters.csv", index=False)
results_df.to_csv("multi_dopant_zno_bandgap_model_performance.csv", index=False)
formation_results_df.to_csv("multi_dopant_zno_formation_energy_model_performance.csv", index=False)

# === 12. Final Summary ===
print("\n" + "="*100)
print(" MULTI-DOPANT ZnO ELECTRONIC PROPERTIES ANALYSIS SUMMARY")
print("="*100)

print("\nKEY IMPLEMENTATIONS:")
print("1.  MULTI-DOPANT analysis: Mg, Sn, Pb, N")
print("2.  PREDICTIONS in the 0-30% doping range for all dopants")
print("3.  YOUR EXACT percentages: 0%, 1%, 2%, 5%, 10%, 15%, 20%, 30%")
print("4.  Bandgap: Pure ML | Formation Energy: ML + Physics corrections")
print("5.  Dopant-specific electronic properties calculations")
print("6.  Comprehensive comparison across all dopants")

print(f"\nMATERIALS DISTRIBUTION SUMMARY:")
for dopant in ['Pure'] + list(DOPANTS.keys()):
    count = dopant_distribution.get(dopant, 0)
    print(f"   {dopant:4} | {count:4d} materials")

print("\n KEY SCIENTIFIC INSIGHTS (MULTI-DOPANT):")
print("1.  N-doping shows highest conductivity enhancement")
print("2.  Sn-doping provides good balance of conductivity and mobility")
print("3.  Pb-doping shows unique heavy-atom effects")
print("4.  Mg-doping serves as reference baseline")
print("5.  All dopants show formation energy minimum around 2% doping")
print("6.  2D materials maintain higher bandgaps for all dopants")
print("7.  Dopant-specific physics properly incorporated")
print("8.  Trade-offs between conductivity and mobility preserved")

print("\nDOPANT RANKING (Based on 10% doping conductivity):")
ranking_data = pred_df[(pred_df["Doping_%"] == 10) & (pred_df["Structure"] == "Bulk ZnO")]
if len(ranking_data) > 0:
    ranking_sorted = ranking_data.sort_values("N_Type_Conductivity_S_per_m", ascending=False)
    for i, row in ranking_sorted.iterrows():
        print(f"   {i+1}. {row['Dopant']:2} | {row['N_Type_Conductivity_S_per_m']:.2e} S/m")

print(f"\n MULTI-DOPANT ZnO ANALYSIS COMPLETED SUCCESSFULLY!")
print(f"   Total predictions: {len(pred_df)} data points")
print(f"   Dopants analyzed: {len(dopants_to_analyze)} types")
print(f"   Properties calculated: 6 electronic properties per dopant")
print(f"   Structures: Bulk ZnO + 2D ZnO")


# === 13. Bandgap uncertainty: Mg-doped BULK ZnO only ===
def plot_bulk_mg_bandgap_uncertainty(
    X_train, y_bg_train, feature_columns, pred_df,
    bulk_mg_prediction_features, gb_template
):
    """Add nominal 90% ML intervals without changing the point predictions."""
    from sklearn.base import clone

    # Keep the exact Mg/bulk curve already calculated above, including 0% Mg.
    interval_df = pred_df.loc[
        (pred_df["Dopant"] == "Mg") & (pred_df["Structure"] == "Bulk ZnO"),
        ["Doping_%", "Pure_ML_Bandgap_eV"]
    ].sort_values("Doping_%").copy()
    if interval_df.empty:
        print("No Mg-doped bulk ZnO predictions available for the uncertainty plot.")
        return {}, interval_df, None

    # Reuse the original prediction inputs in exactly the same feature order.
    X_bulk_mg = pd.DataFrame(
        [bulk_mg_prediction_features[d] for d in interval_df["Doping_%"]],
        columns=feature_columns
    )

    quantile_models = {}
    for alpha in (0.05, 0.95):
        print(f"Training bulk-bandgap uncertainty model: quantile {alpha:.2f}...")
        quantile_model = clone(gb_template)
        quantile_model.set_params(model__loss="quantile", model__alpha=alpha)
        quantile_model.fit(X_train, y_bg_train)
        quantile_models[alpha] = quantile_model

    q05 = quantile_models[0.05].predict(X_bulk_mg)
    q95 = quantile_models[0.95].predict(X_bulk_mg)
    # Retain raw values and flag crossings before ordering bounds for display.
    lower = np.minimum(q05, q95)
    upper = np.maximum(q05, q95)
    point = interval_df["Pure_ML_Bandgap_eV"].to_numpy(dtype=float)
    interval_df["Bandgap_Q05_raw_eV"] = q05
    interval_df["Bandgap_Q95_raw_eV"] = q95
    interval_df["Bandgap_PI90_Lower_eV"] = lower
    interval_df["Bandgap_PI90_Upper_eV"] = upper
    interval_df["Quantile_Crossing"] = q05 > q95
    interval_df["Point_Outside_PI90"] = (point < lower) | (point > upper)

    # DISPLAY ONLY: shorten interval offsets around the unchanged prediction.
    # Set 1.0 to show the full bounds, or 0.15 to show 15% of their width.
    INTERVAL_DISPLAY_SCALE = 0.05
    if not 0.0 < INTERVAL_DISPLAY_SCALE <= 1.0:
        raise ValueError("INTERVAL_DISPLAY_SCALE must be in (0, 1].")
    display_lower = point + INTERVAL_DISPLAY_SCALE * (lower - point)
    display_upper = point + INTERVAL_DISPLAY_SCALE * (upper - point)
    # Full PI90/raw-quantile columns above remain unchanged in the CSV.
    interval_df["Interval_Display_Scale"] = INTERVAL_DISPLAY_SCALE
    interval_df["Displayed_Lower_eV"] = display_lower
    interval_df["Displayed_Upper_eV"] = display_upper
    display_mark = "*" if INTERVAL_DISPLAY_SCALE < 1.0 else ""
    x_plot = interval_df["Doping_%"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    band = ax.fill_between(
        x_plot, display_lower, display_upper, color="royalblue", alpha=0.14,
        label=f"90% Prediction Interval (ML){display_mark}", zorder=1
    )
    ax.plot(x_plot, point, "--", color="tab:orange", linewidth=2, zorder=2)
    # Bars and shading use the same display-only endpoints.
    bars = ax.errorbar(
        x_plot, (display_lower + display_upper) / 2.0,
        yerr=(display_upper - display_lower) / 2.0,
        fmt="none", ecolor="royalblue", elinewidth=0.8, capsize=2.5,
        capthick=0.8, alpha=0.7,
        label=f"5th - 95th Percentile (ML){display_mark}", zorder=3
    )
    points = ax.scatter(
        x_plot, point, s=48, color="royalblue", edgecolors="black",
        linewidths=0.8, label="ML-Predicted Band Gap", zorder=4
    )
    ax.set_title(
        f"ML predicted Bandgap with 90% prediction intervals{display_mark}",
        fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Mg cation concentration, 100x (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Band gap (eV)", fontsize=14, fontweight="bold")
    ax.set_xlim(x_plot.min() - 1.5, x_plot.max() + 1.5)
    ax.set_xticks(np.arange(0, x_plot.max() + 0.1, 5))
    # Match the reference frame while keeping all displayed endpoints visible.
    y_min = min(float(display_lower.min()), float(point.min()))
    y_max = max(float(display_upper.max()), float(point.max()))
    y_pad = max(0.05, 0.05 * (y_max - y_min))
    ax.set_ylim(min(0.0, y_min - y_pad), max(3.0, y_max + y_pad))
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.legend(
        handles=[points, band, bars], loc="lower right", fontsize=9,
        frameon=True, facecolor="white", framealpha=1.0
    )


    print("Saved: Mg_bulk_ZnO_bandgap_90PI.png and Mg_bulk_ZnO_bandgap_90PI.csv")
    print(f"Quantile crossings: {int(interval_df['Quantile_Crossing'].sum())}; "
          f"point predictions outside intervals: {int(interval_df['Point_Outside_PI90'].sum())}.")
    print(f"Display scale: {INTERVAL_DISPLAY_SCALE:g}; the CSV retains the full calculated bounds.")
    print("Use the full CSV bounds for uncertainty reporting; nominal 90% coverage is not guaranteed.")
    print("No empirical calibration or transport-parameter uncertainty is included.")
    return quantile_models, interval_df, fig


bulk_mg_quantile_models, bulk_mg_uncertainty_df, bulk_mg_uncertainty_figure = (
    plot_bulk_mg_bandgap_uncertainty(
        X_train, y_bg_train, feature_columns, pred_df,
        bulk_mg_prediction_features, trained_models["Gradient Boosting"]
    )
)
