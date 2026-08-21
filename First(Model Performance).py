
# 0) Colab One-time Installs
# ==========================================================

!pip install -q mp-api pymatgen shap

# ==========================================================
# 1) Imports & Configuration
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import json
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from mp_api.client import MPRester
from pymatgen.core import Composition, Element
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import KNNImputer

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    WhiteKernel,
    ConstantKernel as C
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# Plot settings
plt.style.use("default")
sns.set_theme(style="whitegrid")

API_KEY = "4QoUiunPSMRpqOLTwA6qRu8edPSBArZD"
# ============================================================
# Reproducibility
# ============================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
# ============================================================
# Cross-validation strategy
# ============================================================

CV = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_SEED
)

print("\nCross-validation strategy:")
print("5-fold shuffled K-fold cross-validation")
print(f"Random seed: {RANDOM_SEED}")

import sklearn
print(sklearn.__version__)
print(GridSearchCV)
# ============================================================
# 2) Fetch ZnO & doped-ZnO materials from Materials Project
# ============================================================

with MPRester(API_KEY) as mpr:
    mp_res = mpr.materials.summary.search(
        elements=["Zn", "O"],
        exclude_elements=["H"],
        fields=[
            "material_id",
            "band_gap",
            "density",
            "volume",
            "nsites",
            "cbm",
            "vbm",
            "elements",
            "formula_pretty"
        ]
    )

df = pd.DataFrame([r.dict() for r in mp_res])
print(f" fetched             : {len(df)} rows")

# Modified filtering to include physically relevant ZnO bandgap range
df["nelements"] = df["elements"].apply(len)

# Retain binary and ternary Zn-O based systems
df = df[df["nelements"].isin([2, 3])]

# Remove entries with missing band-gap values
df = df.dropna(subset=["band_gap"])

# ============================================================
# Physically relevant band-gap filtering
# ============================================================
# Exclude zero/near-zero band-gap systems and extreme outliers.
# This focuses the ML model on semiconducting ZnO-based systems.

df = df[
    (df["band_gap"] >= 0.1) &
    (df["band_gap"] <= 8.0)
].copy()

print("\nBand-gap filtering:")
print("Retained range: 0.1 eV <= Eg <= 8.0 eV")
print(f"Samples after band-gap filtering: {len(df)}")
print(f"Minimum band gap: {df['band_gap'].min():.4f} eV")
print(f"Maximum band gap: {df['band_gap'].max():.4f} eV")
# ============================================================
# Remove exact duplicate materials
# ============================================================

before_duplicates = len(df)

duplicate_columns = [
    "formula_pretty",
    "band_gap",
    "density",
    "volume",
    "nsites"
]

df = df.drop_duplicates(
    subset=duplicate_columns,
    keep="first"
).reset_index(drop=True)

after_duplicates = len(df)

print("\nDuplicate removal:")
print(f"Rows before duplicate removal : {before_duplicates}")
print(f"Duplicates removed            : {before_duplicates - after_duplicates}")
print(f"Rows after duplicate removal  : {after_duplicates}")
df["volume_per_site"] = df["volume"] / df["nsites"]
df["edge_difference"] = df["cbm"] - df["vbm"]
df["structure_type"] = df.apply(lambda x: "2D ZnO" if (
    (x["volume_per_site"] > df["volume_per_site"].median()) and
    (x["density"] < df["density"].median()) and
    (x["edge_difference"] > 2.0)
) else "Normal ZnO", axis=1)

# Verify and adjust bandgap ranges
print("\nInitial Bandgap Distribution:")
print(df.groupby("structure_type")["band_gap"].describe())

# ============================================================
# Enhanced Bandgap Analysis and Visualization
# ============================================================
print("\n" + "="*50)
print(" BANDGAP ANALYSIS: 2D ZnO vs Normal ZnO")
print("="*50)

comparison_stats = df.groupby("structure_type")["band_gap"].agg([
    ("Average Bandgap (eV)", "mean"),
    ("Min Bandgap (eV)", "min"),
    ("Max Bandgap (eV)", "max"),
    ("Standard Deviation", "std"),
    ("Count", "size")
]).round(3)

# Print formatted statistics
print("\n Statistical Summary:")
print("-"*50)
for idx, row in comparison_stats.iterrows():
    print(f"\n {idx}:")
    print(f"   ├── Average Bandgap: {row['Average Bandgap (eV)']:.3f} eV")
    print(f"   ├── Range: {row['Min Bandgap (eV)']:.3f} - {row['Max Bandgap (eV)']:.3f} eV")
    print(f"   ├── Standard Deviation: {row['Standard Deviation']:.3f} eV")
    print(f"   └── Sample Size: {row['Count']} structures")

# Enhanced visualizations
fig = plt.figure(figsize=(15, 10))

# 1. Enhanced Boxplot
plt.subplot(2, 2, 1)
sns.boxplot(data=df, x="structure_type", y="band_gap", palette="Set2")
plt.title("Bandgap Distribution Comparison", fontsize=12, pad=10)
plt.xlabel("Structure Type", fontsize=10)
plt.ylabel("Bandgap (eV)", fontsize=10)
plt.grid(True, alpha=0.3)

# 2. Enhanced Violin Plot
plt.subplot(2, 2, 2)
sns.violinplot(data=df, x="structure_type", y="band_gap", palette="Set2")
plt.title("Bandgap Density Distribution", fontsize=12, pad=10)
plt.xlabel("Structure Type", fontsize=10)
plt.ylabel("Bandgap (eV)", fontsize=10)
plt.grid(True, alpha=0.3)

# 3. Enhanced Scatter Plot
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x="density", y="band_gap", hue="structure_type",
                palette="Set2", alpha=0.6, s=100)
plt.title("Bandgap vs Density Relationship", fontsize=12, pad=10)
plt.xlabel("Density (g/cm³)", fontsize=10)
plt.ylabel("Bandgap (eV)", fontsize=10)
plt.grid(True, alpha=0.3)

# 4. Enhanced Histogram
plt.subplot(2, 2, 4)
for struct_type in df["structure_type"].unique():
    sns.histplot(data=df[df["structure_type"] == struct_type],
                x="band_gap", alpha=0.5, label=struct_type, bins=15)
plt.title("Bandgap Distribution Histogram", fontsize=12, pad=10)
plt.xlabel("Bandgap (eV)", fontsize=10)
plt.ylabel("Count", fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 3) Enhanced Feature engineering
# ============================================================
num_cols = [
    "density",
    "volume",
    "nsites"
]

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Comprehensive feature engineering
df["avg_atomic_volume"] = df["volume"] / df["nsites"]
df["dopant_count"] = df["nelements"] - 2
# ============================================================
# Composition-based Materials descriptors
# ============================================================

def calculate_composition_descriptors(formula):
    """
    Calculate composition-derived descriptors directly from
    the chemical formula.

    These descriptors use only elemental composition and do
    not use band gap, CBM, VBM, or formation energy.
    """

    comp = Composition(formula)

    # Element amounts
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

    # Element objects
    elements = [
        Element(el)
        for el in amounts.keys()
    ]

    # Average Pauling electronegativity
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

    # Average atomic radius
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

    # Average valence electrons
    valence_values = []
    valence_weights = []

    for el, amount in amounts.items():
        element = Element(el)

        if element.group is not None:
            valence_electrons = (
                element.group - 10
                if element.group >= 13
                else element.group
            )

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

        # Use a consistent elemental-radius descriptor
        if element.atomic_radius is not None:
            radii.append(float(element.atomic_radius))
            weights.append(amount / total_atoms)

    if len(radii) < 2:
        return 0.0

    avg_radius = np.average(radii, weights=weights)

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
# =====================================================
# Physics-informed descriptors
# =====================================================

df["density_volume_ratio"] = (
    df["density"] /
    (df["volume"] + 1e-6)
)

df["compactness"] = df["nsites"] / df["volume"]
df["coordination_factor"] = (
    df["density"] *
    df["nsites"] /
    (df["volume"] + 1e-6)
)
df["phase_indicator"] = np.where(
    df["structure_type"]=="2D ZnO",
    1,
    0
)

df["confinement_parameter"] = (
    1 /
    (df["volume_per_site"] + 1e-6)
)


# ============================================================
# Select final features
# ============================================================

base_feats = [
    # Existing non-target structural descriptors
    "density",
    "volume",
    "nsites",

    "avg_atomic_volume",
    "dopant_count",
    "volume_per_site",

    "density_volume_ratio",
    "compactness",

    "coordination_factor",
    "phase_indicator",
    "confinement_parameter",

    # New composition-based Materials descriptors
    "Zn_fraction",
    "O_fraction",
    "Zn_O_ratio",
    "avg_electronegativity",
    "electronegativity_difference",
    "avg_atomic_radius",
    "avg_valence_electrons",
    "ionic_radius_mismatch"
]

final_feats = base_feats

X = df[final_feats].copy()
y = df["band_gap"].copy()

print("\nFinal features:")
for i, feature in enumerate(final_feats, 1):
    print(f"{i:2d}. {feature}")

print(f"\nNumber of final features: {len(final_feats)}")

# ============================================================
# 4) Train / Validation / Test Split
# ============================================================

RANDOM_SEED = 42

# 70% model-development training set
# 15% internal validation set
# 15% final internal test set

# Create target bins for stratification.
# These bins are NOT included as ML features.
y_bins = pd.qcut(
    y,
    q=10,
    labels=False,
    duplicates="drop"
)

X_train, X_temp, y_train, y_temp, bins_train, bins_temp = train_test_split(
    X,
    y,
    y_bins,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=y_bins
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=bins_temp
)

print("\nDataset split:")
print(f"Training set          : {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Internal validation   : {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Internal test         : {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
# ============================================================
# Save final ML dataset with train/validation/test assignment
# ============================================================

df_ml = df.loc[X.index].copy()

df_ml["dataset_split"] = "Training"
df_ml.loc[X_val.index, "dataset_split"] = "Validation"
df_ml.loc[X_test.index, "dataset_split"] = "Test"

final_dataset_columns = (
    ["material_id", "structure_type", "band_gap"]
    + final_feats
    + ["dataset_split"]
)

final_dataset_columns = [
    col for col in final_dataset_columns
    if col in df_ml.columns
]

df_ml[final_dataset_columns].to_csv(
    "final_ZnO_2D_ZnO_ML_dataset.csv",
    index=False
)

print(
    "\nFinal ML dataset saved to: "
    "final_ZnO_2D_ZnO_ML_dataset.csv"
)
scaler = StandardScaler()

# ============================================================
# Keep raw feature matrices.
# Scaling will be performed inside each CV fold.
# ============================================================

X_train_raw = X_train.copy()
X_val_raw = X_val.copy()
X_test_raw = X_test.copy()

print("\nFeature scaling:")
print("StandardScaler fitted on training data only.")
print("Validation and test sets transformed using training scaler.")
param_grid = {
    "n_estimators":[1000,1500],
    "learning_rate":[0.02,0.03,0.05],
    "max_depth":[3,4],
    "subsample":[0.8,0.9],
    "min_samples_split":[2,4],
    "min_samples_leaf":[1,2]
}
# ============================================================
# 5) Optimized Model zoo with regularization
# ============================================================
kernel = (
    C(1.0, (1e-3, 1e3))
    * RBF(
        length_scale=np.ones(X_train_raw.shape[1]),
        length_scale_bounds=(1e-2, 1e2)
    )
    + WhiteKernel(noise_level=1e-2)
)
gb = GradientBoostingRegressor(random_state=RANDOM_SEED)

gb_search = GridSearchCV(
    estimator=gb,
    param_grid=param_grid,
    scoring="r2",
    cv=CV,
    n_jobs=-1,
    verbose=1
)
# ============================================================
# Scale training, validation, and test data
# Fit scaler ONLY on the training data
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)

print("\nFeature scaling:")
print("StandardScaler fitted on training data only.")
print("Validation and test sets transformed using training scaler.")

gb_search.fit(X_train_scaled, y_train)

print(gb_search.best_params_)
print(gb_search.best_score_)

best_gb = gb_search.best_estimator_
models = {
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=800,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=0.8,
            bootstrap=True,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ))
    ]),

    "GradientBoosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model", best_gb)
    ]),

    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            epsilon=0.1
        ))
    ]),

    "MLPRegressor": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(200, 100, 50),
            activation="relu",
            solver="adam",
            alpha=0.01,
            batch_size="auto",
            learning_rate="adaptive",
            max_iter=5000,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=RANDOM_SEED
        ))
    ]),

    "GPR": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
            random_state=RANDOM_SEED
        ))
    ])
}
# ============================================================
# 5-fold Cross-Validation of all five models
# ============================================================

cv_rows = []

for name, model in models.items():

    print(f"\nCross-validating {name} ...")

    cv_scores = cross_val_score(
        model,
        X_train_raw,
        y_train,
        cv=CV,
        scoring="r2",
        n_jobs=-1
    )

    cv_rows.append({
        "Model": name,
        "CV R² Mean": np.mean(cv_scores),
        "CV R² Std": np.std(cv_scores),
        "CV R² Min": np.min(cv_scores),
        "CV R² Max": np.max(cv_scores)
    })

cv_results_df = pd.DataFrame(cv_rows)

print("\n5-Fold Cross-Validation Results:")
print("=" * 80)
print(
    cv_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ============================================================
# 6) Training with overfitting checks
# ============================================================
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=CV,
                       n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring="r2")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

rows = []
predictions = {}

for name, model in models.items():
    print(f"\nTraining {name} ...")

    # Train model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_val_pred = model.predict(X_val_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Calculate metrics
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Store results
    rows.append({
        "Model": name,
        "Validation R²": val_r2,
        "Test R²": test_r2,
        "Validation MAE": val_mae,
        "Test MAE": test_mae,
        "Validation RMSE": val_rmse,
        "Test RMSE": test_rmse,
        "R² Difference": abs(val_r2 - test_r2)
    })

    # Plot learning curve
    plot_learning_curve(model, f'Learning Curve - {name}',
                       X_train_scaled, y_train, ylim=(0, 1), cv=CV)
    plt.show()

# Create comparison DataFrame
# ============================================================
# Combine CV and validation/test performance
# ============================================================

results_df = pd.DataFrame(rows)

# Merge 5-fold CV results with validation/test results
results_df = results_df.merge(
    cv_results_df[
        ["Model", "CV R² Mean", "CV R² Std", "CV R² Min", "CV R² Max"]
    ],
    on="Model",
    how="left"
)

# ============================================================
# Reliability-based model selection
# ============================================================

# Normalize the four selection criteria to comparable 0–1 scales

# Higher CV R² is better
cv_score = (
    results_df["CV R² Mean"] -
    results_df["CV R² Mean"].min()
) / (
    results_df["CV R² Mean"].max() -
    results_df["CV R² Mean"].min()
)

# Higher Test R² is better
test_r2_score = (
    results_df["Test R²"] -
    results_df["Test R²"].min()
) / (
    results_df["Test R²"].max() -
    results_df["Test R²"].min()
)

# Lower Test MAE is better
mae_score = (
    results_df["Test MAE"].max() -
    results_df["Test MAE"]
) / (
    results_df["Test MAE"].max() -
    results_df["Test MAE"].min()
)

# Lower R² difference is better
stability_score = (
    results_df["R² Difference"].max() -
    results_df["R² Difference"]
) / (
    results_df["R² Difference"].max() -
    results_df["R² Difference"].min()
)

# ============================================================
# Weighted reliability score
# ============================================================

# CV performance receives the highest weight because
# cross-validation evaluates generalization across multiple folds.

results_df["Reliability Score"] = (
    0.40 * cv_score +
    0.30 * stability_score +
    0.20 * test_r2_score +
    0.10 * mae_score
)

# Sort according to overall reliability
results_df = results_df.sort_values(
    "Reliability Score",
    ascending=False
).reset_index(drop=True)
# ============================================================
# Display reliability-based model ranking
# ============================================================

print("\n" + "=" * 100)
print("RELIABILITY-BASED MODEL RANKING")
print("=" * 100)

display_columns = [
    "Model",
    "CV R² Mean",
    "CV R² Std",
    "Validation R²",
    "Test R²",
    "Test MAE",
    "R² Difference",
    "Reliability Score"
]

print(
    results_df[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# Select the most reliable model
best_model_name = results_df.iloc[0]["Model"]

print("\n" + "=" * 100)
print(f"SELECTED BEST MODEL: {best_model_name}")
print("=" * 100)

best_row = results_df.iloc[0]

print(f"5-Fold CV R² Mean : {best_row['CV R² Mean']:.4f}")
print(f"Validation R²    : {best_row['Validation R²']:.4f}")
print(f"Test R²          : {best_row['Test R²']:.4f}")
print(f"Test MAE         : {best_row['Test MAE']:.4f} eV")
print(f"R² Difference    : {best_row['R² Difference']:.4f}")
print(f"Reliability Score: {best_row['Reliability Score']:.4f}")

print("\n Reliability-Based Model Performance Comparison:")
print("="*80)
print(results_df.to_string(float_format=lambda x: '{:.4f}'.format(x)))

# ============================================================
#  Combined Performance Visualization
# ============================================================

fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot R²
ax1.bar(results_df["Model"], results_df["Test R²"],
        alpha=0.6, label="Test R²")
ax1.set_ylabel("R² Score")
ax1.set_ylim(0, 1)

# Second axis for error metrics
ax2 = ax1.twinx()

ax2.plot(results_df["Model"], results_df["Test MAE"],
         marker='o', linewidth=2, label="Test MAE")

ax2.plot(results_df["Model"], results_df["Test RMSE"],
         marker='s', linewidth=2, label="Test RMSE")

ax2.set_ylabel("Error (eV)")

plt.title("Comparison of 5 ML Models for ZnO Bandgap Prediction")
ax1.set_xlabel("Model")

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Print results with overfitting analysis
print("\n Model Performance and Overfitting Analysis:")
print("="*80)
print(results_df.to_string(float_format=lambda x: '{:.4f}'.format(x)))
print("\nOverfitting Analysis:")
print("-"*80)
for _, row in results_df.iterrows():
    print(f"\n{row['Model']}:")
    diff = row['R² Difference']
    if diff > 0.1:
        status = "HIGH risk of overfitting"
    elif diff > 0.05:
        status = "MODERATE risk of overfitting"
    else:
        status = "LOW risk of overfitting"
    print(f"R² difference (Val-Test): {diff:.4f} - {status}")

# Plot final comparison
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Model", y="Test R²", palette="Set2")
plt.axhline(y=0.8, color='r', linestyle='--', label='Target R² = 0.8')
plt.title("Model Performance Comparison", pad=20)
plt.xlabel("Model")
plt.ylabel("Test R² Score")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 7) Save best model
# ============================================================
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
joblib.dump(best_model, f"best_model_ZnO.pkl")
print(f"\n Saved best model: {best_model_name}")

# ============================================================
# 8) Final Predictions Analysis
# ============================================================
# Get predictions for test set
y_pred = best_model.predict(X_test_scaled)

# Create DataFrame with actual and predicted values
predictions_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Structure_Type": df.loc[X_test.index, "structure_type"]
})

# Calculate metrics by structure type
print("\nPerformance by Structure Type:")
print("-"*50)
for struct_type in ["2D ZnO", "Normal ZnO"]:
    mask = predictions_df["Structure_Type"] == struct_type
    struct_r2 = r2_score(predictions_df[mask]["Actual"],
                        predictions_df[mask]["Predicted"])
    struct_mae = mean_absolute_error(predictions_df[mask]["Actual"],
                                   predictions_df[mask]["Predicted"])
    print(f"\n{struct_type}:")
    print(f"R² Score: {struct_r2:.4f}")
    print(f"MAE: {struct_mae:.4f} eV")

# Plot actual vs predicted by structure type
plt.figure(figsize=(10, 8))
for struct_type in ["2D ZnO", "Normal ZnO"]:
    mask = predictions_df["Structure_Type"] == struct_type
    plt.scatter(predictions_df[mask]["Actual"],
                predictions_df[mask]["Predicted"],
                alpha=0.6,
                label=struct_type)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label='Ideal')
plt.xlabel("Actual Band Gap (eV)")
plt.ylabel("Predicted Band Gap (eV)")
plt.title(f"Actual vs Predicted Band Gap\nBest Model: {best_model_name}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save predictions
predictions_df.to_csv("predicted_band_gaps_ZnO.csv", index=False)
print("\n Predictions saved to predicted_band_gaps_ZnO.csv")
predictions_df.to_csv("predicted_band_gaps_ZnO.csv", index=False)
print("\n Predictions saved to predicted_band_gaps_ZnO.csv")
