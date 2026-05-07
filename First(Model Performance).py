# ============================================================
# 0)  (Colab) one-time installs
# ============================================================
!pip install -q mp-api pymatgen scikit-learn pandas matplotlib seaborn shap joblib

# ============================================================
# 1) Imports & config
# ============================================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from mp_api.client import MPRester
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap, joblib, warnings, os, json
warnings.filterwarnings("ignore")

# Set default plot style
plt.style.use('default')
sns.set_theme(style="whitegrid")

API_KEY = "4QoUiunPSMRpqOLTwA6qRu8edPSBArZD"

# ============================================================
# 2) Fetch ZnO & doped-ZnO materials
# ============================================================
with MPRester(API_KEY) as mpr:
    mp_res = mpr.materials.summary.search(
        elements=["Zn", "O"],   # Changed Sn → Zn
        exclude_elements=["H"],
        fields=["material_id", "band_gap", "density", "volume",
                "nsites", "formation_energy_per_atom", "cbm", "vbm",
                "elements", "formula_pretty"]
    )

df = pd.DataFrame([r.dict() for r in mp_res])
print(f" fetched             : {len(df)} rows")


# Modified filtering to include normal ZnO bandgap range
df["nelements"] = df["elements"].apply(len)
df = df[df["nelements"].isin([2, 3])]
df = df.dropna(subset=["band_gap"])

# Improved 2D vs normal ZnO classification
df["volume_per_site"] = df["volume"] / df["nsites"]
df["formation_energy_density"] = df["formation_energy_per_atom"] / df["volume"]

# More accurate classification based on multiple criteria
df["structure_type"] = df.apply(lambda x: "2D ZnO" if (
    (x["volume_per_site"] > df["volume_per_site"].median()) and
    (x["density"] < df["density"].median()) and
    (x["band_gap"] > 2.0)  # Higher bandgap characteristic of 2D structures
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
num_cols = ["density", "volume", "nsites",
            "formation_energy_per_atom", "cbm", "vbm"]
df[num_cols] = KNNImputer(n_neighbors=5).fit_transform(df[num_cols])

# Comprehensive feature engineering
df["avg_atomic_volume"] = df["volume"] / df["nsites"]
df["dopant_count"] = df["nelements"] - 2
df["energy_density"] = df["formation_energy_per_atom"] * df["density"]
df["band_width"] = df["cbm"] - df["vbm"]
df["volume_per_site"] = df["volume"] / df["nsites"]
df["density_squared"] = df["density"] ** 2
df["volume_squared"] = df["volume"] ** 2
df["energy_squared"] = df["formation_energy_per_atom"] ** 2
df["density_volume"] = df["density"] * df["volume"]
df["energy_volume"] = df["formation_energy_per_atom"] * df["volume"]
df["band_gap_density_ratio"] = df["band_gap"] / df["density"]
df["formation_energy_ratio"] = df["formation_energy_per_atom"] / df["volume"]

# One-hot encode dopant element
df["dopant"] = df["elements"].apply(
    lambda els: sorted(set(els) - {"Zn", "O"}) or ["none"]
).str[0]


ohe = OneHotEncoder(sparse_output=False)
dopant_ohe = pd.DataFrame(
    ohe.fit_transform(df[["dopant"]]),
    columns=ohe.get_feature_names_out(["dopant"])
)
df = pd.concat([df.reset_index(drop=True), dopant_ohe], axis=1)

# Select features
base_feats = [
    "density", "volume", "nsites", "formation_energy_per_atom",
    "cbm", "vbm", "avg_atomic_volume", "dopant_count",
    "energy_density", "band_width", "volume_per_site",
    "density_squared", "volume_squared", "energy_squared",
    "density_volume", "energy_volume", "band_gap_density_ratio",
    "formation_energy_ratio"
]
final_feats = base_feats + list(dopant_ohe.columns)
X, y = df[final_feats], df["band_gap"]

# ============================================================
# 4) Enhanced preprocessing with validation set
# ============================================================
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 5) Optimized Model zoo with regularization
# ============================================================
kernel = (C(1.0, (1e-3, 1e3)) *
         RBF([1.0] * X_train_scaled.shape[1], (1e-2, 1e2)) +
         WhiteKernel(noise_level=1e-2))

models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=1000,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        min_samples_split=5,
        random_state=42
    ),
    "SVR": SVR(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        epsilon=0.1
    ),
    "MLPRegressor": MLPRegressor(
        hidden_layer_sizes=(200, 100, 50),
        activation='relu',
        solver='adam',
        alpha=0.01,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=5000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=42
    ),
    "GPR": GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=42
    )
}

# ============================================================
# 6) Training with overfitting checks
# ============================================================
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
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
                       X_train_scaled, y_train, ylim=(0, 1), cv=5)
    plt.show()

# Create comparison DataFrame
results_df = pd.DataFrame(rows).sort_values("Test R²", ascending=False)

print("\n🔍 Model Performance Comparison:")
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
