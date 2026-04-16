import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# LOAD DATA + MODELS
# ==============================
properties = pd.read_csv("properties_New.csv", index_col=0).T
heatmix = pd.read_excel("Heat_of_Mixing.xlsx", index_col=0)

density_model = joblib.load("ExtraTrees_density_BO.pkl")
phase_model = joblib.load("best_knn.pkl")

prop_dict = properties.to_dict(orient="index")

VALENCE_ELECTRONS = {
    "Al": 3, "Si": 4,
    "Ti": 4, "V": 5, "Cr": 6, "Mn": 7,
    "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11,
    "Zr": 4, "Nb": 5, "Mo": 6, "Mg":2, "Zn":2
}

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Alloy Design Tool", layout="wide")
st.title("🧪 Alloy Density + Phase Prediction App")

# ==============================
# ELEMENT SELECTION
# ==============================
elements = list(prop_dict.keys())
selected_elements = st.multiselect("Select Elements", elements)

# ==============================
# COMPOSITION INPUT
# ==============================
st.subheader("Composition (%)")

composition = {}
total = 0

for el in selected_elements:
    val = st.number_input(f"{el}", min_value=0.0, max_value=100.0, value=0.0)
    composition[el] = val
    total += val

# Normalize
if st.checkbox("Normalize to 100%"):
    if total > 0:
        composition = {k: v/total*100 for k, v in composition.items()}

st.write(f"Total Composition = {total:.2f} %")

# ==============================
# FEATURE FUNCTION
# ==============================
def compute_features(comp):
    elems = list(comp.keys())
    x = np.array(list(comp.values())) / 100

    aw  = np.array([prop_dict[e]["Atomic_weight"] for e in elems])
    r   = np.array([prop_dict[e]["Rm"] for e in elems])
    rho = np.array([prop_dict[e]["Density"] for e in elems])
    chi = np.array([prop_dict[e]["Xa"] for e in elems])
    Tm  = np.array([prop_dict[e]["Tm"] for e in elems])
    vec = np.array([VALENCE_ELECTRONS[e] for e in elems])

    avg_aw  = np.dot(x, aw)
    avg_vec = np.dot(x, vec)
    avg_Tm  = np.dot(x, Tm)
    avg_en  = np.dot(x, chi)
    r_avg   = np.dot(x, r)

    delta = np.sqrt(np.sum(x * (1 - r / r_avg) ** 2))
    EN_mismatch = np.sum(x * (chi - avg_en) ** 2)

    Hmix = 0.0
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            if elems[i] in heatmix.index and elems[j] in heatmix.columns:
                Hmix += 4 * x[i] * x[j] * heatmix.loc[elems[i], elems[j]]

    mix_vol = np.dot(x, aw / rho)
    theoretical_density = avg_aw / (mix_vol + 1e-12)

    return {
        "avg_aw": avg_aw,
        "avg_vec": avg_vec,
        "avg_Tm": avg_Tm,
        "delta": delta,
        "Hmix": Hmix,
        "theoretical_density": theoretical_density,
        "EN_mismatch": EN_mismatch
    }

# ==============================
# RUN PREDICTION
# ==============================
if st.button("🔍 Run Prediction"):

    if len(composition) == 0 or total == 0:
        st.warning("Please select elements and define composition")
    else:
        comp_norm = {k: v/total*100 for k, v in composition.items()}

        features = compute_features(comp_norm)
        df = pd.DataFrame([features])

        # Align for phase model
        try:
            df_phase = df.reindex(columns=phase_model.feature_names_in_, fill_value=0)
        except:
            df_phase = df

        # ==============================
        # DENSITY WITH ERROR BAR
        # ==============================
        all_preds = np.array([tree.predict(df)[0] for tree in density_model.estimators_])
        density_pred = np.mean(all_preds)
        density_std = np.std(all_preds)

        # Phase prediction
        phase_pred = phase_model.predict(df_phase)[0]

        # ==============================
        # DISPLAY RESULTS
        # ==============================
        st.success("Prediction Complete")

        st.write("### 🔬 Results")
        st.write(f"**Predicted Density:** {density_pred:.2f} ± {density_std:.2f} g/cm³")
        st.write(f"**Predicted Phase:** {phase_pred}")

        st.write("### 📊 Feature Values")
        st.dataframe(df.round(3))
