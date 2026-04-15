import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Klasifikasi PCOS",
    page_icon="🩺",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

SAVE_PATH = "model/model_pcos.pkl"

with open(SAVE_PATH, "rb") as f:
    saved = pickle.load(f)

rfe_model = saved["rfe"]["model"]
rfe_selector = saved["rfe"]["selector"]
rfe_features = saved["rfe"]["features"]

rfecv_model = saved["rfecv"]["model"]
rfecv_selector = saved["rfecv"]["selector"]
rfecv_features = saved["rfecv"]["features"]

scaler = saved["scaler"]
to_drop = saved["to_drop"]
all_features = saved["all_features"]

# =========================
# SIDEBAR
# =========================

st.sidebar.title("🩺 Pengaturan Model")

model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["RFE", "RFECV"]
)

if model_choice == "RFE":
    active_features = rfe_features
    model = rfe_model
    selector = rfe_selector
else:
    active_features = rfecv_features
    model = rfecv_model
    selector = rfecv_selector

st.sidebar.metric("Jumlah Fitur", len(active_features))

# =========================
# DEFAULT VALUES
# =========================

default_values = {
    'Age (yrs)': 33,
    'Weight (Kg)': 68.5,
    'Height(Cm)': 165,
    'Marraige Status (Yrs)': 10,
    'Cycle length(days)': 5,
    'Hb(g/dl)': 11.8,
    'FSH(mIU/mL)': 5.54,
    'LH(mIU/mL)': 0.88,
    'TSH (mIU/L)': 2.54,
    'AMH(ng/mL)': 6.63,
    'PRL(ng/mL)': 10.52,
    'PRG(ng/mL)': 0.36,
    'Vit D3 (ng/mL)': 49.7,
    'I beta-HCG(mIU/mL)': 494.08,
    'Waist(inch)': 36,
    'Hip(inch)': 40,
    'Follicle No. (R)': 15,
    'Follicle No. (L)': 13,
    'Avg. F size (L) (mm)': 18,
    'Avg. F size (R) (mm)': 20,
    'Endometrium (mm)': 10
}

# =========================
# LABEL INDONESIA
# =========================

label_map = {
    'Age (yrs)': 'Umur (tahun)',
    'Weight (Kg)': 'Berat Badan (kg)',
    'Height(Cm)': 'Tinggi Badan (cm)',
    'Marraige Status (Yrs)': 'Lama Menikah (tahun)',

    'Cycle(R/I)': 'Siklus Menstruasi',
    'Cycle length(days)': 'Panjang Siklus (hari)',

    'Hip(inch)': 'Lingkar Pinggul (inch)',
    'Waist(inch)': 'Lingkar Pinggang (inch)',
    'Waist:Hip Ratio': 'Rasio Pinggang-Pinggul',

    'Weight gain(Y/N)': 'Kenaikan Berat Badan (Ya/Tidak)',
    'hair growth(Y/N)': 'Pertumbuhan Rambut Berlebih (Ya/Tidak)',
    'Skin darkening (Y/N)': 'Penggelapan Kulit (Ya/Tidak)',
    'Pimples(Y/N)': 'Jerawat (Ya/Tidak)',
    'Fast food (Y/N)': 'Konsumsi Fast Food (Ya/Tidak)',

    'Follicle No. (L)': 'Jumlah Folikel Kiri',
    'Follicle No. (R)': 'Jumlah Folikel Kanan',

    'Avg. F size (L) (mm)': 'Ukuran Rata-rata Folikel Kiri (mm)',
    'Avg. F size (R) (mm)': 'Ukuran Rata-rata Folikel Kanan (mm)',

    'Endometrium (mm)': 'Ketebalan Endometrium (mm)'
}

# =========================
# GROUP MAP
# =========================

group_map = {
    "Data Demografi": [
        'Age (yrs)',
        'Weight (Kg)',
        'Height(Cm)',
        'Marraige Status (Yrs)'
    ],
    "Siklus Menstruasi": [
        'Cycle(R/I)',
        'Cycle length(days)'
    ],
    "Data Hormon": [
        'FSH(mIU/mL)',
        'LH(mIU/mL)',
        'TSH (mIU/L)',
        'AMH(ng/mL)',
        'PRL(ng/mL)',
        'PRG(ng/mL)',
        'Vit D3 (ng/mL)',
        'I beta-HCG(mIU/mL)',
        'Hb(g/dl)'
    ],
    "Antropometri": [
        'Hip(inch)',
        'Waist(inch)',
        'Waist:Hip Ratio'
    ],
    "Gejala Klinis": [
        'Weight gain(Y/N)',
        'hair growth(Y/N)',
        'Skin darkening (Y/N)',
        'Pimples(Y/N)',
        'Fast food (Y/N)'
    ],
    "Ultrasonografi": [
        'Follicle No. (L)',
        'Follicle No. (R)',
        'Avg. F size (L) (mm)',
        'Avg. F size (R) (mm)',
        'Endometrium (mm)'
    ]
}

# =========================
# HEADER
# =========================

st.title("🩺 Prediksi PCOS")
st.markdown("Masukkan data pasien sesuai fitur model.")

st.divider()

# =========================
# INPUT
# =========================

input_data = {}
used_features = set()

st.subheader("📋 Input Data")

for group_name, group_features in group_map.items():

    valid_features = [
        f for f in group_features
        if f in active_features
    ]

    if not valid_features:
        continue

    st.subheader(group_name)

    cols = st.columns(4)

    for i, feature in enumerate(valid_features):

        used_features.add(feature)

        col = cols[i % 4]

        with col:

            label = label_map.get(feature, feature)

            # Cycle
            if feature == "Cycle(R/I)":

                value = st.selectbox(
                    label,
                    [2, 4],
                    format_func=lambda x:
                    "Teratur" if x == 2 else "Tidak Teratur",
                    key=feature
                )

            # Y/N
            elif "(Y/N)" in feature:

                value = st.selectbox(
                    label,
                    [0, 1],
                    format_func=lambda x:
                    "Ya" if x == 1 else "Tidak",
                    key=feature
                )

            # Umur max 2 digit
            elif feature == "Age (yrs)":

                value = st.number_input(
                    label,
                    min_value=0,
                    max_value=99,
                    step=1,
                    value=int(default_values.get(feature, 26)),
                    key=feature
                )
                
            # Follicle No. max 2 digit
            elif "Follicle No. (L)" in feature:

                value = st.number_input(
                    label,
                    min_value=0,
                    max_value=99,
                    step=1,
                    value=int(default_values.get(feature, 0)),
                    key=feature
                )

            # Follicle No. max 2 digit
            elif "Follicle No. (R)" in feature:

                value = st.number_input(
                    label,
                    min_value=0,
                    max_value=99,
                    step=1,
                    value=int(default_values.get(feature, 0)),
                    key=feature
                )

            # Panjang siklus max 2 digit
            elif feature == "Cycle length(days)":

                value = st.number_input(
                    label,
                    min_value=0,
                    max_value=99,
                    step=1,
                    value=int(default_values.get(feature, 28)),
                    key=feature
                )

            # Ratio otomatis
            elif feature == "Waist:Hip Ratio":

                waist = input_data.get("Waist(inch)", 0)
                hip = input_data.get("Hip(inch)", 1)

                ratio = waist / hip if hip != 0 else 0

                st.number_input(
                    label,
                    value=float(round(ratio, 2)),
                    disabled=True,
                    key=feature
                )

                value = ratio

            else:

                value = st.number_input(
                    label,
                    value=float(default_values.get(feature, 0.0)),
                    key=feature
                )

            input_data[feature] = value

st.divider()

# =========================
# BUTTON
# =========================

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_btn = st.button(
        "🔍 Mulai Analisa",
        use_container_width=True
    )

# =========================
# PREDICTION
# =========================

if predict_btn:

    full_input = {f: default_values.get(f, 0) for f in all_features}
    full_input.update(input_data)

    df_input = pd.DataFrame([full_input])
    df_input = df_input[all_features]

    df_filtered = df_input.drop(columns=to_drop)

    scaled = scaler.transform(df_filtered)
    selected = selector.transform(scaled)

    prediction = model.predict(selected)
    prob = model.predict_proba(selected)[0][1]

    st.divider()
    st.subheader("📊 Hasil Klasifikasi")
    
    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Diagnosis")

        if prediction[0] == 1:
            st.error("⚠️ Terdeteksi PCOS")
        else:
            st.success("✅ Tidak Terdeteksi PCOS")

    with colB:
        st.markdown("### Risiko")

        if prob < 0.3:
            st.success("Rendah")
        elif prob < 0.7:
            st.warning("Sedang")
        else:
            st.error("Tinggi")

    persen = int(prob * 100)

    st.progress(persen)

    st.markdown(
        f"## **Kemungkinan PCOS: {persen}%**"
    )

    st.divider()

    # =========================
    # INTERPRETASI
    # =========================

    importances = model.feature_importances_

    df_imp = pd.DataFrame({
        "Fitur": active_features,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    top_features = df_imp.head(5)["Fitur"].tolist()

    important_values = {
        f: input_data.get(f, "-")
        for f in top_features
    }

    if prediction[0] == 1:

        st.markdown("""
        <div style="
            background-color: #929428;
            padding: 15px;
            font-size: 20px;
            font-weight: bold;
            border-radius: 8px;
            margin-bottom: 20px;
        ">
        ⚠️ Disarankan untuk konsultasi lebih lanjut dengan tenaga medis.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "Model mendeteksi kemungkinan PCOS berdasarkan beberapa indikator berikut:"
        )

        for f, val in important_values.items():

            label = label_map.get(f, f)

            st.markdown(f"- **{label}**: {val}")

    else:

        st.success(
            "Tidak ditemukan indikasi kuat PCOS berdasarkan data yang dimasukkan."
        )

        for f, val in important_values.items():

            label = label_map.get(f, f)

            st.markdown(f"- **{label}**: {val}")