import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# CẤU HÌNH TRANG
# =============================
st.set_page_config(
    page_title="Heart Disease AI - Hồ Duy Khánh",
    page_icon="❤️",
    layout="wide"
)

# =============================
# LOAD MODEL + DATASET
# =============================
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("heart_model.pkl", "rb"))
        df = pd.read_csv("heart.csv")
        return model, df
    except Exception as e:
        return None, None

model, df = load_assets()

if model is None or df is None:
    st.error("Lỗi: Không tìm thấy file 'heart_model.pkl' hoặc 'heart.csv' trong cùng thư mục!")
    st.stop()

# =============================
# SIDEBAR
# =============================
st.sidebar.title("🫀 Heart Disease AI")
st.sidebar.info(
"""
**ĐỒ ÁN MACHINE LEARNING**
* **Sinh viên:** Hồ Duy Khánh
* **Đề tài:** Dự đoán nguy cơ bệnh tim
* **Model:** Logistic Regression
"""
)

st.sidebar.markdown("---")
show_analysis = st.sidebar.checkbox("Hiển thị phân tích dữ liệu", value=True)

# =============================
# HEADER
# =============================
st.title("❤️ Hệ thống dự đoán bệnh tim bằng AI")
st.write("Nhập thông tin bệnh nhân để hệ thống AI dự đoán nguy cơ mắc bệnh.")
st.divider()

# =============================
# INPUT FORM
# =============================
col1, col2 = st.columns(2)

with col1:
    age = st.slider("1. Tuổi", 20, 80, 35)
    sex = st.selectbox("2. Giới tính", ["Nữ", "Nam"])
    cp = st.selectbox(
        "3. Loại đau ngực",
        ["Không có triệu chứng", "Đau thắt ngực điển hình", "Đau thắt ngực không điển hình", "Đau không do tim"]
    )
    trestbps = st.slider("4. Huyết áp nghỉ (mm Hg)", 90, 200, 120)
    chol = st.slider("5. Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.selectbox("6. Đường huyết > 120 mg/dl", ["Không", "Có"])

with col2:
    restecg = st.selectbox(
        "7. Kết quả điện tâm đồ",
        ["Bình thường", "Bất thường sóng ST-T", "Phì đại thất trái"]
    )
    thalach = st.slider("8. Nhịp tim tối đa", 70, 210, 150)
    exang = st.selectbox("9. Đau thắt ngực khi vận động", ["Không", "Có"])
    oldpeak = st.slider("10. Oldpeak (ST depression)", 0.0, 6.0, 0.0, 0.1)
    slope = st.selectbox("11. Độ dốc ST", ["Giảm", "Phẳng", "Tăng"])
    ca = st.selectbox("12. Số mạch máu chính", [0, 1, 2, 3])
    thal = st.selectbox(
        "13. Thalassemia",
        ["Bình thường", "Khiếm khuyết cố định", "Khiếm khuyết có thể đảo ngược"]
    )

# =============================
# MAPPING DỮ LIỆU
# =============================
input_sex = 1 if sex == "Nam" else 0
cp_map = {"Không có triệu chứng": 0, "Đau thắt ngực điển hình": 1, "Đau thắt ngực không điển hình": 2, "Đau không do tim": 3}
input_cp = cp_map[cp]
input_fbs = 1 if fbs == "Có" else 0
restecg_map = {"Bình thường": 0, "Bất thường sóng ST-T": 1, "Phì đại thất trái": 2}
input_restecg = restecg_map[restecg]
input_exang = 1 if exang == "Có" else 0
slope_map = {"Giảm": 0, "Phẳng": 1, "Tăng": 2}
input_slope = slope_map[slope]
thal_map = {"Khiếm khuyết cố định": 1, "Bình thường": 2, "Khiếm khuyết có thể đảo ngược": 3}
input_thal = thal_map[thal]

# =============================
# DỰ ĐOÁN
# =============================
st.divider()
if st.button("🔍 Dự đoán ngay"):
    data = np.array([[age, input_sex, input_cp, trestbps, chol, input_fbs, 
                      input_restecg, thalach, input_exang, oldpeak, 
                      input_slope, ca, input_thal]])

    prediction = model.predict(data)
    
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(data)[0][1]
    else:
        prob = 1.0 if prediction[0] == 1 else 0.0

    st.subheader("Kết quả phân tích")
    if prediction[0] == 1:
        st.error(f"⚠️ Nguy cơ mắc bệnh tim cao: {prob*100:.1f}%")
        st.progress(prob)
    else:
        st.success(f"✅ Nguy cơ thấp: {(1-prob)*100:.1f}%")
        st.progress(prob)

# =============================
# PHÂN TÍCH DATASET
# =============================
if show_analysis:
    st.divider()
    st.subheader("📊 Phân tích dữ liệu gốc")
    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots()
        sns.countplot(x='target', data=df, ax=ax1, palette='Set2')
        ax1.set_title("Tỉ lệ Mắc bệnh (1) vs Khỏe mạnh (0)")
        st.pyplot(fig1)
    with c2:
        fig2, ax2 = plt.subplots()
        sns.histplot(df['age'], kde=True, ax=ax2, color='blue')
        ax2.set_title("Phân bố độ tuổi")
        st.pyplot(fig2)
