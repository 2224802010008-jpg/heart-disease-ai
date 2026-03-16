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
# Đảm bảo file heart_model.pkl và heart.csv nằm cùng thư mục với app.py
try:
    model = pickle.load(open("heart_model.pkl", "rb"))
    df = pd.read_csv("heart.csv")
except FileNotFoundError:
    st.error("Không tìm thấy file model hoặc dataset. Vui lòng kiểm tra lại!")

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
* **Dataset:** UCI Heart Disease
"""
)

st.sidebar.markdown("---")
st.sidebar.write("📊 **Dashboard phân tích**")
show_analysis = st.sidebar.checkbox("Hiển thị phân tích dữ liệu", value=True)

# =============================
# HEADER
# =============================
st.title("❤️ Hệ thống dự đoán bệnh tim bằng AI")
st.write("Nhập thông tin lâm sàng của bệnh nhân để hệ thống đưa ra nhận định về nguy cơ mắc bệnh.")
st.divider()

# =============================
# INPUT FORM
# =============================
col1, col2 = st.columns(2)

with col1:
    age = st.slider("1. Tuổi", 20, 80, value=30)
    sex = st.selectbox("2. Giới tính", ["Nữ", "Nam"])
    
    # Mapping chuẩn UCI: 0: typical, 1: atypical, 2: non-anginal, 3: asymptomatic
    cp = st.selectbox(
        "3. Loại đau ngực",
        ["Không có triệu chứng", "Đau thắt ngực điển hình", "Đau thắt ngực không điển hình", "Đau không do tim"]
    )
    
    trestbps = st.slider("4. Huyết áp nghỉ (mm Hg)", 90, 200, value=120)
    chol = st.slider("5. Cholesterol (mg/dl)", 100, 400, value=200)
    fbs = st.selectbox("6. Đường huyết lúc đói > 120 mg/dl", ["Không", "Có"])

with col2:
    # Mapping chuẩn UCI: 0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy
    restecg = st.selectbox(
        "7. Kết quả điện tâm đồ",
        ["Bình thường", "Bất thường sóng ST-T", "Phì đại thất trái"]
    )
    
    thalach = st.slider("8. Nhịp tim tối đa đạt được", 70, 210, value=150)
    exang = st.selectbox("9. Đau thắt ngực khi vận động", ["Không", "Có"])
    oldpeak = st.slider("10. Oldpeak (Chỉ số ST trầm cảm)", 0.0, 6.0, value=0.0, step=0.1)
    
    # Mapping chuẩn UCI: 0: downsloping, 1: flat, 2: upsloping
    slope = st.selectbox("11. Độ dốc đoạn ST", ["Giảm", "Phẳng", "Tăng"])
    
    ca = st.selectbox("12. Số mạch máu chính (0-3) soi qua Fluoroscopy", [0, 1, 2, 3])
    
    # Mapping chuẩn UCI: 1: fixed defect, 2: normal, 3: reversable defect
    thal = st.selectbox(
        "13. Thalassemia (Tình trạng máu)",
        ["Bình thường", "Khiếm khuyết cố định", "Khiếm khuyết có thể đảo ngược"]
    )

# =============================
# XỬ LÝ DỮ LIỆU ĐẦU VÀO (MAPPING CHUẨN)
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
if st.button("🔍 PHÂN TÍCH NGUY CƠ"):
    
    # Tạo mảng dữ liệu khớp với thứ tự các cột khi train model
    features = np.array([[age, input_sex, input_cp, trestbps, chol, input_fbs, 
                          input_restecg, thalach, input_exang, oldpeak, 
                          input_slope, ca, input_thal]])

    prediction = model.predict(features)
    
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features)[0][1]
    else:
        prob = 1.0 if prediction[0] == 1 else 0.0

    st.subheader("📋 Kết quả phân tích từ AI")
    
    if prediction[0] == 1:
        st.error(f"⚠️ **CẢNH BÁO:** Bệnh nhân có nguy cơ mắc bệnh tim cao.")
        st.write(f"Độ tin cậy của mô hình: **{prob*100:.2f}%**")
        st.progress(prob)
    else:
        st.success(f"✅ **AN TOÀN:** Bệnh nhân có nguy cơ mắc bệnh tim thấp.")
        st.
