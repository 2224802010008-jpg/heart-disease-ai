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
    page_title="Heart Disease AI",
    page_icon="❤️",
    layout="wide"
)

# =============================
# LOAD MODEL + DATASET
# =============================

model = pickle.load(open("heart_model.pkl","rb"))
df = pd.read_csv("heart.csv")

# =============================
# SIDEBAR
# =============================

st.sidebar.title("🫀 Heart Disease AI")

st.sidebar.info(
"""
Ứng dụng **Machine Learning** dự đoán nguy cơ mắc bệnh tim.

Sinh viên: (Điền tên của bạn)  
Môn học: Machine Learning  
Model: Logistic Regression  
Dataset: Heart Disease Dataset
"""
)

st.sidebar.markdown("---")
st.sidebar.write("Dashboard phân tích dữ liệu")

# =============================
# HEADER
# =============================

st.title("❤️ Hệ thống dự đoán bệnh tim bằng AI")

st.write(
"""
Nhập thông tin bệnh nhân để hệ thống AI dự đoán nguy cơ mắc bệnh tim.
"""
)

st.divider()

# =============================
# INPUT FORM
# =============================

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Tuổi",20,80)

    sex = st.selectbox(
        "Giới tính",
        ["Nữ","Nam"]
    )

    cp = st.selectbox(
        "Loại đau ngực",
        [
        "Đau thắt ngực điển hình",
        "Đau thắt ngực không điển hình",
        "Đau không do tim",
        "Không có triệu chứng"
        ]
    )

    trestbps = st.slider("Huyết áp nghỉ (mm Hg)",90,200)

    chol = st.slider("Cholesterol (mg/dl)",100,400)

    fbs = st.selectbox(
        "Đường huyết >120 mg/dl",
        ["Không","Có"]
    )

with col2:

    restecg = st.selectbox(
        "Kết quả điện tâm đồ",
        [
        "Bình thường",
        "Bất thường ST-T",
        "Phì đại thất trái"
        ]
    )

    thalach = st.slider("Nhịp tim tối đa",70,210)

    exang = st.selectbox(
        "Đau thắt ngực khi vận động",
        ["Không","Có"]
    )

    oldpeak = st.slider("Oldpeak (ST depression)",0.0,6.0)

    slope = st.selectbox(
        "Độ dốc ST",
        ["Tăng","Phẳng","Giảm"]
    )

    ca = st.selectbox("Số mạch máu chính",[0,1,2,3])

    thal = st.selectbox(
        "Thalassemia",
        [
        "Bình thường",
        "Khiếm khuyết cố định",
        "Khiếm khuyết có thể đảo ngược"
        ]
    )

# =============================
# CHUYỂN DỮ LIỆU
# =============================

sex = 1 if sex=="Nam" else 0

cp_map = {
"Đau thắt ngực điển hình":0,
"Đau thắt ngực không điển hình":1,
"Đau không do tim":2,
"Không có triệu chứng":3
}

cp = cp_map[cp]

fbs = 1 if fbs=="Có" else 0

restecg_map = {
"Bình thường":0,
"Bất thường ST-T":1,
"Phì đại thất trái":2
}

restecg = restecg_map[restecg]

exang = 1 if exang=="Có" else 0

slope_map = {
"Tăng":0,
"Phẳng":1,
"Giảm":2
}

slope = slope_map[slope]

thal_map = {
"Bình thường":0,
"Khiếm khuyết cố định":1,
"Khiếm khuyết có thể đảo ngược":2
}

thal = thal_map[thal]

st.divider()

# =============================
# PREDICTION
# =============================

if st.button("🔍 Dự đoán nguy cơ bệnh tim"):

    data = np.array([[age,sex,cp,trestbps,chol,fbs,
                      restecg,thalach,exang,oldpeak,
                      slope,ca,thal]])

    prediction = model.predict(data)

    if hasattr(model,"predict_proba"):
        probability = model.predict_proba(data)[0][1]
    else:
        probability = 0.5

    st.subheader("Kết quả phân tích")

    st.progress(int(probability*100))

    if prediction[0]==0:

       st.error(f"⚠️ Nguy cơ mắc bệnh tim: {(1-probability)*100:.1f}%")

    else:

        st.success(f"✅ Nguy cơ thấp: {(1-probability)*100:.1f}%")

st.divider()

# =============================
# DATA ANALYSIS
# =============================

st.subheader("📊 Phân tích Dataset")

col3, col4 = st.columns(2)

with col3:

    st.write("Phân bố bệnh tim")

    fig, ax = plt.subplots()

    df["target"].value_counts().plot(kind="bar",ax=ax)

    ax.set_xlabel("Target")
    ax.set_ylabel("Count")

    st.pyplot(fig)

with col4:

    st.write("Phân bố độ tuổi")

    fig2, ax2 = plt.subplots()

    df["age"].hist(ax=ax2)

    ax2.set_xlabel("Age")

    st.pyplot(fig2)

# =============================
# HEATMAP
# =============================

st.subheader("🔥 Correlation Heatmap")

fig3, ax3 = plt.subplots(figsize=(10,6))

sns.heatmap(df.corr(),annot=False,cmap="coolwarm",ax=ax3)

st.pyplot(fig3)

st.caption("Machine Learning Project - Heart Disease Prediction")
