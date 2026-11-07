
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Classification")
st.write("Machine Learning Based Pass/Fail Prediction")

# Load dataset
df = pd.read_csv("dataset ai.csv")

# Convert GradeClass into Pass and Fail
df["Result"] = df["GradeClass"].apply(
    lambda x: "Fail" if x == 4 else "Pass"
)

# Dashboard metrics
total = len(df)
passed = (df["Result"] == "Pass").sum()
failed = (df["Result"] == "Fail").sum()

c1, c2, c3 = st.columns(3)

c1.metric("Total Students", total)
c2.metric("Pass Students", passed)
c3.metric("Fail Students", failed)

st.divider()

# Visualizations
st.subheader("Student Performance Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("Pass vs Fail Distribution")
    st.bar_chart(df["Result"].value_counts())

with col2:
    st.write("Study Hours Distribution")
    st.bar_chart(df["StudyTimeWeekly"])

st.subheader("Dataset Overview")
st.dataframe(df.head(10), use_container_width=True)

# Machine Learning
st.subheader("Machine Learning Model")

features = [
    "Age", "Gender", "Ethnicity",
    "ParentalEducation", "StudyTimeWeekly",
    "Absences", "Tutoring", "ParentalSupport",
    "Extracurricular", "Sports", "Music",
    "Volunteering"
]

X = df[features]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

st.write("Confusion Matrix")
cm = confusion_matrix(
    y_test, predictions,
    labels=["Fail", "Pass"]
)

st.dataframe(
    pd.DataFrame(
        cm,
        index=["Actual Fail", "Actual Pass"],
        columns=["Predicted Fail", "Predicted Pass"]
    )
)

# Individual Prediction
st.divider()
st.subheader("Individual Student Prediction")

with st.form("prediction_form"):

    age = st.number_input("Age", 15, 18, 16)
    gender = st.selectbox("Gender", [0, 1])
    ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3, 4])
    parental_education = st.selectbox(
        "Parental Education", [0, 1, 2, 3, 4]
    )
    study = st.number_input("Study Time Weekly", 0.0, 50.0, 10.0)
    absences = st.number_input("Absences", 0, 100, 5)
    tutoring = st.selectbox("Tutoring", [0, 1])
    support = st.selectbox("Parental Support", [0, 1, 2, 3, 4])
    extracurricular = st.selectbox("Extracurricular", [0, 1])
    sports = st.selectbox("Sports", [0, 1])
    music = st.selectbox("Music", [0, 1])
    volunteering = st.selectbox("Volunteering", [0, 1])

    submit = st.form_submit_button("Predict Student Result")

if submit:

    new_student = pd.DataFrame([[
        age, gender, ethnicity, parental_education,
        study, absences, tutoring, support,
        extracurricular, sports, music, volunteering
    ]], columns=features)

    result = model.predict(new_student)[0]

    if result == "Pass":
        st.success("Predicted Result: PASS")
    else:
        st.error("Predicted Result: FAIL")
