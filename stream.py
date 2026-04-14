import streamlit as st
import joblib

st.title("Student Dropout Prediction")
st.header("Data analysis")

Age = st.number_input("Enter Age:")
Gender = st.selectbox("Select Gender:",["Male","Female"])
Family_Income = st.number_input("Enter Family Income:")
Internet_Access = st.selectbox("Internet_access(yes/no):",["Yes","No"])
Study_Hours_per_Day = st.number_input("Study_hour_per_day:")
Attendance_Rate = st.number_input("Attendance_rate:")
Assignment_Delay_Days = st.number_input("Assignment_delay_days:")
Travel_Time_Minutes = st.number_input("Travel_Time_minutes:")
Part_Time_Job = st.selectbox("Partime_job(yes/no):",["Yes","No"])
Scholarship = st.selectbox("Scholarship(yes/no):",["Yes","No"])
Stress_Index = st.number_input("Stress_index:")
GPA = st.number_input("GPA:")
Semester_GPA = st.number_input("Semester_gpa:")
CGPA = st.number_input("CGPA:")
Semester = st.selectbox("Semester:",["Year 1","Year 2","Year 3","Year 4"])
Department = st.selectbox("Department",["Science","Arts","Business","CS","Engineering"])
Parental_Education = st.selectbox("Parental_Education:",["Bachelor","High School","Master","PhD"])

model=joblib.load(r'st_dropout.pkl')
l1=joblib.load(r'lb.pkl')
l2=joblib.load(r'lb1.pkl')
l3=joblib.load(r'lb2.pkl')
l4=joblib.load(r'lb3.pkl')
l5=joblib.load(r'lb4.pkl')
l6=joblib.load(r'lb5.pkl')
l7=joblib.load(r'lb6.pkl')
scaler=joblib.load(r'st.pkl')

Gender=l1.transform([Gender])[0]
Internet_Access=l2.transform([Internet_Access])[0]
Part_Time_Job=l3.transform([Part_Time_Job])[0]
Scholarship=l4.transform([Scholarship])[0]
Semester=l5.transform([Semester])[0]
Department=l6.transform([Department])[0]
Parental_Education=l7.transform([Parental_Education])[0]

if st.button("Predict"):
    result=model.predict(
        scaler.transform([[Age, Gender, Family_Income, Internet_Access,
       Study_Hours_per_Day, Attendance_Rate, Assignment_Delay_Days,
       Travel_Time_Minutes, Part_Time_Job, Scholarship, Stress_Index,
       GPA, Semester_GPA, CGPA, Semester, Department,
       Parental_Education]])
    )[0]
    
    if result==1:
        st.error("Drop_out")
    
    else:
        st.success("Retained")