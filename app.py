import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------
# 1. Generate Example Dataset
# ---------------------
np.random.seed(42)
activities = {
    'Resting': (60, 80),
    'Walking (slow)': (80, 100),
    'Walking (fast)': (100, 120),
    'Jogging': (120, 140),
    'Running': (140, 160),
    'Cycling': (120, 150),
    'HIIT': (160, 180),
    'Yoga': (70, 90)
}

activity_list = np.random.choice(list(activities.keys()), 200)
data = []
for act in activity_list:
    age = np.random.randint(18, 60)
    weight = np.random.randint(50, 100)
    gender = np.random.choice(['Male', 'Female'])
    duration = np.random.randint(5, 60)
    hr_range = activities[act]
    heart_rate = np.random.randint(hr_range[0], hr_range[1])
    calories = 0.035 * weight + (heart_rate * duration / weight) * 0.029 * weight
    calories += np.random.normal(0, 5)
    data.append([age, weight, gender, duration, heart_rate, act, calories])

df = pd.DataFrame(data, columns=['Age', 'Weight', 'Gender', 'Duration', 'Heart_Rate', 'Activity', 'Calories'])
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Activity'] = df['Activity'].astype('category').cat.codes

# ---------------------
# 2. Train Model
# ---------------------
X = df[['Age', 'Weight', 'Gender', 'Duration', 'Heart_Rate', 'Activity']]
y = df['Calories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# ---------------------
# 3. Streamlit App UI
# ---------------------
def main():
    st.title("Calorie Prediction App")
    st.write("Predict calories burned based on age, weight, gender, duration, heart rate, and activity.")

    age = st.slider("Age", 18, 60, 25)
    weight = st.slider("Weight (kg)", 50, 100, 70)
    gender = st.selectbox("Gender", ["Male", "Female"])
    duration = st.slider("Duration (minutes)", 5, 60, 30)
    heart_rate = st.slider("Heart Rate", 60, 180, 120)
    activity = st.selectbox("Activity", list(activities.keys()))

    if st.button("Predict"):
        input_data = pd.DataFrame({
            'Age': [age],
            'Weight': [weight],
            'Gender': [0 if gender == "Male" else 1],
            'Duration': [duration],
            'Heart_Rate': [heart_rate],
            'Activity': [list(activities.keys()).index(activity)]
        })
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Calories Burned: {prediction:.2f}")

    st.subheader("Model Evaluation")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"**Mean Absolute Error:** {mae:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

# ---------------------
# 4. Auto-run with Streamlit if executed directly
# ---------------------
if __name__ == "__main__":
    if any(arg.endswith(".py") for arg in sys.argv):
        # Run inside Streamlit
        main()
    else:
        # If running directly with python app.py
        os.system(f"streamlit run {os.path.abspath(__file__)}")
