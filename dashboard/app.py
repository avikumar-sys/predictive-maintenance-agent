import streamlit as st
import requests
import pandas as pd
import os
import time

st.set_page_config(page_title="Predictive Maintenance Agent", layout="wide")

API_URL = "https://predictive-maintenance-agent.onrender.com"
LOG_PATH = os.path.join("logs", "predictions.csv")

# ✅ Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

# ✅ Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["Predict", "History", "Analytics", "Live Monitor"])


# Small helper
def load_history():
    if not os.path.exists(LOG_PATH):
        return None
    df = pd.read_csv(LOG_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# =========================
# 🔹 PREDICTION PAGE
# =========================
if page == "Predict":
    st.title("🛠️ Predictive Maintenance Dashboard")
    st.markdown("Enter machine sensor data below to predict failure risk.")

    col1, col2 = st.columns(2)

    with st.form("prediction_form"):
        with col1:
            type_input = st.selectbox("Type", ["L", "M", "H"])
            air_temp = st.number_input("Air temperature [K]", value=298.1)
            process_temp = st.number_input("Process temperature [K]", value=308.6)
        with col2:
            rpm = st.number_input("Rotational speed [rpm]", value=1551)
            torque = st.number_input("Torque [Nm]", value=42.8)
            tool_wear = st.number_input("Tool wear [min]", value=0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # ✅ Correct field names for backend
        input_data = {
            "Type": type_input,
            "Air_temperature_K": air_temp,
            "Process_temperature_K": process_temp,
            "Rotational_speed_rpm": rpm,
            "Torque_Nm": torque,
            "Tool_wear_min": tool_wear,
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=input_data)
            response.raise_for_status()
            result = response.json()

            left, right = st.columns(2)

            with left:
                st.success(result.get("status", "Prediction completed"))
                st.metric("Failure Probability", result.get("probability", 0))
                st.write("**Recommended Action:**", result.get("recommended_action", ""))

            with right:
                with st.expander("View input data"):
                    st.json(input_data)
                with st.expander("View raw response"):
                    st.json(result)

            # ✅ Save to history
            log_entry = {
                "timestamp": pd.Timestamp.now(),
                "type": type_input,
                "air_temperature_K": air_temp,
                "process_temperature_K": process_temp,
                "rotational_speed_rpm": rpm,
                "torque_Nm": torque,
                "tool_wear_min": tool_wear,
                "prediction": result.get("prediction"),
                "probability": result.get("probability"),
            }

            df = pd.DataFrame([log_entry])

            if not os.path.exists(LOG_PATH):
                df.to_csv(LOG_PATH, index=False)
            else:
                df.to_csv(LOG_PATH, mode="a", header=False, index=False)

        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to API at the deployed URL.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Server returned an error: {e}")
            try:
                st.json(response.json())
            except:
                st.write("No JSON body in error response.")
        except Exception as e:
            st.error("❌ Unexpected error occurred.")
            st.exception(e)


# =========================
# 🔹 HISTORY PAGE
# =========================
if page == "History":
    st.title("📜 Prediction History")

    df = load_history()
    if df is None or df.empty:
        st.warning("No history found yet. Make at least one prediction first.")
    else:
        st.dataframe(df, use_container_width=True)


# =========================
# 🔹 ANALYTICS PAGE
# =========================
if page == "Analytics":
    st.title("📊 Analytics & Trends")

    df = load_history()
    if df is None or df.empty:
        st.warning("No history available for analytics yet.")
    else:
        df = df.sort_values("timestamp")

        st.subheader("Failure probability over time")
        st.line_chart(df.set_index("timestamp")["probability"])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("RPM vs failure probability")
            st.scatter_chart(df[["rotational_speed_rpm", "probability"]])

            st.subheader("Tool wear vs failure probability")
            st.scatter_chart(df[["tool_wear_min", "probability"]])

        with col2:
            st.subheader("Air temperature vs failure probability")
            st.scatter_chart(df[["air_temperature_K", "probability"]])

            st.subheader("Process temperature vs failure probability")
            st.scatter_chart(df[["process_temperature_K", "probability"]])

        st.subheader("Failure vs normal counts")
        fail_counts = df["prediction"].value_counts().rename(index={0: "Normal", 1: "Failure"})
        st.bar_chart(fail_counts)


# =========================
# 🔹 LIVE MONITOR PAGE
# =========================
if page == "Live Monitor":
    st.title("📡 Live Monitoring")

    refresh_sec = st.sidebar.slider("Auto-refresh every (seconds)", 2, 30, 5)
    placeholder = st.empty()

    while True:
        df = load_history()
        with placeholder.container():
            if df is None or df.empty:
                st.warning("No predictions yet.")
            else:
                df = df.sort_values("timestamp", ascending=False)
                st.subheader("Latest predictions")
                st.dataframe(df.head(20), use_container_width=True)

                if "probability" in df.columns:
                    high_risk = df[df["probability"] > 0.7]
                    if not high_risk.empty:
                        st.error("⚠️ High-risk predictions detected!")
                        st.dataframe(high_risk.head(10))
                    else:
                        st.success("✅ No high-risk predictions detected.")

        time.sleep(refresh_sec)