import streamlit as st
import requests
import pandas as pd
import os
import time

st.set_page_config(page_title="Predictive Maintenance Agent", layout="wide")

API_URL = "http://127.0.0.1:9000"
LOG_PATH = os.path.join("logs", "predictions.csv")

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
        input_data = {
            "Type": type_input,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=input_data)
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                st.error("❌ Backend validation error.")
                st.json(result)
            else:
                left, right = st.columns(2)
                with left:
                    st.success(result["status"])
                    st.metric("Failure Probability", result["probability"])
                    st.write("**Recommended Action:**", result["recommended_action"])
                with right:
                    with st.expander("View input data"):
                        st.json(input_data)
                    with st.expander("View raw response"):
                        st.json(result)

        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to API. Is the FastAPI server running on port 9000?")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Server returned an error: {e}")
            try:
                st.json(response.json())
            except Exception:
                st.write("No JSON body in error response.")
        except Exception as e:
            st.error("❌ An unexpected error occurred while making the prediction.")
            st.exception(e)


# =========================
# 🔹 HISTORY PAGE (with filters)
# =========================
if page == "History":
    st.title("📜 Prediction History")

    df = load_history()
    if df is None or df.empty:
        st.warning("No history found yet. Make at least one prediction first.")
    else:
        # Sidebar filters
        st.sidebar.markdown("### Filters")

        # Date range
        if "timestamp" in df.columns:
            min_date = df["timestamp"].min()
            max_date = df["timestamp"].max()
            date_range = st.sidebar.date_input(
                "Date range",
                value=(min_date.date(), max_date.date())
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                df = df[(df["timestamp"].dt.date >= start_date) &
                        (df["timestamp"].dt.date <= end_date)]

        # Machine type filter
        if "type" in df.columns:
            types = df["type"].dropna().unique().tolist()
            selected_types = st.sidebar.multiselect("Machine type", types, default=types)
            df = df[df["type"].isin(selected_types)]

        # Failure only
        if "prediction" in df.columns:
            failure_only = st.sidebar.checkbox("Show failures only (prediction = 1)", value=False)
            if failure_only:
                df = df[df["prediction"] == 1]

        # Probability threshold
        if "probability" in df.columns:
            prob_min, prob_max = float(df["probability"].min()), float(df["probability"].max())
            prob_filter = st.sidebar.slider(
                "Minimum failure probability",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01
            )
            df = df[df["probability"] >= prob_filter]

        st.subheader("Filtered prediction history")
        st.dataframe(df, use_container_width=True)

        # Summary
        if "prediction" in df.columns and len(df) > 0:
            st.subheader("Summary (after filters)")
            total = len(df)
            failures = df["prediction"].sum()
            failure_rate = (failures / total) * 100 if total > 0 else 0
            st.write(f"**Total predictions:** {total}")
            st.write(f"**Failures predicted:** {failures}")
            st.write(f"**Failure rate:** {round(failure_rate, 2)}%")


# =========================
# 🔹 ANALYTICS PAGE (charts)
# =========================
if page == "Analytics":
    st.title("📊 Analytics & Trends")

    df = load_history()
    if df is None or df.empty:
        st.warning("No history available for analytics yet.")
    else:
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")

        # Convert columns to numeric just in case
        num_cols = ["air_temperature_K", "process_temperature_K",
                    "rotational_speed_rpm", "torque_Nm", "tool_wear_min", "probability"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        st.subheader("Failure probability over time")
        if "timestamp" in df.columns and "probability" in df.columns:
            st.line_chart(df.set_index("timestamp")["probability"])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("RPM vs failure probability")
            if "rotational_speed_rpm" in df.columns and "probability" in df.columns:
                st.scatter_chart(df[["rotational_speed_rpm", "probability"]])

            st.subheader("Tool wear vs failure probability")
            if "tool_wear_min" in df.columns and "probability" in df.columns:
                st.scatter_chart(df[["tool_wear_min", "probability"]])

        with col2:
            st.subheader("Air temperature vs failure probability")
            if "air_temperature_K" in df.columns and "probability" in df.columns:
                st.scatter_chart(df[["air_temperature_K", "probability"]])

            st.subheader("Process temperature vs failure probability")
            if "process_temperature_K" in df.columns and "probability" in df.columns:
                st.scatter_chart(df[["process_temperature_K", "probability"]])

        if "prediction" in df.columns:
            st.subheader("Failure vs normal counts")
            fail_counts = df["prediction"].value_counts().rename(index={0: "Normal", 1: "Failure"})
            st.bar_chart(fail_counts)


# =========================
# 🔹 LIVE MONITOR PAGE (auto-refresh)
# =========================
if page == "Live Monitor":
    st.title("📡 Live Monitoring")

    st.markdown(
        "This mode shows the most recent predictions and highlights potential anomalies "
        "based on failure probability."
    )

    refresh_sec = st.sidebar.slider("Auto-refresh every (seconds)", 2, 30, 5)

    placeholder = st.empty()

    while True:
        df = load_history()
        with placeholder.container():
            if df is None or df.empty:
                st.warning("No predictions yet. Go to the Predict page and start generating data.")
            else:
                df = df.sort_values("timestamp", ascending=False)

                # Show last N rows
                st.subheader("Latest predictions")
                st.dataframe(df.head(20), use_container_width=True)

                # Simple anomaly logic: probability > 0.7
                if "probability" in df.columns and "prediction" in df.columns:
                    recent_high_risk = df[(df["probability"] > 0.7) | (df["prediction"] == 1)]
                    if not recent_high_risk.empty:
                        st.error("⚠️ High-risk or failure predictions detected recently!")
                        st.dataframe(recent_high_risk.head(10), use_container_width=True)
                    else:
                        st.success("✅ No high-risk predictions detected in the recent data.")

        time.sleep(refresh_sec)