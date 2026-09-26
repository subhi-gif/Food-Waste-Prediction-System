import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("final_food_waste_model.pkl")


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Food Waste Prediction",
    page_icon="🍽️",
    layout="wide"
)


# ==========================================
# HEADER
# ==========================================

st.title("🍽️ Food Waste Prediction System")

st.write(
    "An AI-based system that predicts expected food waste "
    "using operational and environmental factors."
)

st.divider()


# ==========================================
# NAVIGATION TABS
# ==========================================

prediction_tab, performance_tab, about_tab = st.tabs(
    ["🔮 Prediction", "📊 Model Performance", "ℹ️ About"]
)


# ============================================================
# PREDICTION TAB
# ============================================================

with prediction_tab:

    st.header("🔮 Predict Food Waste")

    st.write(
        "Enter the required information below to estimate "
        "the expected food waste."
    )

    st.divider()

    # ------------------------------------------
    # INPUT COLUMNS
    # ------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Kitchen Details")

        meals_served = st.number_input(
            "Meals Served",
            min_value=0,
            value=100,
            step=1
        )

        kitchen_staff = st.number_input(
            "Kitchen Staff",
            min_value=0,
            value=5,
            step=1
        )

        temperature_C = st.number_input(
            "Temperature (°C)",
            value=25.0,
            step=0.1
        )

        humidity_percent = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=0.1
        )

        past_waste_kg = st.number_input(
            "Previous Waste (kg)",
            min_value=0.0,
            value=20.0,
            step=0.1
        )

    with col2:

        st.subheader("Operational Details")

        selected_date = st.date_input(
            "Date"
        )

        special_event = st.selectbox(
            "Special Event",
            ["No", "Yes"]
        )

        staff_experience = st.selectbox(
            "Staff Experience",
            [
                "beginner",
                "intermediate",
                "expert"
            ]
        )

        waste_category = st.selectbox(
            "Waste Category",
            [
                "dairy",
                "meat",
                "vegetables",
                "grains"
            ]
        )

    st.write("")

    # ==========================================
    # PREDICT BUTTON
    # ==========================================

    predict_button = st.button(
        "🔮 Predict Food Waste",
        use_container_width=True,
        type="primary"
    )


    # ==========================================
    # PREDICTION
    # ==========================================

    if predict_button:

        # Convert selected date
        date = pd.to_datetime(selected_date)

        year = date.year
        month = date.month
        day = date.day

        week_of_year = int(date.isocalendar().week)

        day_of_week = date.dayofweek

        is_weekend = 1 if day_of_week >= 5 else 0

        special_event_value = 1 if special_event == "Yes" else 0


        # --------------------------------------
        # Create input data
        # --------------------------------------

        input_data = pd.DataFrame({
            "meals_served": [meals_served],
            "kitchen_staff": [kitchen_staff],
            "temperature_C": [temperature_C],
            "humidity_percent": [humidity_percent],
            "day_of_week": [day_of_week],
            "special_event": [special_event_value],
            "past_waste_kg": [past_waste_kg],
            "staff_experience": [staff_experience],
            "waste_category": [waste_category],
            "year": [year],
            "month": [month],
            "day": [day],
            "week_of_year": [week_of_year],
            "is_weekend": [is_weekend]
        })


        # --------------------------------------
        # Make prediction
        # --------------------------------------

        prediction = model.predict(input_data)[0]

        # Prevent negative prediction
        prediction = max(0, prediction)


        # ======================================
        # WASTE RISK
        # ======================================

        if prediction < 15:

            risk = "LOW"

            recommendation = (
                "The predicted waste level is relatively low. "
                "The current preparation strategy appears reasonable."
            )

        elif prediction < 30:

            risk = "MEDIUM"

            recommendation = (
                "Monitor preparation quantities and previous "
                "leftovers. Small adjustments may help reduce waste."
            )

        else:

            risk = "HIGH"

            recommendation = (
                "Consider reducing preparation quantities and "
                "closely monitoring leftovers."
            )


        # ======================================
        # RESULT
        # ======================================

        st.divider()

        st.header("🎯 Prediction Result")


        result_col1, result_col2, result_col3 = st.columns(3)


        with result_col1:

            st.metric(
                "Predicted Food Waste",
                f"{prediction:.3f} kg"
            )


        with result_col2:

            st.metric(
                "Waste Risk",
                risk
            )


        with result_col3:

            if meals_served > 0:
                waste_per_meal = prediction / meals_served
            else:
                waste_per_meal = 0

            st.metric(
                "Waste per Meal",
                f"{waste_per_meal:.3f} kg"
            )


        # ======================================
        # RISK MESSAGE
        # ======================================

        st.subheader("💡 Recommendation")


        if risk == "LOW":

            st.success(recommendation)

        elif risk == "MEDIUM":

            st.warning(recommendation)

        else:

            st.error(recommendation)


        # ======================================
        # INPUT SUMMARY
        # ======================================

        st.subheader("📋 Prediction Details")


        detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)


        with detail_col1:

            st.metric(
                "Meals Served",
                f"{meals_served}"
            )


        with detail_col2:

            st.metric(
                "Previous Waste",
                f"{past_waste_kg:.3f} kg"
            )


        with detail_col3:

            st.metric(
                "Temperature",
                f"{temperature_C:.1f} °C"
            )


        with detail_col4:

            st.metric(
                "Humidity",
                f"{humidity_percent:.1f}%"
            )


# ============================================================
# MODEL PERFORMANCE TAB
# ============================================================

with performance_tab:

    st.header("📊 Model Performance")

    st.write(
        "The models were evaluated using a time-based split. "
        "Older observations were used for training and newer "
        "observations were used for testing."
    )

    st.divider()


    # ==========================================
    # MODEL RESULTS
    # ==========================================

    results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting"
        ],
        "MAE": [
            5.222,
            5.609,
            5.471
        ],
        "MSE": [
            61.823,
            68.873,
            83.995
        ],
        "RMSE": [
            7.863,
            8.295,
            9.165
        ],
        "R2 Score": [
            0.919,
            0.910,
            0.890
        ]
    })


    # ==========================================
    # SELECTED MODEL
    # ==========================================

    st.subheader("🏆 Selected Model")

    st.success(
        "Linear Regression was selected as the final model "
        "because it achieved the best performance on the "
        "time-based test data."
    )


    # ==========================================
    # PERFORMANCE METRICS
    # ==========================================

    st.subheader("Final Model Performance")


    metric1, metric2, metric3, metric4 = st.columns(4)


    with metric1:

        st.metric(
            "R² Score",
            f"{0.919:.3f}"
        )


    with metric2:

        st.metric(
            "MAE",
            f"{5.222:.3f} kg"
        )


    with metric3:

        st.metric(
            "MSE",
            f"{61.823:.3f}"
        )


    with metric4:

        st.metric(
            "RMSE",
            f"{7.863:.3f} kg"
        )


    st.divider()


    # ==========================================
    # COMPARISON TABLE
    # ==========================================

    st.subheader("Model Comparison")

    display_results = results.copy()

    display_results["MAE"] = display_results["MAE"].map(
        lambda x: f"{x:.3f}"
    )

    display_results["MSE"] = display_results["MSE"].map(
        lambda x: f"{x:.3f}"
    )

    display_results["RMSE"] = display_results["RMSE"].map(
        lambda x: f"{x:.3f}"
    )

    display_results["R2 Score"] = display_results["R2 Score"].map(
        lambda x: f"{x:.3f}"
    )


    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )


    # ==========================================
    # R2 CHART
    # ==========================================

    st.subheader("📈 R² Score Comparison")

    r2_chart = results.set_index("Model")[["R2 Score"]]

    st.bar_chart(r2_chart)


    # ==========================================
    # ERROR CHART
    # ==========================================

    st.subheader("📉 Error Comparison")

    error_chart = results.set_index("Model")[["MAE", "RMSE"]]

    st.bar_chart(error_chart)


    # ==========================================
    # METRIC EXPLANATION
    # ==========================================

    st.subheader("Understanding the Metrics")


    with st.expander("What is R² Score?"):

        st.write(
            "R² Score measures how well the model explains the "
            "variation in food waste. A value closer to 1 means "
            "the model explains more of the variation."
        )


    with st.expander("What is MAE?"):

        st.write(
            "Mean Absolute Error represents the average absolute "
            "difference between the actual and predicted food waste."
        )

        st.write(
            "Our final model has an MAE of approximately "
            "5.222 kg."
        )


    with st.expander("What is MSE?"):

        st.write(
            "Mean Squared Error calculates the average squared "
            "difference between actual and predicted values. "
            "Larger errors receive greater penalty."
        )


    with st.expander("What is RMSE?"):

        st.write(
            "Root Mean Squared Error is the square root of MSE. "
            "It is expressed in the same unit as the target, "
            "which is kilograms of food waste."
        )


# ============================================================
# ABOUT TAB
# ============================================================

with about_tab:

    st.header("ℹ️ About the Project")


    st.write(
        "The Food Waste Prediction System uses machine learning "
        "to estimate the amount of food waste generated under "
        "different kitchen and operational conditions."
    )


    st.divider()


    # ==========================================
    # PROBLEM
    # ==========================================

    st.subheader("🎯 Problem")

    st.write(
        "Food waste can occur when food preparation does not "
        "match actual demand. Predicting expected waste can help "
        "kitchens make better preparation decisions."
    )


    # ==========================================
    # ML APPROACH
    # ==========================================

    st.subheader("🤖 Machine Learning Approach")

    st.write("Problem Type: Regression")

    st.write("Target Variable: food_waste_kg")

    st.write("Final Model: Linear Regression")

    st.write("Evaluation Method: Time-Based Train/Test Split")


    # ==========================================
    # FEATURES
    # ==========================================

    st.subheader("📋 Input Features")

    features = [
        "Meals served",
        "Kitchen staff",
        "Temperature",
        "Humidity",
        "Day of week",
        "Special event",
        "Previous food waste",
        "Staff experience",
        "Waste category",
        "Year",
        "Month",
        "Day",
        "Week of year",
        "Weekend indicator"
    ]

    for feature in features:

        st.write(f"• {feature}")


    # ==========================================
    # FINAL RESULTS
    # ==========================================

    st.subheader("🏆 Final Model Results")

    st.info(
        "Linear Regression achieved an R² score of 0.919, "
        "MAE of 5.222 kg, and RMSE of 7.863 kg on the "
        "time-based test set."
    )


    # ==========================================
    # DISCLAIMER
    # ==========================================

    st.subheader("⚠️ Note")

    st.write(
        "The prediction is intended as a decision-support "
        "tool. Actual food waste may vary depending on "
        "real-world kitchen conditions."
    )