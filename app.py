# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib # For loading 
import time
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Sustainable Fuel Blend Property Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Feature Engineering Function (from your script) ---
# This function must be available for the app to process new data
def create_advanced_fuel_features(df):
    df_out = df.copy()
    fractions = df_out[[f'Component{j}_fraction' for j in range(1, 6)]].values
    properties = {}
    for i in range(1, 11):
        properties[i] = df_out[[f'Component{j}_Property{i}' for j in range(1, 6)]].values

    # 1. ENHANCED BLEND PROPERTIES
    for i in range(1, 11):
        df_out[f'blend_prop_{i}'] = np.sum(properties[i] * fractions, axis=1)
        weighted_mean = df_out[f'blend_prop_{i}'].values.reshape(-1, 1)
        weighted_var = np.sum(fractions * (properties[i] - weighted_mean) ** 2, axis=1)
        df_out[f'blend_prop_{i}_wstd'] = np.sqrt(weighted_var)
        df_out[f'blend_prop_{i}_skew'] = np.sum(fractions * (properties[i] - weighted_mean) ** 3, axis=1)
        prop_range = np.max(properties[i], axis=1) - np.min(properties[i], axis=1)
        total_fraction = np.sum(fractions, axis=1) + 1e-8
        df_out[f'blend_prop_{i}_range_weighted'] = prop_range * total_fraction
    
    # 2. BASE FUEL ANALYSIS
    total_fraction = np.sum(fractions, axis=1) + 1e-8
    df_out['comp4_dominance'] = fractions[:, 3] / total_fraction
    for prop in range(1, 11):
        df_out[f'comp4_prop_{prop}_influence'] = properties[prop][:, 3] * fractions[:, 3]
        df_out[f'comp4_prop_{prop}_deviation'] = np.abs(properties[prop][:, 3] - df_out[f'blend_prop_{prop}'])

    # 3. COMPONENT INTERACTION
    component_pairs = [(0, 1), (0, 3), (1, 3), (2, 3), (3, 4)]
    for i, j in component_pairs:
        df_out[f'comp_{i+1}_{j+1}_interaction'] = fractions[:, i] * fractions[:, j]
        for prop in range(1, 6):
            prop_diff = np.abs(properties[prop][:, i] - properties[prop][:, j])
            fraction_product = fractions[:, i] * fractions[:, j]
            df_out[f'comp_{i+1}_{j+1}_prop_{prop}_synergy'] = prop_diff * fraction_product
            
    # 4. BLEND COMPLEXITY
    normalized_fractions = fractions / (np.sum(fractions, axis=1, keepdims=True) + 1e-8)
    df_out['blend_entropy'] = -np.sum(normalized_fractions * np.log(normalized_fractions + 1e-8), axis=1)
    
    # 5. PROPERTY CORRELATION
    all_blend_props = [f'blend_prop_{i}' for i in range(1, 11)]
    blend_prop_matrix = df_out[all_blend_props].values
    critical_property_pairs = [(1, 2), (3, 4), (5, 7), (8, 10)]
    for p1, p2 in critical_property_pairs:
        df_out[f'blend_prop_{p1}_{p2}_product'] = (df_out[f'blend_prop_{p1}'] * df_out[f'blend_prop_{p2}'])
        df_out[f'blend_prop_{p1}_{p2}_ratio'] = (df_out[f'blend_prop_{p1}'] / (df_out[f'blend_prop_{p2}'] + 1e-8))

    return df_out

# --- Load Model and Feature List ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('catboost_fuel_blend_model.joblib')
        features = joblib.load('feature_columns.joblib')
        return model, features
    except FileNotFoundError:
        st.error("Model files not found. Please run the training script first.")
        return None, None

model, feature_cols = load_model()
# --- UI Layout ---
st.title("Sustainable Fuel Blend Property Predictor")

st.markdown("""
This tool predicts key properties of sustainable fuel blends using a pre-trained machine learning model.  
It was developed as part of the **Shell AI Hackathon for Sustainable and Efficient Energy**.

To begin, upload a CSV file containing the component-wise data of your fuel blend.
""")

# --- Sidebar ---
st.sidebar.header("Upload Input")
uploaded_file = st.sidebar.file_uploader("Select a CSV file", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("About This Tool")
st.sidebar.info(
    """
    This app uses a CatBoost-based regression model to predict ten blend properties
    based on the composition and characteristics of five source components.
    """
)

# --- Main Application Logic ---
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    
    st.markdown("---")
    st.subheader("1. Uploaded Data Preview")
    st.dataframe(input_df.head())

    with st.spinner("Generating advanced features and making predictions..."):
        start_time = time.time()

        # Preserve ID
        ids = input_df['ID']
        
        # Feature Engineering
        processed_df = create_advanced_fuel_features(input_df)
        X_test = processed_df[feature_cols].fillna(0)

        # Prediction
        if model:
            predictions = model.predict(X_test)
            target_cols = [f'BlendProperty{i}' for i in range(1, 11)]
            submission_df = pd.DataFrame(predictions, columns=target_cols)
            submission_df.insert(0, 'ID', ids)
            end_time = time.time()

            st.success(f"Predictions completed in {end_time - start_time:.2f} seconds.")

            st.markdown("---")
            st.subheader("2. Predicted Blend Properties")
            st.dataframe(submission_df.head(10))

            # Download Button
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv(submission_df)

            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='predicted_blend_properties.csv',
                mime='text/csv'
            )

            # Feature Importance
            st.markdown("---")
            st.subheader("3. Top 20 Most Influential Features")
            try:
                feature_importance = np.mean(
                    [est.get_feature_importance() for est in model.estimators_], axis=0
                )
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': feature_importance
                }).sort_values(by='Importance', ascending=False).head(20)

                st.bar_chart(importance_df.set_index('Feature'))
                st.caption("The chart above shows which features most affect the prediction output.")
            except Exception as e:
                st.warning(f"Unable to display feature importance. Error: {e}")
else:
    st.info("Please upload a CSV file from the sidebar to begin.")
    st.image(
        "https://images.unsplash.com/photo-1621761192059-b14524335c4a?q=80&w=2070&auto=format&fit=crop",
        caption="Sustainable aviation fuels are pivotal for decarbonizing air travel.",
        use_column_width=True
    )
