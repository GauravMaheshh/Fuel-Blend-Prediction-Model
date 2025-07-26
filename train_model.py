# train_model.py
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- Feature Engineering Function (Identical to the one in app.py) ---
def create_advanced_fuel_features(df):
    df_out = df.copy()
    fractions = df_out[[f'Component{j}_fraction' for j in range(1, 6)]].values
    properties = {}
    for i in range(1, 11):
        properties[i] = df_out[[f'Component{j}_Property{i}' for j in range(1, 6)]].values
    # (The rest of the function is the same as in app.py)
    # ... [omitted for brevity, but should be included in your file] ...
    for i in range(1, 11):
        df_out[f'blend_prop_{i}'] = np.sum(properties[i] * fractions, axis=1)
        weighted_mean = df_out[f'blend_prop_{i}'].values.reshape(-1, 1)
        weighted_var = np.sum(fractions * (properties[i] - weighted_mean) ** 2, axis=1)
        df_out[f'blend_prop_{i}_wstd'] = np.sqrt(weighted_var)
        df_out[f'blend_prop_{i}_skew'] = np.sum(fractions * (properties[i] - weighted_mean) ** 3, axis=1)
        prop_range = np.max(properties[i], axis=1) - np.min(properties[i], axis=1)
        total_fraction = np.sum(fractions, axis=1) + 1e-8
        df_out[f'blend_prop_{i}_range_weighted'] = prop_range * total_fraction
    total_fraction = np.sum(fractions, axis=1) + 1e-8
    df_out['comp4_dominance'] = fractions[:, 3] / total_fraction
    for prop in range(1, 11):
        df_out[f'comp4_prop_{prop}_influence'] = properties[prop][:, 3] * fractions[:, 3]
        df_out[f'comp4_prop_{prop}_deviation'] = np.abs(properties[prop][:, 3] - df_out[f'blend_prop_{prop}'])
    component_pairs = [(0, 1), (0, 3), (1, 3), (2, 3), (3, 4)]
    for i, j in component_pairs:
        df_out[f'comp_{i+1}_{j+1}_interaction'] = fractions[:, i] * fractions[:, j]
        for prop in range(1, 6):
            prop_diff = np.abs(properties[prop][:, i] - properties[prop][:, j])
            fraction_product = fractions[:, i] * fractions[:, j]
            df_out[f'comp_{i+1}_{j+1}_prop_{prop}_synergy'] = prop_diff * fraction_product
    normalized_fractions = fractions / (np.sum(fractions, axis=1, keepdims=True) + 1e-8)
    df_out['blend_entropy'] = -np.sum(normalized_fractions * np.log(normalized_fractions + 1e-8), axis=1)
    all_blend_props = [f'blend_prop_{i}' for i in range(1, 11)]
    blend_prop_matrix = df_out[all_blend_props].values
    critical_property_pairs = [(1, 2), (3, 4), (5, 7), (8, 10)]
    for p1, p2 in critical_property_pairs:
        df_out[f'blend_prop_{p1}_{p2}_product'] = (df_out[f'blend_prop_{p1}'] * df_out[f'blend_prop_{p2}'])
        df_out[f'blend_prop_{p1}_{p2}_ratio'] = (df_out[f'blend_prop_{p1}'] / (df_out[f'blend_prop_{p2}'] + 1e-8))
    return df_out

# --- Main Training Logic ---
if __name__ == "__main__":
    print("🔄 Loading training data...")
    try:
        train_df = pd.read_csv('/Users/gaurav/Documents/Shell_Khans/dataset/train.csv')
    except FileNotFoundError:
        print("❌ Error: 'dataset/train.csv' not found. Make sure the dataset is in the correct directory.")
        exit()

    print("🚀 Starting feature engineering...")
    train_df = create_advanced_fuel_features(train_df)
    print("✅ Feature engineering complete.")

    # Prepare data for training
    target_cols = [f'BlendProperty{i}' for i in range(1, 11)]
    feature_cols = [col for col in train_df.columns if col not in target_cols + ['ID']]
    
    X = train_df[feature_cols].fillna(0)
    y = train_df[target_cols]

    print(f"📊 Training with {len(feature_cols)} features.")

    # Model parameters (from your script)
    catboost_params = {
        'iterations': 3000, 'learning_rate': 0.02, 'depth': 8,
        'l2_leaf_reg': 6, 'random_strength': 0.5, 'bagging_temperature': 0.3,
        'border_count': 254, 'random_state': 42, 'verbose': 200,
        'thread_count': -1, 'task_type': 'CPU', 'bootstrap_type': 'Bayesian',
        'grow_policy': 'Lossguide', 'od_type': 'Iter', 'od_wait': 100
    }

    # Create and train the model
    print("🏋️ Training the final CatBoost model... (This may take several minutes)")
    catboost_model = MultiOutputRegressor(CatBoostRegressor(**catboost_params))
    catboost_model.fit(X, y)
    print("✅ Model training complete.")

    # Save the model and feature columns
    print("💾 Saving model and feature list...")
    joblib.dump(catboost_model, 'catboost_fuel_blend_model.joblib')
    joblib.dump(feature_cols, 'feature_columns.joblib')
    print("✅ Files saved successfully.")
    print("\n🎉 You can now run the Streamlit app with: streamlit run app.py")

