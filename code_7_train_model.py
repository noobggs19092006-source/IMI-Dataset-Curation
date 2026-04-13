"""
Code 7: Academic QSPR Regression Trainer evaluating Test Set Isolation via MLPRegressor and ColumnTransformer.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

import config

def main():
    print("Loading raw transformed pipeline matrices...")
    df = pd.read_csv('ready_polymer_dataset.csv')
    
    # Isolate targets and predictors properly
    X = df.drop(columns=['SMILES', 'Target_Eb'])
    y = df['Target_Eb']
    
    print("Partitioning Train/Test Isolation Boundaries...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED
    )
    
    # Extract strict column namespaces for Leak-Free PCA 
    morgan_cols = [f"Morgan_Bit_{i}" for i in range(1024)]
    polybert_cols = [f"PolyBERT_Dim_{i}" for i in range(600)]
    
    print("Spinning up zero-leak Preprocessing ColumnTransformer...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('pca_morgan', PCA(n_components=config.NUM_MORGAN_PCA, random_state=config.RANDOM_SEED), morgan_cols),
            ('pca_polybert', PCA(n_components=config.NUM_POLYBERT_PCA, random_state=config.RANDOM_SEED), polybert_cols)
        ],
        remainder='passthrough'
    )
    
    print("Establishing MLPRegressor Deep Learning Pipeline...")
    from sklearn.impute import SimpleImputer
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('scaler', StandardScaler()),
        ('collinearity', config.CollinearityDropper(threshold=0.95)),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(128, 64), 
            activation='relu', 
            early_stopping=True, 
            random_state=config.RANDOM_SEED
        ))
    ])
    
    print("Fitting model strictly over training sets...")
    pipeline.fit(X_train, y_train)
    
    print("Validating model across test set margins...")
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n--- EVALUATION BENCHMARKS ---\n")
    print(f"Test Set R-Squared (R²): {r2:.4f}")
    print(f"Test Set MAE Margin:     {mae:.2f} MV/m")
    print(f"Test Set RMSE Margin:    {rmse:.2f} MV/m\n")
    
    # Serializing core wrapper
    joblib.dump(pipeline, 'mlp_pipeline.pkl')
    
    # Plot Generation Output
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.75, color='royalblue', edgecolors='black', s=50)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2.5, label='Perfect Fit Benchmark')
    plt.title('Validation Predictions vs Actual Target Eb', fontsize=14, pad=15)
    plt.xlabel('Simulated Target Eb (MV/m)', fontsize=12)
    plt.ylabel('Model Prediction Eb (MV/m)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('eb_predictions.png', dpi=300)
    
    print("Model serialized to 'mlp_pipeline.pkl'")
    print("Mapped visual evaluation scatter plot efficiently mapped: eb_predictions.png")

    # Plot MLP Loss Curve
    mlp_model = pipeline.named_steps['mlp']
    plt.figure(figsize=(8, 6))
    plt.plot(mlp_model.loss_curve_, color='crimson', lw=2)
    plt.title('MLPRegressor Training Loss Curve', fontsize=14, pad=15)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('mlp_loss_curve.png', dpi=300)
    print("Mapped loss generation curve: mlp_loss_curve.png")
    
    # Plot Error Margin Distribution
    abs_errors = np.abs(y_test - y_pred)
    plt.figure(figsize=(8, 6))
    plt.hist(abs_errors, bins=25, color='darkorange', edgecolor='black', alpha=0.75)
    plt.title('Absolute Error Distribution (Test Set)', fontsize=14, pad=15)
    plt.xlabel('Absolute Error Margin (MV/m)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('error_distribution_hist.png', dpi=300)
    print("Mapped visual error distribution: error_distribution_hist.png")

if __name__ == "__main__":
    main()
