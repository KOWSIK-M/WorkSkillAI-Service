import pandas as pd
import os
import sys

# Ensure we can import from local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml_research.data.preprocessing import DataPreprocessor
from ml_research.models.logistic_regression import LogisticRegressionModel
from ml_research.models.random_forest import RandomForestModel
from ml_research.models.xgboost_model import XGBoostModel
from ml_research.models.neural_network import NeuralNetworkModel
from ml_research.evaluation.metrics import calculate_metrics, print_metrics
from ml_research.visualization.plots import plot_confusion_matrix_heatmap, plot_roc_curve, plot_model_comparison
from ml_research.config import RESULTS_DIR

def run_pipeline():
    print("=" * 60)
    print("🔬 STARTING SKILL GAP PREDICTION EXPERIMENT")
    print("=" * 60)
    
    # 1. Data Processing
    dp = DataPreprocessor()
    X_train, X_test, y_train, y_test = dp.preprocess()
    
    label_names = dp.mlb.classes_
    print(f"📚 Classes: {len(label_names)} skills")
    
    # 2. Define Models
    models = [
        LogisticRegressionModel(),
        RandomForestModel(),
        XGBoostModel(),
        NeuralNetworkModel()
    ]
    
    results = []
    
    # 3. Train & Evaluate Loop
    for model in models:
        print("\n" + "-" * 40)
        
        # Train
        model.train(X_train, y_train)
        model.save()
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        print_metrics(metrics, model.name)
        
        # Collect results
        metrics['Model'] = model.name
        results.append(metrics)
        
        # Confusion Matrix Plot (only for top skills to save space)
        if hasattr(y_test, 'toarray'): # sparse support
            y_test_dense = y_test.toarray()
            y_pred_dense = y_pred.toarray() if hasattr(y_pred, 'toarray') else y_pred
        else:
            y_test_dense = y_test
            y_pred_dense = y_pred
            
        # Plot Specifics
        if model.name == "logistic_regression": # Generate ROC for baseline
            plot_roc_curve(y_test_dense, y_prob, len(label_names), filename=f"roc_{model.name}.png")
        if model.name == "random_forest":
             plot_confusion_matrix_heatmap(y_test_dense, y_pred_dense, label_names, filename=f"cm_{model.name}.png")

    # 4. Save Consolidated Results
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)
    results_csv_path = os.path.join(RESULTS_DIR, "metrics", "model_comparison.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n✅ All results saved to {results_csv_path}")
    
    # 5. Comparative Plots
    plot_model_comparison(results_df, metric='f1_weighted')
    plot_model_comparison(results_df, metric='roc_auc_micro', filename='comparison_roc.png')
    
    print("\n🎉 EXPERIMENT COMPLETE! Ready for conference paper.")

if __name__ == "__main__":
    run_pipeline()
