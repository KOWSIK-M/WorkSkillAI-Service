from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, hamming_loss, classification_report
import pandas as pd
import numpy as np

def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {}
    
    # Basic metrics (Micro/Macro/Weighted)
    for avg in ['micro', 'macro', 'weighted']:
        metrics[f'precision_{avg}'] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f'recall_{avg}'] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f'f1_{avg}'] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['subset_accuracy'] = accuracy_score(y_true, y_pred) # Same as above for multi-label in sklearn
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    
    # ROC AUC (only if probabilities provided)
    if y_prob is not None:
        try:
            metrics['roc_auc_micro'] = roc_auc_score(y_true, y_prob, average='micro')
            metrics['roc_auc_macro'] = roc_auc_score(y_true, y_prob, average='macro')
        except ValueError:
            metrics['roc_auc_micro'] = 0.0
            metrics['roc_auc_macro'] = 0.0

    return metrics

def print_metrics(metrics, model_name):
    print(f"\n📊 --- Performance Report: {model_name} ---")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"   F1 Score (Micro): {metrics['f1_micro']:.4f}")
    print(f"   F1 Score (Macro): {metrics['f1_macro']:.4f}")
    if 'roc_auc_micro' in metrics:
        print(f"   ROC AUC (Micro): {metrics['roc_auc_micro']:.4f}")
    print("------------------------------------------\n")
