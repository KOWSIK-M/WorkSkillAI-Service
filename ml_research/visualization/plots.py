import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
from ..config import RESULTS_DIR

def plot_confusion_matrix_heatmap(y_true, y_pred, labels, filename="confusion_matrix.png"):
    # Multi-label confusion matrix is complex (one per label)
    # We aggregate or plot for top 5 common labels to keep it readable
    
    # Selecting top 5 labels for visualization
    top_n = min(5, y_true.shape[1])
    fig, axes = plt.subplots(1, top_n, figsize=(20, 4))
    
    if top_n == 1: axes = [axes]
    
    for i in range(top_n):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f"Skill: {labels[i]}")
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        
    plt.tight_layout()
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, "figures", filename))
    plt.close()

def plot_roc_curve(y_true, y_prob, n_classes, filename="roc_curves.png"):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
             
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Multi-label)')
    plt.legend(loc="lower right")
    
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, "figures", filename))
    plt.close()

def plot_model_comparison(results_df, metric="f1_weighted", filename="model_comparison.png"):
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y=metric, data=results_df, palette="viridis")
    plt.title(f"Model Comparison - {metric}")
    plt.ylim(0, 1)
    for index, row in results_df.iterrows():
        plt.text(index, row[metric], round(row[metric], 3), color='black', ha="center", va="bottom")
        
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, "figures", filename))
    plt.close()
