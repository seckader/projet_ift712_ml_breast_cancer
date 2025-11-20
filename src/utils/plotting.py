import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, title: str, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def save_classification_report(y_true, y_pred, target_names, out_path: str):
    report = classification_report(y_true, y_pred, target_names=target_names)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
