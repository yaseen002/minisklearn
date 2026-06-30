import numpy as np

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error for regression."""
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R^2)."""
    y_true = y_true.ravel() # <--- Force 1D
    y_pred = y_pred.ravel() # <--- Force 1D
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy for classification."""
    y_true = y_true.ravel() # <--- Force 1D
    y_pred = y_pred.ravel() # <--- Force 1D
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Computes the confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """Computes macro-averaged Precision, Recall, and F1-Score."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    precisions, recalls, f1s = [], [], []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        
    return {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1s)
    }