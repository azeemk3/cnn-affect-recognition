import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, average_precision_score
import krippendorff

def basic_cls_metrics(y_true, y_pred_logits):
    y_true = np.asarray(y_true)
    y_pred = y_pred_logits.argmax(1)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"acc": acc, "f1_macro": f1m}

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def sagr(y_true, y_pred):
    st = np.sign(y_true); sp = np.sign(y_pred)
    return float((st == sp).mean())

def ccc(y_true, y_pred, eps=1e-8):
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    mt, mp = y_t.mean(axis=0), y_p.mean(axis=0)
    vt, vp = y_t.var(axis=0),  y_p.var(axis=0)
    cov = ((y_t - mt) * (y_p - mp)).mean(axis=0)
    ccc_dim = (2*cov) / (vt + vp + (mt - mp)**2 + eps)
    return float(ccc_dim.mean())

def cohen_kappa(y_true, y_pred_logits):
    y_pred = y_pred_logits.argmax(1)
    return float(cohen_kappa_score(y_true, y_pred))

def kripp_alpha(y_true, y_pred_logits, n_classes=8):
    """
    Treat ground truth + predictions as two 'coders'
    and compute Krippendorff's alpha.
    """
    y_pred = y_pred_logits.argmax(1)
    data = np.vstack([y_true, y_pred])
    return float(krippendorff.alpha(reliability_data=data, level_of_measurement="nominal"))

def macro_auc(y_true, y_pred_logits, n_classes=8):
    y_true_oh = np.eye(n_classes)[y_true]
    try:
        auc = roc_auc_score(y_true_oh, y_pred_logits, average="macro", multi_class="ovr")
    except Exception:
        auc = 0.0
    return float(auc)

def macro_auc_pr(y_true, y_pred_logits, n_classes=8):
    y_true_oh = np.eye(n_classes)[y_true]
    scores = []
    for i in range(n_classes):
        if np.sum(y_true_oh[:, i]) > 0:
            ap = average_precision_score(y_true_oh[:, i], y_pred_logits[:, i])
            scores.append(ap)
    return float(np.mean(scores)) if scores else 0.0
