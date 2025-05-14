from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import dump
import math
import matplotlib.pyplot as plt
import seaborn as sns

##描述符
# def load_data(filepath):
#     df = pd.read_csv(filepath)
#     df = df.dropna()
#     X = np.array(df.iloc[:, :209].to_numpy())
#     y = df['class'].values
#     return X, y


##指纹
# def load_data(filepath):
#     df = pd.read_csv(filepath)
#     # X = np.array([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=2, nBits=1024) for smi in list(df.iloc[:, 0])])
#     # X = np.array([AllChem.GetMACCSKeysFingerprint(Chem.MolFromSmiles(smi)) for smi in list(df.iloc[:, 0])])
#     X = np.array(
#         [AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(Chem.MolFromSmiles(smi)) for smi in list(df.iloc[:, 0])])
#     # X = np.array([Chem.RDKFingerprint(Chem.MolFromSmiles(smi)) for smi in list(df.iloc[:, 0])])
#     y = df['class'].values
#     return X, y


from transformers import AutoTokenizer, AutoModel
import torch

def load_data(filepath):
    df = pd.read_csv(filepath)
    smiles_list = df.iloc[:, 0].tolist()
    labels = df['class'].values

    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model.eval()

    embeddings = []

    with torch.no_grad():
        for smi in smiles_list:
            inputs = tokenizer(smi, return_tensors="pt", padding=True, truncation=True, max_length=128)
            outputs = model(**inputs)
            # 使用 [CLS] token 表示整分子的向量（等效于第一个 token）
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)

    X = np.array(embeddings)
    return X, labels
def train_model(X_train, y_train, base_model, params, cv_splitter):
    """
    Train a model using GridSearchCV. After finding the best parameters,
    fit a CalibratedClassifierCV for probability calibration.
    """
    # 使用 GridSearchCV 查找最佳的 SVC 参数
    gc = GridSearchCV(base_model, param_grid=params, cv=cv_splitter, scoring='roc_auc', return_train_score=True, verbose=2)
    gc.fit(X_train, y_train)

    # 使用找到的最佳模型进行概率校准
    best_model = gc.best_estimator_
    calibrated_model = CalibratedClassifierCV(best_model, cv='prefit')
    calibrated_model.fit(X_train, y_train)
    return calibrated_model, gc.best_params_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)[:, 1]
    auc_roc_score = roc_auc_score(y_test, y_pred)
    y_pred_binary = [round(y, 0) for y in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
    metrics = calculate_metrics(tp, tn, fp, fn)
    metrics['auc_roc_score'] = auc_roc_score
    return metrics, y_pred_binary

def calculate_metrics(tp, tn, fp, fn):
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    q = (tp + tn) / (tp + fn + tn + fp)
    mcc = (tp * tn - fn * fp) / math.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
    P = tp / (tp + fp)
    F1 = (P * se * 2) / (P + se)
    BA = (se + sp) / 2
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'se': se, 'sp': sp, 'mcc': mcc, 'q': q, 'P': P, 'F1': F1, 'BA': BA}

def save_results_to_file(filename, best_params, metrics):
    """
    Save the best parameters and evaluation metrics to a file.
    """
    with open(filename, 'w') as f:
        f.write("Best Parameters:\n")
        f.write(str(best_params) + "\n\n")
        f.write("Metrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

def plot_confusion_matrix(cm, class_names):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted label', labelpad=10)
    ax.set_ylabel('True label', labelpad=10)
    ax.set_title('SVM Confusion Matrix (Percentage)', pad=20)
    ax.set_xticks(np.arange(cm.shape[1]) + 0.5)
    ax.set_yticks(np.arange(cm.shape[0]) + 0.5)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    plt.setp(ax.get_yticklabels(), rotation=0, va="center")

def save_model(model, filename):
    dump(model, filename)

if __name__ == "__main__":
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X, y = load_data('all_data_brain.csv')  # 替换为正确的路径
    # X, y = load_data('descriptors_output.csv')#描述符输入使用
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化 SVM 模型
    svm_model = SVC(random_state=123)
    svm_params = {
        'C': [0.01, 0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    # 训练模型并进行概率校准
    calibrated_svm, best_params = train_model(X_train, y_train, svm_model, svm_params, kf)
    print("Model training and calibration completed.")

    # 模型评估
    svm_metrics, svm_pred_binary = evaluate_model(calibrated_svm, X_test, y_test)
    print(svm_metrics)

    # 结果保存和显示
    best_model_filename = 'calibrated_svm_model.joblib'
    save_model(calibrated_svm, best_model_filename)
    print(f"The calibrated model is saved as {best_model_filename}")

    save_results_to_file('SVM_results.txt', best_params, svm_metrics)

    cm = confusion_matrix(y_test, svm_pred_binary)
    plot_confusion_matrix(cm, class_names=['Class 0', 'Class 1'])
    plt.show()