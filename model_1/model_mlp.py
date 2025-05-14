import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import dump
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
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



def train_model(X_train, y_train, model, params, cv_splitter):
    gc = GridSearchCV(model, param_grid=params, cv=cv_splitter, scoring='roc_auc', return_train_score=True, verbose=2)
    gc.fit(X_train, y_train)
    return gc

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
    ax.set_title('MLP Confusion Matrix (Percentage)', pad=20)
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
    X, y = load_data('all_data_brain.csv')
    # X, y = load_data('descriptors_output.csv')#描述符输入使用
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlp_model = MLPClassifier(random_state=123)
    mlp_params = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "activation": ['tanh', 'relu'],
        "solver": ['sgd', 'adam'],
        "learning_rate_init": [0.001, 0.01],
        "max_iter": [10000]
    }

    mlp_grid_search = train_model(X_train, y_train, mlp_model, mlp_params, kf)
    print(f"Best parameters: {mlp_grid_search.best_params_}")

    best_model_filename = 'best_mlp_model.joblib'
    save_model(mlp_grid_search.best_estimator_, best_model_filename)
    print(f"The best model is saved as {best_model_filename}")

    mlp_metrics, mlp_pred_binary = evaluate_model(mlp_grid_search, X_test, y_test)
    print(mlp_metrics)

    save_results_to_file('MLP_results.txt', mlp_grid_search.best_params_, mlp_metrics)

    cm = confusion_matrix(y_test, mlp_pred_binary)
    plot_confusion_matrix(cm, class_names=['Class 0', 'Class 1'])
    plt.show()
