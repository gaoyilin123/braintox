import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import dump
from sklearn.ensemble import RandomForestClassifier  # 添加随机森林模型
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
import math
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import e3fp

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = np.array([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=2, nBits=1024) for smi in list(df.iloc[:, 0])])
    y = df['class'].values
    return X, y


def train_model(X_train, y_train, model, params, cv_splitter):
    """
    Train a model using GridSearchCV with a pre-defined cross-validation splitter.
    """
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
    ax.set_title('RF Confusion Matrix (Percentage)', pad=20)
    ax.set_xticks(np.arange(cm.shape[1]) + 0.5)
    ax.set_yticks(np.arange(cm.shape[0]) + 0.5)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    plt.setp(ax.get_yticklabels(), rotation=0, va="center")

def save_model(model, filename):
    dump(model, filename)
    

kf = KFold(n_splits=5, shuffle=True, random_state=42)

X, y = load_data('all_data.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用随机森林模型
rf_model = RandomForestClassifier(random_state=123)
rf_params = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 15]}

rf_grid_search = train_model(X_train, y_train, rf_model, rf_params, kf)
print(f"Best parameters: {rf_grid_search.best_params_}")

best_model_filename = 'best_rf_model.joblib'
save_model(rf_grid_search.best_estimator_, best_model_filename)
print(f"The best model is saved as {best_model_filename}")

rf_metrics, rf_pred_binary = evaluate_model(rf_grid_search, X_test, y_test)
print(rf_metrics)

save_results_to_file('RF_results.txt', rf_grid_search.best_params_, rf_metrics)

cm = confusion_matrix(y_test, rf_pred_binary)
plot_confusion_matrix(cm, class_names=['Class 0', 'Class 1'])
plt.show()

rfr = rf_grid_search.best_estimator_
X_train = pd.DataFrame(X_train,columns = ['Bit'+str(i) for i in range(1024)])
X_test = pd.DataFrame(X_test,columns = ['Bit'+str(i) for i in range(1024)])
rfr.fit(X_train,y_train)

#计算值
explainer = shap.TreeExplainer(rfr, X_train)
#生成结果
shap_values = explainer.shap_values(X_test)

# 打印SHAP值的形状
print("Shape of X_test:", X_test.shape)
print("Shape of X_train:", X_train.shape)
print("Shape of shap_values:", np.array(shap_values).shape)

shap_values_to_plot = shap_values[..., 1]  # 选择第二个通道


# 绘制 SHAP 总结图
shap.summary_plot(shap_values_to_plot, X_test)

shap.dependence_plot(ind='Bit64', shap_values=shap_values[..., 1], features=X_test)
plt.show()

shap.dependence_plot(ind='Bit64', shap_values=shap_values[..., 1], features=X_test)
plt.show()