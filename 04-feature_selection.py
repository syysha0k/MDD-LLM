import os
import time

import pandas as pd
import numpy as np

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import shap

import time


def load_csv(csv_path):
    assert os.path.exists(csv_path)
    frames = pd.read_csv(csv_path, low_memory=False)

    return frames


def get_clf_method(method='LR'):
    if method == 'LR':
        fr_func = LogisticRegression(penalty='l2', random_state=None)
    elif method == 'RF':
        fr_func = RandomForestClassifier()
    elif method == 'DT':
        fr_func = DecisionTreeClassifier()
    elif method == 'SVM':
        fr_func = SVC()
    elif method == 'XGBoost':
        fr_func = XGBClassifier()
    elif method == 'LightGBM':
        fr_func = LGBMClassifier()
    elif method == 'CatBoost':
        fr_func = CatBoostClassifier()
    elif method == 'KNN':
        fr_func = KNeighborsClassifier()
    elif method == 'ET':
        fr_func = ExtraTreesClassifier()
    else:
        raise NotImplementedError('Not support method of {}'.format(method))

    return fr_func


if __name__ == '__main__':
    # csv_path = 'LTRV-Qtrap_method_ginseng.csv'
    # csv_path = '/Users/shayuyang/Desktop/B3/MPU/paper/03-Processing/中药-人参-论文-2023/P-1/Panax-1/script/sczj_0412/LTRV-zj-0415-train.csv'
    # csv_path = '/Users/shayuyang/Desktop/B3/MPU/paper/03-Processing/中药-人参-论文-2023/P-2/code/dataset/panax_0412_select.csv'
    # csv_path = '/Users/shayuyang/Desktop/B3/MPU/paper/03-Processing/中药-人参-论文-2023/P-2/code/dataset/zj_0415_select.csv'
    # csv_path = '/Users/shayuyang/Desktop/B3/MPU/paper/03-Processing/中药-人参-论文-2023/P-2/dataset/0424/select_feature_file_gs_new/F_09.csv'
    # csv_path = '/Users/shayuyang/Desktop/B3/MPU/project/2024/Pan-0507/data/train_data_01.csv'
    # csv_path = '/Users/shayuyang/Desktop/B3/MPU/project/2024/Pan-0507/data/label_train_data_0515.csv'
    # csv_path = '/Users/shayuyang/Desktop/B3/MPU/project/2024/Pan-0507/data/label_train_data_0517.csv'
    # csv_path = '/Users/shayuyang/Desktop/B3/MPU/project/2024/Pan-0507/Timeline/0617/Train-0617.csv'
    # csv_path = '/Users/shayuyang/Desktop/B3/MPU/project/2024/Pan-0507/Timeline/0617/Train-select-0617.csv'
    csv_path = '/Users/shayuyang/Desktop/B3/MPU/project/2024/Pan-0507/Timeline/0617/Train-select-0617-75.csv'

    assert os.path.exists(csv_path)
    frames = pd.read_csv(csv_path)
    X_cont = frames.values[:, 1:]
    Y = frames.values[:, 0].astype(np.int64)

    scaler = StandardScaler()
    X_cont = scaler.fit_transform(X_cont)
    X_cont: np.ndarray = X_cont.astype(np.float32)
    # n_cont_features = X_cont.shape[1]
    all_idx = np.arange(len(Y))

    trainval_idx, test_idx = sklearn.model_selection.train_test_split(
        all_idx, train_size=0.8
    )

    # method list : LR, RF, DT, SVM, XGBoost, LightGBM, CatBoost, KNN
    fr_func = get_clf_method(method='KNN')
    fr_func.fit(X_cont[trainval_idx], Y[trainval_idx])
    # fr_func.fit(X_cont, Y)
    y_pred = fr_func.predict(X_cont[test_idx])

    y_true = Y[test_idx]

    # print(X_cont[test_idx].shape, y_pred.shape)

    score_acc = accuracy_score(y_true, y_pred)
    score_prec = precision_score(y_true, y_pred, average='macro')
    score_rec = recall_score(y_true, y_pred, average='macro')
    score_f1 = f1_score(y_true, y_pred, average='macro')

    print('Acc: {}, F1: {}, P: {}, R: {}'.format(score_acc, score_f1, score_prec, score_rec))

    # # 因为我们使用的catboost的树集成算法，所以我们用shap.explainers.Tree
    # # explainer = shap.explainers.Tree(fr_func)
    # # explainer = shap.TreeExplainer(fr_func)
    # # shap_values = explainer.shap_values(X_cont)
    # explainer = shap.Explainer(fr_func, X_cont)
    #
    # shap_values = explainer(X_cont)
    # # shap_values = np.sum(shap_values, axis=-1)
    # print(shap_values.data.shape)
    # # shap_interaction_values = explainer.shap_interaction_values(X_cont)
    # # print(X_cont[test_idx].shape)
    # # print(shap_values.shape, shap_interaction_values.shape)
    #
    # shap.initjs()
    # # shap.plots.beeswarm(shap_values, max_display=20)
    #
    # # shap.summary_plot(shap_values, X_cont)
    # # shap.summary_plot(shap_values[1], X_cont[test_idx])
    # # print(shap_values.shape)
    # # print(explainer.expected_value)
    #
    # shap.plots.beeswarm(shap_values[:, :, 0], max_display=15)
    # # shap.plots.bar(shap_values)
    #
    # # # for idx in range(10000):
    # # tic = time.time()
    # # gen_inps = np.random.randn(10000, 1, 202)
    # # for inp in gen_inps:
    # #
    # #     pred = fr_func.predict(inp)
    # #
    # # toc = time.time()
    # #
    # # print((toc - tic) / 10000)
