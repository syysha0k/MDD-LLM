import os
import tqdm
import time
import pandas as pd
import numpy as np
from skfeature.function.similarity_based import fisher_score, reliefF, lap_score, trace_ratio, SPEC
from skfeature.function.sparse_learning_based import ll_l21, ls_l21, MCFS, NDFS, RFS, UDFS
from skfeature.function.information_theoretical_based import CIFE, CMIM, DISR, FCBF, ICAP, JMI, MIFS, MIM, MRMR
from skfeature.function.statistical_based import CFS, chi_square, f_score, gini_index, t_score
from skfeature.function.streaming import alpha_investing
from skfeature.function.wrapper import decision_tree_forward, decision_tree_backward
from skfeature.function.wrapper import svm_forward, svm_backward
from skfeature.utility.sparse_learning import feature_ranking, construct_label_matrix_pan
from skfeature.utility import construct_W
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from threading import Thread


def split_dataset(x_train, y_train, n_split=10):
    idx_dise = np.where(y_train == 1)[0]
    idx_norm = np.where(y_train == 0)[0]

    num_dise = len(idx_dise)
    num_norm = len(idx_norm)
    data_mapping = {}

    for idx in range(n_split):
        sel_dise = np.random.choice(idx_dise, int(num_dise * (1 / n_split)), replace=False)
        sel_norm = np.random.choice(idx_norm, int(num_norm * (1 / n_split)), replace=False)

        sel_dise_x_feature = x_train[sel_dise, :]
        sel_dise_y_label = y_train[sel_dise]
        sel_norm_x_feature = x_train[sel_norm, :]
        sel_norm_y_label = y_train[sel_norm]

        sel_x_train = np.vstack((sel_dise_x_feature, sel_norm_x_feature))
        sel_y_train = np.hstack((sel_dise_y_label, sel_norm_y_label))

        data_mapping[idx] = [sel_x_train, sel_y_train]

    return data_mapping


def load_csv(csv_path):
    assert os.path.exists(csv_path)
    frames = pd.read_csv(csv_path, low_memory=False)

    return frames


def feature_selection(x_train, y_train, selector='fisher_score', topk=50, save_folder=None, flag=None):
    tic = time.time()
    if selector == 'fisher_score':  # 1
        idx = fisher_score.fisher_score(x_train, y_train, mode='rank')
    elif selector == 'reliefF':  # 2
        idx = reliefF.reliefF(x_train, y_train, mode='rank')
    elif selector == 'lap_score':  # 3
        idx = lap_score.lap_score(x_train, y_train, mode='rank')
    elif selector == 'trace_ratio':  # 4
        idx = trace_ratio.trace_ratio(x_train, y_train, mode='rank')
    elif selector == 'SPEC':  # 5
        idx = SPEC.spec(x_train, y_train, mode='rank')
    elif selector == 'CIFE':  # 6
        idx = CIFE.cife(x_train, y_train, mode='rank')
    elif selector == 'CMIM':  # 7
        idx = CMIM.cmim(x_train, y_train, mode='rank')
    elif selector == 'DISR':  # 8
        idx = DISR.disr(x_train, y_train, mode='rank')
    elif selector == 'FCBF':  # 9
        idx = FCBF.fcbf(x_train, y_train, mode='rank')
    elif selector == 'ICAP':  # 10
        idx = ICAP.icap(x_train, y_train, mode='rank')
    elif selector == 'JMI':  # 11
        idx = JMI.jmi(x_train, y_train, mode='rank')
    elif selector == 'MIFS':  # 12
        idx = MIFS.mifs(x_train, y_train, mode='rank')
    elif selector == 'MIM':  # 13
        idx = MIM.mim(x_train, y_train, mode='rank')
    elif selector == 'MRMR':  # 14
        idx = MRMR.mrmr(x_train, y_train, mode='rank')
    elif selector == 'CFS':  # 15
        idx = CFS.cfs(x_train, y_train, mode='rank')
    elif selector == 'chi_square':  # 16
        idx = chi_square.chi_square(x_train, y_train, mode='rank')
        idx = idx.astype(int)
    elif selector == 'f_score':  # 17
        idx = f_score.f_score(x_train, y_train, mode='rank')
        idx = idx.astype(int)
    elif selector == 'gini_index':  # 18
        y_train = y_train.astype(int)
        idx = gini_index.gini_index(x_train, y_train, mode='rank')
    elif selector == 't_score':  # 19
        idx = t_score.t_score(x_train, y_train, mode='rank')
    elif selector == 'alpha_investing':  # 20
        idx = alpha_investing.alpha_investing(x_train, y_train, 0.5, 0.5)
    elif selector == 'decision_tree_forward':  # 21
        idx = decision_tree_forward.decision_tree_forward(x_train, y_train, mode='rank')
    elif selector == 'decision_tree_backward':  # 22
        idx = decision_tree_backward.decision_tree_backward(x_train, y_train, mode='rank')
    elif selector == 'svm_forward':  # 23
        idx = svm_forward.svm_forward(x_train, y_train, mode='rank')
    elif selector == 'svm_backward':  # 24
        idx = svm_backward.svm_backward(x_train, y_train, mode='rank')
    elif selector == 'll_l21':  # 25
        # y_train = construct_label_matrix_pan(y_train)
        idx = ll_l21.proximal_gradient_descent(x_train, y_train, 0.1, mode='rank')
    elif selector == 'ls_l21':  # 26
        idx = ls_l21.proximal_gradient_descent(x_train, y_train, 0.1, mode='rank')
    elif selector == 'MCFS':  # 27
        idx = MCFS.mcfs(x_train, mode='rank', n_clusters=2)
    elif selector == 'NDFS':  # 28
        idx = NDFS.ndfs(x_train, mode='rank', n_clusters=2)
    elif selector == 'RFS':  # 29
        idx = RFS.rfs(x_train, y_train, mode='rank')
    elif selector == 'UDFS':  # 30
        idx = UDFS.udfs(x_train, mode='rank', gamma=0.1, n_clusters=2)

    else:
        raise NotImplementedError('Not support selector of {}.'.format(selector))
    toc = time.time()

    print('==> Feature selection with {}'.format(selector))
    print('==> The processing time is {} @ {}'.format(toc - tic, flag))

    select_feature = []
    for item in idx[:topk]:
        select_feature.append(feature_name[item])

    if save_folder is not None:
        if flag is None:
            save_path = os.path.join(save_folder, selector + '.txt')
        else:
            save_path = os.path.join(save_folder, selector + '{}.txt'.format(str(flag)))

        rec_txt = open(save_path, 'w')
        save_line = ','.join(select_feature)
        rec_txt.write(save_line)
        rec_txt.write('\n')
        use_time = 'Method: {} \nUse Time: {}'.format(selector, toc - tic)
        rec_txt.write(use_time)
        rec_txt.write('\n')
        rec_txt.close()

    return select_feature, idx, feature_name


if __name__ == '__main__':
    csv_path = './Training_Test_20211030.csv'
    save_folder = './Selector_feature'

    KF = True
    MPL = True
    # select_feature, idx, feature_name = feature_selection(csv_path, selector='chi_square',
    #                                                       topk=50, save_folder=save_folder)
    #
    # selectors_1 = ['fisher_score', 'reliefF', 'lap_score', 'trace_ratio']
    # selectors_2 = ['CIFE', 'CMIM', 'DISR', 'FCBF', 'ICAP', 'JMI', 'MIFS', 'MIM', 'MRMR']
    # selectors_3 = ['CFS', 'chi_square', 'f_score', 'gini_index', 't_score']
    # selectors_4 = ['alpha_investing']
    # selectors_5 = ['decision_tree_forward', 'decision_tree_backward', 'svm_forward', 'svm_backward']
    # selectors_6 = ['ll_l21', 'ls_l21', 'MCFS', 'RFS', 'UDFS']

    frames = load_csv(csv_path)
    selectors = ['fisher_score']

    frames = frames.dropna(axis=1)
    feature_name = frames.columns.tolist()[1:]
    y_train = frames.values[:, 0].astype(int)
    x_train = frames.values[:, 1:]

    split_dataset(x_train, y_train)

    for selector in selectors:
        if KF is False:
            select_feature, idx, feature_name = feature_selection(x_train, y_train, selector=selector,
                                                                  topk=50, save_folder=save_folder, flag=None)
        elif KF is True and MPL is False:
            feature_mapping = split_dataset(x_train, y_train, n_split=10)
            for k, v in feature_mapping.items():
                x_train, y_train = v
                select_feature, idx, feature_name = feature_selection(x_train, y_train,
                                                                      selector, topk=50,
                                                                      save_folder=save_folder,
                                                                      flag=k)
        elif KF and MPL:
            feature_mapping = split_dataset(x_train, y_train, n_split=10)

            for idx in range(2):
                x_train_0, y_train_0 = feature_mapping.get(5 * idx + 0)
                x_train_1, y_train_1 = feature_mapping.get(5 * idx + 1)
                x_train_2, y_train_2 = feature_mapping.get(5 * idx + 2)
                x_train_3, y_train_3 = feature_mapping.get(5 * idx + 3)
                x_train_4, y_train_4 = feature_mapping.get(5 * idx + 4)

                t0 = Thread(target=feature_selection,
                            args=(x_train_0, y_train_0, selector, 50, save_folder, 5 * idx + 0))
                t1 = Thread(target=feature_selection,
                            args=(x_train_1, y_train_1, selector, 50, save_folder, 5 * idx + 1))
                t2 = Thread(target=feature_selection,
                            args=(x_train_2, y_train_2, selector, 50, save_folder, 5 * idx + 2))
                t3 = Thread(target=feature_selection,
                            args=(x_train_3, y_train_3, selector, 50, save_folder, 5 * idx + 3))
                t4 = Thread(target=feature_selection,
                            args=(x_train_4, y_train_4, selector, 50, save_folder, 5 * idx + 4))

                t0.start()
                t1.start()
                t2.start()
                t3.start()
                t4.start()

                t0.join()
                t1.join()
                t2.join()
                t3.join()
                t4.join()

