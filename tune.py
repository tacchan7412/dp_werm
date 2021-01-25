import itertools
import os
import sys
import uuid

import numpy as np
from sklearn.model_selection import train_test_split

from solver import Solver

from data import generate_synthetic_data
from data import generate_kdd99_big_data
from data import generate_rnaseq_data

from metric import auc_roc
from metric import f1score
from metric import balanced_acc

import time


def calc_auc_numerator(Y, pred_Y):
    res = 0
    for i in np.where(Y == 1)[0]:
        for j in np.where(Y == -1)[0]:
            if pred_Y[i] >= pred_Y[j]:
                res += 1
    return res


def no_private_tune_auc(X, Y, parameters, solver, seed):
    train_X, val_X, train_Y, val_Y = train_test_split(
        X, Y, test_size=0.3, random_state=seed)

    best_auc = 0
    best_w = None
    best_para = None
    stime = time.time()
    for para in parameters:
        w = solver.solve(train_X, train_Y, None, *para)
        if w is not None:
            scores = val_X @ w
            auc = auc_roc(scores, val_Y)
            if auc > best_auc:
                best_auc = auc
                best_w = w
                best_para = para
    print('collapsed time:', time.time() - stime)
    return best_w, best_para


def private_tune_auc(X, Y, eps, parameters, solver, seed):
    m = len(parameters)
    indices = range(len(Y))
    split_indices = np.array_split(
        np.random.RandomState(seed=seed).permutation(indices),
        m+1)
    eval_indices = split_indices[-1]
    val_X = X[eval_indices]
    val_Y = Y[eval_indices]
    zs = [0.] * m
    sens = len(val_Y)

    stime = time.time()
    ws = [None] * m
    for i, index in enumerate(split_indices[:-1]):
        para = parameters[i]
        w = solver.solve(X[index], Y[index], eps, *para)
        if w is not None:
            ws[i] = w
            scores = val_X @ w
            preds = 1 / (1 + np.exp(-scores))
            numerator = calc_auc_numerator(val_Y, preds)
            zs[i] = eps * numerator / (2 * sens)
    print('collapsed time:', time.time() - stime)
    i = np.random.choice(list(range(m)), p=softmax(zs))
    return ws[i], parameters[i]


def softmax(a):
    c = np.max(a)
    # prevent overflow
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


if __name__ == '__main__':
    args = sys.argv
    '''
    args[1]: loss function
    args[2]: solver name
    args[3]: dataset name
    args[4]: random seed for dataset
    args[5]: eps
    args[6]: ratio of positive labels (for synthetic)
    '''
    loss_func = args[1]
    solver_name = args[2]
    data = args[3]
    seed = int(args[4])
    eps = float(args[5])

    solver = Solver(loss_func, solver_name)

    if data == 'synthetic_dif':
        X, Y, test_X, test_Y =\
            generate_synthetic_data(50000, 10, float(args[6]))
        lams = [10**(-2*i) for i in range(5)]
        weights = [0.5, 0.1, 0.05, 0.01]
    elif data == 'kdd99_big':
        X, Y, test_X, test_Y = generate_kdd99_big_data()
        lams = [10**(-2*i) for i in range(5)]
        weights = [0.5, 0.25, 0.125, 0.0625]
    elif data == 'rnaseq':
        X, Y, test_X, test_Y = generate_rnaseq_data()
        lams = [10**-1, 10**-2, 10**-3]
        weights = [0.2, 0.1, 0.05]

    if solver_name == 'dp_sgd_gw':
        params = list(itertools.product(lams, [50], [5, 10], weights))
    elif solver_name == 'dp_sgd':
        params = list(itertools.product(lams, [50], [5, 10]))
    elif 'gw' in solver_name:
        params = list(itertools.product(lams, weights))
    else:
        params = list(itertools.product(lams))

    if 'no_privacy' in solver_name:
        w, para = no_private_tune_auc(X, Y, params, solver, seed)
    else:
        w, para = private_tune_auc(X, Y, eps, params, solver, seed)

    # save results on the test set
    scores = test_X @ w
    auc = auc_roc(scores, test_Y)
    f1 = f1score(scores, test_Y)
    bacc = balanced_acc(scores, test_Y)
    print(auc, f1, bacc)

    if 'synthetic' in data:
        data += '_' + args[6]
    save_dir = os.path.join('result/tune', loss_func, data, str(seed),
                            solver_name, str(eps))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, uuid.uuid4().hex + '.txt')
    with open(save_path, 'w') as f:
        f.write(' '.join([str(s) for s in list(para)]) + '\n')
        f.write(' '.join([str(auc), str(f1), str(bacc)]) + '\n')
