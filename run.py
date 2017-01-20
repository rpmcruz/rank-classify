from rank import RankSVM
from threshold import Threshold
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import numpy as np
import argparse
import sys
import os


def get_args():
    parser = argparse.ArgumentParser(
                        description="Rank-based classification.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', metavar="TR", nargs='?',
                        default="", help='Training data')
    parser.add_argument('--test', metavar="T", nargs='?',
                        default="", help='Test data')
    parser.add_argument('--cv', metavar="CV", nargs='?',
                        default=0, help='Use cross-validation (0-1)')
    parser.add_argument('--strategy', metavar="S", nargs='?',
                        default='inverse',
                        help='Thresholding strategy (uniform, '
                        'inverse, absolute)')
    parser.add_argument('--model', metavar="M", nargs='?',
                        default='rank',
                        help='Classifier (rank, svm, svm-balanced)')
    parser.add_argument('--coarse', metavar="C", type=float, nargs='+',
                        default=None,
                        help='Inter-class Threshold')
    return parser.parse_args()


def to_coarse(y, coarse_intervals):
    if coarse_intervals is None:
        return y

    coarse_intervals = sorted(coarse_intervals, reverse=True)
    yret = np.copy(y).astype(np.int)
    n = len(coarse_intervals)
    prev_v = np.inf
    for vi, v in enumerate(coarse_intervals + [-np.inf]):
        yret[(y >= v) & (y < prev_v)] = n - vi
        prev_v = v

    return yret


args = get_args()

tr_filename = args.train
ts_filename = args.test

try:
    Xtr = np.loadtxt(tr_filename)
    Xts = np.loadtxt(ts_filename)
except Exception as err:
    print(err)
    sys.exit(2)

if args.model == 'svm':
    model = LinearSVC(
        penalty='l1', tol=1e-3, dual=False)
    cv_param = 'C'
elif args.model == 'svm-balanced':
    model = LinearSVC(
        class_weight='balanced', penalty='l1', tol=1e-3, dual=False)
    cv_param = 'C'
elif args.model == 'rank':
    model = Threshold(RankSVM(), args.strategy)
    cv_param = 'model__model__C'


if args.cv:
    model = GridSearchCV(
        model, {cv_param: np.logspace(-3, 3, 7)}, 'neg_mean_absolute_error',
        verbose=2, n_jobs=-1)

pp_filename = os.path.basename(ts_filename) + '-%s-pp.txt' % args.model
yp_filename = os.path.basename(ts_filename) + '-%s-yp.txt' % args.model

ytr = Xtr[:, -1].astype(int)
min_y = np.min(ytr)
ytr -=  min_y
coarse_ytr = to_coarse(ytr, args.coarse)

yts = Xts[:, -1].astype(int)
yts -= min_y
coarse_yts = to_coarse(yts, args.coarse)

Xtr = Xtr[:, :-1]
Xts = Xts[:, :-1]

if args.coarse is None:
    model.fit(Xtr, coarse_ytr)
else:
    model.fit(Xtr, coarse_ytr, ytr)

pp = model.predict_proba(Xts)
yp = model.predict(Xts)

# add back 1 to turn back to [1,k]
yp += min_y

np.savetxt(yp_filename, yp, '%d')
np.savetxt(pp_filename, pp)

if args.cv:
    import pandas as pd
    results = model.cv_results_
    results = {k: v for k, v in results.items() if k in (
        'params', 'mean_test_score', 'std_test_score', 'mean_fit_time')}
    print(pd.DataFrame(results))
