import sys
from rank import RankSVM
from threshold import Threshold
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import numpy as np
import os

if len(sys.argv) < 3:
    print('Usage: %s TR_FILENAME TS_FILENAME' % sys.argv[0])
    sys.exit(1)

tr_filename = sys.argv[1]
ts_filename = sys.argv[2]
try:
    Xtr = np.loadtxt(tr_filename)
    Xts = np.loadtxt(ts_filename)
except Exception as err:
    print(err)
    sys.exit(2)

CV = False
threshold_strategy = 'inverse'
use_model = 'rank'

# optional parameters
for arg in sys.argv[4:]:
    if arg == '--cv':
        CV = True
    elif arg.startswith('--strategy='):
        threshold_strategy = arg[12:]
    elif arg == '--svm':
        use_model = 'svm'
    elif arg == '--svm-balanced':
        use_model = 'svm-balanced'
    else:
        print('Unknown option: %s' % arg)
        sys.exit(3)

if use_model == 'svm':
    model = LinearSVC(
        penalty='l1', tol=1e-3, dual=False)
    cv_param = 'C'
elif use_model == 'svm-balanced':
    model = LinearSVC(
        class_weight='balanced', penalty='l1', tol=1e-3, dual=False)
    cv_param = 'C'
elif use_model == 'rank':
    model = Threshold(RankSVM(), threshold_strategy)
    cv_param = 'model__model__C'

if CV:
    model = GridSearchCV(
        model, {cv_param: np.logspace(-3, 3, 7)}, 'neg_mean_absolute_error',
        verbose=2, n_jobs=-1)

pp_filename = os.path.basename(ts_filename) + '-%s-pp.txt' % use_model
yp_filename = os.path.basename(ts_filename) + '-%s-yp.txt' % use_model

# NOTE: we subtract by 1 because our methods assume classes [0,k[
ytr = Xtr[:, -1].astype(int)-1
yts = Xts[:, -1].astype(int)-1

Xtr = Xtr[:, :-1]
Xts = Xts[:, :-1]

model.fit(Xtr, ytr)
pp = model.predict_proba(Xts)
yp = model.predict(Xts)

# add back 1 to turn back to [1,k]
yp += 1

np.savetxt(yp_filename, yp, '%d')
np.savetxt(pp_filename, pp)

if CV:
    import pandas as pd
    results = model.cv_results_
    results = {k: v for k, v in results.items() if k in (
        'params', 'mean_test_score', 'std_test_score', 'mean_fit_time')}
    print(pd.DataFrame(results))

