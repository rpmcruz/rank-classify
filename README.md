# Pairwise Scoring Rank -> Classify

This is based on a couple of articles we written (to appear).

The ideia is we train with a pairwise scoring rank (based on ranksvm) using an ordinal dataset. From these we get ranking scores for new observations (from -infinity to +infinity) and we then find a threshold rule to convert those prediction back to classes.

**Model:** Data -> Pairwise Differences Data -> SVM

**Prediction:** SVM -> Ranking Score -> Threshold -> Class

## Usage

    python3 run.py train.txt test.txt [OPTIONS]

Options are:
- **--strategy=xxx** where *xxx* is the threshold strategy. Options are: uniform, inverse and absolute. (default is uniform)
- **--cv** to enable cross-validation of C (from 10^-3 to 10^3)
- **--svm** to use ordinary svm (not our ranking)
- **--svm-balanced** to use ordinary svm with inverse frequencies as the cost matrix

## Requirements

- scikit-learn (and all its dependencies)
- pandas (optional) to print cross-validation results
