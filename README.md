# Pairwise Scoring Rank -> Classify

This is based on a couple of articles we written (to appear).

The ideia is we train with a pairwise scoring rank (based on ranksvm) using an ordinal dataset. From these we get ranking scores for new observations (from -infinity to +infinity) and we then find a threshold rule to convert those prediction back to classes.

Model: Data -> Pairwise Differences Data -> SVM

Prediction: SVM -> Ranking Score -> Threshold -> Class
