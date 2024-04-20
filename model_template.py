from xgboost import XGBClassifier

model = XGBClassifier(
    max_depth        = 9,
    n_estimators     = 100,
    learning_rate    = 0.4,
    min_child_weight = 1.4,
    gamma            = 0.5,
    subsample        = .85,
    colsample_bytree = 0.7,
    scale_pos_weight = 1.,
    reg_lambda       = 3,
    n_jobs           = 8,
    objective        = 'binary:logitraw',
    eval_metric      = ['logloss'],
)
