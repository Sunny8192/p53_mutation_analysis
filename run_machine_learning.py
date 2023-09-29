import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score, recall_score, precision_score, roc_auc_score

script_path = os.path.abspath(os.path.dirname(__file__))

# helper function
def cal_performance(actual, predicted):
    bacc = round(balanced_accuracy_score(y_true=actual, y_pred=predicted), 3)
    f1 = round(f1_score(y_true=actual, y_pred=predicted, pos_label=np.unique(actual)[0]), 3)
    mcc = round(matthews_corrcoef(y_true=actual, y_pred=predicted), 3)
    recall = round(recall_score(y_true=actual, y_pred=predicted, pos_label=np.unique(actual)[0]), 3)
    precision = round(precision_score(y_true=actual, y_pred=predicted, pos_label=np.unique(actual)[0]), 3)
    return (bacc, f1, mcc, recall, precision)

def cal_roc_auc(actual, predicted_proba):
    return round(roc_auc_score(y_true=actual, y_score=predicted_proba), 3)

def run_stratified_group_cross_validation(train_csv, algorithm):
    # load data
    df_train = pd.read_csv(train_csv)

    # preprocessing
    remove_cols = ["ID", "wt", "pos", "mt", "mutation", "label"]
    group_col_name = "pos"

    group_column = df_train[group_col_name]
    train_labels = df_train["label"]

    le = preprocessing.LabelEncoder()
    le.fit(train_labels)
    train_labels = le.transform(train_labels)

    train_features = df_train.drop(remove_cols, axis=1)

    # 10-fold Cross Validation - group by "residue_index" to remove redundancy
    kf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=1)

    df_cv_results = pd.DataFrame(columns=["fold", "actual", "predicted", "predicted_proba"])
    fold = 1
    for train, test in kf.split(train_features, train_labels, groups=group_column):

        train_cv_features, test_cv_features, train_cv_labels, test_cv_labels = train_features.iloc[train], train_features.iloc[test], train_labels[train], train_labels[test]
        
        train_cv_features = train_cv_features.values
        test_cv_features = test_cv_features.values

        if algorithm == "XGBOOST":
            cv_clf = XGBClassifier(n_estimators=300, random_state=1, use_label_encoder=False, n_jobs=8)
        elif algorithm == "GB":
            cv_clf = GradientBoostingClassifier(n_estimators=300, random_state=1)
        else:
            print("Sorry, our models were trained on XGBOOST or GB only")
            return False
        
        cv_clf.fit(train_cv_features, train_cv_labels)
        cv_prediction = cv_clf.predict(test_cv_features)
        cv_proba = cv_clf.predict_proba(test_cv_features)

        cv_prediction = le.inverse_transform(cv_prediction)
        test_cv_labels = le.inverse_transform(test_cv_labels)

        fold_results = pd.DataFrame({'fold':fold, 'actual':test_cv_labels, 'predicted':cv_prediction, 'predicted_proba':np.round(cv_proba[:, 1], 5)})
        df_cv_results = pd.concat([df_cv_results, fold_results], axis=0)
        fold += 1
    
    bacc, f1, mcc, recall, precision = cal_performance(df_cv_results["actual"], df_cv_results["predicted"])
    auroc = cal_roc_auc(df_cv_results["actual"], df_cv_results["predicted_proba"])

    # output performance
    print("## --------- Performance ---------")
    print("{:<10}: {:.3f}".format("bacc", bacc))
    print("{:<10}: {:.3f}".format("f1-score", f1))
    print("{:<10}: {:.3f}".format("mcc", mcc))
    print("{:<10}: {:.3f}".format("recall", recall))
    print("{:<10}: {:.3f}".format("precision", precision))
    print("{:<10}: {:.3f}".format("auroc", auroc))


    return True

def run_train_test(train_csv, test_csv, algorithm):

    # load data
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # preprocessing
    remove_cols = ["ID", "wt", "pos", "mt", "mutation", "label"]
    train_labels = df_train["label"]
    test_labels = df_test["label"]

    le = preprocessing.LabelEncoder()
    le.fit(train_labels)

    train_labels = le.transform(train_labels)
    test_labels = le.transform(test_labels)

    train_features = df_train.drop(remove_cols, axis=1)
    train_features = train_features.drop(["splitting"], axis=1)
    test_features = df_test.drop(remove_cols, axis=1)

    train_features = train_features.values
    test_features = test_features.values

    if algorithm == "XGBOOST":
        clf = XGBClassifier(n_estimators=300, random_state=1, use_label_encoder=False, n_jobs=8)
    elif algorithm == "GB":
        clf = GradientBoostingClassifier(n_estimators=300, random_state=1)
    else:
        print("Sorry, our models were trained on XGBOOST or GB only")
        return False

    # train - test
    clf.fit(train_features, train_labels)
    prediction_labels = clf.predict(test_features)
    prediction_proba = clf.predict_proba(test_features)[:, 1]

    prediction_labels = le.inverse_transform(prediction_labels)
    test_labels = le.inverse_transform(test_labels)

    bacc, f1, mcc, recall, precision = cal_performance(test_labels, prediction_labels)
    auroc = cal_roc_auc(test_labels, prediction_proba)

    # output performance
    print("## --------- Performance ---------")
    print("{:<10}: {:.3f}".format("bacc", bacc))
    print("{:<10}: {:.3f}".format("f1-score", f1))
    print("{:<10}: {:.3f}".format("mcc", mcc))
    print("{:<10}: {:.3f}".format("recall", recall))
    print("{:<10}: {:.3f}".format("precision", precision))
    print("{:<10}: {:.3f}".format("auroc", auroc))

    return True


# ExpAssay

"""# 1. data splitting
exp_train_csv = os.path.join(script_path, "ExpAssay", "expassay_train.csv")
exp_blind_csv = os.path.join(script_path, "ExpAssay", "expassay_blind.csv")
exp_base_csv = os.path.join(script_path, "ExpAssay", "expassay_base.csv")

# 2. run machine learning - XGBOOST 
print("## Running Analysis on the ExpAssay Model \n")
print("## Running 10-fold Cross Validation on the whole dataset \n")
run_stratified_group_cross_validation(exp_base_csv, "XGBOOST")
print()
print("## Running blind test \n")
run_train_test(exp_train_csv, exp_blind_csv, "XGBOOST")
print()
print("## End on the ExpAssay Model")
"""

print("\n" + "---" * 10 + "\n")

# noExpAssay

"""# 1. data splitting
noexp_train_csv = os.path.join(script_path, "noExpAssay", "noexpassay_train.csv")
noexp_blind_csv = os.path.join(script_path, "noExpAssay", "noexpassay_blind.csv")
noexp_base_csv = os.path.join(script_path, "noExpAssay", "noexpassay_base.csv")

# 2. run machine learning - GradientBoosting
print("## Running Analysis on the noExpAssay Model \n")
print("## Running 10-fold Cross Validation on the whole dataset \n")
run_stratified_group_cross_validation(noexp_base_csv, "GB")
print()
print("## Running blind test \n")
run_train_test(noexp_train_csv, noexp_blind_csv, "GB")
print()
print("## End on the noExpAssay Model")
"""











