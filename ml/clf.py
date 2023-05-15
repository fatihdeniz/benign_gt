'''
Description: ML end-to-end pipeline
Author: nabeelxy
'''
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, train_test_split
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import matplotlib.pyplot as plt
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import pickle
from collections import Counter
# from imblearn.over_sampling import RandomOverSampler, SMOTE
from numpy import interp

#save data structure to a file
def save_ds(ds, filename):
    pickle.dump(ds, open(filename, 'wb'))

#load the data structure from file
def load_ds(filename):
    return pickle.load(open(filename, 'rb'))

def plot_importance(clf, features, prefix = None, top_n = None):
    imp = list(clf.feature_importances_)
    df_imp = None
    #aggregate feature importance for features starting with the same prefix
    if prefix != None:
        imp2 = dict()
        for i in range(len(imp)):
            is_prefix = False
            for p in prefix:
                if features[i].startswith(p):
                    is_prefix = True
                    if p not in imp2:
                        imp2[p] = 0
                    imp2[p] += imp[i]
                    break
            if is_prefix == False:
                imp2[features[i]] = imp[i]

        df_imp = pd.DataFrame(imp2.items(), columns = ["features", "imp"]).set_index("features").sort_values(by="imp", ascending = False)
    else:
        df_imp = pd.DataFrame({"imp": imp},
                       index = features).sort_values(by="imp", ascending = False)
    if top_n != None and top_n > 0 and top_n < df_imp.shape[0]:
        df_imp = df_imp.head(top_n)
    df_imp.plot(kind="bar", legend = False)

#this method expects the default confusion matrix
#assumes that there are 0 and 1 classes (your true class is 1)
#confusion matrix has the following format:
#              Predicted
#             0   |   1
#       -----------------
#       0  | TN   | FP  |
#Actual -----------------
#       1  | FN   | TP  |
#       -----------------
#
def compute_metrics(cm):
    FN = cm[1][0]
    TN = cm[0][0]
    FP = cm[0][1]
    TP = cm[1][1]
    acc = (TP + TN) /(TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return [acc, precision, recall, FPR]

#this method computes the metrics for each class
#works for both binary as well as multi-class classifiers
def compute_metrics_multi(cm):
    n_classes = cm.shape[0]

    results = [None] * n_classes

    for i in range(n_classes):
        TP = cm[i][i]
        FN = np.sum(cm, axis = 0)[i] - TP
        FP = np.sum(cm, axis = 1)[i] - TP
        TN = np.sum(cm) - TP - FN - FP
        acc = (TP + TN) /(TP + FP + TN + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        FPR = FP / (FP + TN)
        results[i] = [acc, precision, recall, FPR]
    return results

def encode_training_cols(df, cols):
    encoders = dict()
    for col in cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        #le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        #print(le_dict)
        encoders[col] = le
    #save_ds(encoders, "../data/classifier/model/encoders")
    return df, encoders

def encode_testing_cols(df, encoders):
    dicts = dict()
    for col, encoder in encoders.items():
        dicts[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    #set unknown values to -1
    for col, d in dicts.items():
        df[col] = df[col].apply(lambda x: dicts[col][x] if x in dicts[col] else -1)
    return df

def scale_testing(df, scaler):
    X = scaler.transform(df)
    X = pd.DataFrame(X, columns=df.columns)
    return X

def my_train_test_split(filename, test_size = 0.2):
    df = pd.read_csv(filename)
    num_cols = df.shape[1] - 1
    features = df.columns[1:num_cols]
    X = df[features]
    y = df.iloc[:, num_cols]
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = test_size, random_state = 100)
    return X_train, X_test, y_train, y_test, features

# handling new data
# https://stackoverflow.com/questions/59575492/how-to-rescale-new-data-base-on-old-minmaxscale

def prepare_training_data(X, features, cat_cols):
    if isinstance(X, pd.DataFrame):
        df = X
    else:
        df = pd.DataFrame(X, columns = features)
    df.fillna(-1, inplace = True)
    df, encoders = encode_training_cols(df, cat_cols)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(df)
    #save_ds(scaler, "../data/classifier/model/transformer")
    X = pd.DataFrame(X, columns=features)

    return X, encoders, scaler

def prepare_testing_data(X, features, encoders, scaler):
    df = pd.DataFrame(X, columns = features)
    df.fillna(-1, inplace = True)
    df = encode_testing_cols(df, encoders)
    X = scale_testing(df, scaler)
    return X

def hyperparameter_tuning_grid(X, y, fold):
    y = LabelEncoder().fit_transform(y)
    cv = RepeatedStratifiedKFold(n_splits = fold, n_repeats=3, random_state=1)
    model = RandomForestClassifier(random_state=22)
    parameters = {
        "max_depth": list(range(10, 500, 20)) + [None], 
        "n_estimators":  range(10, 250, 10), 
        "max_features": range(1, X.shape[1]+1),
        "bootstrap": [True, False],
        "min_samples_leaf": range(2, 5) 
    }
    n_estimators = [10, 100, 1000]
    # define grid search
    grid_search = GridSearchCV(estimator = model, 
                               param_grid = parameters, 
                               n_jobs = -1, 
                               cv = cv, 
                               scoring = 'accuracy', #['accuracy', 'average_precision', 'f1_macro'],
                               #refit = 'accuracy',
                               error_score = 0)
    grid_result = grid_search.fit(X, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #means = grid_result.cv_results_['mean_test_score']
    #stds = grid_result.cv_results_['std_test_score']
    #params = grid_result.cv_results_['params']

    #for mean, stdev, param in zip(means, stds, params):
    #    print("%f (%f) with: %r" % (mean, stdev, param))
    
    #print("\n The best parameters across ALL searched params:\n",
    #  grid_result.best_params_)

    return grid_result.best_estimator_

def hyperparameter_tuning_random(X, y, fold):
    y = LabelEncoder().fit_transform(y)
    model = RandomForestClassifier(random_state=22)
    cv = RepeatedStratifiedKFold(n_splits = fold, n_repeats=3, random_state=1)
    parameters = {
        "max_depth": list(range(10, 500, 20)) + [None], 
        "n_estimators":  range(10, 250, 10), 
        "max_features": sp_randInt(1, X.shape[1]),
        "bootstrap": [True, False],
        "min_samples_leaf": sp_randInt(2, 4) 
    }

    rscv = RandomizedSearchCV(estimator=model, param_distributions = parameters, cv = cv, n_iter = 50, n_jobs=-1)
    rscv.fit(X, y)

    # Results from Random Search
    print("\n========================================================")
    print(" Results from Random Search " )
    print("========================================================")
    print("\n The best estimator across ALL searched params:\n",
      rscv.best_estimator_)
    print("\n The best score across ALL searched params:\n",
      rscv.best_score_)
    print("\n The best parameters across ALL searched params:\n",
      rscv.best_params_)
    print("\n ========================================================")
    return rscv.best_estimator_


# +
def oversample_min(X, y):
    oversample = RandomOverSampler(sampling_strategy='minority')
    X, y = oversample.fit_resample(X, y)
    return X, y

def oversample_multi(X, y):
    oversample = SMOTE()
    return oversample.fit_resample(X, y)


# -

def validation_run(clf, X, y, splits = 5):
    #cv = StratifiedKFold(n_splits= splits,shuffle=False)
    cv = StratifiedKFold(n_splits= splits,shuffle=True)


    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_dummies = pd.get_dummies(y, drop_first = False)
    y_dummies_values = y_dummies.values
    n_classes = len(y_dummies.columns)

    figs = [None] * n_classes
    axes = [None] * n_classes
    n_tprs = [None] * n_classes
    n_aucs = [None] * n_classes
    n_metrics = [None] * n_classes
    n_prev_auc = [None] * n_classes
    n_best_model = [None] * n_classes

    le = LabelEncoder()
    y = le.fit_transform(y)
    y_dict = dict(zip(le.classes_, le.transform(le.classes_)))

    for i in range(n_classes):
        figs[i] = plt.figure(figsize=[8,8])
        axes[i] = figs[i].add_subplot(111,aspect = 'equal')
        n_tprs[i] = []
        n_aucs[i] = []
        n_prev_auc[i] = None
        n_best_model[i] = None

    metrics = []
    mean_fpr = np.linspace(0,1,100)
    i = 1
    for train,test in cv.split(X,y):
        model = clf.fit(X.iloc[train],y[train])
        prediction = model.predict_proba(X.iloc[test])
        y_dummies_values = pd.get_dummies(y[test], drop_first = False).values
        
        for n in range(n_classes):
            fpr, tpr, t = roc_curve(y_dummies_values[:, n], prediction[:, n])
            n_tprs[n].append(interp(mean_fpr, fpr, tpr))
            roc_auc = auc(fpr, tpr)
            n_aucs[n].append(roc_auc)
        
            if n_best_model[n] == None:
                n_best_model[n] = model
                n_prev_auc[n] = roc_auc
            else:
                if n_prev_auc[n] < roc_auc:
                    n_prev_auc[n] = roc_auc
                    n_best_model[n] = model
            axes[n].plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i= i+1
    
        prediction2 = model.predict(X.iloc[test])
        cm = confusion_matrix(y[test], prediction2)
        metrics.append(compute_metrics_multi(cm))
 
    y_dict_r = dict()
    for c, n in y_dict.items():
        y_dict_r[n] = c 

    for n in range(n_classes):
        metric_class = []
        for metric in metrics:
            metric_class.append(metric[n]) 
        df_metrics = pd.DataFrame(metric_class)
        df_metrics.columns = ["acc", "precision", "recall", "fpr"]
        print("Class: {} Accuracy: {} Precision: {} Recall: {} FPR: {}".format(y_dict_r[n],
                                                             df_metrics["acc"].mean(),
                                                             df_metrics["precision"].mean(),
                                                             df_metrics["recall"].mean(),
                                                             df_metrics["fpr"].mean()))

    for n in range(n_classes):
        axes[n].plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
        mean_tpr = np.mean(n_tprs[n], axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        axes[n].plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)
        axes[n].set_xlabel('False Positive Rate')
        axes[n].set_ylabel('True Positive Rate')
        axes[n].set_title('ROC {}'.format(y_dummies.columns[n]))
        axes[n].legend(loc="lower right")

    plt.show()


def testing_run(clf, X, y, features, encoders=None, scalar=None):
    # X_bar = prepare_testing_data(X, features, encoders, scalar)
    y_dummies_values = pd.get_dummies(y, drop_first = False).values

    le = LabelEncoder()
    y = le.fit_transform(y)
    y_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    y_dict_r = dict()
    n_classes = len(y_dict)
    for c, n in y_dict.items():
        y_dict_r[n] = c

    figs = [None] * n_classes
    axes = [None] * n_classes

    
    X_bar = X
    
    prediction = clf.predict_proba(X_bar)
    prediction2 = clf.predict(X_bar)

    for i in range(n_classes):
        figs[i] = plt.figure(figsize=[8,8])
        axes[i] = figs[i].add_subplot(111,aspect = 'equal')

    mean_fpr = np.linspace(0,1,100)

    for n in range(n_classes):
        fpr, tpr, t = roc_curve(y_dummies_values[:, n], prediction[:, n])
        roc_auc = auc(fpr, tpr)
        axes[n].plot(fpr, tpr, lw=2, alpha=0.3, label='(AUC = %0.2f)' % (roc_auc))

    cm = confusion_matrix(y, prediction2)
    metrics = compute_metrics_multi(cm)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_dict.keys()).plot()
    # print(classification_report(y, prediction2, target_names = y_dict.keys())) 
    for n in range(n_classes):
        metric = metrics[n]
        print("Class: {} Accuracy: {} Precision: {} Recall: {} FPR: {}".format(y_dict_r[n],
                                                                               metric[0],
                                                                               metric[1],
                                                                               metric[2],
                                                                               metric[3]))
    plt.show()


def final_model(clf, filename, cat_cols):
    df = pd.read_csv(filename)
    num_cols = df.shape[1] - 1
    features = df.columns[1:num_cols]
    X_train = df[features]
    y_train = df.iloc[:, num_cols]
    X, encoders, scaler = prepare_training_data(X_train, features, cat_cols)

    le = LabelEncoder()
    y = le.fit_transform(y_train)
    y_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    model = clf.fit(X, y)
    return model, encoders, scaler


from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn import model_selection

def spot_checking_classification(X, Y):
    kfold = model_selection.KFold(n_splits=5)
    # Spot Check Algorithms
    models = []
    models.append(('XGB', XGBClassifier(use_label_encoder=False)))
    models.append(('RF', RandomForestClassifier()))  
    models.append(('SVC', SVC())) 
    models.append(('KNN', KNeighborsClassifier())) 
    models.append(('DT', DecisionTreeClassifier())) 
    models.append(('NB', GaussianNB()))
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))

    # Evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        result = model_selection.cross_val_score(model, X, Y, cv=kfold)
        results.append(result)
        names.append(name)
    
    for i in range(len(results)):
        result = results[i]
        name = names[i]
        print('%s: %f (%f)' % (name, result.mean(), result.std()))
    
    
    # Compare Algorithms
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Spot Checking')
    plt.show()
