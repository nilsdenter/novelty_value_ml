from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, r2_score
from collections import defaultdict
statistics = defaultdict(list)

def generate_statistics(classifier, algorithm, model_pred_train, model_pred_test):
    statistics["Algorithm"].append(algorithm)
    try: statistics["ROC AUC mean validation"].append(round(classifier.best_score_,4))
    except: statistics["ROC AUC mean validation"].append(0.5)
    statistics["ROC AUC test"].append(round(roc_auc_score(y_test, model_pred_test),4))
    statistics["PPV test"].append(round(precision_score(y_test, model_pred_test, pos_label=1),4))
    statistics["TPR test"].append(round(recall_score(y_test, model_pred_test, pos_label=1),4))
    statistics["NPR test"].append(round(precision_score(y_test, model_pred_test, pos_label=0),4))
    statistics["TNR test"].append(round(recall_score(y_test, model_pred_test, pos_label=0),4))
    #statistics["F1 score on test data"].append(round(f1_score(y_test, model_pred_test),4))
    #statistics["MCC on test data"].append(round(matthews_corrcoef(y_test, model_pred_test),4))
    #statistics["Balanced accuracy on train data"].append(round(balanced_accuracy_score(y_train, model_pred_train),4))
    statistics["ROC AUC train"].append(round(roc_auc_score(y_train, model_pred_train),4))
    statistics["PPV train"].append(round(precision_score(y_train, model_pred_train, pos_label=1),4))
    statistics["TPR train"].append(round(recall_score(y_train, model_pred_train, pos_label=1),4))
    statistics["NPR train"].append(round(precision_score(y_train, model_pred_train, pos_label=0),4))
    statistics["TNR train"].append(round(recall_score(y_train, model_pred_train, pos_label=0),4))
    #statistics["F1 score on train data"].append(round(f1_score(y_train, model_pred_train),4))
    #statistics["MCC on train data"].append(round(matthews_corrcoef(y_train, model_pred_train),4))
    model_confusion_matrix = confusion_matrix(y_test, model_pred_test, labels = [1,0])
    model_confusion_matrix = pd.DataFrame(model_confusion_matrix, index= ["Actual True", "Actual False"], columns = ["Predicted True", "Predicted False"])
    model_confusion_matrix.to_excel("%s_confusion_matrix_test.xlsx" %algorithm)
    print(model_confusion_matrix)
    
    model_confusion_matrix = confusion_matrix(y_train, model_pred_train, labels = [1,0])
    model_confusion_matrix = pd.DataFrame(model_confusion_matrix, ["Actual True", "Actual False"], columns = ["Predicted True", "Predicted False"])
    model_confusion_matrix.to_excel("%s_confusion_matrix_train.xlsx" %algorithm)
    print(model_confusion_matrix)
    
    try: 
        print("\nBest model by grid search:")
        print(classifier.best_estimator_)
        print("\nBest Score by grid search:")
        print(classifier.best_score_)
        statistics["Parameter setting"].append(str(classifier.best_estimator_))
        #pd.DataFrame(classifier.cv_results_).to_excel("%s_parameters.xlsx" %method, index=False)
    except:
        print("Dummy!")
        statistics["Parameter setting"].append("Dummy")



def DecisionTree(cv=10):
    from sklearn.tree import DecisionTreeClassifier
    print("\nStarted fitting DecisionTreeClassifier at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
        
    grid = {'model__criterion': ['entropy','gini'],
            'model__max_depth': [i for i in range(1,30)],
            'model__min_samples_leaf': np.unique( np.exp(np.linspace(0, 8, 100)).astype(int)),
            'model__min_impurity_decrease': np.exp(np.linspace(-9, -1, 100))}
    
    grid = {'model__criterion': ['entropy'],
            'model__max_depth': [i for i in range(1,15)],
            'model__min_samples_leaf': np.unique( np.exp(np.linspace(0, 8, 8)).astype(int)),
            'model__min_impurity_decrease': np.exp(np.linspace(-9, -1, 8)),
            'model__max_features': ['auto']}
    
    model = DecisionTreeClassifier(random_state = 0)
    
    pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
    
    
    trained_model = GridSearchCV(estimator = pipe, 
                               param_grid = grid,
                               scoring = 'roc_auc',
                               cv = cv,
                               n_jobs = -1,
                               return_train_score=True,
                               verbose=5)
    trained_model.fit(X_train, y_train)
    joblib.dump(trained_model, "Decision_Tree.joblib")
    pred_test = trained_model.predict(X_test)
    pred_train = trained_model.predict(X_train)
    print("\nAnalytics DecisionTreeClassifier at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    #dtc_fpr, dtc_tpr, dtc_threshold = analytics(dtc_pred, dtc)
    generate_statistics(classifier=trained_model,  algorithm = "Decision Tree", model_pred_train=pred_train, model_pred_test=pred_test)


def RandomForest(cv=10):
    from sklearn.ensemble import RandomForestClassifier
    print("\nStarted fitting RandomForest at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    
    grid = {"model__criterion" : ["entropy"],
    "model__n_estimators": [10,20,30,50,80,100],
    'model__max_depth': [i for i in range(1,15)],
    'model__max_features': ['auto'],
    'model__min_impurity_decrease': np.exp(np.linspace(-9, -1, 8)),
    'model__min_samples_leaf': np.unique( np.exp(np.linspace(0, 8, 8)).astype(int))
    }
    
    model = RandomForestClassifier(random_state = 0)
    
    pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
    trained_model = GridSearchCV(estimator = pipe, 
                               param_grid = grid,
                               scoring = 'roc_auc',
                               cv = cv,
                               n_jobs = -1,
                               return_train_score=True,
                               verbose=5)
    trained_model.fit(X_train, y_train)
    joblib.dump(trained_model, "Random_Forest.joblib")
    pred_test = trained_model.predict(X_test)
    pred_train = trained_model.predict(X_train)
    print("\nAnalytics RandomForest at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    generate_statistics(classifier=trained_model, algorithm = "Random Forest", model_pred_train=pred_train, model_pred_test=pred_test)

def MultiLayerPerceptron(cv=10):
    from sklearn.neural_network import MLPClassifier
    print("\nStarted fitting MultiLayerPerceptron at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    
    grid = {'model__solver': ["adam"],
               'model__activation': ['tanh','relu','logistic', 'identity'],
               'model__alpha': [0.01,0.05,0.005],
               'model__hidden_layer_sizes': [(100,),(30,20,10),(20,20),(25,15),(10,40,10),(20,20,20),(30,10)]}
    
    model = MLPClassifier(random_state = 0)
    pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
    trained_model = GridSearchCV(estimator = pipe, 
                               param_grid = grid,
                               scoring = 'roc_auc',
                               cv = cv,
                               n_jobs = -1,
                               return_train_score=True,
                               verbose=5)
    trained_model.fit(X_train, y_train)
    joblib.dump(trained_model, "MLP.joblib")
    pred_test = trained_model.predict(X_test)
    pred_train = trained_model.predict(X_train)
    print("\nAnalytics MultiLayerPerceptron at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    generate_statistics(classifier=trained_model, algorithm = "Multi Layer Perceptron", model_pred_train=pred_train, model_pred_test=pred_test)

def LogisticRegression_without_Regulization(cv=10):
    from sklearn.linear_model import LogisticRegression
    print("\nStarted fitting LogisticRegression_without_Regulization at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    
    grid = {"model__C" : [1e20]}
    
    model = LogisticRegression()
    pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
    trained_model = GridSearchCV(estimator = pipe, 
                               param_grid = grid,
                               scoring = 'roc_auc',
                               cv = cv,
                               n_jobs = -1,
                               return_train_score=True,
                               verbose=5)
    trained_model.fit(X_train, y_train)
    joblib.dump(trained_model, "LogisticRegression_without_Regulization.joblib")
    pred_test = trained_model.predict(X_test)
    pred_train = trained_model.predict(X_train)
    print("\nAnalytics LogisticRegression_without_Regulization at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    generate_statistics(classifier=trained_model, algorithm = "LogisticRegression_without_Regulization", model_pred_train=pred_train, model_pred_test=pred_test)
 
def RIDGE(cv=10):
    from sklearn.linear_model import RidgeClassifier
    print("\nStarted fitting RIDGE at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    
    grid = {"model__alpha" : [0.001,0.01,0.1,0.2,0.3,0.5,0.7,0.8,0.9,1]}
    
    model = RidgeClassifier(random_state = 0)
    pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
    trained_model = GridSearchCV(estimator = pipe, 
                               param_grid = grid,
                               scoring = 'roc_auc',
                               cv = cv,
                               n_jobs = -1,
                               return_train_score=True,
                               verbose=5)
    trained_model.fit(X_train, y_train)
    joblib.dump(trained_model, "Ridge.joblib")
    pred_test = trained_model.predict(X_test)
    pred_train = trained_model.predict(X_train)
    print("\nAnalytics RIDGE at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    generate_statistics(classifier=trained_model, algorithm = "RIDGE", model_pred_train=pred_train, model_pred_test=pred_test)

def GaussianNaiveBayes(cv=10):
    from sklearn.naive_bayes import GaussianNB
    print("\nStarted fitting GaussianNaiveBayes at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    
    grid = {"model__var_smoothing": [0.001,0.01,0.1,0.2,0.3,0.5,0.7,0.8,0.9,1]}
    
    model = GaussianNB()
    pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
    trained_model = GridSearchCV(estimator = pipe, 
                               param_grid = grid,
                               scoring = 'roc_auc',
                               cv = cv,
                               n_jobs = -1,
                               return_train_score=True,
                               verbose=5)
    trained_model.fit(X_train, y_train)
    joblib.dump(trained_model, "GNB.joblib")
    pred_test = trained_model.predict(X_test)
    pred_train = trained_model.predict(X_train)
    print("\nAnalytics GaussianNaiveBayes at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    generate_statistics(classifier=trained_model, algorithm = "Gaussian Naive Bayes", model_pred_train=pred_train, model_pred_test=pred_test)

def XGBoost(cv=10):
    
    from xgboost import XGBClassifier
    print("\nStarted fitting XGBoost at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
 
    grid = {"model__learning_rate" : [0.1,0.2,0.3,0.5],
            "model__n_estimators": [10, 50,100,200, 300, 500],
            "model__max_depth": [i for i in range(1,21)]}
    model = XGBClassifier(objective="binary:logistic", random_state=0, use_label_encoder = False)
    pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
    trained_model = GridSearchCV(estimator = pipe, 
                               param_grid = grid,
                               scoring = 'roc_auc',
                               cv = cv,
                               n_jobs = -1,
                               return_train_score=True,
                               verbose=5)
    trained_model.fit(X_train, y_train)
    joblib.dump(trained_model, "XGBoost.joblib")
    pred_test = trained_model.predict(X_test)
    pred_train = trained_model.predict(X_train)
    print("\nAnalytics XGBoost at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    generate_statistics(classifier=trained_model, algorithm = "XGBoost", model_pred_train=pred_train, model_pred_test=pred_test)


def Dummy():
    from sklearn.dummy import DummyClassifier
    print("\nStarted fitting Dummy at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    trained_model = DummyClassifier(random_state=0)

    trained_model.fit(X_train, y_train)
    joblib.dump(trained_model, "Dummy.joblib")
    pred_test = trained_model.predict(X_test)
    pred_train = trained_model.predict(X_train)
    print("\nAnalytics Dummy at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    generate_statistics(classifier=trained_model, algorithm = "Dummy", model_pred_train=pred_train, model_pred_test=pred_test)

"""
WITHOUT LN
"""


"""
START WITH BASE MODELS
"""

"""
TOP 10 PERCENT OF CITATIONS 7 YEARS
"""

data_citations = pd.read_excel("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Input_data_scaled_citations.xlsx" , engine = "openpyxl", index_col=0)
print("Data loaded")
number_columns = len(data_citations.columns)
number_iv = 19
columns_iv = [i for i in range(number_columns-number_iv, number_columns)]
X = data_citations.iloc[:,columns_iv].values
os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\citations\\top_10\\7_years")
y = data_citations["TOP10_CIT_7YEARS"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
print("\nCases to train {0}".format(len(y_train)))
print("\nCases to test {0}".format(len(y_test)))

cv_generator = StratifiedKFold(n_splits=10)

LogisticRegression_without_Regulization(cv=cv_generator)
RIDGE(cv=cv_generator)
MultiLayerPerceptron(cv=cv_generator)  
Dummy()
DecisionTree(cv=cv_generator)
RandomForest(cv=cv_generator)
XGBoost(cv=cv_generator) 
df2 = pd.DataFrame(data = statistics)
df2 = df2.sort_values(by=["ROC AUC mean validation"], ascending=False)
df2.to_excel("classification_statistics_TOP10_CIT_7YEARS.xlsx", index=False)

"""
TOP 10 PERCENT OF KPSS_REAL
"""
del statistics
statistics = defaultdict(list)
data_kpss = pd.read_excel("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Input_data_scaled_KPSS.xlsx" , engine = "openpyxl", index_col=0)
print("Data loaded")

X = data_kpss.iloc[:,columns_iv].values

os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\kpss\\top_10")
y = data_kpss["TOP10_KPSS"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
print("\nCases to train {0}".format(len(y_train)))
print("\nCases to test {0}".format(len(y_test)))

LogisticRegression_without_Regulization(cv=cv_generator)
XGBoost(cv=cv_generator)
Dummy()
DecisionTree(cv=cv_generator)
RandomForest(cv=cv_generator)
MultiLayerPerceptron(cv=cv_generator) 
            
df2 = pd.DataFrame(data = statistics)
df2 = df2.sort_values(by=["ROC AUC mean validation"], ascending=False)
df2.to_excel("classification_statistics_TOP10_KPSS.xlsx", index=False)


"""
CONTINUE WITH ROBUSTNESS CHECKS
"""

"""
TOP 10 PERCENT OF CITATIONS 5 YEARS
"""

print("Data loaded")
X = data_citations.iloc[:,columns_iv].values
del statistics
statistics = defaultdict(list)
os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\citations\\top_10\\5_years")
y = data_citations["TOP10_CIT_5YEARS"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
print("\nCases to train {0}".format(len(y_train)))
print("\nCases to test {0}".format(len(y_test)))
LogisticRegression_without_Regulization(cv=cv_generator)
GaussianNaiveBayes(cv=cv_generator)
XGBoost(cv=cv_generator)
Dummy()
DecisionTree(cv=cv_generator)
RandomForest(cv=cv_generator)
MultiLayerPerceptron(cv=cv_generator)  
df2 = pd.DataFrame(data = statistics)
df2 = df2.sort_values(by=["ROC AUC mean validation"], ascending=False)
df2.to_excel("classification_statistics_TOP10_CIT_5YEARS.xlsx", index=False)


"""
TOP 10 PERCENT OF CITATIONS 10 YEARS
"""
X = data_citations.iloc[:,columns_iv].values
del statistics
statistics = defaultdict(list)
os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\citations\\top_10\\10_years")
y = data_citations["TOP10_CIT_10YEARS"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
print("\nCases to train {0}".format(len(y_train)))
print("\nCases to test {0}".format(len(y_test)))
LogisticRegression_without_Regulization(cv=cv_generator)
GaussianNaiveBayes(cv=cv_generator)
XGBoost(cv=cv_generator)
Dummy()
DecisionTree(cv=cv_generator)
RandomForest(cv=cv_generator)
MultiLayerPerceptron(cv=cv_generator)        
df2 = pd.DataFrame(data = statistics)
df2 = df2.sort_values(by=["ROC AUC mean validation"], ascending=False)
df2.to_excel("classification_statistics_TOP10_CIT_10YEARS.xlsx", index=False)

"""
TOP 1 PERCENT OF CITATIONS 5 YEARS
"""

X = data_citations.iloc[:,columns_iv].values
del statistics
statistics = defaultdict(list)
os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\citations\\top_1\\5_years")
y = data_citations["TOP1_CIT_5YEARS"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
print("\nCases to train {0}".format(len(y_train)))
print("\nCases to test {0}".format(len(y_test)))
LogisticRegression_without_Regulization(cv=cv_generator)
GaussianNaiveBayes(cv=cv_generator)
XGBoost(cv=cv_generator)
Dummy()
DecisionTree(cv=cv_generator)
RandomForest(cv=cv_generator)
MultiLayerPerceptron(cv=cv_generator)             
df2 = pd.DataFrame(data = statistics)
df2 = df2.sort_values(by=["ROC AUC mean validation"], ascending=False)
df2.to_excel("classification_statistics_TOP1_CIT_5YEARS.xlsx", index=False)

"""
TOP 1 PERCENT OF CITATIONS 7 YEARS
"""

X = data_citations.iloc[:,columns_iv].values
del statistics
statistics = defaultdict(list)
os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\citations\\top_1\\7_years")
y = data_citations["TOP1_CIT_7YEARS"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
print("\nCases to train {0}".format(len(y_train)))
print("\nCases to test {0}".format(len(y_test)))
LogisticRegression_without_Regulization(cv=cv_generator)
GaussianNaiveBayes(cv=cv_generator)
XGBoost(cv=cv_generator)
Dummy()
DecisionTree(cv=cv_generator)
RandomForest(cv=cv_generator)
MultiLayerPerceptron(cv=cv_generator)             
df2 = pd.DataFrame(data = statistics)
df2 = df2.sort_values(by=["ROC AUC mean validation"], ascending=False)
df2.to_excel("classification_statistics_TOP1_CIT_7YEARS.xlsx", index=False)

"""
TOP 1 PERCENT OF CITATIONS 10 YEARS
"""

X = data_citations.iloc[:,columns_iv].values
del statistics
statistics = defaultdict(list)
os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\citations\\top_1\\10_years")
y = data_citations["TOP1_CIT_10YEARS"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
print("\nCases to train {0}".format(len(y_train)))
print("\nCases to test {0}".format(len(y_test)))
LogisticRegression_without_Regulization(cv=cv_generator)
GaussianNaiveBayes(cv=cv_generator)
XGBoost(cv=cv_generator)
Dummy()
DecisionTree(cv=cv_generator)
RandomForest(cv=cv_generator)
MultiLayerPerceptron(cv=cv_generator)           
df2 = pd.DataFrame(data = statistics)
df2 = df2.sort_values(by=["ROC AUC mean validation"], ascending=False)
df2.to_excel("classification_statistics_TOP1_CIT_10YEARS.xlsx", index=False)

"""
TOP 10 PERCENT OF KPSS_REAL
"""
del statistics
statistics = defaultdict(list)
print("Data loaded")
X = data_kpss.iloc[:,columns_iv].values

os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\kpss\\top_1")
y = data_kpss["TOP1_KPSS"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
print("\nCases to train {0}".format(len(y_train)))
print("\nCases to test {0}".format(len(y_test)))
LogisticRegression_without_Regulization(cv=cv_generator)
GaussianNaiveBayes(cv=cv_generator)
XGBoost(cv=cv_generator)
Dummy()
DecisionTree(cv=cv_generator)
RandomForest(cv=cv_generator)
MultiLayerPerceptron(cv=cv_generator)             
df2 = pd.DataFrame(data = statistics)
df2 = df2.sort_values(by=["ROC AUC mean validation"], ascending=False)
df2.to_excel("classification_statistics_TOP1_KPSS.xlsx", index=False)

"""

WITH LN

"""

"""
START WITH BASE MODELS
"""

"""
TOP 10 PERCENT OF CITATIONS 7 YEARS
"""

data_citations = pd.read_excel("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Input_data_scaled_citations_ln.xlsx" , engine = "openpyxl", index_col=0)

print("Data loaded")
X = data_citations.iloc[:,columns_iv].values

statistics = defaultdict(list)
os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\ln\\citations\\top_10\\7_years")
y = data_citations["TOP10_CIT_7YEARS"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
print("\nCases to train {0}".format(len(y_train)))
print("\nCases to test {0}".format(len(y_test)))
LogisticRegression_without_Regulization(cv=cv_generator)
GaussianNaiveBayes(cv=cv_generator)
XGBoost(cv=cv_generator)
Dummy()
DecisionTree(cv=cv_generator)
RandomForest(cv=cv_generator)
MultiLayerPerceptron(cv=cv_generator)          
df2 = pd.DataFrame(data = statistics)
df2 = df2.sort_values(by=["ROC AUC mean validation"], ascending=False)
df2.to_excel("classification_statistics_TOP10_CIT_7YEARS.xlsx", index=False)

"""
TOP 10 PERCENT OF KPSS_REAL
"""
del statistics
statistics = defaultdict(list)
data_kpss = pd.read_excel("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Input_data_scaled_KPSS_ln.xlsx" , engine = "openpyxl", index_col=0)
print("Data loaded")
X = data_kpss.iloc[:,columns_iv].values

os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\ln\\kpss\\top_10")
y = data_kpss["TOP10_KPSS"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
print("\nCases to train {0}".format(len(y_train)))
print("\nCases to test {0}".format(len(y_test)))
LogisticRegression_without_Regulization(cv=cv_generator)
GaussianNaiveBayes(cv=cv_generator)
XGBoost(cv=cv_generator)
Dummy()
DecisionTree(cv=cv_generator)
RandomForest(cv=cv_generator)
MultiLayerPerceptron(cv=cv_generator)               
df2 = pd.DataFrame(data = statistics)
df2 = df2.sort_values(by=["ROC AUC mean validation"], ascending=False)
df2.to_excel("classification_statistics_TOP10_KPSS.xlsx", index=False)


"""
CONTINUE WITH ROBUSTNESS CHECKS
"""

"""
TOP 10 PERCENT OF CITATIONS 5 YEARS
"""

X = data_citations.iloc[:,columns_iv].values
del statistics
statistics = defaultdict(list)
os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\citations\\ln\\top_10\\5_years")
y = data_citations["TOP10_CIT_5YEARS"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
print("\nCases to train {0}".format(len(y_train)))
print("\nCases to test {0}".format(len(y_test)))
LogisticRegression_without_Regulization(cv=cv_generator)
GaussianNaiveBayes(cv=cv_generator)
XGBoost(cv=cv_generator)
Dummy()
DecisionTree(cv=cv_generator)
RandomForest(cv=cv_generator)
MultiLayerPerceptron(cv=cv_generator)               
df2 = pd.DataFrame(data = statistics)
df2 = df2.sort_values(by=["ROC AUC mean validation"], ascending=False)
df2.to_excel("classification_statistics_TOP10_CIT_5YEARS.xlsx", index=False)


"""
TOP 10 PERCENT OF CITATIONS 10 YEARS
"""

X = data_citations.iloc[:,columns_iv].values
del statistics
statistics = defaultdict(list)
os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\citations\\ln\\top_10\\10_years")
y = data_citations["TOP10_CIT_10YEARS"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
print("\nCases to train {0}".format(len(y_train)))
print("\nCases to test {0}".format(len(y_test)))
LogisticRegression_without_Regulization(cv=cv_generator)
GaussianNaiveBayes(cv=cv_generator)
XGBoost(cv=cv_generator)
Dummy()
DecisionTree(cv=cv_generator)
RandomForest(cv=cv_generator)
MultiLayerPerceptron(cv=cv_generator)             
df2 = pd.DataFrame(data = statistics)
df2 = df2.sort_values(by=["ROC AUC mean validation"], ascending=False)
df2.to_excel("classification_statistics_TOP10_CIT_10YEARS.xlsx", index=False)

"""
TOP 1 PERCENT OF CITATIONS 5 YEARS
"""

X = data_citations.iloc[:,columns_iv].values
del statistics
statistics = defaultdict(list)
os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\citations\\ln\\top_1\\5_years")
y = data_citations["TOP1_CIT_5YEARS"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
print("\nCases to train {0}".format(len(y_train)))
print("\nCases to test {0}".format(len(y_test)))
LogisticRegression_without_Regulization(cv=cv_generator)
GaussianNaiveBayes(cv=cv_generator)
XGBoost(cv=cv_generator)
Dummy()
DecisionTree(cv=cv_generator)
RandomForest(cv=cv_generator)
MultiLayerPerceptron(cv=cv_generator)       
df2 = pd.DataFrame(data = statistics)
df2 = df2.sort_values(by=["ROC AUC mean validation"], ascending=False)
df2.to_excel("classification_statistics_TOP1_CIT_5YEARS.xlsx", index=False)

"""
TOP 1 PERCENT OF CITATIONS 7 YEARS
"""

X = data_citations.iloc[:,columns_iv].values
del statistics
statistics = defaultdict(list)
os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\citations\\ln\\top_1\\7_years")
y = data_citations["TOP1_CIT_7YEARS"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
print("\nCases to train {0}".format(len(y_train)))
print("\nCases to test {0}".format(len(y_test)))
LogisticRegression_without_Regulization(cv=cv_generator)
GaussianNaiveBayes(cv=cv_generator)
XGBoost(cv=cv_generator)
Dummy()
DecisionTree(cv=cv_generator)
RandomForest(cv=cv_generator)
MultiLayerPerceptron(cv=cv_generator)               
df2 = pd.DataFrame(data = statistics)
df2 = df2.sort_values(by=["ROC AUC mean validation"], ascending=False)
df2.to_excel("classification_statistics_TOP1_CIT_7YEARS.xlsx", index=False)

"""
TOP 1 PERCENT OF CITATIONS 10 YEARS
"""

X = data_citations.iloc[:,columns_iv].values
del statistics
statistics = defaultdict(list)
os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\ln\\citations\\top_1\\10_years")
y = data_citations["TOP1_CIT_10YEARS"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
print("\nCases to train {0}".format(len(y_train)))
print("\nCases to test {0}".format(len(y_test)))
LogisticRegression_without_Regulization(cv=cv_generator)
GaussianNaiveBayes(cv=cv_generator)
XGBoost(cv=cv_generator)
Dummy()
DecisionTree(cv=cv_generator)
RandomForest(cv=cv_generator)
MultiLayerPerceptron(cv=cv_generator)            
df2 = pd.DataFrame(data = statistics)
df2 = df2.sort_values(by=["ROC AUC mean validation"], ascending=False)
df2.to_excel("classification_statistics_TOP1_CIT_10YEARS.xlsx", index=False)

"""
TOP 1 PERCENT OF KPSS_REAL
"""
del statistics
statistics = defaultdict(list)

X = data_kpss.iloc[:,columns_iv].values
os.chdir("N:\\Publikationen\\2021 Assessment Novelty measures\\05_Analysis\\Machine_learning\\ln\\kpss\\top_1")
y = data_kpss["TOP1_KPSS"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
print("\nCases to train {0}".format(len(y_train)))
print("\nCases to test {0}".format(len(y_test)))
LogisticRegression_without_Regulization(cv=cv_generator)
GaussianNaiveBayes(cv=cv_generator)
XGBoost(cv=cv_generator)
Dummy()
DecisionTree(cv=cv_generator)
RandomForest(cv=cv_generator)
MultiLayerPerceptron(cv=cv_generator)                
df2 = pd.DataFrame(data = statistics)
df2 = df2.sort_values(by=["ROC AUC mean validation"], ascending=False)
df2.to_excel("classification_statistics_TOP1_KPSS.xlsx", index=False)