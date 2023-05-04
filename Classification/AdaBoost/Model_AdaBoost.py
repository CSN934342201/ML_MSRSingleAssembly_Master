# -*- coding: utf-8 -*-
"""
Created on Mon April 17 11:02 2023
@author: Shuning Chen
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
import xgboost
import lightgbm as lgb
import sklearn.metrics as sm
import sklearn.neighbors
import sklearn.svm
from sklearn import linear_model,neural_network
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor








# Train and save ML model
def TrainSaveMLmodel(Dir_TrainDataSample,Dir_LoadedScaler,Dir_MLModel):
    
    # Load the TrainDataSample, TestDataSample
    TrainDataSample = pd.read_csv(Dir_TrainDataSample, sep='\s+', engine='python')
    
    # Split the features and target
    TrainDataSample_Feature = np.array(TrainDataSample.drop('Judge(1_yes)', axis=1).values.tolist())

    # Split the training set and test set
    Data_TrainVal, Data_Test, Target_TrainVal, Target_Test = train_test_split(TrainDataSample_Feature,TrainDataSample['Judge(1_yes)'],test_size=0.3, random_state=43)

    # Standardlize the features
    ScalerFeature = StandardScaler().fit(Data_TrainVal)
    Data_TrainVal = np.array(ScalerFeature.transform(Data_TrainVal))

    # Save the standardlization
    joblib.dump(ScalerFeature, Dir_LoadedScaler)

    # hyperparameter gridsearch
    n_estimators = np.arange(100,300,10)
    learning_rate = np.linspace(1,2,50)
    max_depth = np.arange(2,30)
    max_feature = np.arange(2,17)
    ParamList = []
    f1= []
    rocauc = []
    for hp1 in n_estimators:
        for hp2 in learning_rate:
            for hp3 in max_depth:
                for hp4 in max_feature:
                    
                    # Hyperparameter markdown
                    print('Hyperparameter in iteration: ',hp1,hp2,hp3,hp4)
                    ParamList.append([hp1,hp2,hp3,hp4])

                    # Instantialization
                    dt = DecisionTreeClassifier(random_state=43,max_depth=hp3,max_features=hp4)
                    adb= AdaBoostClassifier(base_estimator=dt,random_state=43,learning_rate=hp2,n_estimators=hp1)
                    
                    # 5-Fold cross validation
                    f1.append(cross_val_score(adb, Data_TrainVal, Target_TrainVal, cv=5, scoring='f1').mean())
                    rocauc.append(cross_val_score(adb, Data_TrainVal, Target_TrainVal, cv=5, scoring='roc_auc').mean())

                    print(f1[-1])
                    print(rocauc[-1])

    print('The optimized f1 = ',np.max(f1))    
    print('The corresponding rocauc = ',rocauc[f1.index(np.max(f1))]) 
    print('The corresponding hyperparameter = ',ParamList[f1.index(np.max(f1))])

    # Save the ML model
    ParamOptimized = ParamList[f1.index(np.max(f1))]
    dt = DecisionTreeClassifier(random_state=43,max_depth=ParamOptimized[2],max_features=ParamOptimized[3])
    adb= AdaBoostClassifier(base_estimator=dt,random_state=43,learning_rate=ParamOptimized[1],n_estimators=ParamOptimized[0])
    adb.fit(Data_TrainVal,Target_TrainVal)
    joblib.dump(adb, Dir_MLModel)






# Predict
def PredictTest(Dir_TestDataSample,Dir_LoadedScaler,Dir_MLModel):

    # Load the TrainDataSample, TestDataSample
    TestDataSample = pd.read_csv(Dir_TestDataSample, sep='\s+', engine='python')

    # Split the features and target
    TestDataSample_Feature = np.array(TestDataSample.drop('Judge(1_yes)', axis=1).values.tolist())
    TestDataSample_Target = np.array(TestDataSample['Judge(1_yes)'].values.tolist())

    # Load the saved Scaler
    LoadedScaler = joblib.load(Dir_LoadedScaler)

    # Scaler the test feature
    TestDataSample_Feature = LoadedScaler.transform(TestDataSample_Feature)

    # Load the ML model
    model = joblib.load(Dir_MLModel)

    # Predict
    predictions = model.predict(TestDataSample_Feature)

    return TestDataSample_Target, predictions





# Evaluate ML model
def EvaluateModel(TestDataSample_Target, predictions):

    # Metrics
    print('Accuracy_test = ',sm.accuracy_score(TestDataSample_Target, predictions))
    print('Presicion_test = ',sm.precision_score(TestDataSample_Target, predictions))
    print('Recall_test = ',sm.recall_score(TestDataSample_Target, predictions))
    print('Fitscore_test = ',sm.f1_score(TestDataSample_Target, predictions))
    print('Rocauc_test = ',sm.roc_auc_score(TestDataSample_Target, predictions))
    
    # Classification report
    report = sm.classification_report(TestDataSample_Target, predictions)
    print('Classification report: ', report, sep='\n')

    # Confusion matrix
    matrix = sm.confusion_matrix(TestDataSample_Target, predictions, normalize='true')
    print('Confusion matrix: ', matrix, sep='\n')

    # Plot confusion matrix
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.weight'] = 'bold'
    matrix_draw = pd.DataFrame(matrix, index=['Non-eq','Eq'],columns=['Non-eq','Eq'])
    graphic = sns.heatmap(matrix_draw,annot=True,fmt='.3f',cmap='viridis')
    graphic.set_yticklabels(graphic.get_yticklabels(),rotation=0,fontsize=18,fontweight='bold')
    graphic.set_xticklabels(graphic.get_xticklabels(),rotation=0,fontsize=18,fontweight='bold')
    plt.rcParams['font.variant'] = 'small-caps'
    plt.rcParams['font.weight'] = 'bold'
    plt.style.use('seaborn-bright')
    plt.xlabel('Predicted label',fontweight='bold',fontsize=20)
    plt.ylabel('True label',fontweight='bold',fontsize=20)
    plt.title('AdaBoost classifier',fontweight='bold',fontsize=20)
    plt.tight_layout()
    plt.savefig('Adaboost_ConfusionMatrix',dpi=600)
    plt.show()


if __name__ == '__main__':


    # Directories of TrainDataSample.csv, TestDataSample.csv
    Dir_TrainDataSample = 'Your directory of TrainDataSample.csv'
    Dir_TestDataSample = 'Your directory of TestDataSample.csv'
    Dir_LoadedScaler = 'Your directory of StandardScaler.joblib'
    Dir_MLModel = 'Your directory of Model_AdaBoost.pkl'
    
    # Train and save ML model
    # TrainSaveMLmodel(Dir_TrainDataSample,Dir_LoadedScaler,Dir_MLModel)

    # Predict the ML model and get the predictions
    TestDataSample_Target, predictions = PredictTest(Dir_TestDataSample,Dir_LoadedScaler,Dir_MLModel)
    
    # Evaluate ML model
    EvaluateModel(TestDataSample_Target, predictions)
