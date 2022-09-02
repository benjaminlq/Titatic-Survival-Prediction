# <center> Classification of Survival Status for Titanic Passengers </center>
--------------------------------------------------

## TABLE OF CONTENTS
  - [Business Problems](#business-problems)
  - [Explanatory Data Analysis (EDA)](#explanatory-data-analysis-eda)
  - [Feature Engineering](#feature-engineering)
  - [Model Building and Optimization](#model-building-and-optimization)
  - [Model Evaluation](#model-evaluation)
  - [Conclusion](#conclusion)

 ## Business Problems
 In this project, I used the classic Kaggle Titanic dataset. The goal is to predict the survival status of a particular passenger provided his/her demographics information, passenger information (Ticket class, fare) and information related to the number of family and friends travelling together.
 
 ## Explanatory Data Analysis (EDA)
 ### 1. Univariate Distributions
 #### 1.1. Categorical Variables
 <img src="https://user-images.githubusercontent.com/99384454/188139036-d6970061-d27a-4712-90f1-864266ad3fcc.png" width="750">
 <img src="https://user-images.githubusercontent.com/99384454/188139098-4040f880-d02b-4858-a27c-741c53287608.png" width="850"

 #### 1.2. Continuous Variables
  ![continuous](https://user-images.githubusercontent.com/99384454/188139186-585c6c59-62ec-4bd2-becc-fee79286a79d.png)

 ### 2. Multivariate Distributions 
 ![multivariate](https://user-images.githubusercontent.com/99384454/188139225-b4e12d5d-28e3-4d78-a498-2ba53c68b5ee.png) <br>
 Based on the multivariate distribution, no pairs of features have significantly high Pearson correlation. The highest pair with moderate positive correlation is the number of siblings and the number of parents travelling together with the passenger.
 
 ## Feature Engineering
 ### 1. Generate Family feature:
 Combine the number of siblings and parents to generate number of family members.
 ```
data_org['Family'] = data_org['Parch'] + data_org['SibSp'] + 1
data_test['Family'] = data_test['Parch'] + data_test['SibSp'] + 1
sns.countplot(x = 'Family', data = data_org)
 ```
 ![image](https://user-images.githubusercontent.com/99384454/188139934-9eb876ae-761a-4717-9792-e07a4913fdb2.png)

 ### 2. Binarize skewed features
```
data_org['Parch'] = data_org['Parch'].apply(lambda x: str(x) if x < 2 else ">=2")
data_org['SibSp'] = data_org['SibSp'].apply(lambda x: str(x) if x < 2 else ">=2")
data_org['Family'] = data_org['Family'].apply(lambda x: str(x) if x < 4 else ">=4")
data_org['Cabin'] = data_org['Cabin'].apply(lambda x: "Cabin" if x != 'NaN' else "No Cabin")
data_test['Parch'] = data_test['Parch'].apply(lambda x: str(x) if x < 2 else ">=2")
data_test['SibSp'] = data_test['SibSp'].apply(lambda x: str(x) if x < 2 else ">=2")
data_test['Family'] = data_test['Family'].apply(lambda x: str(x) if x < 4 else ">=4")
data_test['Cabin'] = data_test['Cabin'].apply(lambda x: "Cabin" if x != 'NaN' else "No Cabin")
```
<img src="https://user-images.githubusercontent.com/99384454/188140254-40befd38-22a4-42c1-9ef5-607eb09f3129.png" width="900">

 ## Model Building and Optimization
 ### 1. Logistic Regression
```
from sklearn.linear_model import LogisticRegression
lr_grid = {'C':[0.01, 0.1, 1, 10, 100, 1000]}

gcv_lr = model_selection.GridSearchCV(estimator = LogisticRegression(max_iter = 1000),
                                  param_grid = lr_grid,
                                  cv = 10)
gcv_lr.fit(x_train,y_train)
print(gcv_lr.best_params_)
lr = gcv_lr.best_estimator_
lr.fit(x_train,y_train)
```

 ### 2. Random Forest
```
rf_grid = {'max_depth' : [4,5,6,7],
              'min_samples_leaf' : [1,2,3,4]}

gcv_rf = model_selection.GridSearchCV(estimator = ensemble.RandomForestClassifier(n_estimators = 100,
                                                                                 random_state = 2022,
                                                                                 oob_score = True),
                                  param_grid = rf_grid,
                                  cv = 10)

gcv_rf.fit(x_train, y_train)
print('Best  hyperparameters:',gcv_rf.best_params_)
print('Score:',gcv_rf.best_score_)

rf = gcv_rf.best_estimator_
rf.fit(x_train,y_train)
```

 ### 3. GradientBoosting
```
gb_grid = {'learning_rate':[0.01,0.1,1],
          'subsample':[0.25,0.5,1],
          'max_depth':[2,3,4]}
gcv_gb = model_selection.GridSearchCV(estimator = ensemble.GradientBoostingClassifier(n_estimators = 100,
                                                                                      random_state = 2022),
                                      param_grid = gb_grid,
                                      cv = 10)
gcv_gb.fit(x_train,y_train)
print('Best hyperparameters:',gcv_ada.best_params_)
print('Score:',gcv_ada.best_score_)

gb = gcv_gb.best_estimator_
gb.fit(x_train,y_train)
```
 
 ## Model Evaluation
 ![image](https://user-images.githubusercontent.com/99384454/188141681-d0b5be07-c681-4b91-b3e5-895ed3f8cc72.png) <br>
 AdaBoost Classifier has the highest accuracy on the validation dataset of **81.3%**.
  
 
