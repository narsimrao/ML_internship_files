from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

class model:
    def model_processing(X_train, X_test, Y_train, Y_test):

        LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)

        #Decision Tree
        from sklearn.model_selection import GridSearchCV
        param = {}
        param['max_depth'] = [5,10,25,None]
        param['min_samples_split'] = [2,5,10]
        param['class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]

        DT_model = GridSearchCV(DecisionTreeClassifier(),param, cv = 10, return_train_score=False)

        # Random Forest
        param = {}
        param['n_estimators'] = [10, 50, 100, 250]
        param['max_depth'] = [5, 10, 20]
        param['class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]

        RF_model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param,cv=5, n_jobs=-1, scoring='roc_auc')


        LR_model.fit(X_train, Y_train)
        RF_model.fit(X_train, Y_train)
        DT_model.fit(X_train, Y_train)

        return LR_model, RF_model, DT_model