from __future__ import print_function
import os
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor  #GBM algorithm
import xgboost as xgb

from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
# for notebook:
#%matplotlib inline

# PATHS
CDIR = os.path.dirname(os.path.realpath(__file__))
MAIN = os.path.dirname(CDIR)
RESOURCES = os.path.join(MAIN, 'resources')
DATA = os.path.join(RESOURCES, 'data')


def xGBoostModelFit():
    test_preds = np.zeros(test.shape[0])
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test)

    watchlist = [(xg_train, 'train')]
    num_rounds = 100

    xgclassifier = xgb.train(params, xg_train, num_rounds, watchlist, feval = evalerror, early_stopping_rounds= 20, verbose_eval = 10)
    preds = xgclassifier.predict(xg_test, ntree_limit=xgclassifier.best_iteration)
    print('RMSLE Score:', rmsle(y_test, preds))
    fxg_test = xgb.DMatrix(test)
    fold_preds = np.around(xgclassifier.predict(fxg_test, ntree_limit=xgclassifier.best_iteration), decimals = 1)
    test_preds += fold_preds

    submission = pd.DataFrame({'id':ids, 'Demanda_uni_equil': test_preds})
    submission.to_csv('submission.csv', index=False)


def gbm_predict(X_train, y_train, X_test, indep_vars, y_test=None, grid_search=True,
                cv=True, verbose=True, cv_folds=5, scoring='roc_auc',
                submit=False, version='001', y_label='Demanda_uni_equil',
                ids=None, param_grid=None):

    # Train
    y_train_vec = np.array(y_train, dtype=np.float64)
    model = train_gbm(X_train,  y_train_vec, grid_search=grid_search,
                      verbose=verbose, scoring=scoring,
                      param_grid = param_grid, cv_folds=cv_folds)
    # predict on train
    if verbose:
        print('Starting predictions ...')
    test_pred = model.predict(X_test)
    if verbose:
        print('Finished predictions.')


    if scoring=='roc_auc':
        # probability
        test_pred_prob = model.predict_proba(X_test)[:, 1]

    if cv and not submit:
        print('Cross validation')
        cv_score = cross_validation.cross_val_score(model, X_test,
                                                    y_test, cv=cv_folds,
                                                    scoring=scoring)
    if verbose and not submit:
        print("\nModel Report")
        if scoring == 'roc_auc':
            print("Accuracy : %.4g" % metrics.accuracy_score(X_test.values, test_pred))
            print("AUC Score (Train): %f" % metrics.roc_auc_score(X_test, test_pred_prob))
            if cv:
                print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g"
                      % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        elif scoring == 'mean_squared_error':
            # print(type(y_test)) #pandas Series w 300 rows
            # print(type(test_pred)) # numpy array
            mse = metrics.mean_squared_error(y_test.values, test_pred)  # multioutput='raw_values'
            print("mean_squared_error : %.4g" % mse)
        # feature relevance
        if not grid_search:
            predictive_relavance = pd.Series(model.feature_importances_, indep_vars).sort_values(ascending=False)
            predictive_relavance.plot(kind='bar', title='Feature relevance')
            plt.ylabel('Feature relevance score')

    if submit:
        print('Creating prediction file for Kaggle submission...')
        cols = ['id', y_label]
        # Adjustmentsfor submission requirements
        pred_adjusted = np.around(test_pred, decimals=0)
        pred_adjusted[pred_adjusted < 0] = 0
        submission = pd.DataFrame({'id':ids, y_label: pred_adjusted})
        submission = submission[cols]
        print(submission.head())
        submission_path = os.path.join(DATA, 'submission_gbm_{}.csv'.format(version))
        submission.to_csv(submission_path, index=False)
        print('Saved submission csv in: {}'.format(submission_path))
    return model


def train_gbm(X_train,  y_train, type='regression', grid_search=False, verbose=True,
              min_samples_split=None, min_samples_leaf=50, max_depth=8,
              max_features='sqrt', sub_sample=0.8, n_estimators=100,
              learning_rate=0.1, random_state=10, param_grid=None,
              scoring='roc_auc', loss='quantile', cv_folds=5
              ):
    """

    :param X_train:
    :param indep_vars:
    :param dep_var:
    :param verbose:
    :param min_samples_leaf: (int) prevent overfitting, intuition based value..
    :param max_depth: (int) 8 # 5 -8, based on number of features and dataset size
    :param max_features: (str) 'sqrt' # general rule of thumb: sqrt(n_samples)
    :param sub_sample: (float) fraction of observations to be selected for each tree (0.8 commonly used value)
    :param n_estimators: (int) number of sequential trees to be modeled
    :param learning_rate: (float)
    :param random_state: (int)
    :param param_grid: (dict)
    :return:
            (model)
    """
    if not min_samples_split:
        n_samples = X_train.shape[0]
        min_samples_split = n_samples * .01  # prevent overfitting, general rule of thumb: 0.5 - 1%

    if type == 'classification':
        gbm = GradientBoostingClassifier(
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            max_depth=max_depth, max_features=max_features, learning_rate=learning_rate,
            n_estimators=n_estimators, subsample=sub_sample, random_state=random_state)
    else:
        gbm = GradientBoostingRegressor(
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            max_depth=max_depth, max_features=max_features, learning_rate=learning_rate,
            n_estimators=n_estimators, subsample=sub_sample, random_state=random_state,
            loss=loss
        )
    model = gbm
    if grid_search:
        if not param_grid:
            param_grid = dict(n_estimators=range(20,201,10))
        model = GridSearchCV(
            estimator=gbm,
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=4,
            iid=False,
            cv=cv_folds)
    if verbose:
        print('Starting to train model...')
    model.fit(X_train, y_train)
    if grid_search and verbose:
        print('-> Model Grid scores: {0}; \n-> Best params: {1}; \n-> Best score: {2}'
              .format(model.grid_scores_, model.best_params_, model.best_score_))
    if verbose:
        print('Finished training.')
    return model


def load_data(path, file_name, nrows=None, verbose=True):
    """
    convenience func for printing
    side effects
    :param path:
    :param file_name:
    :param nrows:
    :return:
            (pandas.dataframe)
    """
    data_path = os.path.join(path, file_name)
    if verbose:
        print('\n#################################')
        print('Loading data from {0}...'.format(data_path))
    data = pd.read_csv(data_path, nrows=nrows)
    if verbose:
        print('Dataset num rows: {0}, num cols: {1}'
              .format(data.shape[0],data.shape[1]))
        print('Columns: {}'.format(list(data.columns.values)))
        print('Head: ')
        print(data.head())
    return data


if __name__ == '__main__':

    train_path = os.path.join(DATA, 'train.csv')
    test_path = os.path.join(DATA, 'test.csv')
    client_data_path = os.path.join(DATA, 'cliente_tabla.csv')
    product_data_path = os.path.join(DATA, 'producto_tabla.csv')
    town_state_path = os.path.join(DATA, 'town_state.csv')

    print('Loading data..')
    df_train = load_data(path=DATA, file_name='train.csv', nrows=20**4)
    df_client = load_data(path=DATA, file_name='cliente_tabla.csv')
    df_prod = load_data(path=DATA, file_name='producto_tabla.csv')
    df_town = load_data(path=DATA, file_name='town_state.csv')
    df_test = load_data(path=DATA, file_name='test.csv')
    ids = df_test['id']
    df_test = df_test.drop(['id'], axis=1)

    target = 'Demanda_uni_equil'
    indep_vars = list(df_test.columns.values)
    y = df_train[target]
    X = df_train[indep_vars]

    submit_to_kaggle = True

    print('\n\n----------------------')
    print('Finally, loading Kaggle test set to perform predictions...')
    # y_train_vec = np.array(y_train, dtype=np.float64)
    # y_test_vec = np.array(y_test, dtype=np.float64)
    param_grid = dict(n_estimators=range(20, 151, 10))
    # MSE: 2609
    if not submit_to_kaggle:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
        gbm_02 = gbm_predict(X_train=X_train, y_train=y_train, X_test=X_test,
                             y_test=y_test, indep_vars=indep_vars, grid_search=True,
                             cv=False, verbose=True, cv_folds=5, scoring='mean_squared_error',
                             submit=False, ids=ids, y_label=target,
                             param_grid=param_grid)

    ### TO Submit to Kaggle:
    else:
        gbm_12 = gbm_predict(X_train=X, y_train=y, X_test=df_test, indep_vars=indep_vars, grid_search=True,
                             cv=False, verbose=True, cv_folds=5, scoring='mean_squared_error', submit=True, ids=ids,
                             version='002', y_label=target, param_grid=param_grid)

    # With XGBoost
    params = dict(
        objective="reg:linear",
        eta=0.025,
        max_depth=0.8,
        subsample=0.8,
        colsample_bytree=0.6,
        silent=False,
    )







