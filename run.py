import sys
import os
import time

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.preprocessing import StandardScaler


class Const:
    BANDS_ALL = ['u', 'g', 'r', 'i', 'z']
    INDX_ALL = ['0', '1', '2', '3', '4', '5', '6']


class DataPreparer:

    def __init__(self, train, unlabeled, test):
        self.train = train
        self.unlabeled = unlabeled
        self.test = test

    @staticmethod
    def _get_is_na_columns(data, features):
        is_na_df = data[features].isna().astype('int64')

        # rename columns
        renamed_features = dict((f, f + "_na") for f in features)
        is_na_df.rename(columns=renamed_features, inplace=True)

        return is_na_df

    def _fill_nans_based_on_regression(self, col_name, based_on_col_name, data):
        # get only notna values to train regression
        clean_columns_mask = self.unlabeled[[col_name, based_on_col_name]].notna().all(axis=1)
        clean_columns = self.unlabeled[clean_columns_mask][[col_name, based_on_col_name]]

        # get samples where 'col' is NaN and 'based_col' is not NaN
        # (samples, where 'col' value will be predicted)
        col_nan_mask = data[col_name].isna()
        based_col_notna_mask = data[based_on_col_name].notna()
        samples_to_predict = data[col_nan_mask & based_col_notna_mask]

        # reshape to fit the regression
        X = clean_columns[based_on_col_name].values.reshape(-1, 1)
        y = clean_columns[col_name].values.reshape(-1, 1)
        to_predict = samples_to_predict[based_on_col_name].values.reshape(-1, 1)

        if len(to_predict):
            reg = LinearRegression()
            reg.fit(X, y)
            predicted = reg.predict(to_predict).flatten()

            # fill predicted values instead of NaNs
            idx = samples_to_predict.index
            data.loc[idx, col_name] = predicted

    def _fill_0th_with_regression(self):
        zeros_features = ["{}_0".format(b) for b in Const.BANDS_ALL]

        # determine the order of applying regression: order features by correlation strength
        corr_matrix = self.train[zeros_features].corr()
        order = {}
        for f in zeros_features:
            order[f] = corr_matrix[f].argsort()[-2::-1].values

        for step in range(4):
            replacement_scheme = []

            for f in zeros_features:
                base_col = zeros_features[order[f][step]]
                replacement_scheme.append((f, base_col))

            for col, based_on_col in replacement_scheme:
                self._fill_nans_based_on_regression(col, based_on_col, self.train)

            for col, based_on_col in replacement_scheme:
                self._fill_nans_based_on_regression(col, based_on_col, self.test)

    def _handle_missing_values(self):
        self._fill_0th_with_regression()

        # fill rest of '-0' with mean values
        zeros_features = ["{}_0".format(b) for b in Const.BANDS_ALL]
        self.train.fillna(self.train[zeros_features].mean(), inplace=True)
        self.test.fillna(self.test[zeros_features].mean(), inplace=True)

        # fill '-3', '-4' and '-5' with median values
        indx = ['3', '4', '5']
        columns_to_fill = ["{}_{}".format(b, i) for b in Const.BANDS_ALL for i in indx]

        self.train.fillna(self.train[columns_to_fill].median(), inplace=True)
        self.test.fillna(self.test[columns_to_fill].median(), inplace=True)

        # fill all the rest NaNs with mean values (should be only 'rowv' and 'colv')
        self.train.fillna(self.train.mean(), inplace=True)
        self.test.fillna(self.test.mean(), inplace=True)

    @staticmethod
    def _get_above_median_columns_count(data):
        features = ["{}_{}".format(b, str(i)) for i in [0, 3, 4, 5] for b in Const.BANDS_ALL]
        is_above_features = ["{}_above".format(f) for f in features]

        means = data[features].median()

        is_above_df = pd.DataFrame(columns=is_above_features)
        for f, is_above_f in zip(features, is_above_features):
            is_above_df[is_above_f] = data[f] > means[f]

        # sum is_above by measurement (0-th, 3-rd, 4-th and 5-th features)
        num_of_above_columns = ['0th_above', '3rd_above', '4th_above', '5th_above']
        num_of_above = pd.DataFrame(columns=num_of_above_columns)
        for idx, f_name in zip([0, 5, 10, 15], num_of_above_columns):
            num_of_above[f_name] = is_above_df.iloc[:, idx:idx + 5].sum(axis=1)
        return num_of_above

    def _convert_6th_variable(self):
        # convert '-6' variable using one-hot encoding
        features = ["{}_6".format(b) for b in Const.BANDS_ALL]

        dummy_train = pd.DataFrame()
        dummy_test = pd.DataFrame()
        for f in features:
            dummy_train = pd.concat([dummy_train, pd.get_dummies(self.train[f], prefix=f)], axis=1)
            dummy_test = pd.concat([dummy_test, pd.get_dummies(self.test[f], prefix=f)], axis=1)

        # in case test set has values train set doesn't have
        values = ["{}_{}".format(f, i) for f in features for i in range(9)]
        dummy_train = dummy_train.reindex(columns=values).fillna(0).astype('int64')
        dummy_test = dummy_test.reindex(columns=values).fillna(0).astype('int64')

        self.train = pd.concat([self.train, dummy_train], axis=1)
        self.test = pd.concat([self.test, dummy_test], axis=1)

        # drop initial categorical variables
        self.train.drop(features, axis=1, inplace=True)
        self.test.drop(features, axis=1, inplace=True)

    def prepare(self):
        # create '-3' is_na features
        features = ["{}_3".format(b) for b in Const.BANDS_ALL]
        self.train = pd.concat([self.train, DataPreparer._get_is_na_columns(self.train, features)], axis=1)
        self.test = pd.concat([self.test, DataPreparer._get_is_na_columns(self.test, features)], axis=1)

        self._handle_missing_values()

        # create 'above_median' columns
        self.train = pd.concat([self.train, DataPreparer._get_above_median_columns_count(self.train)], axis=1)
        self.test = pd.concat([self.test, DataPreparer._get_above_median_columns_count(self.test)], axis=1)

        # drop not relevant to class prediction columns
        indx = ['1', '2']
        columns_to_drop = ["{}_{}".format(b, i) for b in Const.BANDS_ALL for i in indx]
        columns_to_drop.extend(['ra', 'dec', 'colc', 'rowc'])
        self.train.drop(columns_to_drop, axis=1, inplace=True)
        self.test.drop(columns_to_drop, axis=1, inplace=True)

        # prepare '-6' variable
        self._convert_6th_variable()

    def get_datasets(self):
        return self.train, self.unlabeled, self.test


def read_csv_file(path):
    if not os.path.isfile(path):
        print("Can't read {}".format(path))
        return None
    return pd.read_csv(path, na_values=['na'])


def make_predictions(train, test):
    X_train = train.drop(['objid', 'class'], axis=1)
    y_train = train['class']

    X_test = test.drop(['objid'], axis=1)
    if 'class' in X_test.columns:
        X_test = X_test.drop(['class'], axis=1)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    adb_cls = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_split=10),
                                 n_estimators=120, learning_rate=0.003)
    adb_cls.fit(X_train, y_train)
    predictions = adb_cls.predict(X_test)
    return predictions


def write_prediction(ids, predictions, path):
    submission = pd.DataFrame({'objid': ids, 'prediction': predictions})
    submission.to_csv(path, index=False)


def main():
    parameters = sys.argv
    if len(parameters) != 5:
        print("Wrong number of parameters")
        return

    train, unlabeled, test = [read_csv_file(path) for path in parameters[1:4]]
    predictions_path = parameters[4]

    data_preparer = DataPreparer(train, unlabeled, test)
    data_preparer.prepare()
    train, unlabeled, test = data_preparer.get_datasets()

    predictions = make_predictions(train, test)
    write_prediction(test['objid'], predictions, predictions_path)


if __name__ == '__main__':
    main()
