import pandas as pd
import random, time, csv
import numpy as np
import math, copy, os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler
import warnings
from aif360.metrics import ClassificationMetric
# pandas sklearn numpy
import sys

sys.path.append(os.path.abspath('..'))
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import r2_score

import statsmodels.api as sm
import copy


def run_fairedu(dataset_name, train_dataset_path, test_dataset_path):

    if 'adult' in train_dataset_path:
        dataset_orig = pd.read_csv(train_dataset_path)

        # Drop NULL values
        dataset_orig = dataset_orig.dropna()

        # Drop categorical features
        dataset_orig = dataset_orig.drop(
            ['workclass', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship', 'native-country'],
            axis=1)

        # Change symbolics to numerics
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == ' Male', 1, 0)
        dataset_orig['race'] = np.where(dataset_orig['race'] != ' White', 0, 1)
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == ' <=50K', 0, 1)

        # Discretize age
        dataset_orig['age'] = np.where(dataset_orig['age'] >= 70, 70, dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 60) & (dataset_orig['age'] < 70), 60,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 50) & (dataset_orig['age'] < 60), 50,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 40) & (dataset_orig['age'] < 50), 40,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 30) & (dataset_orig['age'] < 40), 30,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 20) & (dataset_orig['age'] < 30), 20,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where((dataset_orig['age'] >= 10) & (dataset_orig['age'] < 10), 10,
                                       dataset_orig['age'])
        dataset_orig['age'] = np.where(dataset_orig['age'] < 10, 0, dataset_orig['age'])
        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)
        protected_attributes = ['race', 'sex']

    elif 'compas' in train_dataset_path:
        # Load dataset
        dataset_orig = pd.read_csv(train_dataset_path)

        # Drop categorical features
        dataset_orig = dataset_orig.drop(
            ['id', 'name', 'first', 'last', 'compas_screening_date', 'dob', 'age', 'juv_fel_count', 'decile_score',
             'juv_misd_count', 'juv_other_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number',
             'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'c_charge_desc', 'is_recid', 'r_case_number',
             'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out',
             'violent_recid', 'is_violent_recid', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date',
             'vr_charge_desc',
             'type_of_assessment', 'decile_score', 'score_text', 'screening_date', 'v_type_of_assessment',
             'v_decile_score',
             'v_score_text', 'v_screening_date', 'in_custody', 'out_custody', 'start', 'end', 'event'], axis=1)

        # Drop NULL values
        dataset_orig = dataset_orig.dropna()

        # Change symbolics to numerics
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'Female', 1, 0)
        dataset_orig['race'] = np.where(dataset_orig['race'] != 'Caucasian', 0, 1)
        dataset_orig['priors_count'] = np.where(
            (dataset_orig['priors_count'] >= 1) & (dataset_orig['priors_count'] <= 3),
            3,
            dataset_orig['priors_count'])
        dataset_orig['priors_count'] = np.where(dataset_orig['priors_count'] > 3, 4, dataset_orig['priors_count'])
        dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Greater than 45', 45, dataset_orig['age_cat'])
        dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == '25 - 45', 25, dataset_orig['age_cat'])
        dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Less than 25', 0, dataset_orig['age_cat'])
        dataset_orig['c_charge_degree'] = np.where(dataset_orig['c_charge_degree'] == 'F', 1, 0)

        # Rename class column
        dataset_orig.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)

        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)
        protected_attributes = ['race', 'sex']

    elif 'student_oulad' in train_dataset_path:
        # Load dataset
        dataset_orig_train = pd.read_csv(train_dataset_path)
        dataset_orig_test = pd.read_csv(test_dataset_path)

        dataset_orig_train = dataset_orig_train.dropna()

        categorical_columns = ['region', 'imd_band', 'age_band']
        # Initialize the LabelEncoder
        label_encoders = {col: LabelEncoder() for col in categorical_columns}
        numeric_column = 'studied_credits'

        for col in categorical_columns:
            label_encoders[col].fit(dataset_orig_train[col])
        scaler = StandardScaler()
        scaler.fit(dataset_orig_train[[numeric_column]])

        def pre_process(df):

            # Apply label encoding to each column
            for col in categorical_columns:
                df[col] = label_encoders[col].transform(df[col])

            # Normalize the numeric column
            df['disability'] = df['disability'].replace({
                'Y': 0,
                'N': 1
            })
            df['disability'] = df['disability'].replace({
                'Yes': 0,
                'No': 1
            })

            df['gender'] = df['gender'].replace({
                'Female': 0,
                'Male': 1
            })

            df[numeric_column] = scaler.transform(df[[numeric_column]])

            df['Probability'] = np.where(df['Probability'] >= 0.5, 1, 0)

            df = df.astype(float)
            return df

        dataset_orig_train = pre_process(dataset_orig_train)
        dataset_orig_test = pre_process(dataset_orig_test)
        protected_attributes = ['gender', 'disability']

    elif 'default' in train_dataset_path:
        # Load dataset
        dataset_orig = pd.read_csv(train_dataset_path)

        # Drop NULL values
        dataset_orig = dataset_orig.dropna()

        # Change column values
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 2, 0, 1)
        mean_age = dataset_orig.loc[:, "AGE"].mean()
        dataset_orig['AGE'] = np.where(dataset_orig['AGE'] >= mean_age, 1, 0)

        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)
        protected_attributes = ['sex', 'AGE']

    elif 'student_performance' in train_dataset_path:
        # Load dataset

        dataset_orig_train = pd.read_csv(train_dataset_path)

        dataset_orig_test = pd.read_csv(test_dataset_path)
        age_mean = dataset_orig_train.loc[:, "age"].mean()

        def pre_process(dataset_orig):

            dataset_orig = dataset_orig.dropna()
            dataset_orig['age'] = np.where(dataset_orig['age'] >= age_mean, 1, 0)

            dataset_orig['schoolsup'] = np.where(dataset_orig['schoolsup'] == 'yes', 1, 0)
            dataset_orig['sex'] = dataset_orig['sex'].replace({
                'F': 0,
                'M': 1
            })

            prob_mean = dataset_orig.loc[:, "Probability"].mean()

            dataset_orig['Probability'] = np.where(dataset_orig['Probability'] >= prob_mean, 1, 0)
            dataset_orig['famsup'] = np.where(dataset_orig['famsup'] == 'yes', 1, 0)
            dataset_orig['paid'] = np.where(dataset_orig['paid'] == 'yes', 1, 0)
            dataset_orig['activities'] = np.where(dataset_orig['activities'] == 'yes', 1, 0)
            dataset_orig['nursery'] = np.where(dataset_orig['nursery'] == 'yes', 1, 0)
            dataset_orig['higher'] = np.where(dataset_orig['higher'] == 'yes', 1, 0)
            dataset_orig['internet'] = np.where(dataset_orig['internet'] == 'yes', 1, 0)
            dataset_orig['romantic'] = np.where(dataset_orig['romantic'] == 'yes', 1, 0)

            mean = dataset_orig.loc[:, "G1"].mean()
            dataset_orig['G1'] = np.where(dataset_orig['G1'] >= mean, 1, 0)

            mean = dataset_orig.loc[:, "G2"].mean()
            dataset_orig['G2'] = np.where(dataset_orig['G2'] >= mean, 1, 0)
            return dataset_orig

        dataset_orig_train = pre_process(dataset_orig_train)
        dataset_orig_test = pre_process(dataset_orig_test)

        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        dataset_orig_train = pd.DataFrame(scaler.fit_transform(dataset_orig_train), columns=dataset_orig_train.columns)
        dataset_orig_test = pd.DataFrame(scaler.fit_transform(dataset_orig_test), columns=dataset_orig_test.columns)

        protected_attributes = ['sex', 'health']

    elif dataset_name == 'student_dropout':
        # Load dataset
        dataset_orig_train = pd.read_csv(train_dataset_path)
        dataset_orig_test = pd.read_csv(test_dataset_path)

        dataset_orig_train = dataset_orig_train.dropna()

        # Drop NULL values

        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        dataset_orig_train = pd.DataFrame(scaler.fit_transform(dataset_orig_train), columns=dataset_orig_train.columns)
        dataset_orig_test = pd.DataFrame(scaler.fit_transform(dataset_orig_test), columns=dataset_orig_test.columns)

        protected_attributes = ['Debtor', 'Gender']

    elif dataset_name == 'student_academic_performance':
        # Load dataset
        dataset_orig = pd.read_csv('../datasets/xAPI-Edu-Data.csv')

        # Drop categorical features
        dataset_orig = dataset_orig.drop(
            ['PlaceofBirth', 'GradeID', 'SectionID', 'Topic', 'Semester', 'Relation', 'StudentAbsenceDays'], axis=1)

        # Drop NULL values
        dataset_orig = dataset_orig.dropna()

        # Change Column values
        dataset_orig['NationalITy'] = np.where(dataset_orig['NationalITy'] == 'USA', 1, 0)
        dataset_orig['ParentAnsweringSurvey'] = np.where(dataset_orig['ParentAnsweringSurvey'] == 'Yes', 1, 0)
        dataset_orig['ParentschoolSatisfaction'] = np.where(dataset_orig['ParentschoolSatisfaction'] == 'Good', 1, 0)
        dataset_orig['StageID'] = np.where(dataset_orig['StageID'] == 'HighSchool', 1, 0)
        dataset_orig['gender'] = np.where(dataset_orig['gender'] == 'M', 1, 0)

        mean = dataset_orig.loc[:, "raisedhands"].mean()
        dataset_orig['raisedhands'] = np.where(dataset_orig['raisedhands'] >= mean, 1, 0)

        mean = dataset_orig.loc[:, "VisITedResources"].mean()
        dataset_orig['VisITedResources'] = np.where(dataset_orig['VisITedResources'] >= mean, 1, 0)

        mean = dataset_orig.loc[:, "AnnouncementsView"].mean()
        dataset_orig['AnnouncementsView'] = np.where(dataset_orig['AnnouncementsView'] >= mean, 1, 0)

        mean = dataset_orig.loc[:, "Discussion"].mean()
        dataset_orig['Discussion'] = np.where(dataset_orig['Discussion'] >= mean, 1, 0)

        # Make goal column binary
        dataset_orig['Probability'] = np.where(dataset_orig['Class'] == 'H', 1, 0)
        dataset_orig = dataset_orig.drop(['Class'], axis=1)

        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)

        protected_attributes = ['gender', 'NationalITy']
    elif dataset_name == 'DNU':
        # Load dataset
        # dataset_orig = pd.read_csv('../datasets/DNU.csv')
        dataset_orig_train = pd.read_csv(train_dataset_path)
        dataset_orig_test = pd.read_csv(test_dataset_path)

        dataset_orig_train = dataset_orig_train.dropna()

        from sklearn.preprocessing import MinMaxScaler
        mean = dataset_orig_train.loc[:, "Probability"].mean()
        dataset_orig_train['Probability'] = np.where(dataset_orig_train['Probability'] >= mean, 1, 0)
        dataset_orig_train['gender'] = np.where(dataset_orig_train['gender'] == 'Male', 1, 0)

        scaler = MinMaxScaler()

        dataset_orig_train = pd.DataFrame(scaler.fit_transform(dataset_orig_train), columns=dataset_orig_train.columns)
        dataset_orig_test = pd.DataFrame(scaler.fit_transform(dataset_orig_test), columns=dataset_orig_test.columns)

        protected_attributes = ['gender', 'age', 'birthplace']

    # if __name__ == '__main__':

    acc_metric = {'logistic': [],
                  'random_forest': [],
                  'boost': [],
                  'decision_tree': [],
                  'neural_network': []}

    ce_list = []
    recall_a = copy.deepcopy(acc_metric)
    recall_b = copy.deepcopy(acc_metric)
    false_a = copy.deepcopy(acc_metric)
    false_b = copy.deepcopy(acc_metric)
    acc_a = copy.deepcopy(acc_metric)
    acc_b = copy.deepcopy(acc_metric)
    precision_a = copy.deepcopy(acc_metric)
    precision_b = copy.deepcopy(acc_metric)
    f1_score_a = copy.deepcopy(acc_metric)
    f1_score_b = copy.deepcopy(acc_metric)

    DI_b = {
        model: {attr: [] for attr in protected_attributes}
        for model in ['logistic', 'random_forest', 'boost', 'decision_tree', 'neural_network']
    }

    SPD_b = copy.deepcopy(DI_b)
    DI_a = copy.deepcopy(DI_b)
    SPD_a = copy.deepcopy(DI_b)
    aod_b = copy.deepcopy(DI_b)
    eod_b = copy.deepcopy(DI_b)
    aod_a = copy.deepcopy(DI_b)
    eod_a = copy.deepcopy(DI_b)
    running_time = []

    ce_times = []

    iterations = 1
    index = 1
    while (index <= iterations):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            # try:
            if True:
                for k in range(iterations):
                    print('------the {}/{}th turn------'.format(k, iterations))
                    start_time = time.time()
                    # dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.15,
                    #                                                          random_state=None,
                    #                                                          shuffle=True)

                    dataset_orig_train_i, dataset_orig_test_i = dataset_orig_train.copy(
                        deep=True), dataset_orig_test.copy(deep=True)

                    X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], \
                        dataset_orig_train[
                            'Probability']
                    X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], \
                        dataset_orig_test[
                            'Probability']

                    column_train = [column for column in X_train]

                    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                    from sklearn.neural_network import MLPClassifier

                    clfs = {'logistic': LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100),
                            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3),
                            'boost': GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=3,
                                                                random_state=0),
                            'decision_tree': DecisionTreeClassifier(max_depth=3, random_state=13),
                            'neural_network': MLPClassifier(hidden_layer_sizes=(len(dataset_orig_train.columns) // 2),
                                                            max_iter=1000,
                                                            random_state=42)}

                    from aif360.datasets import BinaryLabelDataset


                    dataset_t = BinaryLabelDataset(favorable_label=1.0,
                                                   unfavorable_label=0.0,
                                                   df=dataset_orig_test,
                                                   label_names=['Probability'],
                                                   protected_attribute_names=protected_attributes
                                                   )
                    y_preds = []
                    for clf_name, clf_i in clfs.items():
                        clf_i.fit(X_train, y_train)
                        y_pred = clf_i.predict(X_test)

                        dataset_pred = dataset_t.copy()
                        dataset_pred.labels = y_pred
                        attrs = dataset_t.protected_attribute_names
                        privileged_groups = {
                            attr: dataset_pred.privileged_protected_attributes[
                                dataset_t.protected_attribute_names.index(attr)][0] for
                            attr in attrs}
                        unprivileged_groups = {
                            attr: dataset_pred.unprivileged_protected_attributes[
                                dataset_t.protected_attribute_names.index(attr)][0] for
                            attr in attrs}

                        class_metrics = {
                            attr: ExtendedClassificationMetric(dataset_t, dataset_pred,
                                                       unprivileged_groups=[{attr: unprivileged_groups[attr]}],
                                                       privileged_groups=[{attr: privileged_groups[attr]}]) for attr in
                            attrs}

                        for attr in attrs:
                            DI_b[clf_name][attr].append(class_metrics[attr].disparate_impact())
                            SPD_b[clf_name][attr].append(class_metrics[attr].statistical_parity_difference())
                            aod_b[clf_name][attr].append(class_metrics[attr].average_odds_difference())
                            eod_b[clf_name][attr].append(class_metrics[attr].equal_opportunity_difference())

                        acc_b[clf_name].append(class_metrics[attrs[0]].accuracy())
                        recall_b[clf_name].append(class_metrics[attrs[0]].recall())
                        false_b[clf_name].append(class_metrics[attrs[0]].false_positive_rate())

                        confusion_matrix = class_metrics[attrs[0]].binary_confusion_matrix()
                        TP = confusion_matrix['TP']
                        FP = confusion_matrix['FP']
                        FN = confusion_matrix['FN']

                        # Precision
                        precision = TP / (TP + FP)

                        # Recall
                        recall = TP / (TP + FN)

                        # F1-score
                        f1_score = 2 * (precision * recall) / (precision + recall)

                        precision_b[clf_name].append(precision)
                        f1_score_b[clf_name].append(f1_score)

                    slope_store = []
                    intercept_store = []
                    pvalue_store = []
                    column_u = []
                    flag = 0
                    ce = []

                    def Multiple_Linear_regression(x_values, slope, intercept, p_values):
                        result = intercept
                        for i, attribute in enumerate(protected_attributes):
                            current_slope = slope[i] if p_values[protected_attributes[i]] < 0.05 else 0
                            result += x_values[protected_attributes[i]] * current_slope
                        return result

                    def get_model_parameters(model, current_X, current_feature):
                        slope = model.coef_
                        intercept = model.intercept_
                        r2value = r2_score(current_feature, model.predict(current_X))
                        return slope, intercept, r2value

                    current_running_time = 0
                    for i in column_train:
                        flag = flag + 1
                        if i not in protected_attributes:
                            current_X = X_train[protected_attributes]
                            current_X_sm = sm.add_constant(current_X)
                            current_y = X_train[i]
                            model = sm.OLS(current_y, current_X_sm).fit()
                            p_values = model.pvalues

                            LR_model = linear_model.LinearRegression()

                            current_start_time = time.time()
                            LR_model.fit(current_X, current_y)

                            slope, intercept, rvalue = get_model_parameters(LR_model, current_X, current_y)

                            # pvalue_store.append(p_values)

                            if p_values[protected_attributes[0]] < 0.05 or p_values[protected_attributes[1]] < 0.05:
                                column_u.append(i)
                                ce.append(flag)
                                slope_store.append(slope)
                                intercept_store.append(intercept)

                                X_train.loc[:, i] = X_train.loc[:, i] - Multiple_Linear_regression(X_train,
                                                                                                   slope,
                                                                                                   intercept,
                                                                                                   p_values)

                    for i in range(len(column_u)):
                        X_test.loc[:, column_u[i]] = X_test.loc[:, column_u[i]] - Multiple_Linear_regression(
                            X_test,
                            slope_store[i],
                            intercept_store[i],
                            p_values)

                    X_train = X_train.drop(protected_attributes, axis=1)
                    X_test = X_test.drop(protected_attributes, axis=1)

                    from aif360.datasets import BinaryLabelDataset
                    from aif360.metrics import ClassificationMetric

                    dataset_t = BinaryLabelDataset(favorable_label=1.0,
                                                   unfavorable_label=0.0,
                                                   df=dataset_orig_test,
                                                   label_names=['Probability'],
                                                   protected_attribute_names=protected_attributes,
                                                   )
                    for clf_name, clf_i in clfs.items():
                        clf_i.fit(X_train, y_train)
                        y_pred = clf_i.predict(X_test)
                        dataset_pred = dataset_t.copy()
                        dataset_pred.labels = y_pred
                        attrs = dataset_t.protected_attribute_names
                        privileged_groups = {
                            attr: dataset_pred.privileged_protected_attributes[
                                dataset_t.protected_attribute_names.index(attr)][0] for
                            attr in attrs}
                        unprivileged_groups = {
                            attr: dataset_pred.unprivileged_protected_attributes[
                                dataset_t.protected_attribute_names.index(attr)][0] for
                            attr in attrs}

                        class_metrics = {
                            attr: ExtendedClassificationMetric(dataset_t, dataset_pred,
                                                       unprivileged_groups=[{attr: unprivileged_groups[attr]}],
                                                       privileged_groups=[{attr: privileged_groups[attr]}]) for attr in
                            attrs}
                        for attr in attrs:
                            DI_a[clf_name][attr].append(class_metrics[attr].disparate_impact())
                            SPD_a[clf_name][attr].append(class_metrics[attr].statistical_parity_difference())
                            aod_a[clf_name][attr].append(class_metrics[attr].average_odds_difference())
                            eod_a[clf_name][attr].append(class_metrics[attr].equal_opportunity_difference())

                        acc_a[clf_name].append(class_metrics[attrs[0]].accuracy())
                        recall_a[clf_name].append(class_metrics[attrs[0]].recall())
                        false_a[clf_name].append(class_metrics[attrs[0]].false_positive_rate())

                        confusion_matrix = class_metrics[attrs[0]].binary_confusion_matrix()
                        TP = confusion_matrix['TP']
                        FP = confusion_matrix['FP']
                        FN = confusion_matrix['FN']

                        # Precision
                        precision = TP / (TP + FP)

                        # Recall
                        recall = TP / (TP + FN)

                        # F1-score
                        f1_score = 2 * (precision * recall) / (precision + recall)

                        precision_a[clf_name].append(precision)
                        f1_score_a[clf_name].append(f1_score)

                    current_training_time = time.time() - start_time
                    current_running_time += current_training_time
                    running_time.append(current_running_time)
                    index = index + 1
            # except RuntimeWarning as e:
            #     print(e)
            #     continue
    df = save_results_to_file(aod_b, eod_b, DI_b, SPD_b,
                              acc_b, false_b, precision_b, f1_score_b, recall_b,
                              aod_a, eod_a, DI_a, SPD_a,
                              acc_a, false_a, precision_a, f1_score_a, recall_a,
                              protected_attributes,
                              running_time,
                              out_path=train_dataset_path.replace('merged_output', f'results_{dataset_name}'))  # or "results.csv" or "results.pkl")
    #
    # for clf_name in aod_b.keys():
    #     print(f'\n\nRESULT FROM: {clf_name} model:')
    #     print('---Original---')
    #     print('Acc before:', np.mean(acc_b[clf_name]))
    #     print('Far before:', np.mean(false_b[clf_name]))
    #     print('Precision before', np.mean(precision_b[clf_name]))
    #     print('F1 Score before', np.mean(f1_score_b[clf_name]))
    #     print('recall before:', np.mean(recall_b[clf_name]))
    #     for attr in protected_attributes:
    #         print(f'Aod_{attr} before:', np.mean(np.abs(aod_b[clf_name][attr])))
    #         print(f'Eod_{attr} before:', np.mean(np.abs(eod_b[clf_name][attr])))
    #         print(f'DI_{attr} before:', np.mean(DI_b[clf_name][attr]))
    #         print(f'SPD_{attr} before:', np.mean(np.abs(SPD_b[clf_name][attr])))
    #
    #     print('---FAIREDU---')
    #     print('Acc after:', np.mean(acc_a[clf_name]))
    #     print('Far after:', np.mean(false_a[clf_name]))
    #     print('Precision after', np.mean(precision_a[clf_name]))
    #     print('F1 Score after', np.mean(f1_score_a[clf_name]))
    #     print('recall after:', np.mean(recall_a[clf_name]))
    #     for attr in protected_attributes:
    #         print(f'Aod_{attr} after:', np.mean(np.abs(aod_a[clf_name][attr])))
    #         print(f'Eod_{attr} after:', np.mean(np.abs(eod_a[clf_name][attr])))
    #         print(f'DI_{attr} after:', np.mean(DI_a[clf_name][attr]))
    #         print(f'SPD_{attr} after:', np.mean(np.abs(SPD_a[clf_name][attr])))
    #
    #     print(f'Average running time: ', np.mean(running_time))
    #


def save_results_to_file(
        aod_b, eod_b, DI_b, SPD_b,
        acc_b, false_b, precision_b, f1_score_b, recall_b,
        aod_a, eod_a, DI_a, SPD_a,
        acc_a, false_a, precision_a, f1_score_a, recall_a,
        protected_attributes,
        running_time,
        out_path="results.xlsx",
):
    """
    Collect metrics for each classifier and save them to a file.

    out_path:
        - ends with .xlsx -> saved as Excel
        - ends with .csv  -> saved as CSV
        - anything else   -> saved as pickle
    """
    rows = []

    avg_runtime = float(np.mean(running_time))

    for clf_name in aod_b.keys():
        # --- BEFORE / ORIGINAL ---
        row_before = {
            "model": clf_name,
            "Luong": "2",  # Original
            "Acc": float(np.mean(acc_b[clf_name])),
            "Far": float(np.mean(false_b[clf_name])),
            "Precision": float(np.mean(precision_b[clf_name])),
            "F1": float(np.mean(f1_score_b[clf_name])),
            "Recall": float(np.mean(recall_b[clf_name])),
            "Avg_runtime": avg_runtime,
        }

        for attr in protected_attributes:
            row_before[f"Aod_{attr}"] = float(np.mean(np.abs(aod_b[clf_name][attr])))
            row_before[f"Eod_{attr}"] = float(np.mean(np.abs(eod_b[clf_name][attr])))
            row_before[f"DI_{attr}"] = float(np.mean(DI_b[clf_name][attr]))
            row_before[f"SPD_{attr}"] = float(np.mean(np.abs(SPD_b[clf_name][attr])))

        rows.append(row_before)

        # --- AFTER / FAIREDU ---
        row_after = {
            "model": clf_name,
            "Luong": "1",  # FAIREDU
            "Acc": float(np.mean(acc_a[clf_name])),
            "Far": float(np.mean(false_a[clf_name])),
            "Precision": float(np.mean(precision_a[clf_name])),
            "F1": float(np.mean(f1_score_a[clf_name])),
            "Recall": float(np.mean(recall_a[clf_name])),
            "Avg_runtime": avg_runtime,
        }

        for attr in protected_attributes:
            row_after[f"Aod_{attr}"] = float(np.mean(np.abs(aod_a[clf_name][attr])))
            row_after[f"Eod_{attr}"] = float(np.mean(np.abs(eod_a[clf_name][attr])))
            row_after[f"DI_{attr}"] = float(np.mean(DI_a[clf_name][attr]))
            row_after[f"SPD_{attr}"] = float(np.mean(np.abs(SPD_a[clf_name][attr])))

        rows.append(row_after)

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Save based on extension
    if out_path.endswith(".xlsx"):
        df.to_excel(out_path, index=False)
    elif out_path.endswith(".csv"):
        df.to_csv(out_path, index=False)
    else:
        df.to_pickle(out_path)

    return df




class ExtendedClassificationMetric(ClassificationMetric):
    def performance_measures(self, privileged=None):
        """Compute various performance measures on the dataset, optionally
        conditioned on protected attributes.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Returns:
            dict: True positive rate, true negative rate, false positive rate,
            false negative rate, positive predictive value, negative predictive
            value, false discover rate, false omission rate, and accuracy
            (optionally conditioned).
        """
        TP = self.num_true_positives(privileged=privileged)
        FP = self.num_false_positives(privileged=privileged)
        FN = self.num_false_negatives(privileged=privileged)
        TN = self.num_true_negatives(privileged=privileged)
        GTP = self.num_generalized_true_positives(privileged=privileged)
        GFP = self.num_generalized_false_positives(privileged=privileged)
        GFN = self.num_generalized_false_negatives(privileged=privileged)
        GTN = self.num_generalized_true_negatives(privileged=privileged)
        P = self.num_positives(privileged=privileged)
        N = self.num_negatives(privileged=privileged)

        if N == 0:
            N = 1

        TNR = TN / N if N > 0.0 else np.float64(0.0)
        FPR = FP / N if N > 0.0 else np.float64(0.0)
        FNR = FN / N if N > 0.0 else np.float64(0.0)

        TPR = TP / P if P > 0.0 else np.float64(0.0)
        GTPR = GTP / P if P > 0.0 else np.float64(0.0)
        GFNR = GFN / P if P > 0.0 else np.float64(0.0)
        GTNR = GTN / N if N > 0.0 else np.float64(0.0)
        GFPR = GFP / N if N > 0.0 else np.float64(0.0)

        # TNR = TN / N
        # FPR = FP / N
        # FNR = FN / N
        #
        # TPR = TP / P
        # GTPR = GTP / P
        # GFNR = GFN / P
        # GTNR = GTN / N
        # GFPR = GFP / N


        return dict(
            TPR=TPR, TNR=TNR, FPR=FPR, FNR=FNR,
            GTPR=GTPR, GTNR=GTNR, GFPR=GFPR, GFNR=GFNR,
            PPV=TP / (TP + FP) if (TP + FP) > 0.0 else np.float64(0.0),
            NPV=TN / (TN + FN) if (TN + FN) > 0.0 else np.float64(0.0),
            FDR=FP / (FP + TP) if (FP + TP) > 0.0 else np.float64(0.0),
            FOR=FN / (FN + TN) if (FN + TN) > 0.0 else np.float64(0.0),
            ACC=(TP + TN) / (P + N) if (P + N) > 0.0 else np.float64(0.0)
        )
