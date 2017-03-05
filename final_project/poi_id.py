#!/usr/bin/python

# poi_id.py
# Creates a dataset and classifier definition fot the
# final project in the udacity ud120 machine learning introduction course
# Note: script changed to run on python 3.6
#
# to run: python3 poi_id.py

import matplotlib.pyplot as plt
import numpy
#
from sklearn.naive_bayes import GaussianNB
#
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
#
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
#
from sklearn.grid_search import GridSearchCV
#
import sys
import pickle
#
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


def tuneDecisionTreeClassifier(features, labels):
    classifier = tree.DecisionTreeClassifier()
    parameters = {'min_samples_split':[2, 4, 6, 8, 10], 'max_features':[2,3,4,5]}
    tuned_classifier = GridSearchCV(classifier, parameters)
    tuned_classifier.fit(features, labels)
    print("best estimator = ", tuned_classifier.best_params_)

def getTunedDecisionTree(features, labels):
    #return tree.DecisionTreeClassifier(min_samples_split=8, max_features=4)
    clf = RandomForestClassifier(n_estimators=10, min_samples_split=8, max_features=4)
    clf.fit(features, labels)
    print("Feature importances", clf.feature_importances_ )
    return clf

def tuneSVM(features, labels):
    classifier = svm.SVC()
    selector = SelectKBest(k=4)
    scaler = StandardScaler()
    pca = PCA()
    parameters = {'C':[1, 10, 20, 50, 100, 200, 500], 'kernel':['linear', 'rbf']}
    #unfortunately the pipeline does not work with GridSearchCV
    tuned_classifier = GridSearchCV(classifier.fit(
        scaler.fit_transform(selector.fit_transform(pca.fit_transform(features, labels), labels), labels), labels), parameters)
    tuned_classifier.fit(features, labels)
    print("best estimator = ", tuned_classifier.best_params_)

def getTunedSVM():
    # returns a SVM in a pipeline

    # salary and total_payments are correlated, the same goes for other features, so use PCA to factor those out.
    pca = PCA()
    # to prevent overfitting, a limited number of the best components is selected
    # number of components was determined with trial and error from 3 to all
    selector = SelectKBest(k=4)
    # scale features for svm so each feature is treaded with equal importance
    scaler = StandardScaler()
    # create the tuned classifier (tuned with a GridSearchCV)
    classifier = svm.SVC(kernel = 'linear', C=50)
    # put the pca, selector, scaler and classifier in a pipeline for the grader
    pipeline = Pipeline([("pca",pca),("selector", selector), ("scaler",scaler),("classifier",classifier)])
    #
    return pipeline

def getTunedNB():
    # returns a GaussianNB classifier in a pipeline

    # salary and total_payments are correlated, the same goes for other features, so use PCA to factor those out.
    pca = PCA()
    # to prevent overfitting, a limited number of the best components is selected
    # number of components was determined with trial and error from 3 to all
    selector = SelectKBest(k=4)
    # scale features for svm so each feature is treaded with equal importance
    scaler = StandardScaler()
    # create the tuned classifier (tuned with a GridSearchCV)
    classifier = GaussianNB()
    # put the pca, selector, scaler and classifier in a pipeline for the grader
    pipeline = Pipeline([("pca",pca),("selector", selector), ("scaler",scaler),("classifier",classifier)])
    #
    return pipeline


# create extra derived features
# not all derived features are used, but all were played around with and visualised in a scatterplot
def create_features(dataset):
    for key in data_dict.keys():
        element = data_dict[key]
        from_messages = element['from_messages']
        from_poi_to_this_person = element['from_poi_to_this_person']
        from_this_person_to_poi = element['from_this_person_to_poi']
        to_messages = element['to_messages']
        shared_receipt_with_poi = element['shared_receipt_with_poi']
        total_payments = element['total_payments']
        total_stock_value = element['total_stock_value']
        try:
            # added devide by 1 to force numeric interpretation
            element['total_value'] = (total_payments + total_stock_value) / 1
            element['total_poi_emails'] = (from_poi_to_this_person + from_this_person_to_poi ) / 1
            element['total_poi_emails_as_fraction_of_total'] = (from_this_person_to_poi+from_poi_to_this_person) / (to_messages+from_messages)
            element['from_poi_emails_as_fraction_of_total'] = (from_poi_to_this_person) / (from_messages)
            element['to_poi_emails_as_fraction_of_total'] = (from_poi_to_this_person) / (to_messages)
            element['shared_receipts_with_poi_as_fraction_of_inbox'] = (shared_receipt_with_poi) / (to_messages)
        except:
            element['total_value'] = "NaN"
            element['total_poi_emails'] = "NaN"
            element['total_poi_emails_as_fraction_of_total'] = "NaN"
            element['from_poi_emails_as_fraction_of_total'] = "NaN"
            element['to_poi_emails_as_fraction_of_total'] = "NaN"
            element['shared_receipts_with_poi_as_fraction_of_inbox'] ="NaN"
    return data_dict

# not used
# considered looking for references to LJM in inboxes, sent by POI's before 14 august 2001
# realised that most people in the data set are not in the mail zip file
# but luckily decided to tune decision tree with derived features first.
def count_ljm_references(key, element):
    sys.path.append( "../tools/" )
    from parse_out_email_text import parseOutText
    import os

    # get email references to LJM (Debt hiding vehicle)
    element['LJM_mentions_in_sent_mail'] = 0.
    keyparts = key.split()
    initial = keyparts[len(keyparts)-1]
    lastname = keyparts[0]
    if len(initial) == 1:
        email_address = lastname.lower() + "-" + initial.lower()
        path = "../maildir/"+email_address+"/inbox/"
        if os.path.exists(path):
            temp_counter = 0
            for filename in os.listdir(path):
                temp_counter = temp_counter + 1
                email = open(path + filename, "r")
                email_text = parseOutText(email)
                if email_text.find("LJM"):
                    ljm_mentions = element['LJM_mentions_in_sent_mail']
                    element['LJM_mentions_in_sent_mail'] = ljm_mentions + 1
    print("Processing key ", key, " LJM Mentions: ", element['LJM_mentions_in_sent_mail'])

def visualise_features(data_dictionary, feature_x, feature_y, show=False, text_labels=True):
    # visualise two axes of the feature space
    # last two parameters determine if the interactive python graph is run
    # and if the names of the labels are shown (useful for outlier evaluation)

    plt.clf()
    x = numpy.zeros(len(data_dictionary))
    y = numpy.zeros(len(data_dictionary))
    colours = []
    labels = list(data_dictionary.keys())
    for i in range(0, len(data_dictionary)-1):
        label = labels[i]
        x_data = data_dictionary[label][feature_x]
        y_data = data_dictionary[label][feature_y]
        if x_data!="NaN":
            x[i] = x_data
        if y_data!="NaN":
            y[i] = y_data
        if data_dictionary[label]['poi']:
            colours.append('red')
        else:
            colours.append('green')
        if text_labels:
            plt.annotate(label, (x[i], y[i]))
    plt.scatter(x, y, color=colours)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    if show:
        plt.show()
    filename = feature_x + "-" + feature_y +".png"
    plt.savefig(filename, bbox_inches='tight')


### Task 1: Select what features you'll use.
#Moved further down (after feature creation)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
#visually identified the TOTAL outlier in a scatterplot
data_dict.pop('TOTAL', None)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = create_features(data_dict)

# write scatterplot to disk for use in headless docker instance
visualise_features(my_dataset, 'total_value', 'shared_receipts_with_poi_as_fraction_of_inbox', False, False)
visualise_features(my_dataset, 'total_value', 'shared_receipt_with_poi', False, False)

# features selected by hand after visualising in a scatterplot
features_list = ['poi', 'total_payments', 'total_stock_value', 'exercised_stock_options'
, 'shared_receipts_with_poi_as_fraction_of_inbox', 'long_term_incentive']

# print the used features to the log so they are visible above the scores.
print(features_list)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, remove_all_zeroes=True, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.

# --- Results shown below of initial classifiers
#
# --- Initially SVM gives bad recall: Accuracy: 0.85733	Precision: 0.37085	Recall: 0.10050
# SVC experiments:
# linear kernel initially gives improved precision and recall over rbf kernel
# rbf: Accuracy: 0.85700	Precision: 0.14976	Recall: 0.01550	F1: 0.02809
# linear: Accuracy: 0.85707	Precision: 0.37004	Recall: 0.10250	F1: 0.16053
#
# --- GaussianNB gives reasonable recall and good Precision, but was unable to tune further
# Adding more features reduced the Recall score.
# Results:
# Pipeline(steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
#   svd_solver='auto', tol=0.0, whiten=False)), ('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('classifier', GaussianNB(priors=None))])
# 	Accuracy: 0.85253	Precision: 0.41627	Recall: 0.26350	F1: 0.32272	F2: 0.28437
# 	Total predictions: 15000	True positives:  527	False positives:  739	False negatives: 1473	True negatives: 12261
#
# --- Decision Tree gives best Precision and Recall and is in the final script
# out of the box decision tree gives reasonable recall and precision: Accuracy: 0.79107	Precision: 0.23799	Recall: 0.25750	F1: 0.24736


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# --- Best results with tuned linear SVC still have disappointing recall:
#   Pipeline(steps=[('pca', PCA(copy=True, n_components=None, whiten=False)), ('selector', SelectKBest(k=17, score_func=<function f_classif at 0x7f847445f378>)), ('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('classifier', SVC(C=50, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False))])
# 	Accuracy: 0.83967	Precision: 0.33630	Recall: 0.20800	F1: 0.25703	F2: 0.22518
# 	Total predictions: 15000	True positives:  416	False positives:  821	False negatives: 1584	True negatives: 12179
#
# --- Tuned DecisionTreeClassifier gives adequate Recall and Precision scores
#   DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#             max_features=3, max_leaf_nodes=None, min_samples_leaf=1,
#             min_samples_split=8, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='best')
# 	Accuracy: 0.83080	Precision: 0.34905	Recall: 0.31100	F1: 0.32893	F2: 0.31793
# 	Total predictions: 15000	True positives:  622	False positives: 1160	False negatives: 1378	True negatives: 11840
#
# --- Tuned RandomForestClassifier gives even better scores
#   RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=None, max_features=4, max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=1,
#             min_samples_split=8, min_weight_fraction_leaf=0.0,
#             n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
#             verbose=0, warm_start=False)
# 	Accuracy: 0.87367	Precision: 0.53892	Recall: 0.36350	F1: 0.43416	F2: 0.38881
# 	Total predictions: 15000	True positives:  727	False positives:  622	False negatives: 1273	True negatives: 12378

# --- Tune the classifier
# tuneDecisionTreeClassifier(features, labels)
#
# --- note: tuneSVM Takes hours to run
# tuneSVM(features, labels)

# clf = getTunedSVM()
# clf = getTunedNB()
clf = getTunedDecisionTree(features, labels)

# Evaluation by the cross_validation in tester.py
import tester
tester.main()

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
