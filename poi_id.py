#!/usr/bin/python

import sys
import pickle
from math import isnan
sys.path.append("./tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

"""
  NEW FEATURE DESCRIPTION

  milk - (as in, "MILKing the company")

         (expenses + deferral payments) / 
         1 + (loan advances + long_term_incentive + deferred_income)

         POIs have an incentive to get as much out of the company as possible
         as they don't see a long-term future in the company. Expenses are
         consulting and reimbursements from the company and deferral payments
         are distributions from deferred compensation - these are items that the
         company must pay now. Loan advances, long term incentives and deferred
         income are items that the company must pay at a later time. If you
         believe that the company won't have the money to pay at a later time
         you will want to collect what you can now.
"""
features_list = [
                 'poi',
                 'salary',
                 'bonus',
                 'other',
                 'milk',
                 'fraction_of_deferred_income_to_total_payments'
                ]

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
for outlier in ['TOTAL','THE TRAVEL AGENCY IN THE PARK']:
    data_dict.pop(outlier,0)

### Task 3: Create new feature(s)
# Adding:
#   * milk // See detailed description above or code below
#   * fraction_of_deferred_income_to_total_payments
for poi in data_dict:
    # NaN ---> 0
    for feature in ['expenses',
                    'deferral_payments',
                    'loan_advances',
                    'long_term_incentive',
                    'deferred_income',
                    'total_payments'
                   ]:
        if isnan(float(data_dict[poi][feature])):
            data_dict[poi][feature] = 0

    data_dict[poi]['milk'] = (data_dict[poi]['expenses'] +\
                              data_dict[poi]['deferral_payments']) / \
                             (1 + data_dict[poi]['loan_advances'] + \
                              data_dict[poi]['long_term_incentive'] + \
                              data_dict[poi]['deferred_income'])

    if data_dict[poi]['total_payments'] > 0:
       data_dict[poi]['fraction_of_deferred_income_to_total_payments'] = \
           data_dict[poi]['deferred_income'] / data_dict[poi]['total_payments']
    else:
       data_dict[poi]['fraction_of_deferred_income_to_total_payments'] = 0

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

## NAIVE BAYES
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

## SUPPORT VECTOR MACHINE
#from sklearn.svm import SVC
#clf = SVC(kernel="linear")

## LINEAR SUPPORT VECTOR MACHINE
#from sklearn.svm import LinearSVC
#clf = LinearSVC()

## RANDOM FOREST
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=9)

## DECISION TREE
from sklearn import tree
#clf = tree.DecisionTreeClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# GRID SEARCH
#from sklearn.grid_search import GridSearchCV
#svr = tree.DecisionTreeClassifier()
#clf = GridSearchCV(
#           svr,
#           {'criterion': ('gini','entropy'),
#            'splitter':  ('best','random'),
#            'max_features': [1,2,3,5,9,0.1,0.2,0.25,0.5,0.75,0.8,0.9,0.99,"auto","sqrt","log2",None]})
#            'max_features': [.8],
#            'class_weight': [None, "auto"],
#            'max_leaf_nodes': [None, 2,3,4,5,6,7,8,9,10],
#})

# TUNED RESULT
clf = tree.DecisionTreeClassifier(splitter="random")
test_classifier(clf, my_dataset, features_list)

#### Feature Importance
## Uncomment to review the performance of individual features
##  for this run
#importances = clf.feature_importances_.tolist()
#print "FEATURE IMPORTANCE"
#print "Weight\tFeature"
#print "------\t-------"
#for idx, feature in enumerate(features_list[1:]):
#    print "%.2f \t%s" % (importances[idx], feature)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.
dump_classifier_and_data(clf, my_dataset, features_list)
