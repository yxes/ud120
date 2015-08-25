#!/usr/bin/python

"""
A Methodological Proceedure for Feature Selection
-------------------------------------------------

Output a table that shows the precision and recall
values using the DecisionTreeClassifier as
we run through SelectPercentile features for 
different percentages.

Output an ordered list of features given by
SelectPercentile along with their weights
"""
import sys
import pickle
from math import isnan
sys.path.append("./tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import StratifiedShuffleSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Available features
features_list = [
        'poi',
        'salary',
        'deferral_payments',
        'total_payments',
        'loan_advances',
        'bonus',
        'restricted_stock_deferred',
        'deferred_income',
        'total_stock_value',
        'expenses',
        'exercised_stock_options',
        'other',
        'long_term_incentive',
        'restricted_stock',
        'director_fees',
        'to_messages',
        'from_messages',
        'from_poi_to_this_person',
        'from_this_person_to_poi',
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
                    'total_payments',
                    'from_poi_to_this_person',
                    'from_this_person_to_poi',
                    'to_messages',
                    'from_messages'
                   ]:
        if isnan(float(data_dict[poi][feature])):
            data_dict[poi][feature] = 0.0

  # MILKing the company
    data_dict[poi]['milk'] = (data_dict[poi]['expenses'] +\
                              data_dict[poi]['deferral_payments']) / \
                             (1 + data_dict[poi]['loan_advances'] + \
                              data_dict[poi]['long_term_incentive'] + \
                              data_dict[poi]['deferred_income'])

  # Deferred Income to Total Payments
    if data_dict[poi]['total_payments'] > 0:
       data_dict[poi]['fraction_of_deferred_income_to_total_payments'] = \
           data_dict[poi]['deferred_income'] / data_dict[poi]['total_payments']
    else:
       data_dict[poi]['fraction_of_deferred_income_to_total_payments'] = 0

  # Email: fraction of emails to / from poi based on total emails
    if data_dict[poi]['to_messages'] == 0:
        data_dict[poi]['fraction_of_emails_to_pois'] = 0.0
    else:
        data_dict[poi]['fraction_of_emails_to_pois'] = \
           float(data_dict[poi]['from_poi_to_this_person']) / float(data_dict[poi]['to_messages'])

    if data_dict[poi]['from_messages'] > 0:
        data_dict[poi]['fraction_of_emails_from_pois'] = \
           data_dict[poi]['from_this_person_to_poi'] / data_dict[poi]['from_messages']
    else:
        data_dict[poi]['fraction_of_emails_from_pois'] = 0

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Feature Selection: SelectPercentile
from sklearn.feature_selection import SelectPercentile, f_classif

print "    % | Precision | Recall | Features"
print "="*38;

features_score = { } # feature: score
best_results = { 
        'precision' : 0.0,
        'recall'    : 0.0,
        'features'  : [],
        'percent'   : 0.0
}
for percent in range(0, 101, 5):
    if percent == 0:
        continue

    fs = SelectPercentile(f_classif, percentile=percent)
    features_transformed = fs.fit_transform(features, labels)

    best_features = []
    counter = 0
    for idx, score in sorted(enumerate(fs.scores_), key=lambda score: score[1], reverse=True):
        if len(features_score.keys()) < len(features_list):
            features_score[features_list[idx+1]] = score

        counter = counter + 1
        if counter > len(features_transformed[0]):
            continue
        best_features.append(features_list[idx+1])

    ## DECISION TREE
    if len(best_features) < 2:
        continue; # can't have less than 2 features for a decision tree
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(splitter="random")
    best_features.insert(0, 'poi')
    mydata = featureFormat(my_dataset, best_features, sort_keys = True)
    mylabels, myfeatures = targetFeatureSplit(mydata)
    cv = StratifiedShuffleSplit(mylabels, 1000, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(myfeatures[ii])
            labels_train.append(mylabels[ii])
        for jj in test_idx:
            features_test.append(myfeatures[jj])
            labels_test.append(mylabels[jj])

        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        precision = 1.0 * true_positives / (true_positives + false_positives)
        recall = 1.0 * true_positives / (true_positives + false_negatives)
    except:
        print "Got a divide by zero when trying out:", clf

    print "  %3d |     %.2f |   %.2f |        %2d" % (percent, precision, recall, len(features_transformed[0]))
    if (precision > .3 and recall > .3 and
        precision + recall > best_results['precision'] + best_results['recall']):
        best_results['precision'] = precision
        best_results['recall'] = recall
        best_results['percent'] = percent
        best_results['features'] = best_features

print "="*38;
print ""
print ""
print "="*17
print " WEIGHT | FEATURE"
for feature in sorted(features_score, key=features_score.get, reverse=True):
    print "  %5.2f | %s" % (features_score[feature], feature)

print "="*17
print ""
print ""
print "Best Results for this run:"
print "Percent:",   best_results['percent']
print "Precision:", best_results['precision']
print "Recall:",    best_results['recall']
print "Features:"
for feature in best_results['features'][1:]:
    print "\t", feature

sys.exit(0)
