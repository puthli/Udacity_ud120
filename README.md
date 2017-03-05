# Udacity ud120 Machine Learning Introduction
These are the answers for the final project of the Udacity Machine Learning Introduction course.

* Student: Robin Puthli
* Date: March 2017

#### Requirements
The scripts in this repository require a working installation of:
* **python 3.4** or higher
* **scikit-learn 0.18** or higher
* numpy, scipy (whichever version scikit-learn requires)

####Links
* [Udacity course](https://www.udacity.com/course/intro-to-machine-learning--ud120)
* [scikit-learn](http://scikit-learn.org/stable/)


 ## Enron Submission Free-Response Questions!


#### Question 1
Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

#####My answer
The goal of this project is to predict new 'Persons of Interest' (POI) based on their financial and email data. A POI is someone who we suspect of insider trading or illegal activities and who we might deem (partially) responsible for the eventual collapse of ENRON. Machine learning allows us to extract patterns in data and use those patterns to automatically make predictions. In the case of this project we can use machine learning to predict new POI's based on their email and financial data. The dataset is based on publicly available records of emails and financial statements of executives and major shareholders. Both datasets overlap to a small extent, most of the people in the financial data are not present in the email data and vice versa. Some work on the dataset has been done already to add:
* the known Persons of Interest
* a number of email derivatives such as number of emails from a person to a POI

There are 146 datapoints, which is quite few, and the number of POI's is only 18, so less than 13% of the dataset. There are 20 features, of which some are summations of others.

By visually inspecting scatterplots of the data an outlier was detected and a general feel of the data was obtained. The dataset contains a number of outliers of which one was removed (TOTALS) and another was determined to be real data. A lot of the rows have missing data. Part of the missing data is handled by the starter code for the project, but additional code was written to  make sure missing data is processed correctly in derived features.

#### Question 2
What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]

##### My answer
Finally i used the following features:
* 'total_payments'
* 'total_stock_value'
* 'exercised_stock_options'
* 'shared_receipts_with_poi_as_fraction_of_inbox' (computed)
* 'long_term_incentive'

Initially i used all the features and use principle component analysis and the KBest selector in a pipeline to determine the scores of the algorithm candidates. After seeing that the performance was not improving i viewed the features one by one in a scatterplot to determine if there was any pattern. This immediately showed an outlier, which was removed.

Scaling was required to test the performance of the features with the Naive Bayes and SVM algorithms.
Scaling, PCA and the selector were later removed because they do not add anything to the selected Decision Tree performance. In addition, the max_features parameter of a Decision Tree can be used in a GridSearchCV exploration of parameters, which i did not manage with SelectKBest in a pipeline and had to separately perform.

A number of derived features where added to the dataset:
* total_value = (total_payments + total_stock_value)
* total_poi_emails = (from_poi_to_this_person + from_this_person_to_poi )
* total_poi_emails_as_fraction_of_total = (from_this_person_to_poi + from_poi_to_this_person) / (to_messages + from_messages)
* from_poi_emails_as_fraction_of_total = (from_poi_to_this_person) / (from_messages)
* to_poi_emails_as_fraction_of_total = (from_poi_to_this_person) / (to_messages)
* shared_receipts_with_poi_as_fraction_of_inbox = (shared_receipt_with_poi) / (to_messages)

Each was evaluated using the tester.py script and by visually inspecting a scatterplot. Eventually only the last derived feature shared_receipts_with_poi_as_fraction_of_inbox was left in.

The features i ended up with, have the following feature importances in the Random Forest Classifier: [ 0.12467817  0.15195423  0.32757932  0.3589953   0.03679299]. Which explains why the GridSearchCV mostly came up with max_features = 4

#### Question 3
What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

##### My answer
Because we are after a discrete classification (POI Yes/No) where the POI's in dataset are known, we need a supervised classification algorithm.

After evaluating untrained SVM, Naive Bayes and Decision Tree  Classifier algorithms it looked like SVM would perform best. Initial tuning attempts were hopeful, but eventually the recall score panned out under the limit stated in the project description (precision and recall of .3 or higher). Although the precision score was high for the tuned SVM, the recall score remained well under .3.

Because there is little to tune for Naive Bayes, i switched to a Decision Tree, which quickly yielded precision and recall scores of over 0.3. Because the features are quite weak in predicting POI's, i then moved to an ensemble algorithm; the Random Forest Classifier improved the performance considerably.

#### Question 4
What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

##### My answer
Tuning the parameters of an algorithm changes threshold values and algorithm implementations that are used to create the underlying mathematical models. Examples of tuning parameters are how variance is calculated, whether a linear or other assumption of a decision surface is made, how individual data points should influence the model.

A danger of incorrect tuning is that you can overfit the data or come up with a useless algorithm.
If the email address was added as a feature in a bag of words, it could lead to overfitting. Several of my tuning attempts for example led to no POI's being predicted at all; which gives quite good accuracy, but lousy recall.

The algorithm i picked has a number of tunable parameters that determine when the algorithm makes a decision point (much like a series of if-then-else statements, but based on feature values). Luckily these parameters can be tested in something called a Grid Search (GridSearchCV in sklearn) which allows a list of parameter values to be automatically tested. In my code i used the following parameters to tune the algorithm:
* parameters = {'min_samples_split':[2, 4, 6, 8, 10], 'max_features':[2,3,4,5]}

After tuning the Decision Tree i moved to a Random Forest method and did the same with the n_estimators parameter.


#### Question 5
What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]

##### My answer
Validation is the process of making sure the predictions of your tuned machine learning algorithm work on new data. This is done by testing the algorithm on data which was not used to train on.

A classic mistake is to train the algorithm on the same data as upon which it is tested.

However we have a small dataset (146 entries of which only 18 POI's), so we cannot afford to not use part of the data during training. Therefore I validated my tuned parameters by looking at the accuracy, precision and recall scores in the cross_validation_score of the tester.py script.
Cross validation does the following:
1. takes a number of different combinations of (training data, test data) from the dataset,
2. trains using the training data,
3. predicts using the test data,
4. calculates a number of performance metrics (accuracy, precision, recall, F1 etc),
5. repeats for each combination of (training data, test data),
6. averages the performance metrics for the above

If the recall and precision scores increased during tuning, the parameter changes were succesful.

The StratifiedShuffleSplit used in tester.py splits the dataset into a number of combinations of (training set, test set) while preserving the ratio of POI's to non-POI's. The latter is important because we have far fewer POI's than non-POI's in the dataset, so splitting the data could easily lead to a training set without any POI's.

#### Question 6
Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

##### My answer
Accuracy is important because it gives a fraction of all correct predictions. However for this project if we classify POI's falsely (measured by precision), we will be less worried than missing a POI (measured by recall).
In terms of the performance metrics we should therefore focus on tuning for a high recall.

In my algorithm the scores were:
Metric  | score  
--|--
Accuracy  |  0.87553
Precision  |  0.55019
Recall  |  0.36450
F1 | 0.43850
F2 | 0.39088
