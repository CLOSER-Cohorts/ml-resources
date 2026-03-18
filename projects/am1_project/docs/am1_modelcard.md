---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for AM1 Topic classification model

<!-- Provide a quick summary of what the model is/does. -->

This model predicts the topic that an item (e.g. Question/Variable) belongs to. The topics
that can be assigned to items are defined by the CLOSER Discovery team and are defined on
a wiki page by them.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

Given text that describes a question/variable, we assign it to one of a set of predefined topics. We will use a logistic regression model for this purpose.  The input features for this logistic regression model are text embeddings generated from the text descriptions of questions and variables and of various attributes associated with them. The output targets are numeric labels representing the topics that these embeddings. The user will be presented with the top N most likely topics as calculated by the logistic regression model (where N can be defined by users), 
and they can choose which (if any) of these is the most appropriate topic for that variable/question. 

- **Developed by:** Oliver Lyttleton
- **Model type:** Logistic regression

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** https://discovery.closer.ac.uk/
- **Paper [optional]:** https://closer.ac.uk/

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

The CLOSER project collects a large number of questions and variables from UK longitudinal population studies. The questions/variables already present in the CLOSER metadata repository have each been mapped to a topic from the CLOSER topic controlled vocabulary, which is a list of the topic areas covered by the studies. This allows users to refine search results by topics of interest, or browse questions and variables in a particular topic using the explore function. Validating the existing topic mappings for all questions/variables in the repository would not be feasible given our current resources. At present the questions/variables are manually assigned topics when they are being ingested into the CLOSER repository, which is a potentially error-prone and time-consuming task.

Providing a machine learning model that accepts input text describing the questions/ variables and suggests topics for them would reduce the amount of errors/inconsistencies that occur when topics are being manually assigned to questions or variables, and make this process more efficient. This model could also be used to find questions/variables that are already in the CLOSER metadata repository that have been assigned incorrect/inaccurate topics.



### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

Potential users of the model include CLOSER staff performing tasks like annotating study questions/
variables and validating existing topic assignments, or external studies that partner with 
CLOSER who can use the model to obtain topic suggestions for items when they are preparing data
for CLOSER.

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

We may auto suggest topics extracted from questions from an archivist xml extract, and send that to the studies, then we just need to change the workflow a bit.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

The model should not be used to fully automate topic assignment for items. It is intended to
guide and inform, not replace, human decisions. 

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Buy-in from users – work with internal users who currently work on the ingest process for new questions/variables from studies aware of this classification tool, to develop a prototype which could then be rolled out to external data providers?

Model performance – need to monitor the model’s performance over time and check for model drift.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

TBC

## How to Get Started with the Model

Use the code below to get started with the model.

y_pred=trainedModel.predict(X_test)
predictions_with_probabilities=trainedModel.predict_proba(X_test)

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

Refer to the documentation at (https://github.com/CLOSER-Cohorts/ml-resources/tree/main/projects/am1_project/docs) for information on the training data and filtering/pre-processing that is
performed on this data.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

Refer to the documentation at (https://github.com/CLOSER-Cohorts/ml-resources/tree/main/projects/am1_project/docs), in particular the data_lineage.pptx file, for information on the training
procedure.


#### Preprocessing [optional]

We may filter items where the raw input doesn't have enough information to deliver predictive
value, e.g. the question summary/variable lable is too short, the question/variable has fewer 
than N categories associated with it, a question has a set of categories associated with it that are not deemed to have predictive value, e.g. yes/no. We may need to filter items with these
characteristics from training data.


#### Training Hyperparameters

- **Training regime:** TBC

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

TBC

#### Factors

TBC

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

Accuracy, Precision, Recall, F1 score

### Results

Output from scikit-learn classification_report function:

              precision    recall  f1-score   support

       10603       0.00      0.00      0.00         4
       10605       1.00      1.00      1.00         5
       11001       1.00      0.85      0.92        13
       11002       0.00      0.00      0.00         2
       10801       0.00      0.00      0.00         1
       10704       0.75      1.00      0.86         3
       10304       1.00      0.50      0.67         4
       10317       0.65      0.68      0.67        25
       10107       0.88      0.64      0.74        11
       10807       0.75      0.82      0.78        11
       10815       1.00      0.50      0.67         2
       10106       0.33      0.14      0.20         7
       11612       0.00      0.00      0.00         2
         101       0.00      0.00      0.00         1
         103       1.00      0.60      0.75         5
       10306       0.00      0.00      0.00         1
       10908       1.00      0.67      0.80         3
       10202       0.67      1.00      0.80         8
       10404       0.00      0.00      0.00         1
       10804       0.00      0.00      0.00         1
       10205       0.00      0.00      0.00         1
       10405       0.00      0.00      0.00         1
       10816       0.80      1.00      0.89         4
       10702       0.00      0.00      0.00         1
         110       0.00      0.00      0.00         1
         104       0.80      1.00      0.89         4
       10323       1.00      0.50      0.67         2
       11609       0.00      0.00      0.00         2
       10314       0.65      0.88      0.75        17
       10320       0.60      0.75      0.67         8
       10311       0.99      0.99      0.99       136
       11003       0.76      0.52      0.62        25
         114       0.91      0.94      0.92        71
         115       1.00      0.67      0.80         3
       10906       0.50      0.50      0.50         8
       10402       0.00      0.00      0.00         2
       10901       0.00      0.00      0.00         1
       11101       0.80      0.80      0.80        10
       11615       0.88      1.00      0.94        15
       11607       1.00      1.00      1.00        11
       11608       0.50      0.25      0.33         4
         111       0.85      1.00      0.92        11
       10803       0.64      0.82      0.72        11
       10312       0.94      0.89      0.91        18
       10808       1.00      0.69      0.82        13
       11102       1.00      1.00      1.00         5
       10813       0.00      0.00      0.00         1
       10318       0.50      0.20      0.29        10
       10504       0.64      0.60      0.62        15
       10805       0.50      0.20      0.29         5
         106       0.52      1.00      0.69        11
       10310       0.76      0.89      0.82        63
       11601       1.00      0.60      0.75         5
       10309       1.00      0.50      0.67         6
       10705       1.00      0.67      0.80         9
       11603       0.64      0.89      0.74        18
       10101       0.75      1.00      0.86         6
       11614       1.00      0.80      0.89         5
       11604       0.93      1.00      0.96        27
       10403       1.00      0.50      0.67         2
       10902       1.00      0.50      0.67         2
       10322       0.83      1.00      0.91        15
       10709       0.00      0.00      0.00         2
       10321       1.00      1.00      1.00         1
         109       0.00      0.00      0.00         1
       10604       0.80      0.67      0.73         6
       10812       0.00      0.00      0.00         4
       10708       0.67      0.50      0.57         4
       10606       0.00      0.00      0.00         3
       10502       0.00      0.00      0.00         3
       10103       1.00      0.12      0.22         8
       11104       0.69      0.85      0.76        47
       10501       1.00      1.00      1.00         2
       10601       0.73      0.84      0.78        19
       10907       0.79      0.69      0.73        16
         107       0.00      0.00      0.00         5
       10904       0.00      0.00      0.00         2
       10608       0.78      1.00      0.88         7
       10701       0.71      0.77      0.74        22
         102       1.00      0.67      0.80         9
       10302       0.86      1.00      0.92         6
       11103       1.00      0.89      0.94         9
       10810       1.00      0.75      0.86         4
       11201       0.00      0.00      0.00         2
         108       0.88      1.00      0.93         7
       11602       0.91      0.89      0.90        66
       10203       0.86      0.55      0.67        11
       10711       1.00      1.00      1.00        10
       10903       0.70      0.88      0.78         8
       10301       0.64      0.86      0.73        21
       10602       0.00      0.00      0.00         1
       10303       0.62      0.67      0.65        15
       11605       0.81      0.86      0.83        29
       11606       0.77      1.00      0.87        34
       10201       0.76      0.93      0.84        28
       10703       0.82      0.64      0.72        14
       10809       0.63      0.79      0.70        48
       10104       0.70      0.89      0.78        18
       11610       0.91      0.91      0.91        64
       10607       1.00      0.50      0.67         8
       10316       1.00      1.00      1.00         6
       10706       0.00      0.00      0.00         1
       10609       1.00      0.75      0.86         4

    accuracy                           0.81      1270
   macro avg       0.62      0.57      0.58      1270
weighted avg       0.79      0.81      0.79      1270

#### Summary

TBC

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** {{ hardware_type | default("[More Information Needed]", true)}}
- **Hours used:** {{ hours_used | default("[More Information Needed]", true)}}
- **Cloud Provider:** {{ cloud_provider | default("[More Information Needed]", true)}}
- **Compute Region:** {{ cloud_region | default("[More Information Needed]", true)}}
- **Carbon Emitted:** {{ co2_emitted | default("[More Information Needed]", true)}}


## Model Card Authors [optional]

Oliver Lyttleton

## Model Card Contact

o.lyttleton@ucl.ac.uk