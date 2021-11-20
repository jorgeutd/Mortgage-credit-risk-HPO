# Mortgage credit risk HPO modeling

I am using Tune-sklearn & Optuna with ray here primarily for tuning Scikit-Learn / XGBoost / LightGBMmodels with cutting edge hyperparameter tuning techniques. This was some of the code for my class DAAN 888 project. I only adding the initial EDA, Feature selection & HPO (hyperparameter optimization) notebooks here.


## Dataset

The dataset used was obtained from creditriskanalytics.

The goal was to of accurately predict the likelihood of a borrower’ default of a mortgage loan from two classes (Binary classification):

0: Non-Default
1: Default


## Future Opportunities

After reviewing the obtained results (73.2 % F1-score & 80.4 % Recall), we can further improve our results by:
* Using random Oversampling techniques to improve our accuracy on the default positive class.
* Trying another state of the art model like Roberta and fine tune. 
* Extend our params in our HPO operation.
* collecting more data.

The final parameters of the LGBMClassifier best model are:

LGBMClassifier(bagging_fraction=0.4, learning_rate=0.05, max_depth=30,
               min_split_gain=0.2, n_estimators=2071, num_leaves=49,
               objective='binary', reg_alpha=5.0, reg_lambda=1.0,
               scale_pos_weight=2)