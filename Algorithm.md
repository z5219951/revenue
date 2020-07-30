Among the past works on Box-office revenue prediction, Boosting models (LightGBM, XGBoost, CatBoost), Random Forest, and Neural Networks are the most commonly used and successful models. 
A gradient boosted model (GBM) is a generalization of tree boosting that uses gradient descent to minimize the loss,  and attempts to improve the accuracy and efficiency of decision trees.
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.(Wikipedia).
We will use two modelling techiniques: LightGBM(or CatBoost or LightGBM and CatBoost) and Random Forest.
Both LightGBM and CatBoost are suitable for categorical features, LightGBM is faster and needs lower memory usage, CatBoost performs better on categorical features however it is time expensive in training. We will decide which model of the two to choose or use both.

We will find learning rate (the “step size” with which we descend the gradient), shrinkage (reduction of the learning rate) and loss function as hyperparameters in Gradient Boosting models. Other hyperparameters of Gradient Boosting are similar to those of Random Forests:
•	the number of iterations (i.e. the number of trees to ensemble),
•	the number of observations in each leaf,
•	tree complexity and depth,
•	the proportion of samples and
•	the proportion of features on which to train on.

Hyperparameter optimization:

Random Search or Grid Search

Training
K fold Cross Validation (k = 5, 10)
Compare models

