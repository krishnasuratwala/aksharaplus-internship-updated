from utils.templates.Logistic_Regression_templates import logistic_regression_templates
from utils.templates.k_Nearest_Neighbors_templates import knn_templates
from utils.templates.Random_Forest_templates import random_forest_templates
from utils.templates.Decision_Trees_templates import decision_tree_templates
from utils.templates.Naïve_Bayes_templates import naive_bayes_templates
from utils.templates.Support_Vector_Machines_templates import svm_templates


topics = {
    "Logistic Regression": {
        "subtopics": {
            "Introduction": {
                "easy": logistic_regression_templates["Logistic Regression"]["subtopics"]["Introduction"].get("easy", []),
                "medium": logistic_regression_templates["Logistic Regression"]["subtopics"]["Introduction"].get("medium", []),
                "hard": logistic_regression_templates["Logistic Regression"]["subtopics"]["Introduction"].get("hard", [])
            },
            "Mathematical Foundation": {
                "easy": logistic_regression_templates["Logistic Regression"]["subtopics"]["Mathematical Foundation"].get("easy", []),
                "medium": logistic_regression_templates["Logistic Regression"]["subtopics"]["Mathematical Foundation"].get("medium", []),
                "hard": logistic_regression_templates["Logistic Regression"]["subtopics"]["Mathematical Foundation"].get("hard", [])
            },
            "Cost Function and Optimization": {
                "easy": logistic_regression_templates["Logistic Regression"]["subtopics"]["Cost Function and Optimization"].get("easy", []),
                "medium": logistic_regression_templates["Logistic Regression"]["subtopics"]["Cost Function and Optimization"].get("medium", []),
                "hard": logistic_regression_templates["Logistic Regression"]["subtopics"]["Cost Function and Optimization"].get("hard", [])
            },
            "Regularization": {
                "easy": logistic_regression_templates["Logistic Regression"]["subtopics"]["Regularization"].get("easy", []),
                "medium": logistic_regression_templates["Logistic Regression"]["subtopics"]["Regularization"].get("medium", []),
                "hard": logistic_regression_templates["Logistic Regression"]["subtopics"]["Regularization"].get("hard", [])
            },
            "Multiclass Logistic Regression": {
                "easy": logistic_regression_templates["Logistic Regression"]["subtopics"]["Multiclass Logistic Regression"].get("easy", []),
                "medium": logistic_regression_templates["Logistic Regression"]["subtopics"]["Multiclass Logistic Regression"].get("medium", []),
                "hard": logistic_regression_templates["Logistic Regression"]["subtopics"]["Multiclass Logistic Regression"].get("hard", [])
            },
            "Implementation": {
                "easy": logistic_regression_templates["Logistic Regression"]["subtopics"]["Implementation"].get("easy", []),
                "medium": logistic_regression_templates["Logistic Regression"]["subtopics"]["Implementation"].get("medium", []),
                "hard": logistic_regression_templates["Logistic Regression"]["subtopics"]["Implementation"].get("hard", [])
            }
        }
    },
    "k-Nearest Neighbors": {
        "subtopics": {
            "Introduction": {
                "easy": knn_templates["k-Nearest Neighbors"]["subtopics"]["Introduction"].get("easy", []),
                "medium": knn_templates["k-Nearest Neighbors"]["subtopics"]["Introduction"].get("medium", []),
                "hard": knn_templates["k-Nearest Neighbors"]["subtopics"]["Introduction"].get("hard", [])
            },
            "Distance Metrics": {
                "easy": knn_templates["k-Nearest Neighbors"]["subtopics"]["Distance Metrics"].get("easy", []),
                "medium": knn_templates["k-Nearest Neighbors"]["subtopics"]["Distance Metrics"].get("medium", []),
                "hard": knn_templates["k-Nearest Neighbors"]["subtopics"]["Distance Metrics"].get("hard", [])
            },
            "Choosing k": {
                "easy": knn_templates["k-Nearest Neighbors"]["subtopics"]["Choosing k"].get("easy", []),
                "medium": knn_templates["k-Nearest Neighbors"]["subtopics"]["Choosing k"].get("medium", []),
                "hard": knn_templates["k-Nearest Neighbors"]["subtopics"]["Choosing k"].get("hard", [])
            },
            "Algorithm Mechanics": {
                "easy": knn_templates["k-Nearest Neighbors"]["subtopics"]["Algorithm Mechanics"].get("easy", []),
                "medium": knn_templates["k-Nearest Neighbors"]["subtopics"]["Algorithm Mechanics"].get("medium", []),
                "hard": knn_templates["k-Nearest Neighbors"]["subtopics"]["Algorithm Mechanics"].get("hard", [])
            },
            "Scaling and Normalization": {
                "easy": knn_templates["k-Nearest Neighbors"]["subtopics"]["Scaling and Normalization"].get("easy", []),
                "medium": knn_templates["k-Nearest Neighbors"]["subtopics"]["Scaling and Normalization"].get("medium", []),
                "hard": knn_templates["k-Nearest Neighbors"]["subtopics"]["Scaling and Normalization"].get("hard", [])
            },
            "Implementation": {
                "easy": knn_templates["k-Nearest Neighbors"]["subtopics"]["Implementation"].get("easy", []),
                "medium": knn_templates["k-Nearest Neighbors"]["subtopics"]["Implementation"].get("medium", []),
                "hard": knn_templates["k-Nearest Neighbors"]["subtopics"]["Implementation"].get("hard", [])
            }
        }
    },
    "Random Forest": {
        "subtopics": {
            "Introduction": {
                "easy": random_forest_templates["Random Forest"]["subtopics"][ "Introduction"].get("easy", []),
                "medium": random_forest_templates["Random Forest"]["subtopics"][ "Introduction"].get("medium", []),
                "hard": random_forest_templates["Random Forest"]["subtopics"][ "Introduction"].get("hard", [])
            },
            "Bootstrap Aggregation": {
                "easy": random_forest_templates["Random Forest"]["subtopics"]["Bootstrap Aggregation"].get("easy", []),
                "medium": random_forest_templates["Random Forest"]["subtopics"]["Bootstrap Aggregation"].get("medium", []),
                "hard": random_forest_templates["Random Forest"]["subtopics"]["Bootstrap Aggregation"].get("hard", [])
            },
            "Feature Randomness": {
                "easy": random_forest_templates["Random Forest"]["subtopics"]["Feature Randomness"].get("easy", []),
                "medium": random_forest_templates["Random Forest"]["subtopics"]["Feature Randomness"].get("medium", []),
                "hard": random_forest_templates["Random Forest"]["subtopics"]["Feature Randomness"].get("hard", [])
            },
            "Voting and Averaging": {
                "easy": random_forest_templates["Random Forest"]["subtopics"]["Voting and Averaging"].get("easy", []),
                "medium": random_forest_templates["Random Forest"]["subtopics"]["Voting and Averaging"].get("medium", []),
                "hard": random_forest_templates["Random Forest"]["subtopics"]["Voting and Averaging"].get("hard", [])
            },
            "Hyperparameters": {    
                "easy": random_forest_templates["Random Forest"]["subtopics"]["Hyperparameters"].get("easy", []),
                "medium": random_forest_templates["Random Forest"]["subtopics"]["Hyperparameters"].get("medium", []),
                "hard": random_forest_templates["Random Forest"]["subtopics"]["Hyperparameters"].get("hard", [])
            },
            "Implementation": {
                "easy": random_forest_templates["Random Forest"]["subtopics"]["Implementation"].get("easy", []),
                "medium": random_forest_templates["Random Forest"]["subtopics"]["Implementation"].get("medium", []),
                "hard": random_forest_templates["Random Forest"]["subtopics"]["Implementation"].get("hard", [])
            }
        }
    },
    "Decision Trees": {
        "subtopics": {
            "Introduction": {
                "easy": decision_tree_templates["Decision Trees"]["subtopics"]["Introduction"].get("easy", []),
                "medium": decision_tree_templates["Decision Trees"]["subtopics"]["Introduction"].get("medium", []),
                "hard": decision_tree_templates["Decision Trees"]["subtopics"]["Introduction"].get("hard", [])
            },
            "Splitting Criteria": {
                "easy": decision_tree_templates["Decision Trees"]["subtopics"]["Splitting Criteria"].get("easy", []),
                "medium": decision_tree_templates["Decision Trees"]["subtopics"]["Splitting Criteria"].get("medium", []),
                "hard": decision_tree_templates["Decision Trees"]["subtopics"]["Splitting Criteria"].get("hard", [])
            },
            "Tree Construction": { 
                "easy": decision_tree_templates["Decision Trees"]["subtopics"]["Tree Construction"].get("easy", []),
                "medium": decision_tree_templates["Decision Trees"]["subtopics"]["Tree Construction"].get("medium", []),
                "hard": decision_tree_templates["Decision Trees"]["subtopics"]["Tree Construction"].get("hard", [])
            },
            "Pruning": {
                "easy": decision_tree_templates["Decision Trees"]["subtopics"]["Pruning"].get("easy", []),
                "medium": decision_tree_templates["Decision Trees"]["subtopics"]["Pruning"].get("medium", []),
                "hard": decision_tree_templates["Decision Trees"]["subtopics"]["Pruning"].get("hard", [])
            },
            "Prediction": {
                "easy": decision_tree_templates["Decision Trees"]["subtopics"]["Prediction"].get("easy", []),
                "medium": decision_tree_templates["Decision Trees"]["subtopics"]["Prediction"].get("medium", []),
                "hard": decision_tree_templates["Decision Trees"]["subtopics"]["Prediction"].get("hard", [])
            },
            "Implementation": {
                "easy": decision_tree_templates["Decision Trees"]["subtopics"]["Implementation"].get("easy", []),
                "medium": decision_tree_templates["Decision Trees"]["subtopics"]["Implementation"].get("medium", []),
                "hard": decision_tree_templates["Decision Trees"]["subtopics"]["Implementation"].get("hard", [])
            }


        }
    },
    "Naïve Bayes": {
        "subtopics": {
            "Introduction": {
                "easy": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Introduction"].get("easy", []),
                "medium": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Introduction"].get("medium", []),
                "hard": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Introduction"].get("hard", [])
            },
            "Probability Basics": {
                "easy": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Probability Basics"].get("easy", []),
                "medium": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Probability Basics"].get("medium", []),
                "hard": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Probability Basics"].get("hard", [])
            },
            "Gaussian Naïve Bayes": {
                "easy": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Gaussian Naïve Bayes"].get("easy", []),
                "medium": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Gaussian Naïve Bayes"].get("medium", []),
                "hard": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Gaussian Naïve Bayes"].get("hard", [])
            },
            "Multinomial Naïve Bayes": {
                "easy": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Multinomial Naïve Bayes"].get("easy", []),
                "medium": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Multinomial Naïve Bayes"].get("medium", []),
                "hard": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Multinomial Naïve Bayes"].get("hard", [])
            },
            "Bernoulli Naïve Bayes": {
                "easy": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Bernoulli Naïve Bayes"].get("easy", []),
                "medium": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Bernoulli Naïve Bayes"].get("medium", []),
                "hard": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Bernoulli Naïve Bayes"].get("hard", [])
            },
            "Implementation": {
                "easy": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Implementation"].get("easy", []),
                "medium": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Implementation"].get("medium", []),
                "hard": naive_bayes_templates["Naïve Bayes"]["subtopics"]["Implementation"].get("hard", [])
            }

        }
    },
    "Support Vector Machines": {
        "subtopics": {
            "Introduction": {
                "easy": svm_templates["Support Vector Machines"]["subtopics"]["Introduction"].get("easy", []),
                "medium": svm_templates["Support Vector Machines"]["subtopics"]["Introduction"].get("medium", []),
                "hard": svm_templates["Support Vector Machines"]["subtopics"]["Introduction"].get("hard", [])
            },
            "Linear SVM": {
                "easy": svm_templates["Support Vector Machines"]["subtopics"]["Linear SVM"].get("easy", []),
                "medium": svm_templates["Support Vector Machines"]["subtopics"]["Linear SVM"].get("medium", []),
                "hard": svm_templates["Support Vector Machines"]["subtopics"]["Linear SVM"].get("hard", [])
            },
            "Kernel Trick": {
                "easy": svm_templates["Support Vector Machines"]["subtopics"]["Kernel Trick"].get("easy", []),
                "medium": svm_templates["Support Vector Machines"]["subtopics"]["Kernel Trick"].get("medium", []),
                "hard": svm_templates["Support Vector Machines"]["subtopics"]["Kernel Trick"].get("hard", [])
            },
            "Mathematical Foundation": {
                "easy": svm_templates["Support Vector Machines"]["subtopics"]["Mathematical Foundation"].get("easy", []),
                "medium": svm_templates["Support Vector Machines"]["subtopics"]["Mathematical Foundation"].get("medium", []),
                "hard": svm_templates["Support Vector Machines"]["subtopics"]["Mathematical Foundation"].get("hard", [])
            },
            "Regularization and Cost": {
                "easy": svm_templates["Support Vector Machines"]["subtopics"]["Regularization and Cost"].get("easy", []),
                "medium": svm_templates["Support Vector Machines"]["subtopics"]["Regularization and Cost"].get("medium", []),
                "hard": svm_templates["Support Vector Machines"]["subtopics"]["Regularization and Cost"].get("hard", [])
            },
            "Implementation": {
                "easy": svm_templates["Support Vector Machines"]["subtopics"]["Implementation"].get("easy", []),
                "medium": svm_templates["Support Vector Machines"]["subtopics"]["Implementation"].get("medium", []),
                "hard": svm_templates["Support Vector Machines"]["subtopics"]["Implementation"].get("hard", [])
            }
        }
    }
}