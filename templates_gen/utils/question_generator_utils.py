import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from spacy import displacy

# # Load SpaCy NLP model
# nlp = spacy.load("en_core_web_sm")

from utils.topics import topics
# Options for random values (optional to override in main)
options = {
    "goal": ["classification", "prediction", "data analysis"],
    "application": ["machine learning", "medical diagnosis", "finance", "text classification", "marketing", "natural language processing"],  # NB text, others broad
    "model": ["Logistic Regression", "Support Vector Machines", "k-Nearest Neighbors", "Naïve Bayes", "Decision Trees"],  # All four + extra
    "aspect": ["output type", "training process", "application domain", "assumption type"],
    "feature": ["numeric features", "continuous features", "word counts", "binary presence", "class frequencies"],  # LR/SVM/k-NN: numeric/continuous, NB: counts/presence
    "user": ["beginner", "data scientist", "engineer"],
    "dimension": ["accuracy", "interpretability", "complexity", "speed"],
    "task": ["classification", "probability estimation", "text categorization"],  # NB text focus
    "method": ["sigmoid function", "gradient descent", "regularization", "Bayes’ theorem", "likelihood estimation", "prior calculation", "distance calculation"],  # LR: sigmoid, SVM: grad/reg, NB: Bayes, k-NN: distance
    "scenario": ["binary classification", "multiclass classification", "noisy data", "imbalanced classes", "text data", "non-linear data"],  # LR: binary, NB: text, SVM/k-NN: non-linear/noisy
    "alternative": ["Logistic Regression", "Support Vector Machines", "k-Nearest Neighbors", "Naïve Bayes", "Decision Trees"],  # All as alternatives
    "input": ["features", "raw data", "logits", "frequency counts"],  # LR: logits, NB: counts, SVM/k-NN: features
    "output": ["probabilities", "class labels"],
    "context": ["theory", "practice", "real-world"],
    "principle": ["probability", "optimization", "decision boundaries", "independence", "conditional probability", "distance-based"],  # LR/SVM: boundaries, NB: prob/indep, k-NN: distance
    "theory": ["statistics", "probability theory", "linear algebra", "information theory"],  # LR/SVM: algebra, NB: info, k-NN: stats
    "practice": ["model training", "prediction", "evaluation"],
    "problem": ["imbalanced data", "high dimensions", "non-linear data", "feature dependence", "sparse data", "noisy data"],  # LR: non-linear, NB: dependence, SVM: high dims, k-NN: noisy
    "predecessor": ["Linear Regression", "perceptron", "Bayesian classifiers", "simple probabilistic models", "nearest neighbor methods"],  # LR: lin reg, NB: Bayes, SVM: perceptron, k-NN: NN
    "field": ["data science", "healthcare", "finance", "natural language processing"],  # NB NLP, others broad
    "noise": ["outliers", "mislabels", "random noise"],
    "parameter": ["weights", "bias", "learning rate", "priors", "likelihoods", "variances", "C parameter", "k value"]  # LR/SVM: weights/C, NB: priors/vars, k-NN: k
}


def get_question(topic, subtopic, difficulty, displayed_questions):
    # Validate inputs
    if topic not in topics or subtopic not in topics[topic]["subtopics"]:
        return {"question": "Topic/Subtopic not found.", "answer": "N/A", "plot": None, "table": None}

    templates = topics[topic]["subtopics"][subtopic][difficulty]
    if not templates:
        return {"question": "No templates for this context.", "answer": "N/A", "plot": None, "table": None}

    # Select a unique, unseen template
    available_templates = [t for t in templates if t not in displayed_questions]
    if not available_templates:
        return {"question": "No more new questions available for this selection.", "answer": "N/A", "plot": None, "table": None}
    
    template = random.choice(available_templates)
    question = template

    # Numerical and structural placeholders
    params = {
        "x": random.randint(-10, 10),
        "p": random.uniform(0, 1),
        "w": random.uniform(-2, 2),
        "w1": random.uniform(-2, 2),
        "w2": random.uniform(-2, 2),
        "b": random.uniform(-1, 1),
        "l": random.uniform(-5, 5),
        "n": random.randint(5, 20),   # For scatter plots
        "a": 0,
        "b": 1,
        "y": random.randint(0, 1),
        "d": random.uniform(-0.5, 0.5),
        "c": random.uniform(0.1, 10),
        "f": random.randint(2, 3),    # Features in table
        "t": random.randint(1, 10),   # Iterations for cost
        "r": random.uniform(0.01, 0.1),
        "g": random.uniform(-1, 1),
        "k": random.randint(1, 5),
        "m": random.randint(5, 10),   # Data samples
        "s": random.uniform(0.1, 2)   # Slope for sigmoid
    }

    # Textual/contextual placeholders
    params.update({
        key: random.choice(values)
        for key, values in options.items()
        if f"{{{key}}}" in question
    })

    # Visualizations
    fig = None
    table_data = None

    if "scatter plot below" in question:
        n_points = params["n"]
        separable = random.choice([True, False])
        x1 = np.random.normal(0, 1, n_points) if separable else np.random.normal(0, 2, n_points)
        y1 = np.random.normal(0, 1, n_points) if separable else np.random.normal(0, 2, n_points)
        x2 = np.random.normal(3, 1, n_points) if separable else np.random.normal(0, 2, n_points)
        y2 = np.random.normal(3, 1, n_points) if separable else np.random.normal(0, 2, n_points)

        fig, ax = plt.subplots()
        ax.scatter(x1, y1, color='blue', label='Class 0')
        ax.scatter(x2, y2, color='red', label='Class 1')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'Scatter Plot with {n_points} Points per Class')
        ax.legend()
        plt.tight_layout()

    elif "line graph below" in question and "sigmoid" in question:
        x = np.linspace(-10, 10, 100)
        z = params["w"] * x + params["b"]
        sigmoid = 1 / (1 + np.exp(-z))

        fig, ax = plt.subplots()
        ax.plot(x, sigmoid, color='blue', label=f'Sigmoid (w={params['w']:.2f}, b={params['b']:.2f})')
        ax.set_xlabel('Input (x)')
        ax.set_ylabel('Probability')
        ax.set_title('Sigmoid Curve')
        ax.legend()
        plt.tight_layout()

    elif "line graph below" in question:
        iterations = np.arange(params["t"])
        cost = np.exp(-iterations * params["r"])

        fig, ax = plt.subplots()
        ax.plot(iterations, cost, color='green', label='Cost Decrease')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Cost')
        ax.set_title(f'Cost Over {params['t']} Iterations')
        ax.legend()
        plt.tight_layout()

    elif "data table below" in question:
        m_samples = params["m"]
        f_features = params["f"]
        data = {f"Feature_{i+1}": np.random.uniform(-5, 5, m_samples) for i in range(f_features)}
        data["Label"] = np.random.randint(0, 2, m_samples)
        table_data = pd.DataFrame(data)

    # Fill in placeholders in the template
    try:
        question = question.format(topic=topic, subtopic=subtopic, **params)
    except KeyError as e:
        return {"question": f"Error: Missing placeholder {e} in template '{template}'", "answer": "N/A", "plot": None, "table": None}

    return {
        "question": question,
        "answer": "Sample answer for demonstration purposes.",
        "plot": fig,
        "table": table_data
    }
