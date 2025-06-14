import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import time
import random

# Set page title and layout
st.set_page_config(page_title="Automatic Scatter Plot Analysis App", layout="wide")

# Step 1: Load the fine-tuned BLIP model and processor
model_path = "./blip_finetuned"  # Update this path if your model is elsewhere
try:
    processor = BlipProcessor.from_pretrained(model_path)
    model = BlipForConditionalGeneration.from_pretrained(model_path)
except Exception as e:
    st.error(f"Error loading BLIP model: {e}. Please ensure the model is in the correct directory.")
    st.stop()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
st.write(f"Using device: {device}")

# Step 2: Define algorithm implementations and traits
algorithm_configs = {
    "Logistic Regression": {
        "model": LogisticRegression(),
        "linear": True,
        "strength": "Logistic Regression performs well on linearly separable data because it can find a linear decision boundary to separate the classes.",
        "weakness": "Logistic Regression struggles with highly overlapping classes because it assumes a linear decision boundary, which may not effectively separate the data when the classes are not linearly separable.",
        "code": """
# Example code for Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_squared_error

# Sample data (replace X and y with your data)
X = [[1, 2], [2, 3], [3, 4], [4, 5]]  # Feature data
y = [0, 0, 1, 1]  # Class labels

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
mse = mean_squared_error(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"MSE: {mse:.2f}")
"""
    },
    "Support Vector Machines": {
        "model": SVC(kernel="linear"),
        "linear": True,
        "strength": "A Linear SVM is effective for linearly separable data because it maximizes the margin between classes, creating a robust linear decision boundary.",
        "weakness": "A Linear SVM struggles with highly overlapping classes because it relies on a linear decision boundary, which cannot effectively separate the data without a kernel transformation.",
        "code": """
# Example code for Support Vector Machines (Linear SVM)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_squared_error

# Sample data (replace X and y with your data)
X = [[1, 2], [2, 3], [3, 4], [4, 5]]  # Feature data
y = [0, 0, 1, 1]  # Class labels

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
mse = mean_squared_error(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"MSE: {mse:.2f}")
"""
    },
    "Naïve Bayes": {
        "model": GaussianNB(),
        "linear": False,
        "strength": "Naïve Bayes can work on this data if the features are independent, as it models the probability of each class based on feature distributions.",
        "weakness": "Naïve Bayes struggles with highly overlapping classes because it assumes feature independence, which may not hold when the classes are highly mixed, leading to poor classification.",
        "code": """
# Example code for Naïve Bayes (GaussianNB)
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_squared_error

# Sample data (replace X and y with your data)
X = [[1, 2], [2, 3], [3, 4], [4, 5]]  # Feature data
y = [0, 0, 1, 1]  # Class labels

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
mse = mean_squared_error(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"MSE: {mse:.2f}")
"""
    },
    "Decision Trees": {
        "model": DecisionTreeClassifier(max_depth=5),
        "linear": False,
        "strength": "Decision Trees can handle this data because they can create non-linear decision boundaries, potentially separating the classes even if they overlap.",
        "weakness": "Decision Trees may struggle with this data due to the small number of points, as they can overfit to noise or small variations in the scatter plot.",
        "code": """
# Example code for Decision Trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_squared_error

# Sample data (replace X and y with your data)
X = [[1, 2], [2, 3], [3, 4], [4, 5]]  # Feature data
y = [0, 0, 1, 1]  # Class labels

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
mse = mean_squared_error(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"MSE: {mse:.2f}")
"""
    },
    "Random Forest": {
        "model": RandomForestClassifier(n_estimators=10, max_depth=5),
        "linear": False,
        "strength": "Random Forest can handle this data because it combines multiple decision trees, creating robust non-linear decision boundaries that can separate the classes even if they overlap.",
        "weakness": "Random Forest may still struggle with highly overlapping classes, especially with a small number of points, as the ensemble might not generalize well to such noisy data.",
        "code": """
# Example code for Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_squared_error

# Sample data (replace X and y with your data)
X = [[1, 2], [2, 3], [3, 4], [4, 5]]  # Feature data
y = [0, 0, 1, 1]  # Class labels

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=10, max_depth=5)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
mse = mean_squared_error(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"MSE: {mse:.2f}")
"""
    }
}

# Step 3: Function to determine if classes are separable
def determine_separability(X, y):
    X_class0 = X[y == 0]
    X_class1 = X[y == 1]
    # Compute bounding boxes for each class
    min0, max0 = X_class0.min(axis=0), X_class0.max(axis=0)
    min1, max1 = X_class1.min(axis=0), X_class1.max(axis=0)
    # Check for overlap in both dimensions
    overlap_x = not (max0[0] < min1[0] or max1[0] < min0[0])
    overlap_y = not (max0[1] < min1[1] or max1[1] < min0[1])
    # Classes are separable if there is no overlap in at least one dimension
    separable = not (overlap_x and overlap_y)
    return separable

# Step 4: Function to perform visual analysis
def visual_analysis(X, y, num_points_per_class, separable, is_linear):
    X_class0 = X[y == 0]
    X_class1 = X[y == 1]
    std_class0 = np.std(X_class0, axis=0).mean()
    std_class1 = np.std(X_class1, axis=0).mean()
    clustering = "tightly clustered" if (std_class0 < 1.5 and std_class1 < 1.5) else "spread out"
    center0 = np.mean(X_class0, axis=0)
    center1 = np.mean(X_class1, axis=0)
    center_distance = np.linalg.norm(center0 - center1)
    overlap_description = (
        "minimal overlap" if separable and center_distance > 2 else
        "slight overlap" if center_distance > 1 else
        "significant overlap"
    )

    if is_linear:
        if separable and center_distance > 2:
            visual_effectiveness = "Yes"
            visual_reason = (
                f"the scatter plot shows two distinct clusters of blue and red points with {overlap_description}, "
                f"suggesting that a linear boundary can easily separate the classes. "
                f"The points in each class are {clustering}, making separation straightforward"
            )
        elif center_distance > 1:
            visual_effectiveness = "No"
            visual_reason = (
                f"the scatter plot shows blue and red points with {overlap_description}, "
                f"suggesting that a linear boundary cannot effectively separate the classes due to overlap. "
                f"The points in each class are {clustering}, which {'helps' if clustering == 'tightly clustered' else 'makes separation harder'}"
            )
        else:
            visual_effectiveness = "No"
            visual_reason = (
                f"the scatter plot shows blue and red points with {overlap_description}, "
                f"suggesting that a linear boundary cannot effectively separate the classes. "
                f"The points in each class are {clustering}, which {'adds to the challenge' if clustering == 'spread out' else 'does not help enough'}"
            )
    else:
        if separable or center_distance > 1:
            visual_effectiveness = "Yes"
            visual_reason = (
                f"the scatter plot shows blue and red points with {overlap_description}, "
                f"suggesting that a non-linear boundary can separate the classes effectively. "
                f"The points in each class are {clustering}, which {'makes separation easier' if clustering == 'tightly clustered' else 'may require a more complex boundary'}"
            )
        else:
            visual_effectiveness = "No"
            visual_reason = (
                f"the scatter plot shows blue and red points with {overlap_description}, "
                f"suggesting that a non-linear boundary struggles to separate the classes. "
                f"The points in each class are {clustering}, which {'adds to the challenge' if clustering == 'spread out' else 'does not help enough'}"
            )

    return visual_effectiveness, visual_reason

# Step 5: Function to compute metrics with detailed calculations
def evaluate_algorithm(model, X, y, is_linear):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute confusion matrix for detailed calculations
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    # Accuracy: (TP + TN) / (TP + TN + FP + FN)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    accuracy_calc = f"(TP + TN) / (TP + TN + FP + FN) = ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn}) = {accuracy:.2f}"

    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    precision_calc = f"TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision:.2f}"

    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall_calc = f"TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall:.2f}"

    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_calc = f"2 * (Precision * Recall) / (Precision + Recall) = 2 * ({precision:.2f} * {recall:.2f}) / ({precision:.2f} + {recall:.2f}) = {f1:.2f}"

    # MSE: Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    errors = [(y_test[i] - y_pred[i])**2 for i in range(len(y_test))]
    mse_calc = f"Sum of squared errors / n = ({' + '.join([f'{e:.2f}' for e in errors])}) / {len(y_test)} = {mse:.2f}"

    decision_boundary = ""
    if is_linear:
        coef = model.coef_[0]
        intercept = model.intercept_[0]
        slope = -coef[0] / coef[1] if coef[1] != 0 else float('inf')
        decision_boundary = f"a line with slope {slope:.2f} and intercept {intercept:.2f}"

    return accuracy, precision, f1, mse, decision_boundary, model, {
        "Accuracy": accuracy_calc,
        "Precision": precision_calc,
        "F1 Score": f1_calc,
        "MSE": mse_calc
    }, y_test, y_pred

# Step 6: Function to find the best algorithm
def find_best_algorithm(X, y):
    best_f1 = -1
    best_algo = None
    best_trained_model = None
    best_is_linear = False
    best_metrics = {}

    for algo_name, config in algorithm_configs.items():
        model = config["model"]
        is_linear = config["linear"]
        accuracy, precision, f1, mse, decision_boundary, trained_model, _, _, _ = evaluate_algorithm(model, X, y, is_linear)
        if f1 > best_f1:
            best_f1 = f1
            best_algo = algo_name
            best_trained_model = trained_model
            best_is_linear = is_linear
            best_metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "f1": f1,
                "mse": mse,
                "decision_boundary": decision_boundary
            }

    return best_algo, best_trained_model, best_is_linear, best_metrics

# Step 7: Function to plot scatter plot with decision boundary (for individual algorithm)
def plot_scatter_with_boundary(X, y, model, is_linear, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='Class 0', alpha=0.6)
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Class 1', alpha=0.6)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(X_grid)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.3)

    if is_linear:
        coef = model.coef_[0]
        intercept = model.intercept_[0]
        if coef[1] != 0:
            x_range = np.array([x_min, x_max])
            y_range = -(coef[0] * x_range + intercept) / coef[1]
            ax.plot(x_range, y_range, 'k-', label='Decision Boundary')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    plt.close()
    return image

# Step 8: Function to plot decision boundaries for all algorithms in a 2x3 grid
def plot_all_decision_boundaries(X, y):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (algo_name, config) in enumerate(algorithm_configs.items()):
        model = config["model"]
        is_linear = config["linear"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        axes[idx].scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='Class 0', alpha=0.6)
        axes[idx].scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Class 1', alpha=0.6)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(X_grid)
        Z = Z.reshape(xx.shape)
        axes[idx].contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.3)

        if is_linear:
            coef = model.coef_[0]
            intercept = model.intercept_[0]
            if coef[1] != 0:
                x_range = np.array([x_min, x_max])
                y_range = -(coef[0] * x_range + intercept) / coef[1]
                axes[idx].plot(x_range, y_range, 'k-', label='Decision Boundary')

        axes[idx].set_xlabel('X1')
        axes[idx].set_ylabel('X2')
        axes[idx].set_title(algo_name)
        axes[idx].legend()
        axes[idx].grid(True)

    # Remove the last (empty) subplot
    fig.delaxes(axes[-1])
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    plt.close()
    return image

# Step 9: Function to generate synthetic scatter plot with exactly 10 points
def generate_synthetic_plot():
    num_points = 10  # Fixed to 10 points (5 per class)
    separability_shift = random.uniform(0.0, 5.0)  # Random separability shift between 0.0 and 5.0
    noise_level = random.uniform(0.0, 2.0)  # Random noise level between 0.0 and 2.0

    np.random.seed(int(time.time()))
    X = np.random.randn(num_points, 2) * 2
    y = np.array([0] * (num_points // 2) + [1] * (num_points - num_points // 2))  # 5 points for Class 0, 5 for Class 1
    X[num_points // 2:, 0] += separability_shift
    X[num_points // 2:, 1] += separability_shift
    X += np.random.randn(num_points, 2) * noise_level

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='Class 0', alpha=0.6)
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Class 1', alpha=0.6)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(f'Scatter Plot with {num_points} Points')
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    plt.close()
    return image, X, y

# Step 10: Function to generate a random question
def generate_random_question():
    algorithm = random.choice(list(algorithm_configs.keys()))
    question = f"Would {algorithm} be effective for this scatter plot? Why?"
    return question, algorithm

# Step 11: Main function to process the question and plot
def process_question_and_plot(question, plot_image, X, y):
    # Extract the algorithm (topic) from the question
    topic = None
    for algo in algorithm_configs.keys():
        if algo.lower() in question.lower():
            topic = algo
            break
    if not topic:
        st.error("Algorithm not found in the question. Please specify one of: Logistic Regression, Support Vector Machines, Naïve Bayes, Decision Trees, Random Forest.")
        return

    num_points_per_class = len(y) // 2
    separable = determine_separability(X, y)

    # Step 11.2: Display structured tabular data (x, y coordinates and class)
    st.subheader("Structured Data (Points)")
    data = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "Class": y})
    st.dataframe(data.style.format({"X1": "{:.2f}", "X2": "{:.2f}"}))

    # Step 11.3: Generate and display caption using BLIP
    st.subheader("Image Caption (Generated by BLIP Model)")
    inputs = processor(images=plot_image, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=inputs.pixel_values,
            max_length=50,
            num_beams=5,
            early_stopping=True
        )
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    st.write(f"Caption: {caption}")

    # Step 11.4: Algorithmic Approach
    st.subheader(f"Analysis for Question: {question}")
    
    # Get algorithm config
    config = algorithm_configs.get(topic, {
        "model": LogisticRegression(),
        "linear": True,
        "strength": f"{topic} can classify the data if the classes are separable.",
        "weakness": f"{topic} struggles with overlapping classes."
    })

    # Visual analysis
    visual_effectiveness, visual_reason = visual_analysis(X, y, num_points_per_class, separable, config["linear"])
    st.subheader("Visual Analysis")
    st.write(f"{visual_reason}. Therefore, {topic} is {'not ' if visual_effectiveness == 'No' else ''}likely to work well based on the plot alone.")

    # Programmatic analysis
    accuracy, precision, f1, mse, decision_boundary, trained_model, metric_calcs, y_test, y_pred = evaluate_algorithm(config["model"], X, y, config["linear"])
    F1_THRESHOLD = 0.7
    is_effective = f1 >= F1_THRESHOLD

    # Find the best algorithm
    best_algo, best_trained_model, best_is_linear, best_metrics = find_best_algorithm(X, y)

    # Programmatic reasoning
    if separable:
        if config["linear"]:
            effectiveness = "Yes" if is_effective else "No"
            prog_reason = (
                f"Evaluation metrics show an accuracy of {accuracy:.2f}, an F1 score of {f1:.2f}, and a precision of {precision:.2f}. "
                f"The decision boundary ({decision_boundary}) separates most of the points effectively. "
                f"{config['strength']}"
            )
        else:
            effectiveness = "Yes" if is_effective else "No"
            prog_reason = (
                f"Evaluation metrics show an accuracy of {accuracy:.2f}, an F1 score of {f1:.2f}, and a precision of {precision:.2f}. "
                f"{config['strength']}"
            )
    else:
        if config["linear"]:
            effectiveness = "Yes" if is_effective else "No"
            prog_reason = (
                f"Evaluation metrics show an accuracy of {accuracy:.2f}, an F1 score of {f1:.2f}, and a precision of {precision:.2f}. "
                f"The decision boundary ({decision_boundary}) separates most of the points, though some misclassifications occur due to overlap. "
                f"In real-world scenarios, datasets are rarely perfectly separable, and {topic} can still be effective if the overlap is not severe, "
                f"as {'supported by the F1 score above 0.7' if is_effective else 'indicated by the low F1 score'}. "
                f"{config['weakness'] if not is_effective else ''}"
            )
        else:
            effectiveness = "Yes" if is_effective else "No"
            prog_reason = (
                f"Evaluation metrics show an accuracy of {accuracy:.2f}, an F1 score of {f1:.2f}, and a precision of {precision:.2f}. "
                f"{config['strength']} However, {config['weakness'].lower()} with only {num_points_per_class} points per class."
            )

    # Add note about the best algorithm
    # best_algo_note = (
    #     f"For comparison, the best algorithm for this data is {best_algo}, with an F1 score of {best_metrics['f1']:.2f}."
    # )

    # Display evaluation metrics with formulas and calculations
    st.subheader("Programmatic Analysis")
    st.write(prog_reason)
    st.write("**Evaluation Metrics with Formulas and Calculations**")
    metrics_data = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "F1 Score", "MSE"],
        "Formula": [
            "Accuracy = (TP + TN) / (TP + TN + FP + FN)",
            "Precision = TP / (TP + FP)",
            "F1 Score = 2 * (Precision * Recall) / (Precision + Recall)",
            "MSE = (1/n) * Σ(y_true - y_pred)²"
        ],
        "Calculation": [
            metric_calcs["Accuracy"],
            metric_calcs["Precision"],
            metric_calcs["F1 Score"],
            metric_calcs["MSE"]
        ],
        "Value": [f"{accuracy:.2f}", f"{precision:.2f}", f"{f1:.2f}", f"{mse:.2f}"]
    })
    st.dataframe(metrics_data)

    # Display predictions and ground truth for clarity
    st.write("**Ground Truth vs. Predictions (Test Set)**")
    prediction_data = pd.DataFrame({
        "Ground Truth (y_test)": y_test,
        "Prediction (y_pred)": y_pred
    })
    st.dataframe(prediction_data)

    # Display decision boundary plot for the specified algorithm
    if config["linear"]:
        boundary_image = plot_scatter_with_boundary(X, y, trained_model, config["linear"], f"Decision Boundary for {topic}")
        st.subheader(f"Decision Boundary Plot for {topic}")
        st.image(boundary_image, use_container_width=True)
    else:
        boundary_image = plot_scatter_with_boundary(X, y, trained_model, config["linear"], f"Decision Regions for {topic}")
        st.subheader(f"Decision Regions Plot for {topic} (Approximation)")
        st.write(f"Note: Since {topic} uses a non-linear decision boundary, the plot shows an approximation of the decision regions.")
        st.image(boundary_image, use_container_width=True)

    # Display decision boundaries for all algorithms in a 2x3 grid
    st.subheader("Decision Boundaries/Regions for All Algorithms (2x3 Grid)")
    st.write("This grid shows how each classification algorithm separates the classes, regardless of their performance. Linear algorithms (Logistic Regression, SVM) show a decision boundary, while non-linear algorithms (Naïve Bayes, Decision Trees, Random Forest) show decision regions.")
    all_boundaries_image = plot_all_decision_boundaries(X, y)
    st.image(all_boundaries_image, use_container_width=True)

    # Final answer
    st.subheader("Final Answer")
    answer = (
        f"{effectiveness}, {topic} would {'not ' if effectiveness == 'No' else ''}be effective "
        f"for this scatter plot with {num_points_per_class} points per class. "
        f"**From a visual perspective**, {visual_reason}. Therefore, {topic} is {'not ' if visual_effectiveness == 'No' else ''}likely to work well based on the plot alone. "
        f"**From a programmatic perspective**, {prog_reason} "
        # f"{best_algo_note} "
        f"The scatter plot above shows two classes (Class 0 in blue and Class 1 in red), and the {'distinct separation' if separable else 'overlap'} "
        f"between them influences the algorithm’s performance."
    )
    st.write(answer)

    # Step 11.5: Display the algorithm code
    st.subheader(f"Code for {topic}")
    st.write("You can use this code to replicate the results manually:")
    st.code(config["code"], language="python")

# Step 12: Streamlit Interface
st.title("Automatic Scatter Plot Analysis App")
# st.markdown("""
# This app automatically generates a scatter plot with 10 points and a question about the plot, then analyzes it using a fine-tuned BLIP model for captioning and an algorithmic approach to answer the question.

# - Click **Generate Question** to create a new scatter plot and question.
# - Click **Next** to generate a new scatter plot and question.
# - Click **Answer** to analyze the current scatter plot and question, viewing the full analysis (structured data, caption, visual and programmatic analysis, evaluation metrics with formulas, decision boundary plots, final answer, and algorithm code).
# """)

# Initialize session state
if "plot_image" not in st.session_state:
    st.session_state.plot_image = None
    st.session_state.X = None
    st.session_state.y = None
    st.session_state.question = None
    st.session_state.show_content = False

# Initial state: Show only the "Generate Question" button
if not st.session_state.show_content:
    if st.button("Generate Question"):
        # Generate a new scatter plot and question
        plot_image, X, y = generate_synthetic_plot()
        question, _ = generate_random_question()

        # Update session state
        st.session_state.plot_image = plot_image
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.question = question
        st.session_state.show_content = True

        # Force a rerun to refresh the UI
        st.rerun()

# Display content if a question has been generated
if st.session_state.show_content:
    st.subheader("Generated Question")
    st.write(st.session_state.question)
    st.subheader("Scatter Plot")
    st.image(st.session_state.plot_image, use_container_width=True)
    
    if st.button("Next"):
        # Reset session state for the next question
        st.session_state.plot_image = None
        st.session_state.X = None
        st.session_state.y = None
        st.session_state.question = None
        st.session_state.show_content = False
        st.rerun()
    
    if st.button("Answer"):
        process_question_and_plot(st.session_state.question, st.session_state.plot_image, st.session_state.X, st.session_state.y)