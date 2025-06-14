import streamlit as st
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import pytesseract
from PIL import Image
import io
import spacy
from spacy import displacy
import json
import os

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Full Logistic Regression syllabus
topics = {
    "Logistic Regression": {
        "subtopics": {
            "Introduction": {
                "easy": [
                    "What is the main goal of {topic} in {application}?",
                    "How does {topic} differ from {model} in terms of {aspect}?",
                    "Based on the scatter plot below with {n} points, would {topic} be effective? Why?",
                    "Why is {topic} commonly used for {scenario} in {field}?",
                    "In the line graph below of probability outputs, what does the curve indicate?",
                    "What advantage does {topic} have over {alternative} for {task}?",
                    "Given the data table below with {m} samples, is {topic} suitable? Why?",
                    "How does {topic} predict {output} in a {context} scenario?",
                    "Based on the scatter plot below, why is {topic} a good fit for {application}?",
                    "What’s one practical use of {topic} in {field}?",
                    "In the data table below, how does {topic} interpret {feature} patterns?",
                    "Why might {topic} be preferred for {goal} over {model}?",
                    "Based on the line graph below, what trend suggests {topic} works well?",
                    "How does {topic} simplify {task} in {scenario}?"
                ],
                "medium": [
                    "Compare {topic} and {model} for {task} based on {dimension}.",
                    "Based on the scatter plot below with {n} points, why choose {topic}?",
                    "In the line graph below with slope {s}, why is {topic} effective for {goal}?",
                    "Given the data table below with {m} samples, how does {topic} handle {feature}?",
                    "Why does {topic} perform well in {scenario} compared to {alternative}?",
                    "If the scatter plot below shows {n} points, how does {topic} classify them?",
                    "Based on the line graph below, what does the cost trend imply for {topic}?",
                    "In a {application} context, why does {topic} fit the data table below?",
                    "Contrast {topic} and {alternative} for {task} using the scatter plot below.",
                    "Given the data table below, why is {topic} ideal for {output} prediction?",
                    "How does {topic} balance {dimension} and {aspect} in {scenario}?",
                    "Based on the scatter plot below, why is {topic} suited for {field}?",
                    "In the line graph below, what does the probability curve suggest about {topic}?",
                    "Why might {topic} face challenges with {problem} in the data table below?"
                ],
                "hard": [
                    "Analyze why {topic} outperforms {alternative} for {goal} in {context}.",
                    "Based on the scatter plot below with w={w}, why is {topic}’s approach optimal?",
                    "Critique {topic} versus {model} for {problem} using the line graph below.",
                    "In the data table below with {m} samples, why prefer {topic} over {alternative}?",
                    "Given the scatter plot below, explain {topic}’s suitability for {task}.",
                    "If the line graph below shows noise {p}, how does {topic} adjust?",
                    "Analyze the data table below: how does {topic} handle {feature} variance?",
                    "Based on the scatter plot below with {n} points, critique {topic}’s performance.",
                    "In the line graph below over {t} iterations, what indicates {topic}’s convergence?",
                    "Why does {topic} excel in {scenario} but struggle with {problem} in the data below?",
                    "Given the scatter plot below, prove {topic}’s effectiveness for {field}.",
                    "How does {topic} mitigate {noise} in the data table below?",
                    "Critique {topic}’s assumptions for {task} based on the line graph below.",
                    "Based on the data table below, justify {topic}’s use in {application}."
                ]
            },
            "Mathematical Foundation": {
                "easy": [
                    "What is the sigmoid function in {topic} and its purpose for {task}?",
                    "Compute the sigmoid output for x={x} in {topic}.",
                    "Based on the scatter plot below, what does the decision boundary do?",
                    "If P(y=0) = {p}, what’s P(y=1) in {topic}?",
                    "In the line graph below, what’s the range of the sigmoid output?",
                    "Given the data table below, how does {topic} assign probabilities?",
                    "Calculate the logit if the probability is {p} in {topic}.",
                    "Based on the scatter plot below, how does the boundary separate classes?",
                    "In the line graph below at x={x}, what does the sigmoid slope mean?",
                    "How does {topic} use {input} to generate probabilities?",
                    "Given the data table below with {m} samples, what’s {topic}’s decision rule?",
                    "What’s the role of the decision boundary in {topic} for {n} features?",
                    "Based on the line graph below, why does the sigmoid fit {task}?",
                    "How are weights used in {topic} based on the scatter plot below?"
                ],
                "medium": [
                    "Derive the sigmoid output for x={x} in {topic} step-by-step.",
                    "Based on the scatter plot below with w={w}, how does the boundary function?",
                    "If logit is {l}, calculate the probability in {topic}.",
                    "In the line graph below, how does the sigmoid ensure outputs between {a} and {b}?",
                    "Compute the sigmoid’s derivative at x={x} in {topic}.",
                    "Given the data table below, how does {topic} use w={w} to predict {output}?",
                    "Based on the scatter plot below, why does the boundary shift with w={w}?",
                    "In the line graph below, how steep is the sigmoid at x={x}?",
                    "Given the data table below with {m} samples, derive {topic}’s probability rule.",
                    "How does {topic} handle {noise} in the scatter plot below?",
                    "Calculate the decision threshold for P(y=1)={p} in {topic}.",
                    "Based on the line graph below with slope {s}, what does it imply for {task}?",
                    "In the scatter plot below with w={w}, justify the boundary’s placement.",
                    "Given the data table below, how does {topic} adjust for {feature} spread?"
                ],
                "hard": [
                    "Prove the sigmoid’s optimality for {topic}’s {task} using {theory}.",
                    "Derive the {n}-dimensional boundary equation for {topic} from the scatter plot below.",
                    "Analyze {topic}’s robustness to {noise} in the data table below.",
                    "If w={w}, b={b}, compute the margin width in the scatter plot below.",
                    "Critique the sigmoid versus {alternative} based on the line graph below.",
                    "Derive the sigmoid’s second derivative at x={x} in {topic}.",
                    "Prove {topic}’s outputs stay between {a} and {b} in the line graph below.",
                    "In the data table below with {m} samples, how does {topic} manage {noise}?",
                    "If logit={l}, derive the inverse mapping in the scatter plot below.",
                    "Analyze how {n} features affect the boundary in the scatter plot below.",
                    "Prove the sigmoid’s convexity for {topic}’s {goal} using the line graph below.",
                    "Based on the data table below, why does the sigmoid suit {task}?",
                    "How does {topic} adapt to {problem} in the scatter plot below?",
                    "Critique {topic}’s linearity in the data table below with {n} features."
                ]
            },
            "Cost Function and Optimization": {
                "easy": [
                    "What is the cost function in {topic} and its purpose?",
                    "Calculate the log loss for y={y}, p={p} in {topic}.",
                    "Based on the line graph below, what does the cost trend show?",
                    "How does {topic} use gradient descent for {task}?",
                    "Given the data table below, why does {topic} minimize errors?",
                    "What’s the role of log loss in {topic} for {output}?",
                    "In the line graph below over {t} iterations, what’s the cost’s behavior?",
                    "How does {topic} adjust weights based on the data table below?",
                    "Based on the scatter plot below, why does {topic} optimize the boundary?",
                    "What’s the simplest explanation of gradient descent in {topic}?",
                    "Given the data table below with {m} samples, how does {topic} reduce cost?",
                    "In the line graph below, why does the cost decrease?",
                    "How does {topic} measure error for {n} points in the scatter plot below?",
                    "What does the cost function penalize in {topic}?"
                ],
                "medium": [
                    "Derive the log loss formula for y={y}, p={p} in {topic}.",
                    "Based on the line graph below, how does gradient descent affect {topic}’s cost?",
                    "Given the data table below with {m} samples, how does {topic} optimize weights?",
                    "If weight w={w} updates by {d}, what’s the new weight in {topic}?",
                    "In the scatter plot below, how does {topic} minimize errors for {n} points?",
                    "Compute the gradient of log loss for y={y}, p={p} in {topic}.",
                    "Based on the line graph below over {t} iterations, why does cost drop?",
                    "Given the data table below, how does {topic} adjust for {feature} errors?",
                    "How does gradient descent update {topic}’s {parameter} in {scenario}?",
                    "In the scatter plot below with w={w}, why does optimization shift the boundary?",
                    "Calculate the cost change with learning rate {r} in {topic}.",
                    "Based on the line graph below, what does a flat cost imply?",
                    "Given the data table below, how does {topic} handle {noise} in optimization?",
                    "Why does {topic}’s cost converge in the line graph below?"
                ],
                "hard": [
                    "Prove why log loss is convex for {topic}’s optimization.",
                    "Analyze the impact of learning rate {r} on {topic}’s convergence in the line graph below.",
                    "Derive the gradient descent update rule for {topic} with {f} features.",
                    "In the data table below with {m} samples, how does {topic} optimize under {noise}?",
                    "If cost is {c} for {n} points in the scatter plot below, estimate the final cost.",
                    "Critique gradient descent versus {alternative} in {topic} based on the line graph below.",
                    "Prove the convergence of {topic}’s cost in {t} steps using the data table below.",
                    "Based on the scatter plot below, how does {topic} balance cost and boundary fit?",
                    "Derive the full optimization process for {topic} with w={w}, b={b}.",
                    "In the line graph below, why does {topic}’s cost stabilize at {c}?",
                    "Analyze how {topic} handles {problem} in the data table below during optimization.",
                    "Prove the effect of {parameter} on {topic}’s cost in the scatter plot below.",
                    "Critique {topic}’s optimization speed based on the line graph below.",
                    "How does {topic} adjust gradients for {feature} in the data table below?"
                ]
            },
            "Regularization": {
                "easy": [
                    "What is the purpose of regularization in {topic}?",
                    "Calculate the L2 penalty for w={w}, lambda={l} in {topic}.",
                    "Based on the scatter plot below, how does regularization affect {topic}?",
                    "What does L1 regularization do to weights in {topic}?",
                    "In the line graph below, how does regularization change the cost?",
                    "Given the data table below, why does {topic} use regularization?",
                    "How does {topic} prevent overfitting with {method}?",
                    "Based on the scatter plot below with {n} points, what’s regularization’s role?",
                    "In the data table below with {m} samples, how does L2 regularization help?",
                    "What’s the difference between L1 and L2 regularization in {topic}?",
                    "Given the line graph below, why does regularization stabilize {topic}?",
                    "How does {topic} adjust weights with lambda={l} in {scenario}?",
                    "Based on the scatter plot below, why does regularization improve {task}?",
                    "What’s the simplest effect of regularization on {topic}?"
                ],
                "medium": [
                    "Derive the L1 regularization term for w={w} in {topic}.",
                    "Based on the scatter plot below, how does regularization shrink weights?",
                    "In the line graph below with lambda={l}, how does {topic}’s cost change?",
                    "Given the data table below with {m} samples, how does L2 regularization work?",
                    "Compare L1 and L2 regularization effects on {topic} for w={w}.",
                    "Based on the scatter plot below with {n} points, why does regularization help?",
                    "In the data table below, how does {topic} balance {feature} with regularization?",
                    "Calculate the regularized cost for loss={c}, w={w}, lambda={l} in {topic}.",
                    "Given the line graph below, how does regularization prevent overfitting?",
                    "How does {topic} adjust the boundary with L1 in the scatter plot below?",
                    "Based on the data table below, why does regularization improve {goal}?",
                    "In the line graph below, what does a high lambda={l} imply for {topic}?",
                    "Given the scatter plot below, how does regularization affect {dimension}?",
                    "Why does {topic} use {method} for {problem} in the data table below?"
                ],
                "hard": [
                    "Prove why L1 regularization induces sparsity in {topic} for {f} features.",
                    "Analyze how regularization impacts {topic}’s fit in the scatter plot below.",
                    "Derive the full cost with L2 regularization for w={w}, lambda={l} in {topic}.",
                    "In the data table below with {m} samples, how does regularization handle {noise}?",
                    "Critique L1 versus L2 regularization in {topic} based on the line graph below.",
                    "Prove the effect of lambda={l} on {topic}’s weights in the scatter plot below.",
                    "Based on the data table below, how does regularization balance {dimension}?",
                    "In the line graph below with lambda={l}, why does {topic} stabilize?",
                    "Analyze {topic}’s boundary shift with regularization in the scatter plot below.",
                    "Derive the gradient of L2 regularization for w={w} in {topic}.",
                    "Given the data table below, critique {topic}’s regularization for {problem}.",
                    "Prove how regularization improves {goal} in the line graph below.",
                    "Based on the scatter plot below, how does {topic} adapt to {feature} with L1?",
                    "Critique {topic}’s regularization trade-offs in the data table below."
                ]
            },
            "Multiclass Logistic Regression": {
                "easy": [
                    "What is the goal of multiclass {topic} in {application}?",
                    "How does OvR work in {topic} for {n} classes?",
                    "Based on the scatter plot below, can {topic} handle {n} classes?",
                    "What’s the difference between OvR and OvO in {topic}?",
                    "In the line graph below, what does the probability curve show for {n} classes?",
                    "Given the data table below with {m} samples, how does {topic} classify {n} classes?",
                    "How does {topic} extend binary classification to {task}?",
                    "Based on the scatter plot below, why use OvR for {n} classes?",
                    "In the data table below, what’s the role of OvO in {topic}?",
                    "Calculate the number of OvR classifiers for {n} classes in {topic}.",
                    "Given the line graph below, how does {topic} predict {n} classes?",
                    "How does {topic} handle {feature} for {n} classes in the scatter plot below?",
                    "Based on the data table below, why does {topic} support {goal}?",
                    "What’s the simplest way {topic} manages {n} classes?"
                ],
                "medium": [
                    "Explain how OvR trains {n} classifiers in {topic}.",
                    "Based on the scatter plot below with {n} classes, why use OvO in {topic}?",
                    "In the line graph below, how does {topic} assign probabilities to {n} classes?",
                    "Given the data table below with {m} samples, how does OvR work in {topic}?",
                    "Compare OvR and OvO in {topic} for {n} classes based on {dimension}.",
                    "Based on the scatter plot below, how does {topic} separate {n} classes?",
                    "In the data table below, derive {topic}’s OvO decision for {n} classes.",
                    "Calculate the number of OvO pairs for {n} classes in {topic}.",
                    "Given the line graph below, how does {topic} handle {feature} for {n} classes?",
                    "How does {topic} adjust its cost for {n} classes in the data table below?",
                    "Based on the scatter plot below, why does OvR suit {scenario}?",
                    "In the line graph below with {n} curves, what does it imply for {topic}?",
                    "Given the data table below, how does {topic} manage {noise} with {n} classes?",
                    "Why does {topic} prefer OvR over OvO in the scatter plot below?"
                ],
                "hard": [
                    "Prove why OvR scales better than OvO in {topic} for {n} classes.",
                    "Analyze the computational cost of OvO in {topic} based on the scatter plot below.",
                    "Derive the decision rule for OvR with {n} classes in the data table below.",
                    "In the line graph below with {n} probabilities, why use {topic} for {task}?",
                    "Critique OvR versus OvO in {topic} for {problem} using the scatter plot below.",
                    "Prove how {topic} handles {n} classes in the data table below with {noise}.",
                    "Based on the scatter plot below, analyze {topic}’s OvO performance for {n} classes.",
                    "In the line graph below, how does {topic} adapt probabilities for {n} classes?",
                    "Derive the full OvR process for {n} classes in {topic} with w={w}.",
                    "Given the data table below, critique {topic}’s multiclass fit for {feature}.",
                    "Prove the efficiency of OvO for {n} classes in the scatter plot below.",
                    "Based on the line graph below, how does {topic} balance {dimension} for {n} classes?",
                    "Analyze {topic}’s multiclass boundary in the scatter plot below with {n} classes.",
                    "Critique {topic}’s OvR approach for {scenario} in the data table below."
                ]
            },
            "Implementation": {
                "easy": [
                    "What Python library is commonly used for {topic}?",
                    "How does {topic}’s fit method work in {context}?",
                    "Based on the scatter plot below, how does {topic} predict {n} points?",
                    "Calculate the prediction for w={w}, x={x} in {topic}.",
                    "In the line graph below, what does {topic}’s output represent?",
                    "Given the data table below with {m} samples, how does {topic} train?",
                    "What’s the role of scikit-learn in {topic}’s {task}?",
                    "Based on the scatter plot below, why does {topic} classify easily?",
                    "In the data table below, how does {topic} use {feature} for prediction?",
                    "How does {topic} compute probabilities in {scenario}?",
                    "Given the line graph below, what does {topic}’s curve predict?",
                    "What’s the simplest way to implement {topic} for {n} points?",
                    "Based on the scatter plot below, how does {topic} handle {output}?",
                    "How does {topic} process {input} in the data table below?"
                ],
                "medium": [
                    "Explain how scikit-learn trains {topic} for {n} samples in the scatter plot below.",
                    "If coefficients are w1={w1}, w2={w2}, compute the output for x1={x}, x2={x} in {topic}.",
                    "Based on the line graph below, how does {topic} predict probabilities?",
                    "Given the data table below with {m} samples, how does {topic} fit the data?",
                    "In the scatter plot below, how does {topic}’s predict method work for {n} points?",
                    "Derive {topic}’s prediction rule for {f} features in the data table below.",
                    "Based on the line graph below, why does {topic}’s output stabilize?",
                    "Given the scatter plot below, how does {topic} handle {feature} in {scenario}?",
                    "In the data table below, how does {topic} compute {output} with w={w}?",
                    "How does {topic}’s predict_proba work for P(y=1)={p} in {context}?",
                    "Based on the line graph below with slope {s}, what does {topic} predict?",
                    "Given the scatter plot below, why is {topic} efficient for {task}?",
                    "In the data table below, how does {topic} adjust {parameter}?",
                    "How does {topic} implement {method} in the scatter plot below?"
                ],
                "hard": [
                    "Prove how scikit-learn optimizes {topic}’s {parameter} in {t} iterations.",
                    "Analyze {topic}’s runtime for {n} samples in the scatter plot below.",
                    "Derive the full prediction equation for {topic} with w={w} in the data table below.",
                    "In the line graph below, critique {topic}’s implementation for {scenario}.",
                    "Based on the scatter plot below, how does {topic} scale with {f} features?",
                    "Given the data table below with {m} samples, prove {topic}’s efficiency.",
                    "Critique manual versus scikit-learn implementation of {topic} in the line graph below.",
                    "In the scatter plot below with {n} points, how does {topic} handle {noise}?",
                    "Derive {topic}’s probability distribution for {n} classes in the data table below.",
                    "Based on the line graph below, analyze {topic}’s convergence for {task}.",
                    "Prove {topic}’s prediction accuracy in the scatter plot below with w={w}.",
                    "Given the data table below, how does {topic} optimize {feature} weights?",
                    "Critique {topic}’s implementation trade-offs in the line graph below.",
                    "Analyze {topic}’s performance for {problem} in the data table below."
                ]
            }
        }
    },
    "Naïve Bayes": {
        "subtopics": {
            "Introduction": {
                "easy": [
                    "What is the main idea behind {topic} in {application}?",
                    "How does {topic} differ from {model} in terms of {aspect}?",
                    "Based on the scatter plot below with {n} points, would {topic} work well? Why?",
                    "Why is {topic} effective for {scenario} in {field}?",
                    "In the line graph below of probability outputs, what does the curve suggest?",
                    "What advantage does {topic} have over {alternative} for {task}?",
                    "Given the data table below with {m} samples, is {topic} suitable? Why?",
                    "How does {topic} classify {output} in a {context} scenario?",
                    "Based on the scatter plot below, why is {topic} a good fit for {application}?",
                    "What’s one key assumption of {topic} in {field}?",
                    "In the data table below, how does {topic} use {feature} patterns?",
                    "Why might {topic} be chosen for {goal} over {model}?",
                    "Based on the line graph below, what trend shows {topic}’s strength?",
                    "How does {topic} simplify {task} in {scenario}?"
                ],
                "medium": [
                    "Compare {topic} and {model} for {task} based on {dimension}.",
                    "Based on the scatter plot below with {n} points, why prefer {topic}?",
                    "In the line graph below with slope {s}, why is {topic} effective for {goal}?",
                    "Given the data table below with {m} samples, how does {topic} classify {feature}?",
                    "Why does {topic} perform well in {scenario} compared to {alternative}?",
                    "If the scatter plot below shows {n} points, how does {topic} predict them?",
                    "Based on the line graph below, what does the probability trend imply?",
                    "In a {application} context, why does {topic} fit the data table below?",
                    "Contrast {topic} and {alternative} for {task} using the scatter plot below.",
                    "Given the data table below, why is {topic} ideal for {output} prediction?",
                    "How does {topic} balance {dimension} and {aspect} in {scenario}?",
                    "Based on the scatter plot below, why is {topic} suited for {field}?",
                    "In the line graph below, what does the probability curve suggest about {topic}?",
                    "Why might {topic} struggle with {problem} in the data table below?"
                ],
                "hard": [
                    "Analyze why {topic} outperforms {alternative} for {goal} in {context}.",
                    "Based on the scatter plot below with w={w}, why is {topic}’s approach effective?",
                    "Critique {topic} versus {model} for {problem} using the line graph below.",
                    "In the data table below with {m} samples, why prefer {topic} over {alternative}?",
                    "Given the scatter plot below, explain {topic}’s suitability for {task}.",
                    "If the line graph below shows noise {p}, how does {topic} adjust?",
                    "Analyze the data table below: how does {topic} handle {feature} dependencies?",
                    "Based on the scatter plot below with {n} points, critique {topic}’s assumptions.",
                    "In the line graph below over {t} iterations, what indicates {topic}’s performance?",
                    "Why does {topic} excel in {scenario} but struggle with {problem} in the data below?",
                    "Given the scatter plot below, prove {topic}’s effectiveness for {field}.",
                    "How does {topic} mitigate {noise} in the data table below?",
                    "Critique {topic}’s independence assumption based on the line graph below.",
                    "Based on the data table below, justify {topic}’s use in {application}."
                ]
            },
            "Probability Basics": {
                "easy": [
                    "What is Bayes’ theorem and its role in {topic}?",
                    "Calculate P(A|B) given P(B|A)={p}, P(A)={p}, P(B)={p} in {topic}.",
                    "Based on the scatter plot below, how does {topic} use probabilities?",
                    "What’s the conditional probability in {topic} for {task}?",
                    "In the line graph below, what does the probability curve represent?",
                    "Given the data table below, how does {topic} apply Bayes’ rule?",
                    "How does {topic} compute posterior probabilities?",
                    "Based on the scatter plot below with {n} points, why does {topic} use priors?",
                    "In the data table below with {m} samples, what’s the likelihood’s role?",
                    "What’s the simplest explanation of Bayes’ theorem in {topic}?",
                    "Given the line graph below, how does {topic} estimate {output}?",
                    "How does {topic} use {feature} probabilities in the scatter plot below?",
                    "Based on the data table below, why does {topic} rely on priors?",
                    "What does P(B|A) mean in {topic}’s {context}?"
                ],
                "medium": [
                    "Derive Bayes’ theorem for {topic} with P(A)={p}, P(B)={p}.",
                    "Based on the scatter plot below with {n} points, how does {topic} compute posteriors?",
                    "In the line graph below, how does {topic} update probabilities?",
                    "Given the data table below with {m} samples, calculate {topic}’s likelihood.",
                    "How does {topic} combine priors and likelihoods for {task}?",
                    "Based on the scatter plot below, why does {topic} assume independence?",
                    "In the data table below, derive {topic}’s posterior for {feature}.",
                    "Calculate the joint probability for {f} features in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} in probabilities?",
                    "How does {topic} use Bayes’ rule in the scatter plot below?",
                    "Based on the data table below, why does {topic} prioritize likelihood?",
                    "In the line graph below with slope {s}, what does it imply for {topic}?",
                    "Given the scatter plot below, how does {topic} adjust priors?",
                    "Why does {topic} simplify probabilities in the data table below?"
                ],
                "hard": [
                    "Prove Bayes’ theorem’s correctness for {topic}’s {task}.",
                    "Analyze how {topic} computes posteriors in the scatter plot below.",
                    "Derive the full probability model for {topic} with {f} features.",
                    "In the data table below with {m} samples, how does {topic} handle {noise}?",
                    "Critique {topic}’s independence assumption using the line graph below.",
                    "Prove the likelihood’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s joint probability.",
                    "In the line graph below, how does {topic} balance priors and likelihoods?",
                    "Analyze {topic}’s probability updates for {n} points in the scatter plot below.",
                    "Prove why {topic}’s simplicity works in the data table below.",
                    "Given the line graph below, critique {topic}’s probability model for {problem}.",
                    "Based on the scatter plot below, how does {topic} adapt to {feature}?",
                    "Derive {topic}’s posterior distribution in the data table below.",
                    "Critique {topic}’s Bayes’ application in the line graph below."
                ]
            },
            "Gaussian Naïve Bayes": {
                "easy": [
                    "What is the assumption of {topic} for continuous data?",
                    "Calculate the Gaussian probability for x={x}, mean={m}, std={s} in {topic}.",
                    "Based on the scatter plot below, how does {topic} classify {n} points?",
                    "What’s the role of variance in {topic} for {task}?",
                    "In the line graph below, what does the Gaussian curve show?",
                    "Given the data table below, how does {topic} predict {output}?",
                    "How does {topic} model {feature} distributions?",
                    "Based on the scatter plot below with {n} points, why use {topic}?",
                    "In the data table below with {m} samples, what’s the Gaussian’s role?",
                    "What’s the simplest way {topic} handles continuous {input}?",
                    "Given the line graph below, how does {topic} estimate probabilities?",
                    "How does {topic} use means in the scatter plot below?",
                    "Based on the data table below, why does {topic} assume normality?",
                    "What does the Gaussian assumption mean for {topic} in {context}?"
                ],
                "medium": [
                    "Derive the Gaussian probability for x={x} in {topic} with mean={m}, std={s}.",
                    "Based on the scatter plot below with {n} points, how does {topic} classify?",
                    "In the line graph below, how does {topic} fit Gaussian curves?",
                    "Given the data table below with {m} samples, calculate {topic}’s likelihood.",
                    "How does {topic} estimate variance for {f} features?",
                    "Based on the scatter plot below, why does {topic} suit continuous data?",
                    "In the data table below, derive {topic}’s Gaussian prediction.",
                    "Calculate the probability density for {feature} in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} in Gaussians?",
                    "How does {topic} use standard deviation in the scatter plot below?",
                    "Based on the data table below, why does {topic} fit {scenario}?",
                    "In the line graph below with slope {s}, what does it imply for {topic}?",
                    "Given the scatter plot below, how does {topic} adjust means?",
                    "Why does {topic} model {feature} as Gaussian in the data table below?"
                ],
                "hard": [
                    "Prove why Gaussian assumptions work for {topic}’s {task}.",
                    "Analyze {topic}’s classification in the scatter plot below with {n} points.",
                    "Derive the full Gaussian model for {topic} with {f} features.",
                    "In the data table below with {m} samples, how does {topic} handle {noise}?",
                    "Critique {topic}’s normality assumption using the line graph below.",
                    "Prove the Gaussian likelihood’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s probability density.",
                    "In the line graph below, how does {topic} adapt Gaussians for {task}?",
                    "Analyze {topic}’s variance estimation in the scatter plot below.",
                    "Prove why {topic} excels for continuous data in the data table below.",
                    "Given the line graph below, critique {topic}’s Gaussian fit for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} skewness?",
                    "Derive {topic}’s Gaussian posterior in the data table below.",
                    "Critique {topic}’s Gaussian approach in the line graph below."
                ]
            },
            "Multinomial Naïve Bayes": {
                "easy": [
                    "What is the purpose of {topic} for discrete data?",
                    "Calculate the probability of a count={c} in {topic} with p={p}.",
                    "Based on the scatter plot below, how does {topic} classify {n} points?",
                    "What’s the role of word counts in {topic} for {task}?",
                    "In the line graph below, what does the probability trend show?",
                    "Given the data table below, how does {topic} predict {output}?",
                    "How does {topic} handle {feature} frequencies?",
                    "Based on the scatter plot below with {n} points, why use {topic}?",
                    "In the data table below with {m} samples, what’s {topic}’s assumption?",
                    "What’s the simplest way {topic} processes text data?",
                    "Given the line graph below, how does {topic} estimate likelihoods?",
                    "How does {topic} use counts in the scatter plot below?",
                    "Based on the data table below, why does {topic} suit {application}?",
                    "What does {topic} assume about {feature} in {context}?"
                ],
                "medium": [
                    "Derive the multinomial probability for count={c} in {topic} with p={p}.",
                    "Based on the scatter plot below with {n} points, how does {topic} classify?",
                    "In the line graph below, how does {topic} model frequency probabilities?",
                    "Given the data table below with {m} samples, calculate {topic}’s likelihood.",
                    "How does {topic} estimate probabilities for {f} features?",
                    "Based on the scatter plot below, why does {topic} fit discrete data?",
                    "In the data table below, derive {topic}’s prediction for {feature}.",
                    "Calculate the likelihood for {feature} counts in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} in counts?",
                    "How does {topic} use frequencies in the scatter plot below?",
                    "Based on the data table below, why does {topic} excel in {scenario}?",
                    "In the line graph below with slope {s}, what does it imply for {topic}?",
                    "Given the scatter plot below, how does {topic} adjust probabilities?",
                    "Why does {topic} model {feature} as multinomial in the data table below?"
                ],
                "hard": [
                    "Prove why multinomial assumptions work for {topic}’s {task}.",
                    "Analyze {topic}’s classification in the scatter plot below with {n} points.",
                    "Derive the full multinomial model for {topic} with {f} features.",
                    "In the data table below with {m} samples, how does {topic} handle {noise}?",
                    "Critique {topic}’s frequency assumption using the line graph below.",
                    "Prove the likelihood’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s multinomial probability.",
                    "In the line graph below, how does {topic} adapt counts for {task}?",
                    "Analyze {topic}’s frequency estimation in the scatter plot below.",
                    "Prove why {topic} excels for text data in the data table below.",
                    "Given the line graph below, critique {topic}’s multinomial fit for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} sparsity?",
                    "Derive {topic}’s multinomial posterior in the data table below.",
                    "Critique {topic}’s multinomial approach in the line graph below."
                ]
            },
            "Bernoulli Naïve Bayes": {
                "easy": [
                    "What is the goal of {topic} for binary data?",
                    "Calculate the probability of presence={y} in {topic} with p={p}.",
                    "Based on the scatter plot below, how does {topic} classify {n} points?",
                    "What’s the role of feature presence in {topic} for {task}?",
                    "In the line graph below, what does the probability curve indicate?",
                    "Given the data table below, how does {topic} predict {output}?",
                    "How does {topic} model {feature} occurrences?",
                    "Based on the scatter plot below with {n} points, why use {topic}?",
                    "In the data table below with {m} samples, what’s {topic}’s assumption?",
                    "What’s the simplest way {topic} handles binary {input}?",
                    "Given the line graph below, how does {topic} estimate likelihoods?",
                    "How does {topic} use presence/absence in the scatter plot below?",
                    "Based on the data table below, why does {topic} suit {application}?",
                    "What does {topic} assume about {feature} in {context}?"
                ],
                "medium": [
                    "Derive the Bernoulli probability for presence={y} in {topic} with p={p}.",
                    "Based on the scatter plot below with {n} points, how does {topic} classify?",
                    "In the line graph below, how does {topic} model binary probabilities?",
                    "Given the data table below with {m} samples, calculate {topic}’s likelihood.",
                    "How does {topic} estimate probabilities for {f} binary features?",
                    "Based on the scatter plot below, why does {topic} fit binary data?",
                    "In the data table below, derive {topic}’s prediction for {feature}.",
                    "Calculate the likelihood for {feature} presence in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} in binaries?",
                    "How does {topic} use absence in the scatter plot below?",
                    "Based on the data table below, why does {topic} excel in {scenario}?",
                    "In the line graph below with slope {s}, what does it imply for {topic}?",
                    "Given the scatter plot below, how does {topic} adjust probabilities?",
                    "Why does {topic} model {feature} as Bernoulli in the data table below?"
                ],
                "hard": [
                    "Prove why Bernoulli assumptions work for {topic}’s {task}.",
                    "Analyze {topic}’s classification in the scatter plot below with {n} points.",
                    "Derive the full Bernoulli model for {topic} with {f} features.",
                    "In the data table below with {m} samples, how does {topic} handle {noise}?",
                    "Critique {topic}’s binary assumption using the line graph below.",
                    "Prove the likelihood’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s Bernoulli probability.",
                    "In the line graph below, how does {topic} adapt binaries for {task}?",
                    "Analyze {topic}’s presence estimation in the scatter plot below.",
                    "Prove why {topic} excels for binary data in the data table below.",
                    "Given the line graph below, critique {topic}’s Bernoulli fit for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} sparsity?",
                    "Derive {topic}’s Bernoulli posterior in the data table below.",
                    "Critique {topic}’s Bernoulli approach in the line graph below."
                ]
            },
            "Implementation": {
                "easy": [
                    "What Python library is used to implement {topic}?",
                    "How does {topic}’s fit method work in {context}?",
                    "Based on the scatter plot below, how does {topic} predict {n} points?",
                    "Calculate the prediction for x={x} in {topic} with p={p}.",
                    "In the line graph below, what does {topic}’s output represent?",
                    "Given the data table below with {m} samples, how does {topic} train?",
                    "What’s the role of scikit-learn in {topic}’s {task}?",
                    "Based on the scatter plot below, why does {topic} classify easily?",
                    "In the data table below, how does {topic} use {feature} for prediction?",
                    "How does {topic} compute probabilities in {scenario}?",
                    "Given the line graph below, what does {topic}’s curve predict?",
                    "What’s the simplest way to implement {topic} for {n} points?",
                    "Based on the scatter plot below, how does {topic} handle {output}?",
                    "How does {topic} process {input} in the data table below?"
                ],
                "medium": [
                    "Explain how scikit-learn trains {topic} for {n} samples in the scatter plot below.",
                    "If probability p={p}, compute the output for x={x} in {topic}.",
                    "Based on the line graph below, how does {topic} predict probabilities?",
                    "Given the data table below with {m} samples, how does {topic} fit the data?",
                    "In the scatter plot below, how does {topic}’s predict method work for {n} points?",
                    "Derive {topic}’s prediction rule for {f} features in the data table below.",
                    "Based on the line graph below, why does {topic}’s output stabilize?",
                    "Given the scatter plot below, how does {topic} handle {feature} in {scenario}?",
                    "In the data table below, how does {topic} compute {output} with p={p}?",
                    "How does {topic}’s predict_proba work for P(y=1)={p} in {context}?",
                    "Based on the line graph below with slope {s}, what does {topic} predict?",
                    "Given the scatter plot below, why is {topic} efficient for {task}?",
                    "In the data table below, how does {topic} adjust {parameter}?",
                    "How does {topic} implement {method} in the scatter plot below?"
                ],
                "hard": [
                    "Prove how scikit-learn optimizes {topic}’s {parameter} in {t} iterations.",
                    "Analyze {topic}’s runtime for {n} samples in the scatter plot below.",
                    "Derive the full prediction equation for {topic} with p={p} in the data table below.",
                    "In the line graph below, critique {topic}’s implementation for {scenario}.",
                    "Based on the scatter plot below, how does {topic} scale with {f} features?",
                    "Given the data table below with {m} samples, prove {topic}’s efficiency.",
                    "Critique manual versus scikit-learn implementation of {topic} in the line graph below.",
                    "In the scatter plot below with {n} points, how does {topic} handle {noise}?",
                    "Derive {topic}’s probability distribution for {n} classes in the data table below.",
                    "Based on the line graph below, analyze {topic}’s convergence for {task}.",
                    "Prove {topic}’s prediction accuracy in the scatter plot below with p={p}.",
                    "Given the data table below, how does {topic} optimize {feature} probabilities?",
                    "Critique {topic}’s implementation trade-offs in the line graph below.",
                    "Analyze {topic}’s performance for {problem} in the data table below."
                ]
            }
        }
    },
    "Support Vector Machines": {
        "subtopics": {
            "Introduction": {
                "easy": [
                    "What is the core idea of {topic} in {application}?",
                    "How does {topic} differ from {model} in terms of {aspect}?",
                    "Based on the scatter plot below with {n} points, would {topic} work well? Why?",
                    "Why is {topic} effective for {scenario} in {field}?",
                    "In the line graph below of decision boundary, what does the separation suggest?",
                    "What advantage does {topic} have over {alternative} for {task}?",
                    "Given the data table below with {m} samples, is {topic} suitable? Why?",
                    "How does {topic} classify {output} in a {context} scenario?",
                    "Based on the scatter plot below, why is {topic} a good fit for {application}?",
                    "What’s the main goal of maximizing the margin in {topic}?",
                    "In the data table below, how does {topic} use {feature} patterns?",
                    "Why might {topic} be chosen for {goal} over {model}?",
                    "Based on the line graph below, what trend shows {topic}’s strength?",
                    "How does {topic} simplify {task} in {scenario}?"
                ],
                "medium": [
                    "Compare {topic} and {model} for {task} based on {dimension}.",
                    "Based on the scatter plot below with {n} points, why prefer {topic}?",
                    "In the line graph below with slope {s}, why is {topic} effective for {goal}?",
                    "Given the data table below with {m} samples, how does {topic} separate {feature}?",
                    "Why does {topic} perform well in {scenario} compared to {alternative}?",
                    "If the scatter plot below shows {n} points, how does {topic} define the margin?",
                    "Based on the line graph below, what does the boundary trend imply?",
                    "In a {application} context, why does {topic} fit the data table below?",
                    "Contrast {topic} and {alternative} for {task} using the scatter plot below.",
                    "Given the data table below, why is {topic} ideal for {output} prediction?",
                    "How does {topic} balance {dimension} and {aspect} in {scenario}?",
                    "Based on the scatter plot below, why is {topic} suited for {field}?",
                    "In the line graph below, what does the margin width suggest about {topic}?",
                    "Why might {topic} struggle with {problem} in the data table below?"
                ],
                "hard": [
                    "Analyze why {topic} outperforms {alternative} for {goal} in {context}.",
                    "Based on the scatter plot below with w={w}, why is {topic}’s margin optimal?",
                    "Critique {topic} versus {model} for {problem} using the line graph below.",
                    "In the data table below with {m} samples, why prefer {topic} over {alternative}?",
                    "Given the scatter plot below, explain {topic}’s suitability for {task}.",
                    "If the line graph below shows noise {p}, how does {topic} adjust?",
                    "Analyze the data table below: how does {topic} handle {feature} variance?",
                    "Based on the scatter plot below with {n} points, critique {topic}’s margin.",
                    "In the line graph below over {t} iterations, what indicates {topic}’s convergence?",
                    "Why does {topic} excel in {scenario} but struggle with {problem} in the data below?",
                    "Given the scatter plot below, prove {topic}’s effectiveness for {field}.",
                    "How does {topic} mitigate {noise} in the data table below?",
                    "Critique {topic}’s margin maximization based on the line graph below.",
                    "Based on the data table below, justify {topic}’s use in {application}."
                ]
            },
            "Linear SVM": {
                "easy": [
                    "What is the goal of a linear {topic} for {task}?",
                    "How does a hard margin work in {topic}?",
                    "Based on the scatter plot below, is {topic} suitable for {n} points?",
                    "What’s the role of the margin in {topic} for {output}?",
                    "In the line graph below, what does the linear boundary show?",
                    "Given the data table below, how does {topic} separate classes?",
                    "How does a soft margin differ from a hard margin in {topic}?",
                    "Based on the scatter plot below with {n} points, why use {topic}?",
                    "In the data table below with {m} samples, what’s the linear assumption?",
                    "What’s the simplest explanation of a linear {topic}?",
                    "Given the line graph below, how does {topic} classify {output}?",
                    "How does {topic} use {feature} in the scatter plot below?",
                    "Based on the data table below, why does {topic} need separability?",
                    "What does a linear boundary mean in {topic}’s {context}?"
                ],
                "medium": [
                    "Explain how {topic} finds the optimal hyperplane for {n} points in the scatter plot below.",
                    "Based on the scatter plot below with w={w}, how does {topic} set the margin?",
                    "In the line graph below, how does a soft margin adjust {topic}’s boundary?",
                    "Given the data table below with {m} samples, how does {topic} handle misclassifications?",
                    "How does {topic} balance margin size and errors in {scenario}?",
                    "Based on the scatter plot below, why does {topic} prefer linear separability?",
                    "In the data table below, derive {topic}’s hyperplane for {feature}.",
                    "Calculate the margin width for w={w} in {topic}.",
                    "Given the line graph below, how does {topic} adapt to {noise}?",
                    "How does {topic} use a soft margin in the scatter plot below?",
                    "Based on the data table below, why does {topic} suit {goal}?",
                    "In the line graph below with slope {s}, what does it imply for {topic}?",
                    "Given the scatter plot below, how does {topic} adjust the boundary?",
                    "Why does {topic} rely on linearity in the data table below?"
                ],
                "hard": [
                    "Prove why a hard margin maximizes separation in {topic} for {task}.",
                    "Analyze {topic}’s soft margin in the scatter plot below with {n} points.",
                    "Derive the hyperplane equation for {topic} with w={w}, b={b}.",
                    "In the data table below with {m} samples, how does {topic} handle {noise}?",
                    "Critique hard versus soft margins in {topic} using the line graph below.",
                    "Prove the margin’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s linear decision rule.",
                    "In the line graph below, how does {topic} adapt the boundary for {task}?",
                    "Analyze {topic}’s separability assumption in the scatter plot below.",
                    "Prove why {topic} excels for linear data in the data table below.",
                    "Given the line graph below, critique {topic}’s linear fit for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} overlap?",
                    "Derive {topic}’s soft margin optimization in the data table below.",
                    "Critique {topic}’s linearity in the line graph below."
                ]
            },
            "Kernel Trick": {
                "easy": [
                    "What is the purpose of the kernel trick in {topic}?",
                    "How does an RBF kernel work in {topic}?",
                    "Based on the scatter plot below, how does {topic} classify {n} points?",
                    "What’s the role of a polynomial kernel in {topic} for {task}?",
                    "In the line graph below, what does the transformed boundary show?",
                    "Given the data table below, how does {topic} handle non-linear data?",
                    "How does {topic} use kernels to improve {output}?",
                    "Based on the scatter plot below with {n} points, why use a kernel?",
                    "In the data table below with {m} samples, what’s the kernel’s effect?",
                    "What’s the simplest explanation of the kernel trick in {topic}?",
                    "Given the line graph below, how does {topic} map {feature}?",
                    "How does {topic} apply a kernel in the scatter plot below?",
                    "Based on the data table below, why does {topic} need kernels?",
                    "What does a kernel do in {topic}’s {context}?"
                ],
                "medium": [
                    "Explain how an RBF kernel transforms {n} points in the scatter plot below.",
                    "Based on the scatter plot below with w={w}, how does {topic} use a polynomial kernel?",
                    "In the line graph below, how does {topic} adjust with a kernel?",
                    "Given the data table below with {m} samples, how does {topic} classify non-linearly?",
                    "How does {topic} choose between RBF and polynomial kernels for {scenario}?",
                    "Based on the scatter plot below, why does {topic} need a kernel for {problem}?",
                    "In the data table below, derive {topic}’s kernel transformation for {feature}.",
                    "Calculate the kernel value for x1={x}, x2={x} with gamma={g} in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} with kernels?",
                    "How does {topic} apply an RBF kernel in the scatter plot below?",
                    "Based on the data table below, why does {topic} improve {goal} with kernels?",
                    "In the line graph below with slope {s}, what does the kernel imply?",
                    "Given the scatter plot below, how does {topic} adjust with a polynomial kernel?",
                    "Why does {topic} rely on kernels in the data table below?"
                ],
                "hard": [
                    "Prove why the kernel trick enables {topic} for {task} in non-linear data.",
                    "Analyze {topic}’s RBF kernel in the scatter plot below with {n} points.",
                    "Derive the kernel function for {topic} with degree={k} and gamma={g}.",
                    "In the data table below with {m} samples, how does {topic} handle {noise} with kernels?",
                    "Critique RBF versus polynomial kernels in {topic} using the line graph below.",
                    "Prove the kernel’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s transformed decision rule.",
                    "In the line graph below, how does {topic} adapt kernels for {task}?",
                    "Analyze {topic}’s kernel transformation in the scatter plot below.",
                    "Prove why {topic} excels with kernels in the data table below.",
                    "Given the line graph below, critique {topic}’s kernel fit for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} complexity?",
                    "Derive {topic}’s kernel-based margin in the data table below.",
                    "Critique {topic}’s kernel approach in the line graph below."
                ]
            },
            "Mathematical Foundation": {
                "easy": [
                    "What are support vectors in {topic} and their role in {task}?",
                    "How does {topic} use Lagrange multipliers?",
                    "Based on the scatter plot below, what do support vectors do?",
                    "What’s the dual form in {topic} for {output}?",
                    "In the line graph below, what does the margin width represent?",
                    "Given the data table below, how does {topic} identify support vectors?",
                    "How does {topic} optimize the margin mathematically?",
                    "Based on the scatter plot below with {n} points, why use support vectors?",
                    "In the data table below with {m} samples, what’s the dual problem?",
                    "What’s the simplest explanation of {topic}’s math?",
                    "Given the line graph below, how does {topic} define the boundary?",
                    "How does {topic} use {feature} in the scatter plot below mathematically?",
                    "Based on the data table below, why does {topic} rely on support vectors?",
                    "What does the primal form mean in {topic}’s {context}?"
                ],
                "medium": [
                    "Derive the dual form of {topic} for {n} points in the scatter plot below.",
                    "Based on the scatter plot below with w={w}, how does {topic} use Lagrange multipliers?",
                    "In the line graph below, how does {topic} optimize the margin?",
                    "Given the data table below with {m} samples, calculate {topic}’s support vectors.",
                    "How does {topic} solve the primal problem for {f} features?",
                    "Based on the scatter plot below, why does {topic} depend on support vectors?",
                    "In the data table below, derive {topic}’s dual objective for {feature}.",
                    "Calculate the margin distance for w={w}, b={b} in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} mathematically?",
                    "How does {topic} use the dual form in the scatter plot below?",
                    "Based on the data table below, why does {topic} optimize {goal}?",
                    "In the line graph below with slope {s}, what does it imply mathematically?",
                    "Given the scatter plot below, how does {topic} adjust support vectors?",
                    "Why does {topic} use Lagrange multipliers in the data table below?"
                ],
                "hard": [
                    "Prove the optimality of support vectors in {topic} for {task}.",
                    "Analyze {topic}’s dual form in the scatter plot below with {n} points.",
                    "Derive the full Lagrangian for {topic} with w={w}, b={b}.",
                    "In the data table below with {m} samples, how does {topic} handle {noise} mathematically?",
                    "Critique primal versus dual forms in {topic} using the line graph below.",
                    "Prove the margin’s maximization in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s support vector conditions.",
                    "In the line graph below, how does {topic} adapt the dual for {task}?",
                    "Analyze {topic}’s mathematical complexity in the scatter plot below.",
                    "Prove why {topic} relies on support vectors in the data table below.",
                    "Given the line graph below, critique {topic}’s math for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} in the dual?",
                    "Derive {topic}’s KKT conditions in the data table below.",
                    "Critique {topic}’s mathematical foundation in the line graph below."
                ]
            },
            "Regularization and Cost": {
                "easy": [
                    "What is the role of the C parameter in {topic}?",
                    "How does {topic} balance margin and errors?",
                    "Based on the scatter plot below, how does C affect {topic}?",
                    "What’s the cost function in {topic} for {task}?",
                    "In the line graph below, what does the cost trend show?",
                    "Given the data table below, how does {topic} use regularization?",
                    "How does a large C differ from a small C in {topic}?",
                    "Based on the scatter plot below with {n} points, why use C?",
                    "In the data table below with {m} samples, what’s C’s effect?",
                    "What’s the simplest explanation of regularization in {topic}?",
                    "Given the line graph below, how does {topic} adjust errors?",
                    "How does {topic} apply C to {feature} in the scatter plot below?",
                    "Based on the data table below, why does {topic} need regularization?",
                    "What does the cost penalize in {topic}’s {context}?"
                ],
                "medium": [
                    "Explain how C={c} changes {topic}’s margin in the scatter plot below.",
                    "Based on the scatter plot below with w={w}, how does {topic} use regularization?",
                    "In the line graph below, how does {topic} adjust cost with C={c}?",
                    "Given the data table below with {m} samples, how does {topic} tune C?",
                    "How does {topic} trade off margin and errors for {scenario}?",
                    "Based on the scatter plot below, why does {topic} need a small C for {problem}?",
                    "In the data table below, derive {topic}’s cost function for {feature}.",
                    "Calculate the regularized cost for C={c}, w={w} in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} with regularization?",
                    "How does {topic} adjust the margin with C in the scatter plot below?",
                    "Based on the data table below, why does {topic} improve {goal} with C?",
                    "In the line graph below with slope {s}, what does C imply?",
                    "Given the scatter plot below, how does {topic} balance errors?",
                    "Why does {topic} use regularization in the data table below?"
                ],
                "hard": [
                    "Prove why C controls regularization in {topic} for {task}.",
                    "Analyze {topic}’s cost function in the scatter plot below with {n} points.",
                    "Derive the full cost with C={c} for {topic} with w={w}.",
                    "In the data table below with {m} samples, how does {topic} handle {noise} with C?",
                    "Critique large versus small C in {topic} using the line graph below.",
                    "Prove the cost’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s regularized margin.",
                    "In the line graph below, how does {topic} adapt C for {task}?",
                    "Analyze {topic}’s regularization impact in the scatter plot below.",
                    "Prove why {topic} balances errors with C in the data table below.",
                    "Given the line graph below, critique {topic}’s cost fit for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} with C?",
                    "Derive {topic}’s cost optimization in the data table below.",
                    "Critique {topic}’s regularization trade-offs in the line graph below."
                ]
            },
            "Implementation": {
                "easy": [
                    "What Python library is used to implement {topic}?",
                    "How does {topic}’s fit method work in {context}?",
                    "Based on the scatter plot below, how does {topic} predict {n} points?",
                    "Calculate the prediction for w={w}, x={x} in {topic}.",
                    "In the line graph below, what does {topic}’s boundary represent?",
                    "Given the data table below with {m} samples, how does {topic} train?",
                    "What’s the role of scikit-learn in {topic}’s {task}?",
                    "Based on the scatter plot below, why does {topic} classify easily?",
                    "In the data table below, how does {topic} use {feature} for prediction?",
                    "How does {topic} compute the margin in {scenario}?",
                    "Given the line graph below, what does {topic}’s curve predict?",
                    "What’s the simplest way to implement {topic} for {n} points?",
                    "Based on the scatter plot below, how does {topic} handle {output}?",
                    "How does {topic} process {input} in the data table below?"
                ],
                "medium": [
                    "Explain how scikit-learn trains {topic} for {n} samples in the scatter plot below.",
                    "If weights are w1={w1}, w2={w2}, compute the output for x1={x}, x2={x} in {topic}.",
                    "Based on the line graph below, how does {topic} predict the boundary?",
                    "Given the data table below with {m} samples, how does {topic} fit the data?",
                    "In the scatter plot below, how does {topic}’s predict method work for {n} points?",
                    "Derive {topic}’s decision rule for {f} features in the data table below.",
                    "Based on the line graph below, why does {topic}’s margin stabilize?",
                    "Given the scatter plot below, how does {topic} handle {feature} in {scenario}?",
                    "In the data table below, how does {topic} compute {output} with w={w}?",
                    "How does {topic}’s kernel parameter work for C={c} in {context}?",
                    "Based on the line graph below with slope {s}, what does {topic} predict?",
                    "Given the scatter plot below, why is {topic} efficient for {task}?",
                    "In the data table below, how does {topic} adjust {parameter}?",
                    "How does {topic} implement {method} in the scatter plot below?"
                ],
                "hard": [
                    "Prove how scikit-learn optimizes {topic}’s {parameter} in {t} iterations.",
                    "Analyze {topic}’s runtime for {n} samples in the scatter plot below.",
                    "Derive the full prediction equation for {topic} with w={w} in the data table below.",
                    "In the line graph below, critique {topic}’s implementation for {scenario}.",
                    "Based on the scatter plot below, how does {topic} scale with {f} features?",
                    "Given the data table below with {m} samples, prove {topic}’s efficiency.",
                    "Critique manual versus scikit-learn implementation of {topic} in the line graph below.",
                    "In the scatter plot below with {n} points, how does {topic} handle {noise}?",
                    "Derive {topic}’s kernel transformation for {n} points in the data table below.",
                    "Based on the line graph below, analyze {topic}’s convergence for {task}.",
                    "Prove {topic}’s prediction accuracy in the scatter plot below with w={w}.",
                    "Given the data table below, how does {topic} optimize {feature} weights?",
                    "Critique {topic}’s implementation trade-offs in the line graph below.",
                    "Analyze {topic}’s performance for {problem} in the data table below."
                ]
            }
        }
    },
    "k-Nearest Neighbors": {
        "subtopics": {
            "Introduction": {
                "easy": [
                    "What is the basic idea of {topic} in {application}?",
                    "How does {topic} differ from {model} in terms of {aspect}?",
                    "Based on the scatter plot below with {n} points, would {topic} work well? Why?",
                    "Why is {topic} useful for {scenario} in {field}?",
                    "In the line graph below of distances, what does the trend suggest?",
                    "What advantage does {topic} have over {alternative} for {task}?",
                    "Given the data table below with {m} samples, is {topic} suitable? Why?",
                    "How does {topic} classify {output} in a {context} scenario?",
                    "Based on the scatter plot below, why is {topic} a good fit for {application}?",
                    "What’s the main principle behind {topic}’s predictions?",
                    "In the data table below, how does {topic} use {feature} patterns?",
                    "Why might {topic} be chosen for {goal} over {model}?",
                    "Based on the line graph below, what shows {topic}’s strength?",
                    "How does {topic} simplify {task} in {scenario}?"
                ],
                "medium": [
                    "Compare {topic} and {model} for {task} based on {dimension}.",
                    "Based on the scatter plot below with {n} points, why prefer {topic}?",
                    "In the line graph below with slope {s}, why is {topic} effective for {goal}?",
                    "Given the data table below with {m} samples, how does {topic} classify {feature}?",
                    "Why does {topic} perform well in {scenario} compared to {alternative}?",
                    "If the scatter plot below shows {n} points, how does {topic} predict them?",
                    "Based on the line graph below, what does the distance trend imply?",
                    "In a {application} context, why does {topic} fit the data table below?",
                    "Contrast {topic} and {alternative} for {task} using the scatter plot below.",
                    "Given the data table below, why is {topic} ideal for {output} prediction?",
                    "How does {topic} balance {dimension} and {aspect} in {scenario}?",
                    "Based on the scatter plot below, why is {topic} suited for {field}?",
                    "In the line graph below, what does the neighbor distance suggest?",
                    "Why might {topic} struggle with {problem} in the data table below?"
                ],
                "hard": [
                    "Analyze why {topic} outperforms {alternative} for {goal} in {context}.",
                    "Based on the scatter plot below with k={k}, why is {topic}’s approach effective?",
                    "Critique {topic} versus {model} for {problem} using the line graph below.",
                    "In the data table below with {m} samples, why prefer {topic} over {alternative}?",
                    "Given the scatter plot below, explain {topic}’s suitability for {task}.",
                    "If the line graph below shows noise {p}, how does {topic} adjust?",
                    "Analyze the data table below: how does {topic} handle {feature} variance?",
                    "Based on the scatter plot below with {n} points, critique {topic}’s performance.",
                    "In the line graph below over {t} neighbors, what indicates {topic}’s accuracy?",
                    "Why does {topic} excel in {scenario} but struggle with {problem} in the data below?",
                    "Given the scatter plot below, prove {topic}’s effectiveness for {field}.",
                    "How does {topic} mitigate {noise} in the data table below?",
                    "Critique {topic}’s distance-based approach based on the line graph below.",
                    "Based on the data table below, justify {topic}’s use in {application}."
                ]
            },
            "Distance Metrics": {
                "easy": [
                    "What is the role of distance metrics in {topic}?",
                    "Calculate the Euclidean distance between x1={x}, y1={x} and x2={x}, y2={x} in {topic}.",
                    "Based on the scatter plot below, how does {topic} use distances for {n} points?",
                    "What’s the difference between Euclidean and Manhattan distance in {topic}?",
                    "In the line graph below, what does the distance curve show?",
                    "Given the data table below, how does {topic} measure {feature} distances?",
                    "How does {topic} use Minkowski distance for {task}?",
                    "Based on the scatter plot below with {n} points, why use a distance metric?",
                    "In the data table below with {m} samples, what’s the simplest distance metric?",
                    "What’s the purpose of distance in {topic}’s {context}?",
                    "Given the line graph below, how does {topic} rank neighbors?",
                    "How does {topic} apply distances in the scatter plot below?",
                    "Based on the data table below, why does {topic} need distance metrics?",
                    "What does Manhattan distance mean for {topic}?"
                ],
                "medium": [
                    "Derive the Manhattan distance for {f} features in {topic}.",
                    "Based on the scatter plot below with {n} points, how does {topic} use Euclidean distance?",
                    "In the line graph below, how does {topic} compare distance metrics?",
                    "Given the data table below with {m} samples, calculate {topic}’s Minkowski distance.",
                    "How does {topic} choose between Euclidean and Manhattan for {scenario}?",
                    "Based on the scatter plot below, why does {topic} prefer a specific distance metric?",
                    "In the data table below, derive {topic}’s distance for {feature}.",
                    "Calculate the distance with p={p} for Minkowski in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} with distances?",
                    "How does {topic} use Manhattan distance in the scatter plot below?",
                    "Based on the data table below, why does {topic} suit {goal} with Euclidean?",
                    "In the line graph below with slope {s}, what does the distance imply?",
                    "Given the scatter plot below, how does {topic} adjust distance metrics?",
                    "Why does {topic} rely on distances in the data table below?"
                ],
                "hard": [
                    "Prove why Euclidean distance works for {topic}’s {task}.",
                    "Analyze {topic}’s distance metrics in the scatter plot below with {n} points.",
                    "Derive the full Minkowski distance formula for {topic} with p={p}.",
                    "In the data table below with {m} samples, how does {topic} handle {noise} with distances?",
                    "Critique Euclidean versus Manhattan in {topic} using the line graph below.",
                    "Prove the distance metric’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s optimal distance metric.",
                    "In the line graph below, how does {topic} adapt distances for {task}?",
                    "Analyze {topic}’s sensitivity to distance in the scatter plot below.",
                    "Prove why {topic} excels with Manhattan in the data table below.",
                    "Given the line graph below, critique {topic}’s distance fit for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} with Minkowski?",
                    "Derive {topic}’s distance-based ranking in the data table below.",
                    "Critique {topic}’s distance metric choice in the line graph below."
                ]
            },
            "Choosing k": {
                "easy": [
                    "What is the purpose of choosing k in {topic}?",
                    "How does k={k} affect {topic}’s predictions?",
                    "Based on the scatter plot below, how does k={k} classify {n} points?",
                    "What’s the role of k in {topic} for {task}?",
                    "In the line graph below, what does k’s impact show?",
                    "Given the data table below, how does {topic} pick k?",
                    "How does a small k differ from a large k in {topic}?",
                    "Based on the scatter plot below with {n} points, why use k={k}?",
                    "In the data table below with {m} samples, what’s k’s effect?",
                    "What’s the simplest way to choose k in {topic}?",
                    "Given the line graph below, how does {topic} balance k?",
                    "How does {topic} use k with {feature} in the scatter plot below?",
                    "Based on the data table below, why does {topic} need k?",
                    "What does k represent in {topic}’s {context}?"
                ],
                "medium": [
                    "Explain how k={k} changes {topic}’s accuracy in the scatter plot below.",
                    "Based on the scatter plot below with k={k}, how does {topic} classify?",
                    "In the line graph below, how does {topic} adjust with k={k}?",
                    "Given the data table below with {m} samples, how does {topic} optimize k?",
                    "How does {topic} balance bias and variance with k in {scenario}?",
                    "Based on the scatter plot below, why does {topic} need a small k for {problem}?",
                    "In the data table below, derive {topic}’s k selection for {feature}.",
                    "Calculate the error rate for k={k} in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} with k?",
                    "How does {topic} adjust k in the scatter plot below?",
                    "Based on the data table below, why does {topic} improve {goal} with k={k}?",
                    "In the line graph below with slope {s}, what does k imply?",
                    "Given the scatter plot below, how does {topic} tune k?",
                    "Why does {topic} use k in the data table below?"
                ],
                "hard": [
                    "Prove why k controls bias-variance in {topic} for {task}.",
                    "Analyze {topic}’s k={k} performance in the scatter plot below with {n} points.",
                    "Derive the optimal k for {topic} with {f} features.",
                    "In the data table below with {m} samples, how does {topic} handle {noise} with k?",
                    "Critique small versus large k in {topic} using the line graph below.",
                    "Prove k’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s k-based decision rule.",
                    "In the line graph below, how does {topic} adapt k for {task}?",
                    "Analyze {topic}’s k sensitivity in the scatter plot below.",
                    "Prove why {topic} balances errors with k in the data table below.",
                    "Given the line graph below, critique {topic}’s k fit for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} with k={k}?",
                    "Derive {topic}’s k optimization in the data table below.",
                    "Critique {topic}’s k selection trade-offs in the line graph below."
                ]
            },
            "Algorithm Mechanics": {
                "easy": [
                    "What is the main step in {topic}’s algorithm?",
                    "How does {topic} find nearest neighbors?",
                    "Based on the scatter plot below, how does {topic} classify {n} points?",
                    "What’s the role of voting in {topic} for {task}?",
                    "In the line graph below, what does the neighbor trend show?",
                    "Given the data table below, how does {topic} predict {output}?",
                    "How does {topic} use weighted voting?",
                    "Based on the scatter plot below with {n} points, why use neighbors?",
                    "In the data table below with {m} samples, what’s the algorithm’s core?",
                    "What’s the simplest explanation of {topic}’s mechanics?",
                    "Given the line graph below, how does {topic} rank neighbors?",
                    "How does {topic} apply voting in the scatter plot below?",
                    "Based on the data table below, why does {topic} rely on neighbors?",
                    "What does nearest neighbor mean in {topic}’s {context}?"
                ],
                "medium": [
                    "Explain how {topic} finds k={k} neighbors in the scatter plot below.",
                    "Based on the scatter plot below with {n} points, how does {topic} vote?",
                    "In the line graph below, how does {topic} weigh neighbors?",
                    "Given the data table below with {m} samples, how does {topic} classify?",
                    "How does {topic} use distance weighting for {f} features?",
                    "Based on the scatter plot below, why does {topic} rely on voting?",
                    "In the data table below, derive {topic}’s prediction for {feature}.",
                    "Calculate the weighted vote for k={k} in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} in voting?",
                    "How does {topic} apply nearest neighbor search in the scatter plot below?",
                    "Based on the data table below, why does {topic} suit {goal}?",
                    "In the line graph below with slope {s}, what does voting imply?",
                    "Given the scatter plot below, how does {topic} adjust mechanics?",
                    "Why does {topic} use neighbors in the data table below?"
                ],
                "hard": [
                    "Prove why neighbor voting works for {topic}’s {task}.",
                    "Analyze {topic}’s mechanics in the scatter plot below with {n} points.",
                    "Derive the full voting rule for {topic} with k={k}.",
                    "In the data table below with {m} samples, how does {topic} handle {noise}?",
                    "Critique voting versus weighting in {topic} using the line graph below.",
                    "Prove the neighbor search’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s weighted prediction.",
                    "In the line graph below, how does {topic} adapt mechanics for {task}?",
                    "Analyze {topic}’s computational cost in the scatter plot below.",
                    "Prove why {topic} excels with voting in the data table below.",
                    "Given the line graph below, critique {topic}’s mechanics for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} in voting?",
                    "Derive {topic}’s neighbor-based decision in the data table below.",
                    "Critique {topic}’s algorithm efficiency in the line graph below."
                ]
            },
            "Scaling and Normalization": {
                "easy": [
                    "Why does {topic} need feature scaling?",
                    "How does {topic} normalize {feature} data?",
                    "Based on the scatter plot below, how does scaling affect {topic}?",
                    "What’s the role of standardization in {topic} for {task}?",
                    "In the line graph below, what does normalized data show?",
                    "Given the data table below, how does {topic} scale {feature}?",
                    "How does {topic} handle unscaled data?",
                    "Based on the scatter plot below with {n} points, why use normalization?",
                    "In the data table below with {m} samples, what’s scaling’s effect?",
                    "What’s the simplest way to scale data for {topic}?",
                    "Given the line graph below, how does {topic} adjust for scaling?",
                    "How does {topic} use normalized {feature} in the scatter plot below?",
                    "Based on the data table below, why does {topic} require scaling?",
                    "What does standardization mean in {topic}’s {context}?"
                ],
                "medium": [
                    "Explain how normalization improves {topic}’s accuracy in the scatter plot below.",
                    "Based on the scatter plot below with {n} points, how does {topic} use scaled data?",
                    "In the line graph below, how does {topic} adjust with standardization?",
                    "Given the data table below with {m} samples, how does {topic} normalize {feature}?",
                    "How does {topic} handle feature ranges for {f} features?",
                    "Based on the scatter plot below, why does {topic} need scaling for {problem}?",
                    "In the data table below, derive {topic}’s scaled prediction.",
                    "Calculate the normalized value for x={x} in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} with scaling?",
                    "How does {topic} apply standardization in the scatter plot below?",
                    "Based on the data table below, why does {topic} improve {goal} with normalization?",
                    "In the line graph below with slope {s}, what does scaling imply?",
                    "Given the scatter plot below, how does {topic} adjust unscaled data?",
                    "Why does {topic} rely on scaling in the data table below?"
                ],
                "hard": [
                    "Prove why scaling is critical for {topic}’s {task}.",
                    "Analyze {topic}’s performance with unscaled data in the scatter plot below.",
                    "Derive the standardization formula for {topic} with {f} features.",
                    "In the data table below with {m} samples, how does {topic} handle {noise} with scaling?",
                    "Critique scaling versus no scaling in {topic} using the line graph below.",
                    "Prove normalization’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s scaled distance rule.",
                    "In the line graph below, how does {topic} adapt scaling for {task}?",
                    "Analyze {topic}’s sensitivity to scale in the scatter plot below.",
                    "Prove why {topic} excels with normalized data in the data table below.",
                    "Given the line graph below, critique {topic}’s scaling for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} ranges?",
                    "Derive {topic}’s normalization impact in the data table below.",
                    "Critique {topic}’s scaling trade-offs in the line graph below."
                ]
            },
            "Implementation": {
                "easy": [
                    "What Python library is used to implement {topic}?",
                    "How does {topic}’s fit method work in {context}?",
                    "Based on the scatter plot below, how does {topic} predict {n} points?",
                    "Calculate the prediction for k={k}, x={x} in {topic}.",
                    "In the line graph below, what does {topic}’s distance represent?",
                    "Given the data table below with {m} samples, how does {topic} train?",
                    "What’s the role of scikit-learn in {topic}’s {task}?",
                    "Based on the scatter plot below, why does {topic} classify easily?",
                    "In the data table below, how does {topic} use {feature} for prediction?",
                    "How does {topic} compute distances in {scenario}?",
                    "Given the line graph below, what does {topic}’s curve predict?",
                    "What’s the simplest way to implement {topic} for {n} points?",
                    "Based on the scatter plot below, how does {topic} handle {output}?",
                    "How does {topic} process {input} in the data table below?"
                ],
                "medium": [
                    "Explain how scikit-learn trains {topic} for {n} samples in the scatter plot below.",
                    "If k={k}, compute the output for x={x} in {topic}.",
                    "Based on the line graph below, how does {topic} predict neighbors?",
                    "Given the data table below with {m} samples, how does {topic} fit the data?",
                    "In the scatter plot below, how does {topic}’s predict method work for {n} points?",
                    "Derive {topic}’s prediction rule for {f} features in the data table below.",
                    "Based on the line graph below, why does {topic}’s distance stabilize?",
                    "Given the scatter plot below, how does {topic} handle {feature} in {scenario}?",
                    "In the data table below, how does {topic} compute {output} with k={k}?",
                    "How does {topic}’s kneighbors method work for k={k} in {context}?",
                    "Based on the line graph below with slope {s}, what does {topic} predict?",
                    "Given the scatter plot below, why is {topic} efficient for {task}?",
                    "In the data table below, how does {topic} adjust {parameter}?",
                    "How does {topic} implement {method} in the scatter plot below?"
                ],
                "hard": [
                    "Prove how scikit-learn optimizes {topic}’s {parameter} in {t} iterations.",
                    "Analyze {topic}’s runtime for {n} samples in the scatter plot below.",
                    "Derive the full prediction equation for {topic} with k={k} in the data table below.",
                    "In the line graph below, critique {topic}’s implementation for {scenario}.",
                    "Based on the scatter plot below, how does {topic} scale with {f} features?",
                    "Given the data table below with {m} samples, prove {topic}’s efficiency.",
                    "Critique manual versus scikit-learn implementation of {topic} in the line graph below.",
                    "In the scatter plot below with {n} points, how does {topic} handle {noise}?",
                    "Derive {topic}’s distance computation for {n} points in the data table below.",
                    "Based on the line graph below, analyze {topic}’s convergence for {task}.",
                    "Prove {topic}’s prediction accuracy in the scatter plot below with k={k}.",
                    "Given the data table below, how does {topic} optimize {feature} distances?",
                    "Critique {topic}’s implementation trade-offs in the line graph below.",
                    "Analyze {topic}’s performance for {problem} in the data table below."
                ]
            }
        }
    },
    "Decision Trees": {
        "subtopics": {
            "Introduction": {
                "easy": [
                    "What is the core idea of {topic} in {application}?",
                    "How does {topic} differ from {model} in terms of {aspect}?",
                    "Based on the scatter plot below with {n} points, would {topic} work well? Why?",
                    "Why is {topic} useful for {scenario} in {field}?",
                    "In the line graph below of splits, what does the trend suggest?",
                    "What advantage does {topic} have over {alternative} for {task}?",
                    "Given the data table below with {m} samples, is {topic} suitable? Why?",
                    "How does {topic} classify {output} in a {context} scenario?",
                    "Based on the scatter plot below, why is {topic} a good fit for {application}?",
                    "What’s the main structure of {topic}?",
                    "In the data table below, how does {topic} use {feature} patterns?",
                    "Why might {topic} be chosen for {goal} over {model}?",
                    "Based on the line graph below, what shows {topic}’s strength?",
                    "How does {topic} simplify {task} in {scenario}?"
                ],
                "medium": [
                    "Compare {topic} and {model} for {task} based on {dimension}.",
                    "Based on the scatter plot below with {n} points, why prefer {topic}?",
                    "In the line graph below with slope {s}, why is {topic} effective for {goal}?",
                    "Given the data table below with {m} samples, how does {topic} split {feature}?",
                    "Why does {topic} perform well in {scenario} compared to {alternative}?",
                    "If the scatter plot below shows {n} points, how does {topic} build its structure?",
                    "Based on the line graph below, what does the depth trend imply?",
                    "In a {application} context, why does {topic} fit the data table below?",
                    "Contrast {topic} and {alternative} for {task} using the scatter plot below.",
                    "Given the data table below, why is {topic} ideal for {output} prediction?",
                    "How does {topic} balance {dimension} and {aspect} in {scenario}?",
                    "Based on the scatter plot below, why is {topic} suited for {field}?",
                    "In the line graph below, what does the split trend suggest?",
                    "Why might {topic} struggle with {problem} in the data table below?"
                ],
                "hard": [
                    "Analyze why {topic} outperforms {alternative} for {goal} in {context}.",
                    "Based on the scatter plot below with depth {t}, why is {topic}’s approach effective?",
                    "Critique {topic} versus {model} for {problem} using the line graph below.",
                    "In the data table below with {m} samples, why prefer {topic} over {alternative}?",
                    "Given the scatter plot below, explain {topic}’s suitability for {task}.",
                    "If the line graph below shows noise {p}, how does {topic} adjust?",
                    "Analyze the data table below: how does {topic} handle {feature} complexity?",
                    "Based on the scatter plot below with {n} points, critique {topic}’s splits.",
                    "In the line graph below over {t} levels, what indicates {topic}’s performance?",
                    "Why does {topic} excel in {scenario} but struggle with {problem} in the data below?",
                    "Given the scatter plot below, prove {topic}’s effectiveness for {field}.",
                    "How does {topic} mitigate {noise} in the data table below?",
                    "Critique {topic}’s tree structure based on the line graph below.",
                    "Based on the data table below, justify {topic}’s use in {application}."
                ]
            },
            "Splitting Criteria": {
                "easy": [
                    "What is the purpose of splitting criteria in {topic}?",
                    "How does {topic} calculate Gini impurity?",
                    "Based on the scatter plot below, how does {topic} split {n} points?",
                    "What’s the role of entropy in {topic} for {task}?",
                    "In the line graph below, what does the impurity curve show?",
                    "Given the data table below, how does {topic} choose splits?",
                    "How does information gain work in {topic}?",
                    "Based on the scatter plot below with {n} points, why use Gini?",
                    "In the data table below with {m} samples, what’s the simplest criterion?",
                    "What’s the goal of splitting in {topic}’s {context}?",
                    "Given the line graph below, how does {topic} reduce impurity?",
                    "How does {topic} apply entropy to {feature} in the scatter plot below?",
                    "Based on the data table below, why does {topic} need criteria?",
                    "What does Gini impurity measure in {topic}?"
                ],
                "medium": [
                    "Derive the Gini impurity for {f} classes in {topic}.",
                    "Based on the scatter plot below with {n} points, how does {topic} use entropy?",
                    "In the line graph below, how does {topic} compare Gini and entropy?",
                    "Given the data table below with {m} samples, calculate {topic}’s information gain.",
                    "How does {topic} choose between Gini and entropy for {scenario}?",
                    "Based on the scatter plot below, why does {topic} prefer a specific criterion?",
                    "In the data table below, derive {topic}’s split for {feature}.",
                    "Calculate the entropy reduction for a split in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} with criteria?",
                    "How does {topic} use information gain in the scatter plot below?",
                    "Based on the data table below, why does {topic} suit {goal} with Gini?",
                    "In the line graph below with slope {s}, what does the criterion imply?",
                    "Given the scatter plot below, how does {topic} adjust splits?",
                    "Why does {topic} rely on impurity in the data table below?"
                ],
                "hard": [
                    "Prove why Gini impurity optimizes {topic}’s {task}.",
                    "Analyze {topic}’s splitting criteria in the scatter plot below with {n} points.",
                    "Derive the full information gain formula for {topic} with {f} features.",
                    "In the data table below with {m} samples, how does {topic} handle {noise} with entropy?",
                    "Critique Gini versus entropy in {topic} using the line graph below.",
                    "Prove the criterion’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s optimal split criterion.",
                    "In the line graph below, how does {topic} adapt criteria for {task}?",
                    "Analyze {topic}’s sensitivity to impurity in the scatter plot below.",
                    "Prove why {topic} excels with information gain in the data table below.",
                    "Given the line graph below, critique {topic}’s criterion for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} with Gini?",
                    "Derive {topic}’s split optimization in the data table below.",
                    "Critique {topic}’s splitting trade-offs in the line graph below."
                ]
            },
            "Tree Construction": {
                "easy": [
                    "What is the main step in {topic}’s construction?",
                    "How does {topic} recursively split data?",
                    "Based on the scatter plot below, how does {topic} build a tree for {n} points?",
                    "What’s the role of stopping conditions in {topic}?",
                    "In the line graph below, what does the depth trend show?",
                    "Given the data table below, how does {topic} grow its tree?",
                    "How does {topic} decide when to stop splitting?",
                    "Based on the scatter plot below with {n} points, why build a tree?",
                    "In the data table below with {m} samples, what’s the construction process?",
                    "What’s the simplest explanation of {topic}’s growth?",
                    "Given the line graph below, how does {topic} form branches?",
                    "How does {topic} split {feature} in the scatter plot below?",
                    "Based on the data table below, why does {topic} need construction?",
                    "What does recursive splitting mean in {topic}’s {context}?"
                ],
                "medium": [
                    "Explain how {topic} constructs a tree for {n} points in the scatter plot below.",
                    "Based on the scatter plot below with {n} points, how does {topic} set depth?",
                    "In the line graph below, how does {topic} apply stopping conditions?",
                    "Given the data table below with {m} samples, how does {topic} split recursively?",
                    "How does {topic} balance depth and splits for {f} features?",
                    "Based on the scatter plot below, why does {topic} stop at a certain depth?",
                    "In the data table below, derive {topic}’s tree structure for {feature}.",
                    "Calculate the number of splits for depth {t} in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} in construction?",
                    "How does {topic} grow branches in the scatter plot below?",
                    "Based on the data table below, why does {topic} suit {goal} with recursion?",
                    "In the line graph below with slope {s}, what does construction imply?",
                    "Given the scatter plot below, how does {topic} adjust tree growth?",
                    "Why does {topic} use recursion in the data table below?"
                ],
                "hard": [
                    "Prove why recursive splitting works for {topic}’s {task}.",
                    "Analyze {topic}’s construction in the scatter plot below with {n} points.",
                    "Derive the full tree-building process for {topic} with {f} features.",
                    "In the data table below with {m} samples, how does {topic} handle {noise} in growth?",
                    "Critique depth versus splits in {topic} using the line graph below.",
                    "Prove the stopping condition’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s optimal tree structure.",
                    "In the line graph below, how does {topic} adapt construction for {task}?",
                    "Analyze {topic}’s computational cost in the scatter plot below.",
                    "Prove why {topic} excels with recursion in the data table below.",
                    "Given the line graph below, critique {topic}’s growth for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} in construction?",
                    "Derive {topic}’s stopping criteria in the data table below.",
                    "Critique {topic}’s construction efficiency in the line graph below."
                ]
            },
            "Pruning": {
                "easy": [
                    "What is the purpose of pruning in {topic}?",
                    "How does {topic} perform pre-pruning?",
                    "Based on the scatter plot below, how does pruning affect {topic}?",
                    "What’s the role of post-pruning in {topic} for {task}?",
                    "In the line graph below, what does the pruned trend show?",
                    "Given the data table below, how does {topic} reduce overfitting?",
                    "How does pre-pruning differ from post-pruning in {topic}?",
                    "Based on the scatter plot below with {n} points, why use pruning?",
                    "In the data table below with {m} samples, what’s pruning’s effect?",
                    "What’s the simplest way to prune in {topic}?",
                    "Given the line graph below, how does {topic} simplify its tree?",
                    "How does {topic} prune {feature} splits in the scatter plot below?",
                    "Based on the data table below, why does {topic} need pruning?",
                    "What does pruning prevent in {topic}’s {context}?"
                ],
                "medium": [
                    "Explain how pre-pruning improves {topic}’s accuracy in the scatter plot below.",
                    "Based on the scatter plot below with {n} points, how does {topic} use post-pruning?",
                    "In the line graph below, how does {topic} adjust with pruning?",
                    "Given the data table below with {m} samples, how does {topic} prune {feature}?",
                    "How does {topic} balance pruning and accuracy for {scenario}?",
                    "Based on the scatter plot below, why does {topic} need pruning for {problem}?",
                    "In the data table below, derive {topic}’s pruned structure for {feature}.",
                    "Calculate the pruned depth for a tree in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} with pruning?",
                    "How does {topic} apply post-pruning in the scatter plot below?",
                    "Based on the data table below, why does {topic} improve {goal} with pruning?",
                    "In the line graph below with slope {s}, what does pruning imply?",
                    "Given the scatter plot below, how does {topic} adjust pruning?",
                    "Why does {topic} rely on pruning in the data table below?"
                ],
                "hard": [
                    "Prove why pruning reduces overfitting in {topic}’s {task}.",
                    "Analyze {topic}’s pruning in the scatter plot below with {n} points.",
                    "Derive the full pruning process for {topic} with {f} features.",
                    "In the data table below with {m} samples, how does {topic} handle {noise} with pruning?",
                    "Critique pre-pruning versus post-pruning in {topic} using the line graph below.",
                    "Prove pruning’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s optimal pruned tree.",
                    "In the line graph below, how does {topic} adapt pruning for {task}?",
                    "Analyze {topic}’s pruning impact in the scatter plot below.",
                    "Prove why {topic} excels with pruning in the data table below.",
                    "Given the line graph below, critique {topic}’s pruning for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} with pruning?",
                    "Derive {topic}’s pruning criteria in the data table below.",
                    "Critique {topic}’s pruning trade-offs in the line graph below."
                ]
            },
            "Prediction": {
                "easy": [
                    "How does {topic} predict {output}?",
                    "What’s the role of leaf nodes in {topic}?",
                    "Based on the scatter plot below, how does {topic} classify {n} points?",
                    "What’s the simplest prediction method in {topic}?",
                    "In the line graph below, what does the prediction trend show?",
                    "Given the data table below, how does {topic} assign {output}?",
                    "How does {topic} use leaf nodes for {task}?",
                    "Based on the scatter plot below with {n} points, why predict with leaves?",
                    "In the data table below with {m} samples, what’s the prediction process?",
                    "What’s the goal of prediction in {topic}’s {context}?",
                    "Given the line graph below, how does {topic} reach a decision?",
                    "How does {topic} predict {feature} outcomes in the scatter plot below?",
                    "Based on the data table below, why does {topic} use leaves?",
                    "What does a leaf node represent in {topic}?"
                ],
                "medium": [
                    "Explain how {topic} predicts {output} for {n} points in the scatter plot below.",
                    "Based on the scatter plot below with {n} points, how does {topic} use leaf nodes?",
                    "In the line graph below, how does {topic} assign predictions?",
                    "Given the data table below with {m} samples, how does {topic} classify {feature}?",
                    "How does {topic} handle regression versus classification for {f} features?",
                    "Based on the scatter plot below, why does {topic} rely on leaf predictions?",
                    "In the data table below, derive {topic}’s prediction for {feature}.",
                    "Calculate the prediction value for a leaf in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} in predictions?",
                    "How does {topic} use majority voting in the scatter plot below?",
                    "Based on the data table below, why does {topic} suit {goal} with leaves?",
                    "In the line graph below with slope {s}, what does prediction imply?",
                    "Given the scatter plot below, how does {topic} adjust predictions?",
                    "Why does {topic} predict with leaves in the data table below?"
                ],
                "hard": [
                    "Prove why leaf nodes optimize {topic}’s {task}.",
                    "Analyze {topic}’s prediction in the scatter plot below with {n} points.",
                    "Derive the full prediction process for {topic} with {f} features.",
                    "In the data table below with {m} samples, how does {topic} handle {noise} in predictions?",
                    "Critique classification versus regression in {topic} using the line graph below.",
                    "Prove the prediction’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s leaf-based decision.",
                    "In the line graph below, how does {topic} adapt predictions for {task}?",
                    "Analyze {topic}’s prediction accuracy in the scatter plot below.",
                    "Prove why {topic} excels with leaf nodes in the data table below.",
                    "Given the line graph below, critique {topic}’s prediction for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} in predictions?",
                    "Derive {topic}’s prediction rule in the data table below.",
                    "Critique {topic}’s prediction efficiency in the line graph below."
                ]
            },
            "Implementation": {
                "easy": [
                    "What Python library is used to implement {topic}?",
                    "How does {topic}’s fit method work in {context}?",
                    "Based on the scatter plot below, how does {topic} predict {n} points?",
                    "Calculate the prediction for a tree in {topic}.",
                    "In the line graph below, what does {topic}’s split represent?",
                    "Given the data table below with {m} samples, how does {topic} train?",
                    "What’s the role of scikit-learn in {topic}’s {task}?",
                    "Based on the scatter plot below, why does {topic} classify easily?",
                    "In the data table below, how does {topic} use {feature} for prediction?",
                    "How does {topic} build a tree in {scenario}?",
                    "Given the line graph below, what does {topic}’s curve predict?",
                    "What’s the simplest way to implement {topic} for {n} points?",
                    "Based on the scatter plot below, how does {topic} handle {output}?",
                    "How does {topic} process {input} in the data table below?"
                ],
                "medium": [
                    "Explain how scikit-learn trains {topic} for {n} samples in the scatter plot below.",
                    "If depth={t}, compute the splits for {topic}.",
                    "Based on the line graph below, how does {topic} predict outcomes?",
                    "Given the data table below with {m} samples, how does {topic} fit the data?",
                    "In the scatter plot below, how does {topic}’s predict method work for {n} points?",
                    "Derive {topic}’s tree structure for {f} features in the data table below.",
                    "Based on the line graph below, why does {topic}’s depth stabilize?",
                    "Given the scatter plot below, how does {topic} handle {feature} in {scenario}?",
                    "In the data table below, how does {topic} compute {output} with splits?",
                    "How does {topic}’s max_depth parameter work in {context}?",
                    "Based on the line graph below with slope {s}, what does {topic} predict?",
                    "Given the scatter plot below, why is {topic} efficient for {task}?",
                    "In the data table below, how does {topic} adjust {parameter}?",
                    "How does {topic} implement {method} in the scatter plot below?"
                ],
                "hard": [
                    "Prove how scikit-learn optimizes {topic}’s {parameter} in {t} levels.",
                    "Analyze {topic}’s runtime for {n} samples in the scatter plot below.",
                    "Derive the full tree equation for {topic} in the data table below.",
                    "In the line graph below, critique {topic}’s implementation for {scenario}.",
                    "Based on the scatter plot below, how does {topic} scale with {f} features?",
                    "Given the data table below with {m} samples, prove {topic}’s efficiency.",
                    "Critique manual versus scikit-learn implementation of {topic} in the line graph below.",
                    "In the scatter plot below with {n} points, how does {topic} handle {noise}?",
                    "Derive {topic}’s split computation for {n} points in the data table below.",
                    "Based on the line graph below, analyze {topic}’s convergence for {task}.",
                    "Prove {topic}’s prediction accuracy in the scatter plot below.",
                    "Given the data table below, how does {topic} optimize {feature} splits?",
                    "Critique {topic}’s implementation trade-offs in the line graph below.",
                    "Analyze {topic}’s performance for {problem} in the data table below."
                ]
            }
        }
    },
    "Random Forest": {
        "subtopics": {
            "Introduction": {
                "easy": [
                    "What is the main concept of {topic} in {application}?",
                    "How does {topic} differ from {model} in terms of {aspect}?",
                    "Based on the scatter plot below with {n} points, would {topic} work well? Why?",
                    "Why is {topic} effective for {scenario} in {field}?",
                    "In the line graph below of ensemble predictions, what does the trend suggest?",
                    "What advantage does {topic} have over {alternative} for {task}?",
                    "Given the data table below with {m} samples, is {topic} suitable? Why?",
                    "How does {topic} classify {output} in a {context} scenario?",
                    "Based on the scatter plot below, why is {topic} a good fit for {application}?",
                    "What’s the role of multiple trees in {topic}?",
                    "In the data table below, how does {topic} use {feature} patterns?",
                    "Why might {topic} be chosen for {goal} over {model}?",
                    "Based on the line graph below, what shows {topic}’s strength?",
                    "How does {topic} improve {task} in {scenario}?"
                ],
                "medium": [
                    "Compare {topic} and {model} for {task} based on {dimension}.",
                    "Based on the scatter plot below with {n} points, why prefer {topic}?",
                    "In the line graph below with slope {s}, why is {topic} effective for {goal}?",
                    "Given the data table below with {m} samples, how does {topic} aggregate {feature}?",
                    "Why does {topic} perform well in {scenario} compared to {alternative}?",
                    "If the scatter plot below shows {n} points, how does {topic} ensemble them?",
                    "Based on the line graph below, what does the voting trend imply?",
                    "In a {application} context, why does {topic} fit the data table below?",
                    "Contrast {topic} and {alternative} for {task} using the scatter plot below.",
                    "Given the data table below, why is {topic} ideal for {output} prediction?",
                    "How does {topic} balance {dimension} and {aspect} in {scenario}?",
                    "Based on the scatter plot below, why is {topic} suited for {field}?",
                    "In the line graph below, what does the ensemble trend suggest?",
                    "Why might {topic} struggle with {problem} in the data table below?"
                ],
                "hard": [
                    "Analyze why {topic} outperforms {alternative} for {goal} in {context}.",
                    "Based on the scatter plot below with {t} trees, why is {topic}’s approach effective?",
                    "Critique {topic} versus {model} for {problem} using the line graph below.",
                    "In the data table below with {m} samples, why prefer {topic} over {alternative}?",
                    "Given the scatter plot below, explain {topic}’s suitability for {task}.",
                    "If the line graph below shows noise {p}, how does {topic} adjust?",
                    "Analyze the data table below: how does {topic} handle {feature} variance?",
                    "Based on the scatter plot below with {n} points, critique {topic}’s ensemble.",
                    "In the line graph below over {t} trees, what indicates {topic}’s performance?",
                    "Why does {topic} excel in {scenario} but struggle with {problem} in the data below?",
                    "Given the scatter plot below, prove {topic}’s effectiveness for {field}.",
                    "How does {topic} mitigate {noise} in the data table below?",
                    "Critique {topic}’s ensemble approach based on the line graph below.",
                    "Based on the data table below, justify {topic}’s use in {application}."
                ]
            },
            "Bootstrap Aggregation": {
                "easy": [
                    "What is bootstrap aggregation in {topic}?",
                    "How does {topic} sample data with replacement?",
                    "Based on the scatter plot below, how does {topic} use bagging for {n} points?",
                    "What’s the role of bagging in {topic} for {task}?",
                    "In the line graph below, what does the bagged trend show?",
                    "Given the data table below, how does {topic} apply bootstrap?",
                    "How does {topic} reduce variance with bagging?",
                    "Based on the scatter plot below with {n} points, why use bagging?",
                    "In the data table below with {m} samples, what’s bagging’s effect?",
                    "What’s the simplest explanation of bagging in {topic}?",
                    "Given the line graph below, how does {topic} aggregate samples?",
                    "How does {topic} use bagging on {feature} in the scatter plot below?",
                    "Based on the data table below, why does {topic} need bootstrap?",
                    "What does sampling with replacement mean in {topic}’s {context}?"
                ],
                "medium": [
                    "Explain how bagging improves {topic}’s accuracy in the scatter plot below.",
                    "Based on the scatter plot below with {n} points, how does {topic} sample data?",
                    "In the line graph below, how does {topic} reduce variance with bagging?",
                    "Given the data table below with {m} samples, how does {topic} apply bootstrap?",
                    "How does {topic} balance bias and variance with bagging for {f} features?",
                    "Based on the scatter plot below, why does {topic} rely on bagging for {problem}?",
                    "In the data table below, derive {topic}’s bagged prediction for {feature}.",
                    "Calculate the bagged sample size for {t} trees in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} with bagging?",
                    "How does {topic} use bootstrap in the scatter plot below?",
                    "Based on the data table below, why does {topic} improve {goal} with bagging?",
                    "In the line graph below with slope {s}, what does bagging imply?",
                    "Given the scatter plot below, how does {topic} adjust bagging?",
                    "Why does {topic} rely on bagging in the data table below?"
                ],
                "hard": [
                    "Prove why bagging reduces variance in {topic}’s {task}.",
                    "Analyze {topic}’s bootstrap aggregation in the scatter plot below with {n} points.",
                    "Derive the full bagging process for {topic} with {t} trees.",
                    "In the data table below with {m} samples, how does {topic} handle {noise} with bagging?",
                    "Critique bagging’s impact in {topic} using the line graph below.",
                    "Prove bagging’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s bagged decision rule.",
                    "In the line graph below, how does {topic} adapt bagging for {task}?",
                    "Analyze {topic}’s variance reduction in the scatter plot below.",
                    "Prove why {topic} excels with bagging in the data table below.",
                    "Given the line graph below, critique {topic}’s bagging for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} with bagging?",
                    "Derive {topic}’s bootstrap optimization in the data table below.",
                    "Critique {topic}’s bagging efficiency in the line graph below."
                ]
            },
            "Feature Randomness": {
                "easy": [
                    "What is feature randomness in {topic}?",
                    "How does {topic} select random features?",
                    "Based on the scatter plot below, how does {topic} use randomness for {n} points?",
                    "What’s the role of feature selection in {topic} for {task}?",
                    "In the line graph below, what does the random trend show?",
                    "Given the data table below, how does {topic} apply feature randomness?",
                    "How does {topic} increase diversity with randomness?",
                    "Based on the scatter plot below with {n} points, why use random features?",
                    "In the data table below with {m} samples, what’s randomness’s effect?",
                    "What’s the simplest explanation of feature randomness in {topic}?",
                    "Given the line graph below, how does {topic} pick features?",
                    "How does {topic} use randomness on {feature} in the scatter plot below?",
                    "Based on the data table below, why does {topic} need feature randomness?",
                    "What does random selection mean in {topic}’s {context}?"
                ],
                "medium": [
                    "Explain how feature randomness improves {topic}’s accuracy in the scatter plot below.",
                    "Based on the scatter plot below with {n} points, how does {topic} select features?",
                    "In the line graph below, how does {topic} diversify with randomness?",
                    "Given the data table below with {m} samples, how does {topic} apply random splits?",
                    "How does {topic} balance feature subsets for {f} features?",
                    "Based on the scatter plot below, why does {topic} rely on randomness for {problem}?",
                    "In the data table below, derive {topic}’s random feature prediction.",
                    "Calculate the feature subset size for {t} trees in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} with randomness?",
                    "How does {topic} use random features in the scatter plot below?",
                    "Based on the data table below, why does {topic} improve {goal} with randomness?",
                    "In the line graph below with slope {s}, what does randomness imply?",
                    "Given the scatter plot below, how does {topic} adjust feature selection?",
                    "Why does {topic} rely on randomness in the data table below?"
                ],
                "hard": [
                    "Prove why feature randomness enhances {topic}’s {task}.",
                    "Analyze {topic}’s feature randomness in the scatter plot below with {n} points.",
                    "Derive the full randomness process for {topic} with {f} features.",
                    "In the data table below with {m} samples, how does {topic} handle {noise} with randomness?",
                    "Critique randomness’s impact in {topic} using the line graph below.",
                    "Prove randomness’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s random feature decision.",
                    "In the line graph below, how does {topic} adapt randomness for {task}?",
                    "Analyze {topic}’s diversity gain in the scatter plot below.",
                    "Prove why {topic} excels with random features in the data table below.",
                    "Given the line graph below, critique {topic}’s randomness for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} with randomness?",
                    "Derive {topic}’s feature subset optimization in the data table below.",
                    "Critique {topic}’s randomness efficiency in the line graph below."
                ]
            },
            "Voting and Averaging": {
                "easy": [
                    "What is the purpose of voting in {topic}?",
                    "How does {topic} average predictions?",
                    "Based on the scatter plot below, how does {topic} classify {n} points?",
                    "What’s the role of majority voting in {topic} for {task}?",
                    "In the line graph below, what does the voting trend show?",
                    "Given the data table below, how does {topic} combine predictions?",
                    "How does {topic} use averaging for regression?",
                    "Based on the scatter plot below with {n} points, why use voting?",
                    "In the data table below with {m} samples, what’s voting’s effect?",
                    "What’s the simplest way to combine predictions in {topic}?",
                    "Given the line graph below, how does {topic} aggregate trees?",
                    "How does {topic} vote on {feature} in the scatter plot below?",
                    "Based on the data table below, why does {topic} need voting?",
                    "What does averaging mean in {topic}’s {context}?"
                ],
                "medium": [
                    "Explain how majority voting improves {topic}’s accuracy in the scatter plot below.",
                    "Based on the scatter plot below with {n} points, how does {topic} average predictions?",
                    "In the line graph below, how does {topic} combine tree outputs?",
                    "Given the data table below with {m} samples, how does {topic} vote on {feature}?",
                    "How does {topic} balance voting and averaging for {f} features?",
                    "Based on the scatter plot below, why does {topic} rely on voting for {problem}?",
                    "In the data table below, derive {topic}’s voted prediction for {feature}.",
                    "Calculate the averaged output for {t} trees in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} with voting?",
                    "How does {topic} use majority voting in the scatter plot below?",
                    "Based on the data table below, why does {topic} improve {goal} with averaging?",
                    "In the line graph below with slope {s}, what does voting imply?",
                    "Given the scatter plot below, how does {topic} adjust averaging?",
                    "Why does {topic} rely on voting in the data table below?"
                ],
                "hard": [
                    "Prove why voting optimizes {topic}’s {task}.",
                    "Analyze {topic}’s voting in the scatter plot below with {n} points.",
                    "Derive the full averaging process for {topic} with {t} trees.",
                    "In the data table below with {m} samples, how does {topic} handle {noise} with voting?",
                    "Critique voting versus averaging in {topic} using the line graph below.",
                    "Prove voting’s role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s combined prediction.",
                    "In the line graph below, how does {topic} adapt voting for {task}?",
                    "Analyze {topic}’s prediction stability in the scatter plot below.",
                    "Prove why {topic} excels with averaging in the data table below.",
                    "Given the line graph below, critique {topic}’s voting for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} with voting?",
                    "Derive {topic}’s voting optimization in the data table below.",
                    "Critique {topic}’s voting efficiency in the line graph below."
                ]
            },
            "Hyperparameters": {
                "easy": [
                    "What is the role of hyperparameters in {topic}?",
                    "How does {topic} set the number of trees?",
                    "Based on the scatter plot below, how does {t} trees affect {topic}?",
                    "What’s the purpose of max_depth in {topic} for {task}?",
                    "In the line graph below, what does the hyperparameter trend show?",
                    "Given the data table below, how does {topic} tune hyperparameters?",
                    "How does {topic} use feature subset size?",
                    "Based on the scatter plot below with {n} points, why tune {t} trees?",
                    "In the data table below with {m} samples, what’s the effect of depth?",
                    "What’s the simplest way to set hyperparameters in {topic}?",
                    "Given the line graph below, how does {topic} adjust {t} trees?",
                    "How does {topic} tune {feature} splits in the scatter plot below?",
                    "Based on the data table below, why does {topic} need hyperparameters?",
                    "What does n_estimators mean in {topic}’s {context}?"
                ],
                "medium": [
                    "Explain how {t} trees improve {topic}’s accuracy in the scatter plot below.",
                    "Based on the scatter plot below with {n} points, how does {topic} set max_depth?",
                    "In the line graph below, how does {topic} adjust with feature subset size?",
                    "Given the data table below with {m} samples, how does {topic} tune {t} trees?",
                    "How does {topic} balance hyperparameters for {f} features?",
                    "Based on the scatter plot below, why does {topic} need a specific {t} for {problem}?",
                    "In the data table below, derive {topic}’s hyperparameter effect on {feature}.",
                    "Calculate the impact of depth={t} in {topic}.",
                    "Given the line graph below, how does {topic} handle {noise} with hyperparameters?",
                    "How does {topic} adjust n_estimators in the scatter plot below?",
                    "Based on the data table below, why does {topic} improve {goal} with tuning?",
                    "In the line graph below with slope {s}, what does tuning imply?",
                    "Given the scatter plot below, how does {topic} optimize hyperparameters?",
                    "Why does {topic} rely on hyperparameters in the data table below?"
                ],
                "hard": [
                    "Prove why {t} trees optimize {topic}’s {task}.",
                    "Analyze {topic}’s hyperparameters in the scatter plot below with {n} points.",
                    "Derive the full hyperparameter tuning process for {topic} with {f} features.",
                    "In the data table below with {m} samples, how does {topic} handle {noise} with tuning?",
                    "Critique depth versus n_estimators in {topic} using the line graph below.",
                    "Prove hyperparameters’ role in {topic} based on the scatter plot below.",
                    "Based on the data table below, derive {topic}’s optimal hyperparameter settings.",
                    "In the line graph below, how does {topic} adapt tuning for {task}?",
                    "Analyze {topic}’s sensitivity to hyperparameters in the scatter plot below.",
                    "Prove why {topic} excels with tuned trees in the data table below.",
                    "Given the line graph below, critique {topic}’s tuning for {problem}.",
                    "Based on the scatter plot below, how does {topic} handle {feature} with n_estimators?",
                    "Derive {topic}’s hyperparameter optimization in the data table below.",
                    "Critique {topic}’s hyperparameter trade-offs in the line graph below."
                ]
            },
            "Implementation": {
                "easy": [
                    "What Python library is used to implement {topic}?",
                    "How does {topic}’s fit method work in {context}?",
                    "Based on the scatter plot below, how does {topic} predict {n} points?",
                    "Calculate the prediction for {t} trees in {topic}.",
                    "In the line graph below, what does {topic}’s ensemble represent?",
                    "Given the data table below with {m} samples, how does {topic} train?",
                    "What’s the role of scikit-learn in {topic}’s {task}?",
                    "Based on the scatter plot below, why does {topic} classify easily?",
                    "In the data table below, how does {topic} use {feature} for prediction?",
                    "How does {topic} build an ensemble in {scenario}?",
                    "Given the line graph below, what does {topic}’s curve predict?",
                    "What’s the simplest way to implement {topic} for {n} points?",
                    "Based on the scatter plot below, how does {topic} handle {output}?",
                    "How does {topic} process {input} in the data table below?"
                ],
                "medium": [
                    "Explain how scikit-learn trains {topic} for {n} samples in the scatter plot below.",
                    "If n_estimators={t}, compute the ensemble output for {topic}.",
                    "Based on the line graph below, how does {topic} predict outcomes?",
                    "Given the data table below with {m} samples, how does {topic} fit the data?",
                    "In the scatter plot below, how does {topic}’s predict method work for {n} points?",
                    "Derive {topic}’s ensemble structure for {f} features in the data table below.",
                    "Based on the line graph below, why does {topic}’s ensemble stabilize?",
                    "Given the scatter plot below, how does {topic} handle {feature} in {scenario}?",
                    "In the data table below, how does {topic} compute {output} with {t} trees?",
                    "How does {topic}’s max_features parameter work in {context}?",
                    "Based on the line graph below with slope {s}, what does {topic} predict?",
                    "Given the scatter plot below, why is {topic} efficient for {task}?",
                    "In the data table below, how does {topic} adjust {parameter}?",
                    "How does {topic} implement {method} in the scatter plot below?"
                ],
                "hard": [
                    "Prove how scikit-learn optimizes {topic}’s {parameter} in {t} trees.",
                    "Analyze {topic}’s runtime for {n} samples in the scatter plot below.",
                    "Derive the full ensemble equation for {topic} in the data table below.",
                    "In the line graph below, critique {topic}’s implementation for {scenario}.",
                    "Based on the scatter plot below, how does {topic} scale with {f} features?",
                    "Given the data table below with {m} samples, prove {topic}’s efficiency.",
                    "Critique manual versus scikit-learn implementation of {topic} in the line graph below.",
                    "In the scatter plot below with {n} points, how does {topic} handle {noise}?",
                    "Derive {topic}’s voting computation for {n} points in the data table below.",
                    "Based on the line graph below, analyze {topic}’s convergence for {task}.",
                    "Prove {topic}’s prediction accuracy in the scatter plot below with {t} trees.",
                    "Given the data table below, how does {topic} optimize {feature} ensemble?",
                    "Critique {topic}’s implementation trade-offs in the line graph below.",
                    "Analyze {topic}’s performance for {problem} in the data table below."
                ]
            }
        }
    }
}





# Options for theoretical placeholders
# Unified options for Logistic Regression and Naïve Bayes
# Unified options for Logistic Regression, Naïve Bayes, and SVM
# Unified options for Logistic Regression, SVM, k-NN, and Naïve Bayes
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

# Create folder for scatter plots
os.makedirs("scatter_plots", exist_ok=True)

# Initialize dataset
dataset = []
displayed_questions = set()
target_samples = 2000

# List all topic-subtopic-difficulty combinations to ensure variety
combinations = []
for topic in topics:
    for subtopic in topics[topic]["subtopics"]:
        for difficulty in topics[topic]["subtopics"][subtopic]:
            combinations.append((topic, subtopic, difficulty))

# Shuffle combinations to ensure variety
random.shuffle(combinations)

i = 0
combo_idx = 0
while i < target_samples:
    # Cycle through topic-subtopic-difficulty combinations
    topic, subtopic, difficulty = combinations[combo_idx % len(combinations)]
    combo_idx += 1
    
    # Pick a random template
    templates = topics[topic]["subtopics"][subtopic][difficulty]
    if not templates:
        continue
    template = random.choice(templates)
    if "scatter plot below" not in template:
        continue
    
    # Define parameters
    params = {
        "n": random.randint(5, 20),  # Number of points per class
        "w": random.uniform(-2, 2),
        "topic": topic,
        "subtopic": subtopic
    }
    params.update({key: random.choice(values) for key, values in options.items() if f"{{{key}}}" in template})
    
    # Format the question
    try:
        question = template.format(**params)
    except KeyError as e:
        print(f"Error: Missing placeholder {e} in template '{template}'")
        continue
    
    # Generate scatter plot
    n_points = params["n"]
    separable = random.choice([True, False])
    x1 = np.random.normal(0, 1, n_points) if separable else np.random.normal(0, 2, n_points)
    y1 = np.random.normal(0, 1, n_points) if separable else np.random.normal(0, 2, n_points)
    x2 = np.random.normal(3, 1, n_points) if separable else np.random.normal(0, 2, n_points)
    y2 = np.random.normal(3, 1, n_points) if separable else np.random.normal(0, 2, n_points)
    
    # Store coordinates and labels
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.array([0] * n_points + [1] * n_points)
    
    # Create plot
    fig, ax = plt.subplots()
    ax.scatter(x1, y1, color='blue', label='Class 0')
    ax.scatter(x2, y2, color='red', label='Class 1')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'Scatter Plot with {n_points} Points per Class')
    ax.legend()
    plt.tight_layout()
    
    # Save plot
    image_path = f"scatter_plots/plot_{i}.png"
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)
    
    # Add to dataset
    dataset.append({
        "image_path": image_path,
        "question": question,
        "plot_data": {"X": X.tolist(), "y": y.tolist()},
        "num_classes": 2,
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": difficulty,
        "separable": separable
    })
    
    i += 1
    if (i + 1) % 100 == 0:
        print(f"Generated {i + 1} scatter plots")

# Save dataset
with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("Dataset generation complete. 2000 scatter plots saved in 'scatter_plots' folder.")