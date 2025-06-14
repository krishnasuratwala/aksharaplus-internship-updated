# main.py

import streamlit as st
import pandas as pd
from utils.question_generator_utils import get_question
from utils.topics import topics 
# Example structure – replace with real structure from your template files
from utils.templates.Logistic_Regression_templates import logistic_regression_templates
from utils.templates.k_Nearest_Neighbors_templates import knn_templates
from utils.templates.Random_Forest_templates import random_forest_templates
from utils.templates.Decision_Trees_templates import decision_tree_templates
from utils.templates.Naïve_Bayes_templates import naive_bayes_templates
from utils.templates.Support_Vector_Machines_templates import svm_templates

# Organize your topics, subtopics, and difficulty levels


st.title("Supervised Learning Classification Problem Question Generator")
topic_list = list(topics.keys())
selected_topic = st.selectbox("Select Topic", topic_list)

subtopic_list = list(topics[selected_topic]["subtopics"].keys())
selected_subtopic = st.selectbox("Select Subtopic", subtopic_list)

difficulties = ["easy", "medium", "hard"]
selected_difficulty = st.selectbox("Select Difficulty Level", difficulties)

# Session state
if "displayed_questions" not in st.session_state:
    st.session_state.displayed_questions = set()
if "current_question" not in st.session_state:
    st.session_state.current_question = None

# Buttons
if st.button("Generate Questions"):
    st.session_state.displayed_questions.clear()
    result = get_question(selected_topic, selected_subtopic, selected_difficulty, st.session_state.displayed_questions)
    st.session_state.current_question = result
    st.session_state.displayed_questions.add(result["question"])

if st.button("Next"):
    if st.session_state.current_question:
        result = get_question(selected_topic, selected_subtopic, selected_difficulty, st.session_state.displayed_questions)
        st.session_state.current_question = result
        st.session_state.displayed_questions.add(result["question"])
    else:
        st.write("Please generate a question first.")

# Display
if st.session_state.current_question:
    st.write("### Question:")
    st.write(st.session_state.current_question["question"])
    if st.session_state.current_question["plot"]:
        st.write("#### Plot:")
        st.pyplot(st.session_state.current_question["plot"])
    if st.session_state.current_question["table"] is not None:
        st.write("#### Data Table:")
        st.table(st.session_state.current_question["table"])
    st.write("#### Answer:")
    st.write(st.session_state.current_question["answer"])
# Drag and Drop Image Upload
st.write("### Upload an Image (Drag and Drop)")
uploaded_file = st.file_uploader("Drag and drop an image here or click to upload", 
                               type=['png', 'jpg', 'jpeg'],
                               accept_multiple_files=False)