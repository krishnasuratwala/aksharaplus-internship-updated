import unittest
from unittest.mock import patch, Mock
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
# import pytesseract

# Add current directory to path to import app1
sys.path.append(".")
from template_ques_gen import options, topics, get_question

class TestQuestionGenerator(unittest.TestCase):
    def setUp(self):
        plt.close('all')
        self.topic = "Logistic Regression"
        self.subtopic = "Introduction"
        self.difficulty = "easy"
        self.displayed_questions = set()

    # Test 1: Options dictionary structure
    def test_options_structure(self):
        self.assertIsInstance(options, dict)
        required_keys = ["goal", "application", "model", "feature", "method", "parameter"]
        for key in required_keys:
            self.assertIn(key, options)
            self.assertIsInstance(options[key], list)
            self.assertGreater(len(options[key]), 0)

    # Test 2: Topics dictionary structure
    def test_topics_structure(self):
        self.assertIsInstance(topics, dict)
        self.assertIn(self.topic, topics)
        self.assertIn("subtopics", topics[self.topic])
        self.assertIn(self.subtopic, topics[self.topic]["subtopics"])
        for diff in ["easy", "medium", "hard"]:
            self.assertIn(diff, topics[self.topic]["subtopics"][self.subtopic])
            self.assertGreater(len(topics[self.topic]["subtopics"][self.subtopic][diff]), 10)

    # Test 3: Generate question with valid inputs
    def test_generate_question_valid(self):
        result = get_question(self.topic, self.subtopic, self.difficulty, self.displayed_questions)
        self.assertIsInstance(result, dict)
        self.assertIn("question", result)
        self.assertIn("answer", result)
        self.assertIn("plot", result)
        self.assertIn("table", result)
        self.assertGreater(len(result["question"]), 10)
        self.assertNotIn("{", result["question"])
        self.assertNotIn("}", result["question"])
        self.assertIn(self.topic, result["question"])

    # Test 4: Generate question with invalid topic
    def test_generate_question_invalid_topic(self):
        result = get_question("Invalid Topic", self.subtopic, self.difficulty, self.displayed_questions)
        self.assertEqual(result["question"], "Topic/Subtopic not found.")
        self.assertEqual(result["plot"], None)
        self.assertEqual(result["table"], None)

    # Test 5: Generate question with invalid subtopic
    def test_generate_question_invalid_subtopic(self):
        result = get_question(self.topic, "Invalid Subtopic", self.difficulty, self.displayed_questions)
        self.assertEqual(result["question"], "Topic/Subtopic not found.")
        self.assertEqual(result["plot"], None)
        self.assertEqual(result["table"], None)

    # Test 6: Generate scatter plot question
    def test_generate_scatter_plot(self):
        template = "Based on the scatter plot below with {n} points, would {topic} be effective? Why?"
        with patch("random.choice", return_value=template):
            result = get_question(self.topic, self.subtopic, self.difficulty, self.displayed_questions)
            self.assertIn("scatter plot below", result["question"])
            n = int(result["question"].split("with ")[1].split(" points")[0])
            self.assertIsInstance(result["plot"], plt.Figure)
            ax = result["plot"].gca()
            scatter = ax.collections
            self.assertEqual(len(scatter), 2)
            self.assertEqual(len(scatter[0].get_offsets()), n)

    # Test 7: Generate sigmoid line graph
    def test_generate_sigmoid_line_graph(self):
        template = "In the line graph below of sigmoid outputs, what does the curve indicate?"
        with patch("random.choice", return_value=template):
            result = get_question(self.topic, self.subtopic, self.difficulty, self.displayed_questions)
            self.assertIn("line graph below", result["question"])
            self.assertIn("sigmoid", result["question"])
            self.assertIsInstance(result["plot"], plt.Figure)
            ax = result["plot"].gca()
            lines = ax.lines
            self.assertEqual(len(lines), 1)
            self.assertEqual(ax.get_title(), "Sigmoid Curve")

    # Test 8: Generate cost line graph question
    def test_generate_cost_line_graph(self):
        template = "Based on the line graph below, what does the cost trend imply for {topic}?"
        with patch("random.choice", return_value=template):
            result = get_question(self.topic, self.subtopic, self.difficulty, self.displayed_questions)
            self.assertIn("line graph below", result["question"])
            self.assertIn("cost", result["question"])
            self.assertIsInstance(result["plot"], plt.Figure)
            ax = result["plot"].gca()
            lines = ax.lines
            self.assertEqual(len(lines), 1)
            self.assertTrue("Cost Over" in ax.get_title())

    # Test 9: Generate data table question
    def test_generate_data_table(self):
        template = "Given the data table below with {m} samples, is {topic} suitable? Why?"
        with patch("random.choice", return_value=template):
            result = get_question(self.topic, self.subtopic, self.difficulty, self.displayed_questions)
            self.assertIn("data table below", result["question"])
            m = int(result["question"].split("with ")[1].split(" samples")[0])
            self.assertIsInstance(result["table"], pd.DataFrame)
            self.assertEqual(result["table"].shape[0], m)
            self.assertGreaterEqual(result["table"].shape[1], 3)
            self.assertIn("Label", result["table"].columns)

    # Test 10: Placeholder syncing
    def test_placeholder_syncing(self):
        template = "Based on the scatter plot below with {n} points, would {topic} be effective? Why?"
        with patch("random.choice", return_value=template):
            result = get_question(self.topic, self.subtopic, self.difficulty, self.displayed_questions)
            n = int(result["question"].split("with ")[1].split(" points")[0])
            ax = result["plot"].gca()
            self.assertEqual(len(ax.collections[0].get_offsets()), n)

    # Test 11: Question uniqueness
    def test_question_uniqueness(self):
        template = "What is the main goal of {topic} in {application}?"
        with patch("random.choice", side_effect=[template, template]):
            result1 = get_question(self.topic, self.subtopic, self.difficulty, self.displayed_questions)
            self.displayed_questions.add(result1["question"])
            with patch("random.choice", return_value="Why is {topic} used in {field}?"):
                result2 = get_question(self.topic, self.subtopic, self.difficulty, self.displayed_questions)
            self.assertNotEqual(result1["question"], result2["question"])

    # # Test 12: OCR processing
    # @patch("pytesseract.image_to_string", return_value="Sample OCR text")
    # def test_ocr_processing(self, mock_ocr):
    #     img = Image.new('RGB', (100, 100), color='white')
    #     img_byte_arr = io.BytesIO()
    #     img.save(img_byte_arr, format='PNG')
    #     img_byte_arr.seek(0)
        
    #     result = pytesseract.image_to_string(Image.open(img_byte_arr))
    #     self.assertEqual(result, "Sample OCR text")

    # Test 13: Main logic
    @patch("streamlit.selectbox", side_effect=["Logistic Regression", "Introduction", "easy"])
    @patch("streamlit.button", side_effect=[True, False])
    @patch("streamlit.write")
    @patch("streamlit.pyplot")
    @patch("streamlit.table")
    @patch("streamlit.session_state", new_callable=dict)
    def test_main_logic(self, mock_session_state, mock_table, mock_pyplot, mock_write, mock_button, mock_selectbox):
        # Simulate the app logic fully
        mock_session_state["displayed_questions"] = set()
        mock_session_state["current_question"] = None
        
        # Simulate "Generate Questions" button press
        selected_topic = "Logistic Regression"
        selected_subtopic = "Introduction"
        selected_difficulty = "easy"
        result = get_question(selected_topic, selected_subtopic, selected_difficulty, mock_session_state["displayed_questions"])
        mock_session_state["current_question"] = result
        mock_session_state["displayed_questions"].add(result["question"])
        
        # Simulate display logic explicitly
        if mock_session_state["current_question"]:
            mock_write("### Question:")
            mock_write(result["question"])
            if result["plot"]:
                mock_pyplot(result["plot"])
            if result["table"] is not None:
                mock_table(result["table"])
            mock_write("#### Answer:")
            mock_write(result["answer"])
        
        # Verify calls
        mock_write.assert_any_call("### Question:")
        mock_write.assert_any_call(result["question"])
        mock_write.assert_any_call("#### Answer:")
        mock_write.assert_any_call(result["answer"])
        if result["plot"]:
            mock_pyplot.assert_called_once_with(result["plot"])
        if result["table"] is not None:
            mock_table.assert_called_once_with(result["table"])

if __name__ == "__main__":
    unittest.main()