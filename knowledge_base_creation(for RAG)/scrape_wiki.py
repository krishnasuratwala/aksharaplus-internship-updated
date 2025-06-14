
import wikipediaapi
import json
import re
import requests
import mwparserfromhell
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import time
nltk.download('punkt_tab', quiet=True)

# Deduplicated list of 108 topics
topics = [
    "Logistic_regression", "Support_vector_machine", "K-nearest_neighbors_algorithm",
    "Naive_Bayes_classifier", "Decision_tree", "Statistical_classification",
    "Predictive_modelling", "Machine_learning", "Text_classification",
    "Natural_language_processing", "Supervised_learning", "Feature_(machine_learning)",
    "Binary_classification", "Sigmoid_function", "Gradient_descent",
    "Regularization_(mathematics)", "Bayes'_theorem", "Decision_boundary",
    "Probability_theory", "Linear_algebra", "Model_selection",
    "Evaluation_of_binary_classifiers", "Evaluation_metrics", "Multiclass_classification",
    "Data_preprocessing", "Class_imbalance", "Nonlinear_system", "Linear_regression",
    "Perceptron", "Weight", "Bias_(statistics)", "Learning_rate", "Outlier",
    "Noise_(signal_processing)", "Accuracy_and_precision", "Time_complexity",
    "Logit", "Cross_entropy", "Overfitting", "Softmax_function", "Convex_optimization",
    "Scikit-learn", "Convergence_(mathematics)", "Derivative", "Mathematical_optimization",
    "Sparse_matrix", "Computational_complexity_theory", "Receiver_operating_characteristic",
    "Data_visualization", "Scatter_plot", "Line_graph", "Robust_statistics",
    "Margin_(machine_learning)", "Inverse_function", "Multidimensional_scaling",
    "Error_analysis_(mathematics)", "Gradient", "Penalty_method", "Bias-variance_tradeoff",
    "Feature_scaling", "Decision_rule", "Coefficient", "Statistical_assumption",
    "Applications_of_machine_learning", "Hyperplane", "Linear_separability",
    "Radial_basis_function_kernel", "Polynomial_kernel", "Kernel_method", "Feature_space",
    "Lagrange_multipliers", "Duality_(optimization)", "Karush–Kuhn–Tucker_conditions",
    "Hinge_loss", "Nearest_neighbor_search", "Euclidean_distance", "Manhattan_distance",
    "Minkowski_distance", "Distance_measure", "Majority_voting", "Standard_score",
    "Gini_impurity", "Entropy_(information_theory)", "Information_gain",
    "Decision_tree_learning", "Pruning_(decision_trees)", "Random_forest",
    "Ensemble_learning", "Bootstrap_aggregating", "Sampling_with_replacement",
    "Random_subspace_method", "Feature_selection"
]

def get_raw_wikitext(title, retries=3, delay=2):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": title,
        "rvslots": "main",
        "rvprop": "content"
    }
    for attempt in range(retries):
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            try:
                data = response.json()
                pages = data["query"]["pages"]
                page = next(iter(pages.values()))
                wikitext = page["revisions"][0]["slots"]["main"]["*"]
                time.sleep(1)  # polite delay
                return wikitext
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                print(f"Error processing JSON for '{title}': {e}")
                return None

        elif response.status_code == 429:
            wait_time = delay * (attempt + 1)
            print(f"⚠️ Too many requests for '{title}'. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            print(f"❌ Error fetching '{title}': Status code {response.status_code}")
            return None

    print(f"❌ Failed to fetch '{title}' after {retries} retries.")
    return None
   

def replace_math_with_latex(text):
    text = re.sub(r'<math>(.*?)</math>', r'$\1$', text, flags=re.DOTALL)
    def replace_math_template(match):
        inner = match.group(1).strip()
        return f"${inner}$"
    text = re.sub(r'{{\s*math\s*\|\s*(.*?)\s*}}', replace_math_template, text)
    return text

def extract_latex_formulas(wikicode):
    formulas = []
    for template in wikicode.filter_templates():
        if template.name.matches("math") and template.params:
            formulas.append(str(template.params[0].value).strip())
    math_tags = re.findall(r'<math>(.*?)</math>', str(wikicode), re.DOTALL)
    formulas.extend([tag.strip() for tag in math_tags])
    return formulas

def clean_and_chunk(wikitext, min_words=200, max_words=300):
    wikicode = mwparserfromhell.parse(wikitext)
    text = replace_math_with_latex(str(wikicode))
    wikicode_clean = mwparserfromhell.parse(text)
    clean_text = wikicode_clean.strip_code()

    clean_text = re.sub(r'\[\[File:.*?\]\]', '', clean_text)
    clean_text = re.sub(r'thumb\|.*?\|', '', clean_text)
    clean_text = re.sub(r'\n+', '\n', clean_text)
    clean_text = re.sub(r'\{\{.*?\}\}', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    sentences = sent_tokenize(clean_text)
    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:
        words = word_tokenize(sentence)
        word_count += len(words)
        current_chunk.append(sentence)
        if word_count >= min_words:
            chunks.append({
                "text": ' '.join(current_chunk),
                "word_count": word_count
            })
            current_chunk = []
            word_count = 0

    if current_chunk:
        last_chunk_words = sum(len(word_tokenize(s)) for s in current_chunk)
        if last_chunk_words < min_words and chunks:
            chunks[-1]["text"] += ' ' + ' '.join(current_chunk)
            chunks[-1]["word_count"] += last_chunk_words
        else:
            chunks.append({
                "text": ' '.join(current_chunk),
                "word_count": last_chunk_words
            })

    final_chunks = []
    for chunk in chunks:
        words = word_tokenize(chunk["text"])
        if len(words) <= max_words:
            final_chunks.append(chunk)
        else:
            sub_sentences = sent_tokenize(chunk["text"])
            sub_chunk = []
            sub_count = 0
            for s in sub_sentences:
                sub_words = len(word_tokenize(s))
                if sub_count + sub_words > max_words and sub_chunk:
                    final_chunks.append({"text": ' '.join(sub_chunk), "word_count": sub_count})
                    sub_chunk = [s]
                    sub_count = sub_words
                else:
                    sub_chunk.append(s)
                    sub_count += sub_words
            if sub_chunk:
                final_chunks.append({"text": ' '.join(sub_chunk), "word_count": sub_count})

    for chunk in final_chunks:
        if "word_count" in chunk:
            del chunk["word_count"]
    return final_chunks

def extract_structured_article(title):
    wiki = wikipediaapi.Wikipedia(user_agent='my-wiki-scraper/1.0 (krishnawinin30@gmail.com)', language='en')
    page = wiki.page(title)

    if not page.exists():
        print(f"Page not found: {title}")
        return None

    print(f"📘 Fetching article: {title}")
    raw_wikitext = get_raw_wikitext(title)
    wikicode = mwparserfromhell.parse(raw_wikitext)
    latex_formulas = extract_latex_formulas(wikicode)
    chunks = clean_and_chunk(raw_wikitext)

    data = {
        "title": title,
        "chunks": chunks,
        "latex_formulas": latex_formulas
    }

    filename = f"{title.replace(' ', '_')}_structured.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ Saved: {filename}")
    return data

def build_knowledge_base():
    knowledge_base = {}
    for topic in topics:
        data = extract_structured_article(topic)
        if data:
            knowledge_base[topic] = data
        time.sleep(1)  # polite delay between topics
    with open("classification_algorithms_knowledge_base1.json", "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=4)
    print("✅ Combined knowledge base saved to 'classification_algorithms_knowledge_base.json'")

# Run
build_knowledge_base()


