import pandas as pd


def load_evaluation_data():
    """
    Loads evaluation data from multiple directories and returns a dictionary of dataframes.

    Returns:
    - dict: Dictionary of loaded pandas DataFrames where keys are of the format 'category_number' (e.g., 'agentic_breakpoint_1').
    """
    dataframes = {}
    data_paths = {
        "agentic_breakpoint": "../../data/agentic_breakpoint/",
        "agentic_full_text": "../../data/agentic_full_text/",
        "semantic": "../../data/semantic/",
        "fixed_length": "../../data/fixed_length/"
    }

    evaluation_files = ["evaluated_1.csv", "evaluated_3.csv", "evaluated_5.csv"]

    for category, path in data_paths.items():
        for file in evaluation_files:
            file_path = path + file
            key = f"{category}_{file.split('_')[-1].split('.')[0]}"
            dataframes[key] = pd.read_csv(file_path)

    return dataframes


def average_evaluation(data):
    """
    Computes the average evaluation of the critique agent score.
    """
    result = data['llm_evaluation'].mean()
    return result


def average_given_true(data):
    """
    Computes the average evaluation of the critique agent score, given that the search was successful.
    """
    average = data[data["search_evaluation"]]["llm_evaluation"].mean()
    return average


def search_evaluation(data):
    """
    Computes the search accuracy.
    """
    true_count = data['search_evaluation'].sum()
    return true_count / len(data['search_evaluation'])


def average_f1_score(data):
    """
    Computes the average F1 score.
    """
    data = data[data["search_evaluation"]]
    total_f1_score = 0
    for _, row in data.iterrows():
        total_f1_score += calculate_f1_score(row['RAG_answer'], row['answer'])
    average_f1 = total_f1_score / len(data)
    return average_f1


def exact_match_score(data):
    """
    Computes the exact match score.
    """
    data = data[data["search_evaluation"]]
    exact_matches = 0
    total_answers = len(data)

    for _, row in data.iterrows():
        if calculate_exact_match(row['RAG_answer'], row['answer']):
            exact_matches += 1

    return exact_matches / total_answers


def calculate_exact_match(generated_answer, ground_truth):
    """
    Helper function to check if the RAG answer exactly matches the ground truth.
    """
    return generated_answer == ground_truth


def calculate_f1_score(generated_answer, ground_truth):
    """
    Helper function to compute the individual F1 scores between the RAG answer and the ground truth.
    """
    generated_tokens = set(generated_answer.split())
    ground_truth_tokens = set(ground_truth.split())

    true_positives = len(generated_tokens & ground_truth_tokens)
    false_positives = len(generated_tokens - ground_truth_tokens)
    false_negatives = len(ground_truth_tokens - generated_tokens)

    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score


def compute_metrics(data, dataset_name):
    """
    Compute all metrics for a given evaluation dataset and return it as a dataframe.
    """
    return pd.DataFrame({
        'Dataset': [dataset_name],
        'Average Evaluation': [average_evaluation(data)],
        'Average Given Search': [average_given_true(data)],
        'Search Evaluation': [search_evaluation(data)],
        'Average F1 Score': [average_f1_score(data)],
        'Exact Match Score': [exact_match_score(data)]
    })


if __name__ == "__main__":
    data = load_evaluation_data()

    results_1 = pd.concat([
        compute_metrics(data['agentic_breakpoint_1'], 'Agentic_Breakpoint_1'),
        compute_metrics(data['agentic_full_text_1'], 'Agentic_Full_Text_1'),
        compute_metrics(data['semantic_1'], 'Semantic_1'),
        compute_metrics(data['fixed_length_1'], 'FixedLength_1')
    ])

    results_3 = pd.concat([
        compute_metrics(data['agentic_breakpoint_3'], 'Agentic_Breakpoint_3'),
        compute_metrics(data['agentic_full_text_3'], 'Agentic_Full_Text_3'),
        compute_metrics(data['semantic_3'], 'Semantic_3'),
        compute_metrics(data['fixed_length_3'], 'FixedLength_3')
    ])

    results_5 = pd.concat([
        compute_metrics(data['agentic_breakpoint_5'], 'Agentic_Breakpoint_5'),
        compute_metrics(data['agentic_full_text_5'], 'Agentic_Full_Text_5'),
        compute_metrics(data['semantic_5'], 'Semantic_5'),
        compute_metrics(data['fixed_length_5'], 'FixedLength_5')
    ])

    print("Results for dataset size 1:")
    print(results_1.to_string(index=False))

    print("\nResults for dataset size 3:")
    print(results_3.to_string(index=False))

    print("\nResults for dataset size 5:")
    print(results_5.to_string(index=False))
