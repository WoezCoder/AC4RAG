import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from average_metrics import load_evaluation_data

# Load all datasets
data = load_evaluation_data()

datasets = {
    'Agentic_breakpoint_1': data['agentic_breakpoint_1'],
    'Agentic_full_text_1': data['agentic_full_text_1'],
    'Semantic_1': data['semantic_1'],
    'FixedLength_1': data['fixed_length_1'],
    'Agentic_breakpoint_3': data['agentic_breakpoint_3'],
    'Agentic_full_text_3': data['agentic_full_text_3'],
    'Semantic_3': data['semantic_3'],
    'FixedLength_3': data['fixed_length_3'],
    'Agentic_breakpoint_5': data['agentic_breakpoint_5'],
    'Agentic_full_text_5': data['agentic_full_text_5'],
    'Semantic_5': data['semantic_5'],
    'FixedLength_5': data['fixed_length_5'],
}

# Mapping for custom titles
title_mapping = {
    'Agentic_breakpoint': 'AC4RAG V2',
    'Agentic_full_text': 'AC4RAG V1'
}

def calculate_f1_score(generated_answer, ground_truth):
    generated_tokens = set(generated_answer.split())
    ground_truth_tokens = set(ground_truth.split())

    true_positives = len(generated_tokens & ground_truth_tokens)
    false_positives = len(generated_tokens - ground_truth_tokens)
    false_negatives = len(ground_truth_tokens - generated_tokens)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return f1_score


def compute_f1_scores(data):
    f1_scores = []
    for _, row in data.iterrows():
        f1_score = calculate_f1_score(row['RAG_answer'], row['answer'])
        f1_scores.append(f1_score)
    return f1_scores


def check_symmetry_and_plot(differences, pair_name, ax_hist):
    sns.histplot(differences, kde=True, ax=ax_hist)
    ax_hist.set_title(pair_name, fontsize=12)
    ax_hist.set_ylabel('Count')

if __name__ == "__main__":
    f1_scores = {
        name: compute_f1_scores(data)
        for name, data in datasets.items()
    }

    evaluation_levels = ['1', '3', '5']
    for level in evaluation_levels:
        print(f"\nResults for Evaluation Level {level}:")

        # Get datasets with the current suffix
        datasets_with_suffix = {name: f1_scores[name] for name in f1_scores if name.endswith(f'_{level}')}

        # Create all pairs for comparison
        pairs = [
            ('Agentic_breakpoint', 'Agentic_full_text'),
            ('Agentic_breakpoint', 'Semantic'),
            ('Agentic_breakpoint', 'FixedLength'),
            ('Agentic_full_text', 'Semantic'),
            ('Agentic_full_text', 'FixedLength'),
            ('Semantic', 'FixedLength')
        ]

        num_plots = len(pairs)
        num_rows = 3
        num_cols = 2
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12), sharex=True)

        axs = axs.flatten()  # Flatten the 2D array of axes to iterate easily

        for i, (label1, label2) in enumerate(pairs):
            dataset1 = datasets_with_suffix.get(f'{label1}_{level}')
            dataset2 = datasets_with_suffix.get(f'{label2}_{level}')

            if dataset1 is not None and dataset2 is not None and len(dataset1) == len(dataset2):
                # Calculate differences
                differences = [a - b for a, b in zip(dataset1, dataset2)]

                # Prepare titles with custom names
                title1 = title_mapping.get(label1, label1)
                title2 = title_mapping.get(label2, label2)
                pair_title = f'{title1} vs. {title2}'

                # Plot histogram and check symmetry
                check_symmetry_and_plot(differences, pair_title, axs[i])

                # Perform Wilcoxon signed-rank test
                stat, p_value = wilcoxon(dataset1, dataset2)
                print(f'{label1} vs. {label2} at level {level}:')
                print(f'Wilcoxon signed-rank test statistic: {stat}')
                print(f'p-value: {p_value}\n')

                # Interpretation
                alpha = 0.05
                print("Interpretation:")
                if p_value < alpha:
                    print(f'{label1} vs. {label2} (Level {level}): Significant difference between methods (reject H0)')
                else:
                    print(f'{label1} vs. {label2} (Level {level}): No significant difference (fail to reject H0)')
            else:
                print(f"Datasets for {label1} or {label2} at level {level} are missing or mismatched.")

        # Only show x-axis labels on the lowest row
        for ax in axs[num_cols * (num_rows - 1):]:
            ax.set_xlabel('')

        # Adjust font sizes
        for ax in axs:
            ax.title.set_fontsize(12)
            ax.xaxis.label.set_fontsize(10)
            ax.yaxis.label.set_fontsize(10)
            ax.tick_params(axis='both', which='major', labelsize=10)

        # Adjust layout to increase space between plots
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust as needed

        plt.show()
