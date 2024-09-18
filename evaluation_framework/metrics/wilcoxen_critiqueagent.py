import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from average_metrics import load_evaluation_data

# Load all datasets
data = load_evaluation_data()

datasets = {
    'Agentic_breakpoint_1': data['agentic_breakpoint_1']['llm_evaluation'],
    'Agentic_full_text_1': data['agentic_full_text_1']['llm_evaluation'],
    'Semantic_1': data['semantic_1']['llm_evaluation'],
    'FixedLength_1': data['fixed_length_1']['llm_evaluation'],
    'Agentic_breakpoint_3': data['agentic_breakpoint_3']['llm_evaluation'],
    'Agentic_full_text_3': data['agentic_full_text_3']['llm_evaluation'],
    'Semantic_3': data['semantic_3']['llm_evaluation'],
    'FixedLength_3': data['fixed_length_3']['llm_evaluation'],
    'Agentic_breakpoint_5': data['agentic_breakpoint_5']['llm_evaluation'],
    'Agentic_full_text_5': data['agentic_full_text_5']['llm_evaluation'],
    'Semantic_5': data['semantic_5']['llm_evaluation'],
    'FixedLength_5': data['fixed_length_5']['llm_evaluation'],
}

# Mapping for custom titles
title_mapping = {
    'Agentic_breakpoint': 'AC4RAG V2',
    'Agentic_full_text': 'AC4RAG V1'
}


def check_symmetry_and_plot(differences, pair_name, ax_hist):
    sns.histplot(differences, kde=True, ax=ax_hist)
    ax_hist.set_title(pair_name, fontsize=12)  # Increased title font size


if __name__ == "__main__":
    suffixes = ['1', '3', '5']
    for suffix in suffixes:
        print(f"\nResults for Evaluation Level {suffix}:")

        # Get datasets with the current suffix
        datasets_with_suffix = {name: data for name, data in datasets.items() if name.endswith(f'_{suffix}')}

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
            dataset1 = datasets_with_suffix[f'{label1}_{suffix}']
            dataset2 = datasets_with_suffix[f'{label2}_{suffix}']

            # Calculate differences
            differences = dataset1 - dataset2

            # Prepare titles with custom names
            title1 = title_mapping.get(label1, label1)
            title2 = title_mapping.get(label2, label2)
            pair_title = f'{title1} vs. {title2}'

            # Plot histogram and check symmetry
            check_symmetry_and_plot(differences, pair_title, axs[i])

            # Perform Wilcoxon signed-rank test
            stat, p_value = wilcoxon(dataset1, dataset2)
            print(f'{label1} vs. {label2} at level {suffix}:')
            print(f'Wilcoxon signed-rank test statistic: {stat}')
            print(f'p-value: {p_value}\n')

            # Interpretation
            alpha = 0.05
            print("Interpretation:")
            if p_value < alpha:
                print(
                    f'{label1} vs. {label2} (Level {suffix}): There is a significant difference between the two methods (reject H0)')
            else:
                print(
                    f'{label1} vs. {label2} (Level {suffix}): There is no significant difference between the two methods (fail to reject H0)')

        # Only show x-axis labels on the lowest row
        for ax in axs[num_cols * (num_rows - 1):]:
            ax.set_xlabel('')

        # Adjust font sizes
        for ax in axs:
            ax.title.set_fontsize(12)  # Increased title font size
            ax.xaxis.label.set_fontsize(10)
            ax.yaxis.label.set_fontsize(10)
            ax.tick_params(axis='both', which='major', labelsize=10)

        # Adjust layout to increase space between plots
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust as needed

        plt.show()
