import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from average_metrics import load_evaluation_data

data = load_evaluation_data()

datasets = {
    'Agentic_breakpoint_1': data['agentic_breakpoint_1']['search_evaluation'],
    'Agentic_full_text_1': data['agentic_full_text_1']['search_evaluation'],
    'Semantic_1': data['semantic_1']['search_evaluation'],
    'FixedLength_1': data['fixed_length_1']['search_evaluation'],
    'Agentic_breakpoint_3': data['agentic_breakpoint_3']['search_evaluation'],
    'Agentic_full_text_3': data['agentic_full_text_3']['search_evaluation'],
    'Semantic_3': data['semantic_3']['search_evaluation'],
    'FixedLength_3': data['fixed_length_3']['search_evaluation'],
    'Agentic_breakpoint_5': data['agentic_breakpoint_5']['search_evaluation'],
    'Agentic_full_text_5': data['agentic_full_text_5']['search_evaluation'],
    'Semantic_5': data['semantic_5']['search_evaluation'],
    'FixedLength_5': data['fixed_length_5']['search_evaluation'],
}


def perform_mcnemars_test(search1, search2, dataset1, dataset2):
    """
    Perform McNemar's Test to compare two datasets.

    Returns:
    - A string with the test statistic and p-value
    """
    n_01 = np.sum((search1 == True) & (search2 == False))
    n_10 = np.sum((search1 == False) & (search2 == True))

    table = np.array([[np.sum((search1 == True) & (search2 == True)), n_01],
                      [n_10, np.sum((search1 == False) & (search2 == False))]])

    result = mcnemar(table, exact=True)

    return f"{dataset1} vs {dataset2} - Statistic: {result.statistic:.4f}, P-value: {result.pvalue:.8f}"


if __name__ == "__main__":
    suffixes = ['1', '3', '5']
    for suffix in suffixes:
        print(f"\nMcNemar Tests for Suffix _{suffix}:")
        datasets_with_suffix = {name: data for name, data in datasets.items() if name.endswith(f'_{suffix}')}
        dataset_names = list(datasets_with_suffix.keys())

        for i in range(len(dataset_names)):
            for j in range(i + 1, len(dataset_names)):
                dataset1 = dataset_names[i]
                dataset2 = dataset_names[j]
                match1 = datasets_with_suffix[dataset1]
                match2 = datasets_with_suffix[dataset2]
                result = perform_mcnemars_test(match1, match2, dataset1, dataset2)
                print(result)