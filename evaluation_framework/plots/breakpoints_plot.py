import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_cosine_distances(file_path, number):
    with open(file_path, 'r') as file:
        source_documents = json.load(file)
    return [item[f'cosine_distance_to_next_{number}'] for item in source_documents if f'cosine_distance_to_next_{number}' in item]

# Load data
cosine_distances_agentic_breakpoint_1 = load_cosine_distances('../../data/agentic_breakpoint/chunks_agentic_breakpoint.json', number=1)
cosine_distances_agentic_full_text_1 = load_cosine_distances('../../data/agentic_full_text/chunks_agentic_full_text.json', number=1)
cosine_distances_semantic_1 = load_cosine_distances('../../data/semantic/chunks_semantic.json', number=1)
cosine_distances_fixed_length_1 = load_cosine_distances('../../data/fixed_length/chunks_fixed_length.json', number=1)

cosine_distances_agentic_breakpoint_5 = load_cosine_distances('../../data/agentic_breakpoint/chunks_agentic_breakpoint.json', number=5)
cosine_distances_agentic_full_text_5 = load_cosine_distances('../../data/agentic_full_text/chunks_agentic_full_text.json', number=5)
cosine_distances_semantic_5 = load_cosine_distances('../../data/semantic/chunks_semantic.json', number=5)
cosine_distances_fixed_length_5 = load_cosine_distances('../../data/fixed_length/chunks_fixed_length.json', number=5)

cosine_distances_agentic_breakpoint_full = load_cosine_distances('../../data/agentic_breakpoint/chunks_agentic_breakpoint.json', number="full_text")
cosine_distances_agentic_full_text_full = load_cosine_distances('../../data/agentic_full_text/chunks_agentic_full_text.json', number="full_text")
cosine_distances_semantic_full = load_cosine_distances('../../data/semantic/chunks_semantic.json', number="full_text")
cosine_distances_fixed_length_full = load_cosine_distances('../../data/fixed_length/chunks_fixed_length.json', number="full_text")

# Plotting
fig, axes = plt.subplots(3, 4, figsize=(20, 15), sharex=True, sharey=True)

def plot_kde(ax, data, title):
    sns.kdeplot(data, bw_adjust=0.7, fill=True, ax=ax)
    ax.set_title(title)
    ax.set_yticklabels([])
    ax.set_ylabel('Density')
    ax.grid(True)


plot_kde(axes[0, 3], cosine_distances_agentic_breakpoint_1, 'AC4RAG V2 (1 Sentence)')
plot_kde(axes[0, 2], cosine_distances_agentic_full_text_1, 'AC4RAG V1 (1 Sentence)')
plot_kde(axes[0, 1], cosine_distances_semantic_1, 'Semantic (1 Sentence)')
plot_kde(axes[0, 0], cosine_distances_fixed_length_1, 'Fixed Length (1 Sentence)')

plot_kde(axes[1, 3], cosine_distances_agentic_breakpoint_5, 'AC4RAG V2 (5 Sentences)')
plot_kde(axes[1, 2], cosine_distances_agentic_full_text_5, 'AC4RAG V1 (5 Sentences)')
plot_kde(axes[1, 1], cosine_distances_semantic_5, 'Semantic (5 Sentences)')
plot_kde(axes[1, 0], cosine_distances_fixed_length_5, 'Fixed Length (5 Sentences)')

plot_kde(axes[2, 3], cosine_distances_agentic_breakpoint_full, 'AC4RAG V2 (Full Text)')
plot_kde(axes[2, 2], cosine_distances_agentic_full_text_full, 'AC4RAG V1 (Full Text)')
plot_kde(axes[2, 1], cosine_distances_semantic_full, 'Semantic (Full Text)')
plot_kde(axes[2, 0], cosine_distances_fixed_length_full, 'Fixed Length (Full Text)')


min_value = 0
max_value = 1

for ax in axes.flat:
    ax.set_xlim(min_value, max_value)

plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()


def calculate_average(distances):
    return sum(distances) / len(distances) if distances else 0


average_distance_agentic_1 = calculate_average(cosine_distances_agentic_breakpoint_1)
average_distance_agentic_full_text_1 = calculate_average(cosine_distances_agentic_full_text_1)
average_distance_semantic_1 = calculate_average(cosine_distances_semantic_1)
average_distance_fixed_length_1 = calculate_average(cosine_distances_fixed_length_1)

average_distance_agentic_5 = calculate_average(cosine_distances_agentic_breakpoint_5)
average_distance_agentic_full_text_5 = calculate_average(cosine_distances_agentic_full_text_5)
average_distance_semantic_5 = calculate_average(cosine_distances_semantic_5)
average_distance_fixed_length_5 = calculate_average(cosine_distances_fixed_length_5)

average_distance_agentic_full = calculate_average(cosine_distances_agentic_breakpoint_full)
average_distance_agentic_full_text_full = calculate_average(cosine_distances_agentic_full_text_full)
average_distance_semantic_full = calculate_average(cosine_distances_semantic_full)
average_distance_fixed_length_full = calculate_average(cosine_distances_fixed_length_full)

print("Averages for Chunk Size 1:")
print(f"Agentic Breakpoint: {average_distance_agentic_1}")
print(f"Agentic Full Text: {average_distance_agentic_full_text_1}")
print(f"Semantic: {average_distance_semantic_1}")
print(f"Fixed Length: {average_distance_fixed_length_1}")

print("Averages for Chunk Size 5:")
print(f"Agentic Breakpoint: {average_distance_agentic_5}")
print(f"Agentic Full Text: {average_distance_agentic_full_text_5}")
print(f"Semantic: {average_distance_semantic_5}")
print(f"Fixed Length: {average_distance_fixed_length_5}")

print("Averages for Full Text:")
print(f"Agentic Breakpoint: {average_distance_agentic_full}")
print(f"Agentic Full Text: {average_distance_agentic_full_text_full}")
print(f"Semantic: {average_distance_semantic_full}")
print(f"Fixed Length: {average_distance_fixed_length_full}")
