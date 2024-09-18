import json
import nltk
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_content_word_distribution(ax, json_data, title, x_lim=None, y_lim=None):
    """
    Calculate and plot the distribution of word counts in the given JSON data.
    """
    content_word_counts = []

    for entry in json_data:
        content = entry['content']
        words = nltk.word_tokenize(content)
        word_count = len(words)
        content_word_counts.append(word_count)

    # Plot on the specified axes
    sns.kdeplot(content_word_counts, fill=True, ax=ax, label=title)
    ax.set_ylabel('Density')
    ax.set_title(title, fontsize=18)
    ax.grid(True)

    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)


# Load the data
with open('../../data/agentic_breakpoint/chunks_agentic_breakpoint.json', 'r') as file:
    json_data_agentic_breakpoint = json.load(file)

with open('../../data/agentic_full_text/chunks_agentic_full_text.json', 'r') as file:
    json_data_agentic_full_text = json.load(file)

with open('../../data/semantic/chunks_semantic.json', 'r') as file:
    json_data_semantic = json.load(file)

# Create the figure with shared x-axis
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Plot each distribution on a separate subplot by passing the respective axes
calculate_content_word_distribution(
    axes[0],  # Pass the first axes object
    json_data_semantic,
    'Semantic',
    x_lim=(0, 4000),
    y_lim=(0, 0.003)
)

calculate_content_word_distribution(
    axes[1],  # Pass the second axes object
    json_data_agentic_full_text,
    'AC4RAG V1',
    x_lim=(0, 4000),
    y_lim=(0, 0.003)
)

calculate_content_word_distribution(
    axes[2],  # Pass the third axes object
    json_data_agentic_breakpoint,
    'AC4RAG V2',
    x_lim=(0, 4000),
    y_lim=(0, 0.003)
)

# Only the bottom plot needs an x-axis label
axes[2].set_xlabel('Number of Words')

# Adjust layout and spacing
plt.subplots_adjust(hspace=10)
plt.tight_layout()
plt.show()
