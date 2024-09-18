import re


def clean_text(text):
    """
    Clean the input text by performing the following operations:
    1. Remove newline characters and double quotes.
    2. Replace multiple spaces with a single space.
    3. Ensure there's a space after periods followed by non-space characters.
    4. Remove commas that are directly followed by non-space characters.

    Returns:
    str: The cleaned text.
    """

    modified = re.sub(r',\S', lambda match: match.group()[1:], re.sub(r'\.\S', lambda match: f". {match.group()[1]}", re.sub(r'\s+', ' ', text.replace('\n', '').replace('"', ''))))

    return modified


if __name__ == "__main__":
    file_path = '../data/paul_graham_essays.txt'

    with open(file_path, 'r') as file:
        full_input_text = file.read()

    modified_text = clean_text(full_input_text)

    with open(file_path, 'w') as file:
        file.write(modified_text)

    print("Newline characters have been removed, and the file has been updated.")
