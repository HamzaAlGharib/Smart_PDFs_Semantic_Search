import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords_per_file(file_texts: dict, top_n: int = 10) -> dict:
    """
    Picks the most DISTINCTIVE keywords for each file using TF-IDF.

    TF-IDF rewards words that are frequent in one file but rare across the
    others, so the keywords it returns are the ones that best characterise
    each document.

    Input : {filename: "full text of the file", ...}
    Output: {filename: [keyword1, keyword2, ...], ...}
    """
    filenames = list(file_texts.keys())
    corpus = [file_texts[name] for name in filenames]

    # Guard: if every document is empty there is nothing to extract.
    if not any(text.strip() for text in corpus):
        return {name: [] for name in filenames}

    vectorizer = TfidfVectorizer(
        stop_words="english",                    # ignore "the", "and", ...
        token_pattern=r"(?u)\b[a-zA-Z]{4,}\b",   # letters only, len >= 4 (drops code noise like int/sig)
        ngram_range=(1, 2),                      # allow single words AND two-word phrases
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)  # shape: (files, vocabulary)
    terms = vectorizer.get_feature_names_out()

    keywords = {}
    for row_index, filename in enumerate(filenames):
        # The TF-IDF weights for this one file.
        weights = tfidf_matrix[row_index].toarray()[0]
        # Indices of the highest-weight terms, largest first.
        top_indices = np.argsort(weights)[::-1][:top_n]
        # Keep only terms that actually appear (weight > 0).
        keywords[filename] = [terms[i] for i in top_indices if weights[i] > 0]

    return keywords
