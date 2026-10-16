import re


def chunk_text(text: str, chunk_size: int = 500, overlap_sentences: int = 1) -> list[str]:
    """
    Splits text into clean chunks made of WHOLE SENTENCES, so every chunk
    starts at the beginning of a sentence and ends at the end of one
    (never mid-sentence). Sentences are grouped together until a chunk
    reaches about chunk_size characters. A small overlap of sentences is
    shared between neighbouring chunks so context isn't lost at a boundary.
    """
    # PDFs often contain stray line breaks inside sentences. Collapse all
    # runs of whitespace into single spaces so sentence detection works.
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Split into sentences: break AFTER a . ! or ? that is followed by a space.
    # The lookbehind (?<=...) keeps the punctuation attached to the sentence.
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = []
    current_length = 0

    for sentence in sentences:
        current.append(sentence)
        current_length += len(sentence) + 1  # +1 for the joining space

        if current_length >= chunk_size:
            chunks.append(" ".join(current))
            # Carry the last sentence(s) into the next chunk as overlap.
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current_length = sum(len(s) + 1 for s in current)

    # Whatever sentences are left become the final chunk.
    if current:
        chunks.append(" ".join(current))

    return chunks
