from sentence_transformers import SentenceTransformer

# Load the model ONCE when this file is first imported.
# all-mpnet-base-v2 is a higher-quality model (768-dim) that separates
# relevant from irrelevant text much better than the smaller MiniLM model —
# this is what improves search quality and the AUC score.
# local_files_only=True loads the model straight from the local cache and
# never contacts Hugging Face. This avoids startup errors/delays when their
# servers are slow or down (HTTP 504), and lets the project run fully offline.
# Note: the model must already be downloaded (it is). If you ever switch to a
# NEW model, set this to False once so it can download, then set it back.
model = SentenceTransformer("all-mpnet-base-v2", local_files_only=True)


def generate_embeddings(chunks: list[str]) -> list:
    """
    Receives a list of text chunks and returns their embeddings.
    Each embedding is a vector (list of numbers) representing the meaning
    of that chunk.
    """
    embeddings = model.encode(chunks)
    return embeddings
