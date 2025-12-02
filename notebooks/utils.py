# utils.py

import os
import tiktoken
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ----------------------------------------------------
# 1) Load environment variables from .env
# ----------------------------------------------------
load_dotenv()    # <-- IMPORTANT: loads NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

# ----------------------------------------------------
# 2) Local embedding model (NO OpenAI needed)
# ----------------------------------------------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = embedding_model.get_sentence_embedding_dimension()  # = 384


# ----------------------------------------------------
# 3) Local Chat (optional) - Ollama stub
# ----------------------------------------------------
def local_chat(prompt: str) -> str:
    """
    If Ollama is installed, this will call it.
    Otherwise, returns a safe default message.
    """
    try:
        import subprocess
        cmd = ["ollama", "run", "llama3", prompt]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return "(Local LLM not configured)"


# ----------------------------------------------------
# 4) Neo4j Driver - Aura or local depending on .env
# ----------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_URI:
    raise ValueError("❌ NEO4J_URI is missing. Check your .env or Codespace variables.")

neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    notifications_min_severity="OFF",
)


# ----------------------------------------------------
# 5) Chunking
# ----------------------------------------------------
def chunk_text(text, chunk_size, overlap, split_on_whitespace_only=True):
    chunks = []
    index = 0

    while index < len(text):
        if split_on_whitespace_only:
            prev_whitespace = 0
            left_index = index - overlap
            while left_index >= 0:
                if text[left_index] == " ":
                    prev_whitespace = left_index
                    break
                left_index -= 1
            next_whitespace = text.find(" ", index + chunk_size)
            if next_whitespace == -1:
                next_whitespace = len(text)
            chunk = text[prev_whitespace:next_whitespace].strip()
            chunks.append(chunk)
            index = next_whitespace + 1
        else:
            start = max(0, index - overlap + 1)
            end = min(index + chunk_size + overlap, len(text))
            chunk = text[start:end].strip()
            chunks.append(chunk)
            index += chunk_size

    return chunks


# ----------------------------------------------------
# 6) Token counting
# ----------------------------------------------------
def num_tokens_from_string(string: str, model: str = "gpt-4"):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))


# ----------------------------------------------------
# 7) Local embeddings (replacement for OpenAI embeddings)
# ----------------------------------------------------
def embed(texts):
    """Return locally generated embeddings (list of 384-dim vectors)."""
    return embedding_model.encode(texts).tolist()


# ----------------------------------------------------
# 8) Chat + tool_choice (local)
# ----------------------------------------------------
def chat(messages, model="local", temperature=0, config={}):
    # Combine messages into single prompt
    prompt = "\n".join([m["content"] for m in messages])
    return local_chat(prompt)


def tool_choice(messages, model="local", temperature=0, tools=[], config={}):
    return {
        "tool": "none",
        "output": chat(messages)
    }
