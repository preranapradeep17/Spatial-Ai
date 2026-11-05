"""From-scratch TF-IDF Document Retriever for Spatial Quest.

Implements pure Python/NumPy TF-IDF vectorization and cosine similarity
retrieval over facility documents, mission logs, and security protocols.
Traceable to CS3235 (Working with LLMs - Information Retrieval & RAG Foundations).
No external vector databases or pretrained embedding APIs used.
"""
import math
import re
import json
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

def tokenize(text: str) -> List[str]:
    """Simple alphanumeric lowercase word tokenizer."""
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())

class TFIDFRetriever:
    """TF-IDF Document Vectorizer and Cosine Similarity Retriever."""

    def __init__(self, chunk_size_words: int = 60, chunk_overlap_words: int = 15):
        self.chunk_size = chunk_size_words
        self.chunk_overlap = chunk_overlap_words
        self.documents: List[Dict[str, Any]] = []  # List of chunks
        self.vocab: Dict[str, int] = {}            # word -> term_idx
        self.idf: np.ndarray = np.array([])        # idf values
        self.doc_vectors: np.ndarray = np.array([])# (N_chunks, vocab_size)
        self.is_indexed: bool = False

    def chunk_document(self, text: str, doc_id: str, title: str) -> List[Dict[str, Any]]:
        """Splits a document text into sliding window chunks."""
        words = text.split()
        if not words:
            return []

        chunks: List[Dict[str, Any]] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunk_idx = 0

        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                "doc_id": doc_id,
                "title": title,
                "text": chunk_text
            })
            chunk_idx += 1
            if i + self.chunk_size >= len(words):
                break

        return chunks

    def load_documents_from_json(self, filepath: str):
        """Loads and chunks documents from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)

        all_chunks = []
        for item in raw_docs:
            chunks = self.chunk_document(item.get("content", ""), item.get("id", "doc"), item.get("title", "Untitled"))
            all_chunks.extend(chunks)

        self.documents = all_chunks
        self._build_index()

    def load_documents_from_list(self, doc_list: List[Dict[str, str]]):
        """Loads and chunks documents from an in-memory list of dicts."""
        all_chunks = []
        for item in doc_list:
            chunks = self.chunk_document(item.get("content", ""), item.get("id", "doc"), item.get("title", "Untitled"))
            all_chunks.extend(chunks)

        self.documents = all_chunks
        self._build_index()

    def _build_index(self):
        """Computes TF-IDF matrix over all indexed chunks."""
        if not self.documents:
            self.is_indexed = False
            return

        # 1. Build vocabulary and term document frequencies
        df_counts: Dict[str, int] = {}
        chunk_tokens_list: List[List[str]] = []

        for chunk in self.documents:
            tokens = tokenize(chunk["text"])
            chunk_tokens_list.append(tokens)
            unique_in_chunk = set(tokens)
            for t in unique_in_chunk:
                df_counts[t] = df_counts.get(t, 0) + 1

        self.vocab = {term: idx for idx, term in enumerate(sorted(df_counts.keys()))}
        N_docs = len(self.documents)
        vocab_size = len(self.vocab)

        # 2. Compute IDF: ln((1 + N) / (1 + df)) + 1
        self.idf = np.zeros(vocab_size, dtype=np.float32)
        for term, idx in self.vocab.items():
            df = df_counts[term]
            self.idf[idx] = math.log((1.0 + N_docs) / (1.0 + df)) + 1.0

        # 3. Compute TF-IDF matrix
        self.doc_vectors = np.zeros((N_docs, vocab_size), dtype=np.float32)
        for i, tokens in enumerate(chunk_tokens_list):
            if not tokens:
                continue
            tf_counts: Dict[int, int] = {}
            for t in tokens:
                if t in self.vocab:
                    idx = self.vocab[t]
                    tf_counts[idx] = tf_counts.get(idx, 0) + 1

            total_tokens = len(tokens)
            for idx, count in tf_counts.items():
                tf = count / total_tokens
                self.doc_vectors[i, idx] = tf * self.idf[idx]

            # Normalize document vector to unit L2 norm
            norm = np.linalg.norm(self.doc_vectors[i])
            if norm > 1e-8:
                self.doc_vectors[i] /= norm

        self.is_indexed = True

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves the top_k most relevant chunks using Cosine Similarity."""
        if not self.is_indexed or not self.vocab or not self.documents:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        # Vectorize query
        q_vec = np.zeros(len(self.vocab), dtype=np.float32)
        tf_counts: Dict[int, int] = {}
        for t in tokens:
            if t in self.vocab:
                idx = self.vocab[t]
                tf_counts[idx] = tf_counts.get(idx, 0) + 1

        if not tf_counts:
            return []

        total_tokens = len(tokens)
        for idx, count in tf_counts.items():
            tf = count / total_tokens
            q_vec[idx] = tf * self.idf[idx]

        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-8:
            q_vec /= q_norm

        # Cosine similarity: doc_vectors (N, V) @ q_vec (V,) -> (N,)
        similarities = np.dot(self.doc_vectors, q_vec)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.0:
                results.append({
                    "chunk_id": self.documents[idx]["chunk_id"],
                    "doc_id": self.documents[idx]["doc_id"],
                    "title": self.documents[idx]["title"],
                    "text": self.documents[idx]["text"],
                    "score": round(score, 4)
                })

        return results

# Singleton instance
global_retriever = TFIDFRetriever()
