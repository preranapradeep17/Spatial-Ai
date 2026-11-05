"""Unit tests for from-scratch TF-IDF document retriever."""
import os
import pytest
from backend.rag.retriever import TFIDFRetriever

def test_tfidf_retriever_chunking_and_indexing():
    docs = [
        {
            "id": "DOC-1",
            "title": "Emergency Evacuation",
            "content": "In an active containment alarm, evacuate immediately through the green egress door. Maintain safe distance from obstacles."
        },
        {
            "id": "DOC-2",
            "title": "Door Override Code",
            "content": "The manual override passcode for Door 3 is 7391. The physical key is located in the cabinet near the bookshelf."
        }
    ]

    retriever = TFIDFRetriever(chunk_size_words=20, chunk_overlap_words=5)
    retriever.load_documents_from_list(docs)

    assert retriever.is_indexed
    assert len(retriever.documents) >= 2
    assert "override" in retriever.vocab

    # Test retrieval for override passcode
    results = retriever.retrieve("What is the override code for the door?", top_k=1)
    assert len(results) > 0
    assert results[0]["doc_id"] == "DOC-2"
    assert "7391" in results[0]["text"]
    assert results[0]["score"] > 0.0

    # Test retrieval for evacuation
    results_evac = retriever.retrieve("Where do I go during an evacuation alarm?", top_k=1)
    assert len(results_evac) > 0
    assert results_evac[0]["doc_id"] == "DOC-1"
    assert "door" in results_evac[0]["text"]

def test_tfidf_load_from_json(tmp_path):
    json_path = "data/documents/facility_documents.json"
    if os.path.exists(json_path):
        retriever = TFIDFRetriever()
        retriever.load_documents_from_json(json_path)
        assert retriever.is_indexed
        results = retriever.retrieve("chemical spill safety protocol", top_k=1)
        assert len(results) > 0
        assert "Protocol 22" in results[0]["text"] or "DOC-002" == results[0]["doc_id"]
