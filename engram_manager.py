#!/usr/bin/env python3
"""
Engram Manager for LLM Memory Optimization

This module provides the core functionality for storing and retrieving engrams,
which are memory units representing experiences or knowledge in an LLM system.
"""

import json
import uuid
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class EngramStore:
    """
    CRUD operations for engrams.

    Stores engrams in memory (or optionally in a file) and provides
    create, read, update, delete operations.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the engram store.

        Args:
            storage_path: Optional path to a file for persistent storage.
                         If None, engrams are stored in memory only.
        """
        self.storage_path = storage_path
        self._engrams: Dict[str, Dict] = {}

        if storage_path:
            self._load_from_file()

    def _load_from_file(self) -> None:
        """Load engrams from the storage file if it exists."""
        try:
            with open(self.storage_path, 'r') as f:
                for line in f:
                    engram = json.loads(line.strip())
                    self._engrams[engram['id']] = engram
        except FileNotFoundError:
            # If file doesn't exist, start with empty store
            pass
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in storage file: {e}")

    def _save_to_file(self) -> None:
        """Save all engrams to the storage file."""
        if not self.storage_path:
            return

        with open(self.storage_path, 'w') as f:
            for engram in self._engrams.values():
                f.write(json.dumps(engram) + '\n')

    def add_engram(self, engram: Dict[str, Any]) -> str:
        """
        Add a new engram to the store.

        Args:
            engram: Dictionary representing the engram to add.
                   Must conform to the engram schema.

        Returns:
            The ID of the added engram.

        Raises:
            ValueError: If the engram is missing required fields.
        """
        # Validate required fields
        required_fields = ['id', 'timestamp', 'embedding']
        for field in required_fields:
            if field not in engram:
                raise ValueError(f"Missing required field: {field}")

        # Generate ID if not provided
        if not engram['id']:
            engram['id'] = str(uuid.uuid4())

        # Ensure timestamp is present
        if not engram['timestamp']:
            engram['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

        # Store the engram
        self._engrams[engram['id']] = engram

        # Persist if storage path is configured
        if self.storage_path:
            self._save_to_file()

        return engram['id']

    def get_engram(self, engram_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an engram by its ID.

        Args:
            engram_id: The ID of the engram to retrieve.

        Returns:
            The engram dictionary if found, None otherwise.
        """
        return self._engrams.get(engram_id)

    def update_engram(self, engram_id: str, engram: Dict[str, Any]) -> bool:
        """
        Update an existing engram.

        Args:
            engram_id: The ID of the engram to update.
            engram: The updated engram data.

        Returns:
            True if the engram was updated, False if not found.
        """
        if engram_id not in self._engrams:
            return False

        # Preserve the ID and timestamp if not provided in update
        if 'id' not in engram:
            engram['id'] = engram_id
        if 'timestamp' not in engram:
            engram['timestamp'] = self._engrams[engram_id]['timestamp']

        self._engrams[engram_id] = engram

        # Persist if storage path is configured
        if self.storage_path:
            self._save_to_file()

        return True

    def delete_engram(self, engram_id: str) -> bool:
        """
        Delete an engram by its ID.

        Args:
            engram_id: The ID of the engram to delete.

        Returns:
            True if the engram was deleted, False if not found.
        """
        if engram_id not in self._engrams:
            return False

        del self._engrams[engram_id]

        # Persist if storage path is configured
        if self.storage_path:
            self._save_to_file()

        return True

    def list_engrams(self) -> List[Dict[str, Any]]:
        """
        List all engrams in the store.

        Returns:
            A list of all engram dictionaries.
        """
        return list(self._engrams.values())


class EngramIndex:
    """
    Similarity search index for engrams.

    Provides functionality to search for engrams based on embedding similarity
    using cosine similarity.
    """

    def __init__(self, embedding_dim: int):
        """
        Initialize the engram index.

        Args:
            embedding_dim: The dimensionality of the embedding vectors.
        """
        self.embedding_dim = embedding_dim
        # Store embeddings as a list of (id, embedding) tuples
        self._index: List[Tuple[str, np.ndarray]] = []

    def add_engram(self, engram_id: str, embedding: List[float]) -> None:
        """
        Add an engram embedding to the index.

        Args:
            engram_id: The ID of the engram.
            embedding: The embedding vector as a list of floats.
        """
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {len(embedding)}")

        # Convert to numpy array for efficient computation
        embedding_array = np.array(embedding, dtype=np.float32)
        self._index.append((engram_id, embedding_array))

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar engrams based on cosine similarity.

        Args:
            query_embedding: The query embedding vector.
            k: The number of results to return.

        Returns:
            A list of tuples (engram_id, similarity_score) sorted by similarity in descending order.
        """
        if len(query_embedding) != self.embedding_dim:
            raise ValueError(f"Query embedding dimension mismatch. Expected {self.embedding_dim}, got {len(query_embedding)}")

        query_array = np.array(query_embedding, dtype=np.float32)

        # Compute cosine similarity for each indexed embedding
        similarities = []
        for engram_id, embedding in self._index:
            # Cosine similarity: dot product divided by product of norms
            norm_query = np.linalg.norm(query_array)
            norm_embedding = np.linalg.norm(embedding)

            if norm_query == 0 or norm_embedding == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query_array, embedding) / (norm_query * norm_embedding)

            similarities.append((engram_id, float(similarity)))

        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]


# Example usage (for testing purposes)
if __name__ == "__main__":
    # Example of how to use the EngramStore and EngramIndex
    store = EngramStore()
    index = EngramIndex(embedding_dim=3)  # Using 3-dim for example

    # Create a sample engram
    engram = {
        "id": "",
        "timestamp": "",
        "embedding": [0.1, 0.2, 0.3],
        "metadata": {
            "source": "example",
            "tags": ["test"],
            "score": 0.95
        }
    }

    # Add to store
    engram_id = store.add_engram(engram)
    print(f"Added engram with ID: {engram_id}")

    # Add to index
    index.add_engram(engram_id, engram["embedding"])

    # Search
    results = index.search([0.1, 0.2, 0.3], k=1)
    print(f"Search results: {results}")

    # Retrieve
    retrieved = store.get_engram(engram_id)
    print(f"Retrieved engram: {retrieved}")