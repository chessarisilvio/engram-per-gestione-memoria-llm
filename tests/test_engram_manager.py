#!/usr/bin/env python3
"""
Unit tests for the Engram Manager module.
"""

import unittest
import tempfile
import os
import json
from engram_manager import EngramStore, EngramIndex


class TestEngramStore(unittest.TestCase):
    """Test cases for the EngramStore class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary file for storage testing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        self.temp_file.close()
        self.store = EngramStore(storage_path=self.temp_file.name)

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Remove the temporary file
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_add_engram_without_id(self):
        """Test adding an engram without providing an ID (should generate one)."""
        engram = {
            "id": "",
            "timestamp": "",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {
                "source": "test",
                "tags": ["unit_test"],
                "score": 0.8
            }
        }

        engram_id = self.store.add_engram(engram)

        # Check that an ID was generated
        self.assertIsNotNone(engram_id)
        self.assertNotEqual(engram_id, "")

        # Check that the engram was stored correctly
        retrieved = self.store.get_engram(engram_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["id"], engram_id)
        self.assertEqual(retrieved["embedding"], [0.1, 0.2, 0.3])
        self.assertEqual(retrieved["metadata"]["source"], "test")

    def test_add_engram_with_id(self):
        """Test adding an engram with a provided ID."""
        custom_id = "test-engram-123"
        engram = {
            "id": custom_id,
            "timestamp": "2026-06-11T21:22:00Z",
            "embedding": [0.5, 0.6, 0.7],
            "metadata": {
                "source": "manual",
                "tags": ["test", "manual"],
                "score": 0.9
            }
        }

        engram_id = self.store.add_engram(engram)

        # Check that the provided ID was used
        self.assertEqual(engram_id, custom_id)

        # Check that the engram was stored correctly
        retrieved = self.store.get_engram(engram_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["id"], custom_id)
        self.assertEqual(retrieved["timestamp"], "2026-06-11T21:22:00Z")
        self.assertEqual(retrieved["embedding"], [0.5, 0.6, 0.7])

    def test_get_engram_not_found(self):
        """Test retrieving a non-existent engram."""
        result = self.store.get_engram("non-existent-id")
        self.assertIsNone(result)

    def test_update_engram(self):
        """Test updating an existing engram."""
        # Add an engram
        engram = {
            "id": "",
            "timestamp": "2026-06-11T21:22:00Z",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {
                "source": "original",
                "tags": ["original"],
                "score": 0.5
            }
        }

        engram_id = self.store.add_engram(engram)

        # Update the engram
        updated_engram = {
            "id": engram_id,
            "timestamp": "2026-06-11T21:23:00Z",
            "embedding": [0.4, 0.5, 0.6],
            "metadata": {
                "source": "updated",
                "tags": ["updated", "modified"],
                "score": 0.8,
                "custom": {"version": 2}
            }
        }

        result = self.store.update_engram(engram_id, updated_engram)
        self.assertTrue(result)

        # Check that the engram was updated
        retrieved = self.store.get_engram(engram_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["timestamp"], "2026-06-11T21:23:00Z")
        self.assertEqual(retrieved["embedding"], [0.4, 0.5, 0.6])
        self.assertEqual(retrieved["metadata"]["source"], "updated")
        self.assertEqual(retrieved["metadata"]["score"], 0.8)
        self.assertEqual(retrieved["metadata"]["custom"]["version"], 2)

    def test_update_engram_not_found(self):
        """Test updating a non-existent engram."""
        engram = {
            "id": "non-existent-id",
            "timestamp": "2026-06-11T21:22:00Z",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {}
        }

        result = self.store.update_engram("non-existent-id", engram)
        self.assertFalse(result)

    def test_delete_engram(self):
        """Test deleting an existing engram."""
        # Add an engram
        engram = {
            "id": "",
            "timestamp": "2026-06-11T21:22:00Z",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {}
        }

        engram_id = self.store.add_engram(engram)

        # Verify it exists
        self.assertIsNotNone(self.store.get_engram(engram_id))

        # Delete it
        result = self.store.delete_engram(engram_id)
        self.assertTrue(result)

        # Verify it's gone
        self.assertIsNone(self.store.get_engram(engram_id))

    def test_delete_engram_not_found(self):
        """Test deleting a non-existent engram."""
        result = self.store.delete_engram("non-existent-id")
        self.assertFalse(result)

    def test_list_engrams(self):
        """Test listing all engrams."""
        # Add multiple engrams
        engram1 = {
            "id": "",
            "timestamp": "2026-06-11T21:22:00Z",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {"source": "test1"}
        }

        engram2 = {
            "id": "",
            "timestamp": "2026-06-11T21:23:00Z",
            "embedding": [0.4, 0.5, 0.6],
            "metadata": {"source": "test2"}
        }

        id1 = self.store.add_engram(engram1)
        id2 = self.store.add_engram(engram2)

        # List all engrams
        engrams = self.store.list_engrams()
        self.assertEqual(len(engrams), 2)

        # Check that both engrams are present
        ids = [e["id"] for e in engrams]
        self.assertIn(id1, ids)
        self.assertIn(id2, ids)

    def test_persistence(self):
        """Test that engrams are persisted to file and loaded correctly."""
        # Add an engram
        engram = {
            "id": "",
            "timestamp": "2026-06-11T21:22:00Z",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {
                "source": "persistence_test",
                "tags": ["persistent"]
            }
        }

        engram_id = self.store.add_engram(engram)

        # Create a new store instance with the same file
        new_store = EngramStore(storage_path=self.temp_file.name)

        # Check that the engram was loaded
        retrieved = new_store.get_engram(engram_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["id"], engram_id)
        self.assertEqual(retrieved["embedding"], [0.1, 0.2, 0.3])
        self.assertEqual(retrieved["metadata"]["source"], "persistence_test")


class TestEngramIndex(unittest.TestCase):
    """Test cases for the EngramIndex class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.index = EngramIndex(embedding_dim=3)

    def test_add_engram(self):
        """Test adding an engram to the index."""
        engram_id = "test-engram-1"
        embedding = [0.1, 0.2, 0.3]

        self.index.add_engram(engram_id, embedding)

        # Check that the engram was added (we can't directly access _index in a clean way,
        # but we can test through search)
        results = self.index.search([0.1, 0.2, 0.3], k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], engram_id)
        # Should be very similar (cosine similarity of 1.0 for identical vectors)
        self.assertAlmostEqual(results[0][1], 1.0, places=5)

    def test_add_engram_wrong_dimension(self):
        """Test adding an engram with wrong embedding dimension."""
        engram_id = "test-engram-1"
        embedding = [0.1, 0.2]  # Only 2 dimensions, expected 3

        with self.assertRaises(ValueError) as context:
            self.index.add_engram(engram_id, embedding)

        self.assertIn("Embedding dimension mismatch", str(context.exception))

    def test_search(self):
        """Test searching for similar engrams."""
        # Add some test engrams
        self.index.add_engram("engram1", [1.0, 0.0, 0.0])
        self.index.add_engram("engram2", [0.0, 1.0, 0.0])
        self.index.add_engram("engram3", [0.0, 0.0, 1.0])

        # Search for a vector similar to engram1
        results = self.index.search([1.0, 0.0, 0.0], k=2)

        # Should return engram1 first with high similarity
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "engram1")
        self.assertAlmostEqual(results[0][1], 1.0, places=5)

        # Second result should be one of the others with low similarity
        self.assertLess(results[1][1], 0.5)  # Orthogonal vectors have 0 similarity

    def test_search_wrong_dimension(self):
        """Test searching with wrong query embedding dimension."""
        # Add an engram first
        self.index.add_engram("test", [1.0, 0.0, 0.0])

        # Try to search with wrong dimension
        with self.assertRaises(ValueError) as context:
            self.index.search([1.0, 0.0], k=1)  # Only 2 dimensions

        self.assertIn("Query embedding dimension mismatch", str(context.exception))

    def test_search_empty_index(self):
        """Test searching in an empty index."""
        results = self.index.search([1.0, 0.0, 0.0], k=5)
        self.assertEqual(len(results), 0)

    def test_search_k_larger_than_index(self):
        """Test searching when k is larger than the number of engrams."""
        # Add fewer engrams than k
        self.index.add_engram("engram1", [1.0, 0.0, 0.0])
        self.index.add_engram("engram2", [0.0, 1.0, 0.0])

        # Search for k=5 but we only have 2 engrams
        results = self.index.search([1.0, 0.0, 0.0], k=5)

        # Should return only 2 results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "engram1")


if __name__ == "__main__":
    unittest.main()