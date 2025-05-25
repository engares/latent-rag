import unittest
from data.data_processing import clean_text, add_noise, build_dae_dataset, build_contrastive_pairs
import torch

class TestDataProcessing(unittest.TestCase):

    def test_clean_text(self):
        self.assertEqual(clean_text("  Hello   World  "), "Hello World")
        self.assertEqual(clean_text("\nHello\tWorld\n"), "Hello World")

    def test_add_noise(self):
        text = "This is a test sentence"
        noisy_text = add_noise(text, removal_prob=0.5, swap_prob=0.5)
        self.assertNotEqual(text, noisy_text)  # Ensure noise is added

    def test_build_dae_dataset(self):
        samples = ["This is a test", "Another example"]
        dataset = build_dae_dataset(samples)
        self.assertEqual(len(dataset), len(samples))
        for item in dataset:
            self.assertIn("input", item)
            self.assertIn("target", item)

    def test_dae_dataset_uniqueness(self):
        samples = ["This is a test", "Another example", "Yet another test"]
        dataset = build_dae_dataset(samples)
        inputs = [item["input"] for item in dataset]
        self.assertEqual(len(inputs), len(set(inputs)), "All inputs in the dataset should be unique")

    def test_build_contrastive_pairs(self):
        dataset = [ # ai generated test
            {
                "query": "What is AI?",
                "positive_passages": [{"text": "AI is artificial intelligence."}],
                "negative_passages": [
                    {"text": "AI is a type of food."},
                    {"text": "AI is a movie."}
                ]
            }
        ]
        pairs = build_contrastive_pairs(dataset, max_negatives=1)
        self.assertEqual(len(pairs), 1)
        self.assertIn("query", pairs[0])
        self.assertIn("positive", pairs[0])
        self.assertIn("negative", pairs[0])

    def test_contrastive_pairs_uniqueness(self):
        dataset = [
            {
                "query": "What is AI?",
                "positive_passages": [{"text": "AI is artificial intelligence."}],
                "negative_passages": [
                    {"text": "AI is a type of food."},
                    {"text": "AI is a movie."}
                ]
            },
            {
                "query": "What is ML?",
                "positive_passages": [{"text": "ML is machine learning."}],
                "negative_passages": [
                    {"text": "ML is a type of food."},
                    {"text": "ML is a movie."}
                ]
            }
        ]
        pairs = build_contrastive_pairs(dataset, max_negatives=1)
        unique_pairs = {tuple(pair.values()) for pair in pairs}
        self.assertEqual(len(pairs), len(unique_pairs), "All contrastive pairs should be unique")

    def test_load_and_compare_embeddings(self):
        vae_embeddings = torch.load("./data/SQUAD/squad_vae_embeddings.pt")
        dae_embeddings = torch.load("./data/SQUAD/squad_dae_embeddings.pt")

        self.assertIsNotNone(vae_embeddings, "VAE embeddings should not be None")
        self.assertIsNotNone(dae_embeddings, "DAE embeddings should not be None")

        if isinstance(vae_embeddings, dict):
            print("VAE embeddings keys:", vae_embeddings.keys())
        if isinstance(dae_embeddings, dict):
            print("DAE embeddings keys:", dae_embeddings.keys())

        if isinstance(vae_embeddings, dict):
            vae_embeddings = vae_embeddings.get("input", None)
        if isinstance(dae_embeddings, dict):
            dae_embeddings = dae_embeddings.get("input", None)

        self.assertIsInstance(vae_embeddings, torch.Tensor, "VAE embeddings should be a tensor")
        self.assertIsInstance(dae_embeddings, torch.Tensor, "DAE embeddings should be a tensor")

        self.assertFalse(torch.equal(vae_embeddings, dae_embeddings), "VAE and DAE embeddings should be different")

    def test_load_and_compare_contrastive_embeddings(self):
        contrastive_embeddings = torch.load("./data/SQUAD/squad_contrastive_embeddings.pt")

        self.assertIsNotNone(contrastive_embeddings, "Contrastive embeddings should not be None")

        if isinstance(contrastive_embeddings, dict):
            print("Contrastive embeddings keys:", contrastive_embeddings.keys())

        if isinstance(contrastive_embeddings, dict):
            query_embeddings = contrastive_embeddings.get("query", None)
            positive_embeddings = contrastive_embeddings.get("positive", None)
            negative_embeddings = contrastive_embeddings.get("negative", None)

        self.assertIsInstance(query_embeddings, torch.Tensor, "Query embeddings should be a tensor")
        self.assertIsInstance(positive_embeddings, torch.Tensor, "Positive embeddings should be a tensor")
        self.assertIsInstance(negative_embeddings, torch.Tensor, "Negative embeddings should be a tensor")
        self.assertFalse(torch.equal(query_embeddings, positive_embeddings), "Query and positive embeddings should be different")
        self.assertFalse(torch.equal(query_embeddings, negative_embeddings), "Query and negative embeddings should be different")
        self.assertFalse(torch.equal(positive_embeddings, negative_embeddings), "Positive and negative embeddings should be different")



if __name__ == "__main__":
    unittest.main()
