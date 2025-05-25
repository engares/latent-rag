import unittest
from data.data_processing import clean_text, add_noise, build_dae_dataset, build_contrastive_pairs

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


if __name__ == "__main__":
    unittest.main()
