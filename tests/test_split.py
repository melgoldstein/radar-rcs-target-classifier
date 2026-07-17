import tempfile, unittest
from pathlib import Path
from src.generate_returns import generate
from src.train_classifier import train
class ClassifierChecks(unittest.TestCase):
    def test_group_disjoint_training(self):
        with tempfile.TemporaryDirectory() as directory:
            results=train(generate(),Path(directory))
            self.assertEqual(len(results),2)
            self.assertGreater(results['random_forest']['classification']['accuracy'],.7)
