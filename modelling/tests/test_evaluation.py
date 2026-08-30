import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from modelling.evaluation import verify_saved_result_parity


class SavedResultParityTests(unittest.TestCase):
    def test_saved_forecasts_reproduce_reported_metrics(self):
        project_root = Path(__file__).resolve().parents[2]
        results = project_root / "data" / "processed" / "model_results"
        parity = verify_saved_result_parity(results)

        self.assertEqual(len(parity), 7)
        self.assertTrue(parity["matches_saved_result"].all())


if __name__ == "__main__":
    unittest.main()
