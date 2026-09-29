import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from cyclic.imitation import LearnResult
from cyclic.validation import evaluate, output_label, split


class YearSplitTests(unittest.TestCase):
    def test_other_year_outputs_are_labelled_without_relaxing_training_split(self):
        trained = SimpleNamespace(training_years=(2021,))
        self.assertEqual(output_label(trained, 2021), "training-year diagnostic")
        self.assertEqual(output_label(trained, 2022), "unseen-year output (learning off)")
        self.assertIn("unknown", output_label(SimpleNamespace(training_years=None), 2022))
        self.assertEqual(output_label(SimpleNamespace(), 2022), "untrained-policy output")
        # Package et_net with compatible weights is not "untrained".
        et = SimpleNamespace(name="et_net", training_years=())
        with patch("cyclic.et_net.weights_status",
                   return_value=SimpleNamespace(compatible=True, train_years=(2021, 2022))):
            self.assertEqual(output_label(et, 2022), "training-year diagnostic")
            self.assertEqual(et.training_years, (2021, 2022))
        with self.assertRaises(ValueError):
            split(2022, 2021, policies=trained)

    def test_defaults_and_forbidden_splits(self):
        self.assertEqual(split(), (2021, 2022))
        self.assertEqual(split(2022), (2022, 2021))
        with self.assertRaises(ValueError):
            split(2022, 2022)
        with self.assertRaises(ValueError):
            split(2021, policies=SimpleNamespace(training_years=(2022,)))
        with self.assertRaises(ValueError):
            split(2021, policies=SimpleNamespace(training_years=None))
        with self.assertRaises(ValueError):
            split(2021, observed=(pd.DataFrame({"date": ["2022-05-01"]}), None))

    def test_final_evaluation_uses_copy_and_preserves_training_selection(self):
        policy = SimpleNamespace(value=1)
        training_score = SimpleNamespace(loss=0.2)
        result = LearnResult("test", {"North": policy}, training_score, pd.DataFrame([{"loss": .2}]))
        held_out = SimpleNamespace(blocks="held-out blocks")

        def model(year, agents, physics):
            self.assertEqual(year, 2022)
            agents["North"].value = 999
            return SimpleNamespace(run=lambda: held_out)

        with patch("cyclic.model.CyclicModel", side_effect=model), \
             patch("cyclic.imitation.load_observed", return_value=("held-out observations", None, "test")) as observed, \
             patch("cyclic.imitation.score", return_value=SimpleNamespace(loss=9)) as score, \
             patch("cyclic.flyover.from_results", return_value=SimpleNamespace(fidelity=.5)):
            final = evaluate(result, 2021, 2022)
        observed.assert_called_once_with(2022)
        score.assert_called_once_with("held-out blocks", "held-out observations")
        self.assertIs(final.notes["training_score"], training_score)
        self.assertEqual(final.policies["North"].value, 1)
        self.assertEqual(final.policies["North"].training_years, (2021,))
        self.assertEqual(final.score.loss, 9)
        self.assertFalse(final.notes["evaluation_learning_enabled"])


if __name__ == "__main__":
    unittest.main()
