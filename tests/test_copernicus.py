"""Copernicus credential helpers (no network)."""

import json
import os
import tempfile
import unittest
from pathlib import Path


class CopernicusConfigTests(unittest.TestCase):
    def test_save_and_load_credentials(self):
        from cyclic import copernicus

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["CYCLIC_CONFIG_DIR"] = tmp
            try:
                path = copernicus.save_credentials("user@example.com", "secret")
                self.assertTrue(Path(path).is_file())
                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)
                self.assertEqual(payload["username"], "user@example.com")
                loaded = copernicus.load_credentials()
                self.assertEqual(loaded["username"], "user@example.com")
                self.assertEqual(loaded["password"], "secret")
                self.assertTrue(copernicus.clear_credentials())
                self.assertIsNone(copernicus.load_credentials())
            finally:
                os.environ.pop("CYCLIC_CONFIG_DIR", None)

    def test_imbanuma_bbox_covers_paddies(self):
        from cyclic.copernicus import district_bbox, IMBANUMA_BBOX
        from cyclic import data

        west, south, east, north = district_bbox(pad_deg=0.0)
        frame = data.paddies()
        self.assertLessEqual(west, float(frame["lon"].min()) + 1e-6)
        self.assertGreaterEqual(east, float(frame["lon"].max()) - 1e-6)
        self.assertLessEqual(south, float(frame["lat"].min()) + 1e-6)
        self.assertGreaterEqual(north, float(frame["lat"].max()) - 1e-6)
        self.assertEqual(len(IMBANUMA_BBOX), 4)


if __name__ == "__main__":
    unittest.main()
