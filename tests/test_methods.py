import unittest

from cyclic import methods, okura
from cyclic.identify import construct_ratios


class MethodsTests(unittest.TestCase):
    def test_core_okura_identities_close_both_years(self):
        for year in okura.YEARS:
            checks = methods.table_checks(year)
            self.assertTrue(checks.core_ok, msg=checks.core.loc[~checks.core["ok"]].to_string())

    def test_closed_form_capture_is_lfv_over_lfa(self):
        for year in okura.YEARS:
            shares = construct_ratios(year)
            t5 = okura.TABLE5_1000M3[year]
            lfa = (t5["LfA_North"] + t5["LfA_South"]) * 1000.0
            lfv = t5["LFV_pump"] * 1000.0
            self.assertAlmostEqual(shares["pool_capture_share"], lfv / lfa, places=6)
            self.assertAlmostEqual(
                shares["pool_capture_share"] + shares["kx_ratio"] + shares["ky_ratio"]
                + shares["kz_ratio"] + shares["ground_ratio"], 1.0, places=6)
            self.assertAlmostEqual(
                shares["kx_ratio"] + shares["ky_ratio"],
                (1.0 - shares["pool_capture_share"]) * shares["neighbour_of_remain"], places=6)

    def test_methods_pack_writes_files(self):
        import os
        import tempfile
        folder = tempfile.mkdtemp()
        out = methods.write_paper_pack(2022, folder=folder)
        self.assertTrue(os.path.isfile(out.methods))
        self.assertTrue(os.path.isfile(out.figure))
        self.assertTrue(out.core_ok)
        with open(out.methods, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("Okura 3", text)
        self.assertIn("M1", text)
        self.assertIn("M6", text)
        self.assertIn("without soil properties", text.lower())
        self.assertIn("@register_lateral", text)
        self.assertIn("farmer model", text.lower())
        self.assertIn("equation builder", text.lower())
        self.assertIn("Xu et al. (2017)", text)
        self.assertIn("1.29", text)


if __name__ == "__main__":
    unittest.main()
