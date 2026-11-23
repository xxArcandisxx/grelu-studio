"""Scientific convention checks: target scale, missing bases, and SNV coordinates."""
import math
import unittest

import numpy as np

from scripts.benchmark_gm12878 import allele_sequences, dnase_label


class BiologicalBenchmarkTests(unittest.TestCase):
    def test_measured_target_is_log_of_sum_not_sum_of_logs(self):
        signal = np.zeros(1000, dtype=np.float32)
        signal[0], signal[999], signal[10] = 3, 4, np.nan
        self.assertAlmostEqual(dnase_label(signal), math.log(8), places=6)
        self.assertNotAlmostEqual(dnase_label(signal), math.log(4) + math.log(5))
        self.assertEqual(dnase_label(np.full(1000, np.nan)), 0)

    def test_bad_signal_is_rejected(self):
        for signal in [np.zeros(999), np.full(1000, -1), np.full(1000, np.inf)]:
            with self.assertRaises(ValueError):
                dnase_label(signal)

    def test_one_based_variant_and_even_window_match_grelu_convention(self):
        # Position 1500 is index 1499; unique flanks expose off-by-one extraction.
        chrom = 'C' * 1499 + 'G' + 'T' * 1500
        first, second = allele_sequences({'chr1': chrom}, 'chr1', 1500, 'G', 'A')
        self.assertEqual(first, 'C' * 1056 + 'G' + 'T' * 1057)
        self.assertEqual(second, 'C' * 1056 + 'A' + 'T' * 1057)
        # Original table allele1 need not be the reference-genome allele.
        swapped = allele_sequences({'chr1': chrom}, 'chr1', 1500, 'A', 'G')
        self.assertEqual(swapped, (second, first))
        with self.assertRaises(ValueError):
            allele_sequences({'chr1': chrom}, 'chr1', 1500, 'C', 'A')


if __name__ == '__main__':
    unittest.main()
