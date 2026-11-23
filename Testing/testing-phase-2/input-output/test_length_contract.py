"""Metadata regression tests, never substitutes for pretrained inference evidence."""
import io
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from core.model_lengths import ModelLengthError, contract_for_spec, length_contract, validate_length


class LengthContractTests(unittest.TestCase):
    def test_correct_2114_window(self):
        contract = length_contract({'train': {'seq_len': 2114}}, {'model_type': 'DilatedConvModel'})
        self.assertEqual(validate_length(contract, 2114)['seq_len'], 2114)

    def test_incorrect_200_rejected(self):
        contract = length_contract({'train': {'seq_len': 2114}})
        with self.assertRaisesRegex(ModelLengthError, 'configured 200.*2,114'):
            validate_length(contract, 200)

    def test_missing_metadata_remains_unknown_even_with_config(self):
        contract = validate_length(length_contract(), 200, require_known=False)
        self.assertIsNone(contract['seq_len'])
        self.assertFalse(contract['verified_compatibility'])
        with self.assertRaisesRegex(ModelLengthError, 'unknown'):
            validate_length(contract, 200)

    def test_other_model_128_reference_window(self):
        contract = length_contract({'train': {'seq_len': 128}}, {'model_type': 'ConvModel'})
        self.assertEqual(validate_length(contract, 128)['seq_len'], 128)
        with self.assertRaises(ModelLengthError):
            validate_length(contract, 2114)

    def test_fully_convolutional_architecture_is_not_called_fixed(self):
        contract = length_contract({'train': {'seq_len': 2114}}, {'model_type': 'DilatedConvModel', 'crop_len': 557})
        self.assertEqual(contract['architecture_length_mode'], 'variable')
        self.assertEqual(contract['workflow_length_mode'], 'fixed_window')

    def test_variable_architecture_without_workflow_window_is_not_certified(self):
        contract = length_contract({}, {'model_type': 'ConvModel'})
        self.assertEqual(contract['architecture_length_mode'], 'variable')
        self.assertIsNone(contract['seq_len'])
        with self.assertRaises(ModelLengthError):
            validate_length(contract, 512)

    def test_supported_mlp_constructor_window(self):
        contract = length_contract({}, {'model_type': 'ConvMLPModel', 'seq_len': 128})
        self.assertEqual(contract['architecture_length_mode'], 'length_dependent')
        self.assertEqual(validate_length(contract, 128)['seq_len'], 128)

    def test_unreviewed_model_seq_len_not_assumed_a_fixed_requirement(self):
        contract = length_contract({}, {'model_type': 'UnreviewedTransformer', 'seq_len': 999})
        self.assertEqual(contract['status'], 'unknown')

    def test_contradictory_native_fields_rejected(self):
        with self.assertRaisesRegex(ModelLengthError, 'Conflicting native'):
            length_contract({'train': {'seq_len': 128}, 'train_seq_len': 256})

    def test_noninteger_lengths_not_silently_truncated(self):
        for value in [True, False, 0, -1, 2114.5, '2114.5', 'unknown']:
            with self.subTest(value=value), self.assertRaises(ModelLengthError):
                validate_length(length_contract({'train': {'seq_len': 2114}}), value)

    def test_metadata_infers_window_when_config_is_omitted(self):
        self.assertEqual(validate_length(length_contract({'train': {'seq_len': 128}}))['seq_len'], 128)

    def test_builtin_demo_is_a_separate_validated_workflow_spec(self):
        contract = contract_for_spec({'source': 'demo', 'seq_len': 96})
        self.assertEqual(contract['seq_len'], 96)
        self.assertEqual(contract['architecture_length_mode'], 'variable')

    def test_safe_metadata_reader_and_file_cache_invalidation(self):
        # These files contain metadata only, not a fake model or predictions.
        import torch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'metadata-only.ckpt'
            torch.save({'data_params': {'train': {'seq_len': 128}}}, path)
            spec = {'source': 'checkpoint', 'path': str(path), 'seq_len': 128}
            self.assertEqual(contract_for_spec(spec)['seq_len'], 128)
            torch.save({'data_params': {}}, path)
            with self.assertRaisesRegex(ModelLengthError, 'unknown'):
                contract_for_spec(spec)


def run_recorded():
    checks = []
    class Recorded(unittest.TextTestResult):
        def addSuccess(self, test):
            super().addSuccess(test)
            checks.append({'id': test._testMethodName, 'status': 'PASS', 'explanation': test.shortDescription() or test._testMethodName})
        def addFailure(self, test, err):
            super().addFailure(test, err)
            checks.append({'id': test._testMethodName, 'status': 'FAIL', 'explanation': self._exc_info_to_string(err,test)})
        def addError(self, test, err):
            super().addError(test, err)
            checks.append({'id': test._testMethodName, 'status': 'FAIL', 'explanation': self._exc_info_to_string(err,test)})
        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            checks.append({'id': test._testMethodName, 'status': 'SKIP', 'explanation': reason})
        def addSubTest(self, test, subtest, err):
            super().addSubTest(test, subtest, err)
            if err:
                checks.append({'id': str(subtest), 'status': 'FAIL', 'explanation': self._exc_info_to_string(err,test)})
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2, resultclass=Recorded).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(LengthContractTests))
    return {'checks': checks, 'tests_run': result.testsRun, 'all_passed': result.wasSuccessful(),
            'scope': 'Metadata/length-policy regressions only. No model predictions are mocked or used.'}


if __name__ == '__main__':
    sys.exit(0 if run_recorded()['all_passed'] else 1)
