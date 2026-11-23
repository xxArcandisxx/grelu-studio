"""Existing lightweight input/account regressions; no model computations."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import unittest

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[3];sys.path.insert(0,str(ROOT))


def main(out):
    spec=importlib.util.spec_from_file_location('gate6_support',BASE/'verify_storage.py')
    support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
    support.configure(out)
    rows=[]
    class Results(unittest.TextTestResult):
        def addSuccess(self,test):
            super().addSuccess(test);rows.append(dict(id=test.id(),status='PASS',mode='EXISTING_REGRESSION',workflow='input/account regression',evidence='unittest.log'))
        def addFailure(self,test,err):
            super().addFailure(test,err);rows.append(dict(id=test.id(),status='FAIL',mode='EXISTING_REGRESSION',workflow='input/account regression',evidence='unittest.log'))
        def addError(self,test,err):
            super().addError(test,err);rows.append(dict(id=test.id(),status='FAIL',mode='EXISTING_REGRESSION',workflow='input/account regression',evidence='unittest.log'))
        def addSkip(self,test,reason):
            super().addSkip(test,reason);rows.append(dict(id=test.id(),status='SKIP',reason=reason,mode='EXISTING_REGRESSION',workflow='input/account regression',evidence='unittest.log'))
    suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_dataset_overview','tests.test_example_loading','tests.test_account_settings'])
    with (out/'unittest.log').open('w') as log:result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=Results).run(suite)
    support.save(out/'results.json',dict(checks=rows,tests_run=result.testsRun,training_executed=False))
    print('Existing regression tests:',result.testsRun,'failures:',len(result.failures),'errors:',len(result.errors),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
