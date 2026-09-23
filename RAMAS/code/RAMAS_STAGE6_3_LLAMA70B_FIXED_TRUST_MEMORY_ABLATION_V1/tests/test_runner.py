"""Run-boundary and provider guards; these tests never contact an LLM."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_experiment as run
from stage63lib.fixture import build_fixture


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.cfg=run.load_config()
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def args(self,**kwargs):
        a=dict(output=self.root/'out',project_root=self.root,full_dataset=None,
               resume=False,demo=True,audit_only=False)
        a.update(kwargs)
        return SimpleNamespace(**a)

    def test_provider_digest_or_serving_change_rejected_before_calls(self):
        current=dict(model_digest=self.cfg['expected_model_digest'],model='llama3.3:70b',
                     options=dict(seed=16061),ollama_version='fixed')
        audit={'adaptive_reference_identity':dict(provider_identity=copy.deepcopy(current),
               python=run.platform.python_version(),numpy=run.np.__version__,pandas=run.pd.__version__)}
        run.require_matched_provider(self.cfg,current,audit)
        bad=copy.deepcopy(current);bad['model_digest']='different'
        with self.assertRaisesRegex(RuntimeError,'digest differs'):
            run.require_matched_provider(self.cfg,bad,audit)
        bad=copy.deepcopy(current);bad['options']['seed']=99
        with self.assertRaisesRegex(RuntimeError,'serving protocols'):
            run.require_matched_provider(self.cfg,bad,audit)
        audit['adaptive_reference_identity']['numpy']='other-version'
        with self.assertRaisesRegex(RuntimeError,'runtime differs'):
            run.require_matched_provider(self.cfg,current,audit)

    def test_audit_only_never_constructs_ollama(self):
        period,base,accounting,risk=build_fixture()
        cfg=copy.deepcopy(self.cfg);cfg['expected_rows']=len(period.frame)
        with patch.object(run,'verify_package',return_value='controlled-manifest'), \
             patch.object(run,'load_config',return_value=cfg), \
             patch('stage63lib.inputs.load_inputs',return_value=(period,base,accounting,risk,{'fixture':True})), \
             patch('stage63lib.scientific_audit.audit_scientific_inputs',return_value=({'kind':'CONTROLLED_TEST'},[])), \
             patch('stage63lib.provider.OllamaProvider',side_effect=AssertionError('Provider must not be constructed')):
            self.assertEqual(run.execute(self.args(demo=False,audit_only=True)),0)
        result=json.loads((self.root/'out/AUDIT_ONLY_COMPLETE.json').read_text())
        self.assertEqual(result['new_llm_calls'],0)
        self.assertFalse((self.root/'out/00_CONTRACT.json').exists())

    def test_source_failure_never_constructs_ollama(self):
        with patch.object(run,'verify_package',return_value='controlled-manifest'), \
             patch('stage63lib.inputs.load_inputs',side_effect=RuntimeError('corrupt source')), \
             patch('stage63lib.provider.OllamaProvider',side_effect=AssertionError('Provider must not be constructed')):
            with self.assertRaisesRegex(RuntimeError,'corrupt source'):
                run.execute(self.args(demo=False))

    def test_output_cannot_overwrite_stage62_reference(self):
        target=self.root/self.cfg['adaptive_reference_relative']
        with patch.object(run,'verify_package',return_value='controlled-manifest'):
            with self.assertRaisesRegex(ValueError,'historical source'):
                run.execute(self.args(output=target))

    def test_demo_complete_resume_preserves_contract_and_calls(self):
        with patch.object(run,'verify_package',return_value='controlled-manifest'):
            self.assertEqual(run.execute(self.args()),0)
            output=self.root/'out'
            contract=(output/'00_CONTRACT.json').read_bytes()
            calls={p.name:p.read_bytes() for p in (output/'calls').glob('*.json')}
            self.assertEqual(len(calls),92)
            c=json.loads(contract)
            self.assertEqual(c['fixed_beta'],.05)
            self.assertFalse(c['adaptive_beta_updates_enabled'])
            self.assertFalse(c['pretrained_llama_weights_updated'])
            self.assertEqual(c['identity']['evidence_kind'],'CONTROLLED_NON_ECONOMIC')
            self.assertEqual(run.execute(self.args(resume=True)),0)
            self.assertEqual((output/'00_CONTRACT.json').read_bytes(),contract)
            self.assertEqual({p.name:p.read_bytes() for p in (output/'calls').glob('*.json')},calls)
            with self.assertRaisesRegex(RuntimeError,'fresh output'):
                run.execute(self.args(audit_only=True,resume=True))
            final=json.loads((output/'10_FINAL_STATUS.json').read_text())
            self.assertFalse(final['continuous_learning_established'])
            self.assertFalse(final['fully_verified_forecast_and_execution_timing'])
            self.assertEqual(final['fixed_beta'],.05)


if __name__=='__main__':
    unittest.main()
