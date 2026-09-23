"""Causal/accounting/replay tests with labelled synthetic inputs only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from stage63lib.engine import run_pair
from stage63lib.fixture import build_fixture
from stage63lib.journal import Journal, digest
from stage63lib.legacy import legacy, state_vector
from suite64.engine import _new_arm, payload_for, project_action, retrieval_evidence, run_arm
ROOT = Path(__file__).resolve().parents[1]

def spec(name='memory', **changes):
    value = dict(name=name, advisor='llama', trust='fixed', memory='expanding', risk='standard', fixed_beta=.05)
    value.update(changes)
    return value

class Provider:
    def __init__(self, force_action=None, bad_citation=False):
        self.calls, self.payloads = 0, []
        self.force_action, self.bad_citation = force_action, bad_citation
    def complete(self, payload):
        self.calls += 1
        self.payloads.append(copy.deepcopy(payload))
        visible = payload['memory']['similar_completed_episodes']
        action = self.force_action or legacy.deterministic_action(payload['state'])
        citations = ['unseen-episode'] if self.bad_citation else ([visible[0]['episode_id']] if visible else [])
        answer = dict(action=action, confidence=.7, reason_codes=['INSUFFICIENT_EVIDENCE'], cited_memory_ids=citations)
        return json.dumps(answer), dict(done_reason='stop', prompt_eval_count=101, eval_count=17), .25

class InMemoryJournal:
    def __init__(self, provider):
        self.provider, self.new_calls, self.reused_calls, self.requests = provider, 0, 0, {}
    def complete(self, key, payload):
        self.new_calls += 1
        self.requests[key] = copy.deepcopy(payload)
        return self.provider.complete(payload)

class EngineTests(unittest.TestCase):
    def setUp(self):
        self.period, self.base, self.accounting, self.risk = build_fixture()
        self.config = json.loads((ROOT/'source_config.json').read_text())
        self.config['checkpoint_every'] = 100
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def run_spec(self, variant, core=None, provider=None):
        journal = InMemoryJournal(provider or Provider()) if variant['advisor']=='llama' else None
        rows, arm = run_arm(self.period, self.config, self.base, self.accounting, self.risk,
                            journal, self.output/variant['name'], variant, core_desired=core)
        return rows, arm, journal
    def episode(self, state, name, decision, outcome, regime=None, advantage=.01):
        return dict(episode_id=name, decision_date=decision, return_date=outcome,
            hard_regime=regime or state['hard_regime'], state_vector=state_vector(state).tolist(),
            action='BTC', confidence=.7, shadow_log_advantage_vs_ramoe=advantage, asset_return=.02)
    def first_state(self):
        return payload_for(_new_arm(self.config,spec()), self.period,0,self.config,self.base,self.risk,spec())['state']

    def test_default_fixed_arms_match_legacy_exactly(self):
        previous = InMemoryJournal(Provider())
        old_rows, old_arms = run_pair(self.period,self.config,self.base,self.accounting,self.risk,previous,self.output/'legacy')
        for name, mode in [('memory','expanding'),('no_memory','none')]:
            rows, arm, journal = self.run_spec(spec(name,memory=mode))
            for i,row in enumerate(rows):
                key = f'{i+1:06d}-{name}'
                self.assertEqual(journal.requests[key],previous.requests[key])
                self.assertEqual(row['request_sha256'],digest(previous.requests[key]))
                for new,old in [('net_return_after_trading_costs','net_return'),('exposure','exposure'),
                                ('pretrade_exposure','pretrade_exposure'),('shadow_net_return','shadow_net_return')]:
                    self.assertEqual(row[new],old_rows[i][name+'_'+old])
                self.assertEqual((row['input_token_count'],row['output_token_count']),(101,17))
            self.assertEqual(arm.ledger.episodes,old_arms[name].ledger.episodes)
            self.assertEqual(arm.wealth,old_arms[name].wealth)

    def test_retrieval_modes_never_filter_private_adaptive_trust(self):
        results=[]
        for mode in ['expanding','none','frozen2021','current_year','pooled']:
            rows,arm,_=self.run_spec(spec('rule_'+mode,advisor='rule',trust='adaptive',memory=mode))
            results.append((arm.ledger.episodes,arm.trust.events,arm.trust.beta,[r['net_return_after_trading_costs'] for r in rows]))
        for result in results[1:]: self.assertEqual(result,results[0])
        self.assertGreater(len(results[0][1]),0)
        self.assertEqual(len(results[0][0]),len(self.period.frame))

    def test_causal_calendar_filters_use_completed_return_dates(self):
        state=self.first_state();state['decision_date']='2022-01-02'
        episodes=[self.episode(state,'old','2021-12-30','2021-12-31'),
                  self.episode(state,'new','2021-12-31','2022-01-01'),
                  self.episode(state,'just-completed','2022-01-01','2022-01-02'),
                  self.episode(state,'future','2022-01-02','2022-01-03'),
                  self.episode(state,'same-decision','2022-01-02','2022-01-02')]
        before=copy.deepcopy(episodes)
        visible=lambda mode:{e['episode_id'] for e in retrieval_evidence(episodes,state,mode,20)['similar_completed_episodes']}
        self.assertEqual(visible('expanding'),{'old','new','just-completed'})
        self.assertEqual(visible('frozen2021'),{'old'})
        self.assertEqual(visible('current_year'),{'new','just-completed'})
        self.assertEqual(visible('none'),set())
        self.assertEqual(episodes,before)
        state['decision_date']='2021-12-31'
        self.assertEqual(visible('frozen2021'),{'old'})

    def test_pooled_preserves_regime_labels_and_distance_ties(self):
        state=self.first_state();state['decision_date']='2022-01-02'
        episodes=[self.episode(state,'b-bull','2021-12-30','2021-12-31'),
                  self.episode(state,'a-bear','2021-12-30','2021-12-31',regime='bear')]
        pooled=retrieval_evidence(episodes,state,'pooled',5)
        normal=retrieval_evidence(episodes,state,'expanding',5)
        self.assertEqual([e['episode_id'] for e in pooled['similar_completed_episodes']],['a-bear','b-bull'])
        self.assertEqual(pooled['similar_completed_episodes'][0]['hard_regime'],'bear')
        self.assertEqual(pooled['summary']['BTC']['count'],2)
        self.assertEqual(normal['summary']['BTC']['count'],1)

    def test_own_drift_cost_and_wealth_continue_across_year(self):
        outputs=[]
        for label,exposure in [('low',.2),('high',.8)]:
            rows,_,_=self.run_spec(spec(label,advisor='abstain',memory='none',risk='none'),np.full(len(self.period.frame),exposure))
            outputs.append(rows)
            self.assertEqual(rows[0]['pretrade_exposure'],0.)
            for i,row in enumerate(rows):
                self.assertAlmostEqual(row['net_return_after_trading_costs'],(1-row['cost_fraction'])*(1+row['exposure']*row['asset_simple_return'])-1,places=14)
                if i:
                    prior=rows[i-1]
                    expected=prior['exposure']*(1+prior['asset_simple_return'])/(1+prior['exposure']*prior['asset_simple_return'])
                    self.assertAlmostEqual(row['pretrade_exposure'],expected,places=14)
                    self.assertEqual(row['wealth_before'],prior['wealth_after'])
            january=next(r for r in rows if r['decision_date']=='2022-01-01')
            self.assertGreater(january['pretrade_exposure'],0)
            self.assertEqual(january['beta'],0.)
        self.assertNotEqual(outputs[0][1]['pretrade_exposure'],outputs[1][1]['pretrade_exposure'])
        self.assertNotEqual(outputs[0][0]['cost_fraction'],outputs[1][0]['cost_fraction'])

    def test_risk_off_removes_projection_but_retains_bounds_and_cost(self):
        class ForbiddenRisk:
            @staticmethod
            def project_exposure(**kwargs): raise AssertionError('No-risk path invoked projector')
        projected=project_action('BTC',.05,.45,.1,None,self.base,ForbiddenRisk(),'none')
        self.assertEqual(projected.exposure,.95*.45+.05)
        self.assertIsNone(projected.ambiguity_cvar)
        normal=project_action('BTC',.05,.45,.1,self.period.scenarios[0],self.base,self.risk)
        self.assertNotEqual(projected.exposure,normal.exposure)
        rows,_,journal=self.run_spec(spec('no_risk',risk='none'),provider=Provider('BTC'))
        preview=journal.requests['000001-no_risk']['safe_exposure_previews']['BTC']
        self.assertEqual(preview['risk_limited_exposure'],projected.exposure)
        self.assertIsNone(preview['ambiguity_cvar'])
        self.assertGreater(rows[0]['cost_fraction'],0.)
        self.assertEqual(rows[0]['exposure'],rows[0]['desired_exposure'])

    def test_restart_reuses_durable_responses_and_reconstructs_state(self):
        provider=Provider();variant=spec('durable')
        first=Journal(self.output/'journal',provider,{'controlled_fixture':True})
        rows,arm=run_arm(self.period,self.config,self.base,self.accounting,self.risk,first,self.output/'arm',variant)
        second_provider=Provider();second=Journal(self.output/'journal',second_provider,{'controlled_fixture':True})
        replay,replay_arm=run_arm(self.period,self.config,self.base,self.accounting,self.risk,second,self.output/'arm',variant)
        self.assertEqual(second_provider.calls,0)
        self.assertEqual(second.reused_calls,len(self.period.frame))
        self.assertEqual(replay,rows)
        self.assertEqual(replay_arm.ledger.episodes,arm.ledger.episodes)
        self.assertEqual(replay_arm.wealth,arm.wealth)
        self.assertTrue((self.output/'arm'/'episodes.json').exists())

    def test_unseen_citations_rejected_and_recorded_as_abstain(self):
        self.config['maximum_consecutive_invalid']=100
        rows,arm,_=self.run_spec(spec('bad'),provider=Provider('BTC',True))
        self.assertTrue(all(not r['valid'] and r['action']=='ABSTAIN' for r in rows))
        self.assertTrue(all('not shown' in r['error'] for r in rows))
        self.assertTrue(all(e['action']=='ABSTAIN' for e in arm.ledger.episodes))

    def test_altered_core_keeps_common_archived_outcome_reference(self):
        rows,arm,_=self.run_spec(spec('changed_core',advisor='rule',trust='adaptive'),np.full(len(self.period.frame),.1))
        for i,row in enumerate(rows):
            reference=float(self.period.current_trace.iloc[i]['portfolio_net_return'])
            self.assertEqual(row['reference_net_return'],reference)
            self.assertAlmostEqual(arm.ledger.episodes[i]['shadow_log_advantage_vs_ramoe'],np.log1p(row['shadow_net_return'])-np.log1p(reference),places=14)
            self.assertEqual(row['core_desired_exposure'],.1)

    def test_monthly_trust_excludes_future_outcomes(self):
        variant=spec('adaptive',trust='adaptive');arm=_new_arm(self.config,variant)
        first=payload_for(arm,self.period,0,self.config,self.base,self.risk,variant)['state']
        arm.ledger.add_completed(self.episode(first,'future','2022-01-02','2022-01-03',advantage=-10))
        index=next(i for i,d in enumerate(self.period.decision_dates) if str(d.date())=='2022-01-01')
        payload=payload_for(arm,self.period,index,self.config,self.base,self.risk,variant)
        self.assertEqual(payload['memory']['similar_completed_episodes'],[])
        self.assertTrue(all(e['eligible_completed_episodes']==0 for e in arm.trust.events))

    def test_rejects_unbounded_or_wrong_length_core(self):
        for core in [np.full(len(self.period.frame),1.01),np.zeros(2)]:
            with self.assertRaises(ValueError): self.run_spec(spec('invalid',advisor='abstain'),core)

if __name__=='__main__': unittest.main()
