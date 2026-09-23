"""Causal arm separation and durable reconstruction tests using synthetic data."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from stage62lib.engine import Arm, payload_for, run_pair, make_state
from stage62lib.fixture import build_fixture
from stage62lib.journal import Journal
from stage62lib.legacy import RegimeTrust, state_vector


PACKAGE = Path(__file__).resolve().parents[1]


class MemorySensitiveProvider:
    """Deterministic synthetic response: an instrumentation fixture only."""
    kind = 'controlled'

    def __init__(self, fail_on=None, invalid_citation=False):
        self.requests = []
        self.fail_on = fail_on
        self.invalid_citation = invalid_citation

    def complete(self, payload):
        self.requests.append(copy.deepcopy(payload))
        if len(self.requests) == self.fail_on:
            raise OSError('synthetic transport interruption')
        visible = payload['memory']['similar_completed_episodes']
        action = 'BTC' if visible else 'CASH'
        citations = [visible[0]['episode_id']] if visible else []
        if self.invalid_citation is True or (self.invalid_citation == 'sparse' and
                                             ((len(self.requests) - 1) // 2) % 4 == 0):
            citations = ['episode-not-visible']
        raw = json.dumps(dict(action=action, confidence=0.8,
                              reason_codes=['MEMORY_SUPPORT' if visible else 'BEARISH_ROUTER'],
                              cited_memory_ids=citations))
        return raw, {'controlled': True, 'done_reason': 'stop', 'message': {'content': raw}}, 0.


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((PACKAGE / 'config.json').read_text())
        self.period, self.base, self.accounting, self.risk = build_fixture()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.identity = {'run': 'CONTROLLED_SYNTHETIC_ONLY', 'version': 'test-v1'}

    def tearDown(self):
        self.temp.cleanup()

    def run_case(self, output, provider=None, period=None, config=None):
        output.mkdir(parents=True, exist_ok=True)
        provider = provider or MemorySensitiveProvider()
        journal = Journal(output, provider, self.identity)
        with contextlib.redirect_stdout(io.StringIO()):
            rows, arms = run_pair(period or self.period, config or self.config,
                                  self.base, self.accounting, self.risk, journal, output)
        return rows, arms, provider, journal

    def test_fixture_accounting_charges_symmetric_cost_and_carries_drift(self):
        self.assertAlmostEqual(self.accounting.net_return(0.6, 0.4, 0.1, 0.001),
                               (1 - 0.0002) * 1.06 - 1)
        self.assertAlmostEqual(self.accounting.net_return(0.4, 0.6, 0.1, 0.001),
                               (1 - 0.0002) * 1.04 - 1)
        self.assertAlmostEqual(self.accounting.drifted_exposure(0.6, 0.1), 0.66 / 1.06)
        self.assertFalse(self.period.source_audit['economic_evidence'])

    def test_no_memory_has_empty_evidence_but_independent_learning_ledger(self):
        rows, arms, provider, journal = self.run_case(self.root / 'ledger')
        self.assertEqual(journal.new_calls, 80)
        for name, arm in arms.items():
            self.assertEqual(len(arm.ledger.episodes), 40)
            self.assertTrue(arm.trust.events)
            self.assertTrue(any(e['eligible_completed_episodes'] >= 8 for e in arm.trust.events))
        for path in (journal.output / 'calls').glob('*-no_memory.json'):
            p = json.loads(path.read_text())['request']
            self.assertEqual(p['memory'], {'similar_completed_episodes': [],
                                          'summary': 'NO_COMPLETED_SIMILAR_EPISODES'})
        self.assertNotEqual(arms['memory'].trust.beta['bull'], arms['no_memory'].trust.beta['bull'])
        self.assertTrue(all(r['no_memory_retrieved_memory_count'] == 0 for r in rows))

    def test_actual_visible_memories_are_completed_earlier_decisions(self):
        _, arms, provider, _ = self.run_case(self.root / 'visible')
        by_id = {x['episode_id']: x for x in arms['memory'].ledger.episodes}
        links = 0
        for payload in provider.requests:
            for item in payload['memory']['similar_completed_episodes']:
                links += 1
                episode = by_id[item['episode_id']]
                self.assertLess(episode['decision_date'], payload['state']['decision_date'])
                self.assertLessEqual(episode['return_date'], payload['state']['decision_date'])
                self.assertEqual(episode['hard_regime'], payload['state']['hard_regime'])
        self.assertGreater(links, 0)

    def test_caller_injected_future_current_day_and_wrong_regime_hidden(self):
        arm = Arm('memory', RegimeTrust(self.config['trust']))
        state = make_state(self.period, 0, 0., 0.45, 0.05)
        safe = dict(episode_id='safe', decision_date='2021-12-08', return_date='2021-12-09',
                    hard_regime='bull', state_vector=state_vector(state).tolist(), action='BTC',
                    confidence=0.8, shadow_log_advantage_vs_ramoe=0.01, asset_return=0.01)
        arm.ledger.episodes = [safe,
            dict(safe, episode_id='future-outcome', return_date='2022-01-01'),
            dict(safe, episode_id='same-decision', decision_date='2021-12-10', return_date='2021-12-10'),
            dict(safe, episode_id='other-regime', hard_regime='bear')]
        payload = payload_for(arm, self.period, 0, self.config, self.base, self.risk)
        self.assertEqual([x['episode_id'] for x in payload['memory']['similar_completed_episodes']], ['safe'])

    def test_independent_positions_and_same_initial_payload_without_arm_labels(self):
        rows, arms, _, journal = self.run_case(self.root / 'positions')
        entries = {p.stem: json.loads(p.read_text()) for p in (journal.output / 'calls').glob('*.json')}
        self.assertEqual(entries['000001-memory']['request'], entries['000001-no_memory']['request'])
        self.assertEqual(entries['000001-memory']['request']['state']['pretrade_exposure'], 0.)
        self.assertEqual(rows[0]['memory_beta'], 0.05)
        self.assertEqual(rows[0]['no_memory_beta'], 0.05)
        self.assertEqual(rows[1]['memory_action'], 'BTC')
        self.assertEqual(rows[1]['no_memory_action'], 'CASH')
        self.assertNotEqual(rows[2]['memory_pretrade_exposure'], rows[2]['no_memory_pretrade_exposure'])
        self.assertNotEqual(rows[2]['memory_shadow_pretrade_exposure'], rows[2]['no_memory_shadow_pretrade_exposure'])
        for entry in entries.values():
            encoded = json.dumps(entry['request'])
            self.assertNotIn('"arm"', encoded)
            self.assertNotIn('no_memory', encoded)
        self.assertNotEqual(arms['memory'].ledger.episodes[1]['action'],
                            arms['no_memory'].ledger.episodes[1]['action'])

    def test_year_boundary_carries_memory_trust_and_drifted_portfolio(self):
        rows, arms, _, _ = self.run_case(self.root / 'year')
        first_january = next(i for i, r in enumerate(rows) if r['decision_date'] == '2022-01-01')
        for name in self.config['arms']:
            previous, current = rows[first_january - 1], rows[first_january]
            expected = self.accounting.drifted_exposure(previous[name + '_exposure'], previous['asset_simple_return'])
            self.assertAlmostEqual(current[name + '_pretrade_exposure'], expected)
            self.assertEqual(current[name + '_memory_total_before'], first_january)
        self.assertEqual(len(arms['memory'].trust.events), 3)
        self.assertEqual(len(arms['no_memory'].trust.events), 3)

    def test_late_asset_outcome_cannot_change_any_earlier_or_current_request(self):
        _, _, p1, _ = self.run_case(self.root / 'original')
        changed = copy.deepcopy(self.period)
        changed.asset_returns[-1] = -0.7
        _, _, p2, _ = self.run_case(self.root / 'changed', period=changed)
        self.assertEqual(p1.requests, p2.requests)

    def test_interrupted_resume_reconstructs_exact_rows_and_monthly_trust_once(self):
        expected_rows, expected_arms, _, _ = self.run_case(self.root / 'complete')
        destination = self.root / 'resume'
        provider = MemorySensitiveProvider(fail_on=56)
        with self.assertRaisesRegex(OSError, 'synthetic transport'):
            self.run_case(destination, provider=provider)
        durable = len(list((destination / 'calls').glob('*.json')))
        self.assertEqual(durable, 55)
        self.assertEqual(json.loads((destination / 'PROGRESS.json').read_text())['completed_days'], 27)
        rows, arms, resumed_provider, journal = self.run_case(destination)
        self.assertEqual(rows, expected_rows)
        self.assertEqual(journal.reused_calls, durable)
        self.assertEqual(journal.new_calls, 80 - durable)
        self.assertEqual(len(resumed_provider.requests), 80 - durable)
        for name in self.config['arms']:
            self.assertEqual(arms[name].trust.events, expected_arms[name].trust.events)
            self.assertEqual(arms[name].ledger.episodes, expected_arms[name].ledger.episodes)
            self.assertEqual(arms[name].wealth, expected_arms[name].wealth)
        self.assertFalse(list((destination / 'pending_calls').glob('*.json')))

    def test_finished_resume_uses_own_arm_records_without_any_provider_calls(self):
        expected, _, _, _ = self.run_case(self.root / 'finished')
        rows, _, provider, journal = self.run_case(self.root / 'finished', provider=MemorySensitiveProvider(fail_on=1))
        self.assertEqual(rows, expected)
        self.assertEqual(provider.requests, [])
        self.assertEqual(journal.reused_calls, 80)
        self.assertEqual(journal.new_calls, 0)

    def test_pending_request_mismatch_stops_before_provider_call(self):
        destination = self.root / 'pending'
        bad_provider = MemorySensitiveProvider(fail_on=1)
        journal = Journal(destination, bad_provider, self.identity)
        payload = {'memory': {'similar_completed_episodes': []}, 'test_state': 1}
        with self.assertRaises(OSError):
            journal.complete('000001-memory', payload)
        new_provider = MemorySensitiveProvider()
        resumed = Journal(destination, new_provider, self.identity)
        with self.assertRaisesRegex(RuntimeError, 'Pending call does not match'):
            resumed.complete('000001-memory', dict(payload, test_state=2))
        self.assertEqual(new_provider.requests, [])

    def test_journal_rejects_identity_change_and_modified_saved_response(self):
        destination = self.root / 'tamper'
        self.run_case(destination)
        path = destination / 'calls' / '000001-memory.json'
        original = json.loads(path.read_text())
        changed_identity = Journal(destination, MemorySensitiveProvider(), {'run': 'other'})
        with self.assertRaisesRegex(RuntimeError, 'Journal identity/request/hash mismatch'):
            changed_identity.complete('000001-memory', original['request'])
        modified = copy.deepcopy(original)
        modified['raw_response'] = '{}'
        path.write_text(json.dumps(modified))
        with self.assertRaisesRegex(RuntimeError, 'Journal identity/request/hash mismatch'):
            Journal(destination, MemorySensitiveProvider(), self.identity).complete('000001-memory', original['request'])

    def test_unseen_citation_falls_back_to_abstain_and_retains_all_days(self):
        config = copy.deepcopy(self.config)
        rows, arms, _, journal = self.run_case(self.root / 'invalid',
            provider=MemorySensitiveProvider(invalid_citation='sparse'), config=config)
        self.assertEqual(len(rows), 40)
        self.assertEqual(journal.new_calls, 80)
        for name in config['arms']:
            self.assertEqual(len(arms[name].ledger.episodes), 40)
            invalid = [r for r in rows if not r[name + '_valid']]
            self.assertEqual(len(invalid), 10)
            self.assertTrue(all(r[name + '_action'] == 'ABSTAIN' for r in invalid))
            self.assertTrue(all(r[name + '_desired_exposure'] == 0.45 for r in invalid))

    def test_three_consecutive_invalid_outputs_stop_before_long_replay(self):
        destination = self.root / 'invalid-consecutive'
        with self.assertRaisesRegex(RuntimeError, 'Three consecutive invalid model outputs'):
            self.run_case(destination, provider=MemorySensitiveProvider(invalid_citation=True))
        consecutive = self.config['maximum_consecutive_invalid']
        self.assertEqual(json.loads((destination / 'PROGRESS.json').read_text())['completed_days'], consecutive)
        self.assertEqual(len(list((destination / 'calls').glob('*.json'))), consecutive * 2)


if __name__ == '__main__':
    unittest.main()
