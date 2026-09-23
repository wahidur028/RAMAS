import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from suite64.core_variants import CoreSourceError, _arrays, _period_weights, build_core_variants


class CoreVariantsTests(unittest.TestCase):
    def fixture(self):
        q = np.array([[.2, .6, .2], [.6, .1, .3]])
        w = np.tile(np.array([[.8,.2,0,0,0,0], [.2,.8,0,0,0,0], [.5,.5,0,0,0,0]])[None], (2,1,1))
        e = np.array([[0,1,.7,.4,.8,.9], [0,1,.2,.6,.7,1.]])
        initial = np.tile([.5,.5,0,0,0,0], (3,1))
        return q, w, e, initial, np.array([1,1,0,0,0,0], bool)

    def test_aggregation_interventions_leave_source_arrays_unchanged(self):
        args = self.fixture()
        original = [x.copy() for x in args]
        variants, _ = _arrays(*args)
        np.testing.assert_allclose(variants["original"], [.62,.35])
        np.testing.assert_allclose(variants["hard_allocator"], [.8,.2])
        np.testing.assert_allclose(variants["uniform_allocator"], [.5,.5])
        np.testing.assert_allclose(variants["frozen_W"], [.5,.5])
        for now, before in zip(args, original):
            np.testing.assert_array_equal(now, before)

    def test_inactive_policy_proposals_are_real_but_have_zero_effect(self):
        q,w,e,initial,mask = self.fixture()
        result, mixed = _arrays(q,w,e,initial,mask)
        e[:,2:] = 1 - e[:,2:]
        changed, _ = _arrays(q,w,e,initial,mask)
        for key in result:
            np.testing.assert_array_equal(result[key], changed[key])
        np.testing.assert_array_equal(mixed[:,2:], np.zeros((2,4)))

    def test_unadmitted_weight_or_invalid_probabilities_are_rejected(self):
        q,w,e,initial,mask = self.fixture()
        w[0,0,0] -= .1
        w[0,0,2] = .1
        with self.assertRaises(CoreSourceError):
            _arrays(q,w,e,initial,mask)
        q,w,e,initial,mask = self.fixture()
        q[0] = [.1,.1,.1]
        with self.assertRaises(CoreSourceError):
            _arrays(q,w,e,initial,mask)

    def test_hard_allocator_tie_uses_first_recorded_regime(self):
        q,w,e,initial,mask = self.fixture()
        q[0] = [.5,.5,0]
        variants, _ = _arrays(q,w,e,initial,mask)
        self.assertAlmostEqual(variants["hard_allocator"][0], .2)

    def test_completed_month_updates_cannot_affect_earlier_decisions(self):
        names = ["cash","buy_and_hold"]
        config = dict(expert_names=names, regime_names=["bear","bull","mix"], always_admitted_experts=names,
                      transaction_cost_bps=10, trust_eta=.5, cvar_alpha=.95, trust_tail_penalty=1,
                      trust_turnover_penalty=.1, trust_minimum_posterior_mass=20,
                      trust_minimum_effective_sample_size=20)
        frame = pd.DataFrame({"return_date":["2021-01-30","2021-01-31","2021-02-01"],
                              "prob_bear":[0,0,0],"prob_bull":[1,1,1],"prob_mix":[0,0,0],
                              "exposure_cash":[0,0,0],"exposure_buy_and_hold":[1,1,1],
                              "canonical_asset_simple_return":[.01,.02,.03]})
        calls=[]
        def update(w,q,r,t,mask,**kwargs):
            calls.append((q.copy(),r.copy(),t.copy()))
            return SimpleNamespace(trust_matrix=np.tile([.4,.6],(3,1)), support_mask=np.array([0,1,0],bool),
                                   posterior_mass=q.sum(0),effective_sample_size=q.sum(0))
        accounting=SimpleNamespace(net_return=lambda x,p,r,c:(1-c*abs(x-p))*(1+x*r)-1,
                                   drifted_exposure=lambda x,r:x*(1+r)/(1+x*r))
        w,terminal,_ = _period_weights(frame,np.full((3,2),.5),config,np.ones(2,bool),accounting,
                                       SimpleNamespace(risk_aware_trust_update=update))
        np.testing.assert_array_equal(w[:2],np.full((2,3,2),.5))
        np.testing.assert_array_equal(w[2],np.tile([.4,.6],(3,1)))
        self.assertEqual(len(calls),1)
        self.assertEqual(len(calls[0][0]),2)
        self.assertEqual(calls[0][2][0,1],1.)
        self.assertEqual(calls[0][2][1,1],0.)

    def test_missing_corrected_source_is_an_error_not_zero_weights(self):
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CoreSourceError):
                build_core_variants(SimpleNamespace(),{},Path(directory))


if __name__ == "__main__":
    unittest.main()
