"""Reconcile existing comparisons. No new model calls or regenerated portfolios."""
from pathlib import Path
import hashlib, json, math
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE/'recorded'
OUT = HERE/'audit_output'

def bootstrap(a, b):
    delta = (np.log1p(a)-np.log1p(b))*10000
    n = len(delta); rng = np.random.default_rng(16062)
    samples = np.empty(5000); offsets = np.arange(30)
    for first in range(0, 5000, 128):
        count = min(128,5000-first)
        starts = rng.integers(0,n,size=(count,math.ceil(n/30)))
        indices = ((starts[:,:,None]+offsets)%n).reshape(count,-1)[:,:n]
        samples[first:first+count] = delta[indices].mean(axis=1)
    lo,hi = np.quantile(samples,[.025,.975])
    return float(delta.mean()),float(lo),float(hi)

def main():
    OUT.mkdir(exist_ok=True)
    daily = pd.read_csv(DATA/'ALL_ARMS_DAILY.csv',low_memory=False)
    contrasts = pd.read_csv(DATA/'MEMORY_CONTRASTS.csv').fillna('')
    labels = {'adaptive_memory':'Original, adaptive authority',
              'fixed_memory':'Original, fixed authority',
              'closed_loop_memory':'Independent rerun, fixed authority',
              'state_controlled_memory':'Fixed-state diagnostic',
              'llm_adaptive_memory_controller_off':'Factorial, adaptive authority, projection bypassed',
              'llm_adaptive_memory_controller_on':'Factorial, adaptive authority, projection active',
              'llm_fixed_memory_controller_off':'Factorial, fixed authority, projection bypassed',
              'llm_fixed_memory_controller_on':'Factorial, fixed authority, projection active'}
    rows = []
    for _,c in contrasts.iterrows():
        a = daily[daily.arm==c.memory_arm].copy()
        b = daily[daily.arm==c.no_memory_arm].copy()
        rows.append((labels[c.memory_arm],a,b,c.to_dict()))
    balanced = json.loads((DATA/'balanced_comparison.json').read_text())['contrasts'][0]
    balanced.update(trust='fixed',controller='on',sampling_seed=16061,
                    prompt_variant='frozen_llama_role',run_id='20260922T014916Z')
    rows.append(('Action-balanced retrieval and revised episode scoring, fixed authority',
                 pd.read_csv(DATA/'balanced_exposed.csv'),pd.read_csv(DATA/'balanced_hidden.csv'),balanced))
    inventory = []; largest_error = 0.
    for label,a,b,c in rows:
        a = a[a.return_date.between('2022-01-01','2025-05-28')].sort_values('return_date').reset_index(drop=True)
        b = b[b.return_date.between('2022-01-01','2025-05-28')].sort_values('return_date').reset_index(drop=True)
        assert len(a)==len(b)==1244 and a.return_date.equals(b.return_date)
        assert not a.return_date.duplicated().any() and not b.return_date.duplicated().any()
        if 'valid' in a and a.valid.notna().all():
            assert a.valid.astype(str).str.lower().eq('true').all()
            assert b.valid.astype(str).str.lower().eq('true').all()
        point,lo,hi = bootstrap(a.net_return_after_trading_costs.to_numpy(),b.net_return_after_trading_costs.to_numpy())
        err = max(abs(point-float(c['mean_daily_net_log_difference_bps'])),
                  abs(lo-float(c['ci95_lower_bps'])),abs(hi-float(c['ci95_upper_bps'])))
        largest_error = max(largest_error,err)
        assert err<1e-9,(label,err)
        fixed = label=='Fixed-state diagnostic'
        original_prompt = c['prompt_variant']=='frozen_llama_role'
        prompt = HERE/'data'/('original_system_prompt.txt' if original_prompt else 'system_prompt.txt')
        inventory.append({'comparison':label,'estimand':'fixed_state_one_step' if fixed else 'closed_loop_recorded_path',
            'independently_queried_closed_loop':not fixed,'fixed_upstream_projection_bypass_replay':False,
            'authority':c['trust'],'projection':c['controller'],'serving_seed':int(c['sampling_seed']),
            'model':'llama3.3:70b','model_digest':'a6eb4748fd2990ad2952b2335a95a7f952d1a06119a0aa6a2df6cd052a93a3fa',
            'prompt_identifier':c['prompt_variant'], 'prompt_sha256':hashlib.sha256(prompt.read_bytes()).hexdigest(),
            'serving_identifier':c['run_id'],'temperature':0,'context_limit':8192,'output_limit':512,
            'top_p_explicit':1.0 if not original_prompt else '',
            'serving_provenance':'consolidated inventory; original rerun contract requested' if 'rerun' in label or fixed else 'bundled original run contract',
            'first_return_date':a.return_date.iloc[0],'last_return_date':a.return_date.iloc[-1],'paired_dates':len(a),
            'exposed_valid_abstentions':int(a.action.eq('ABSTAIN').sum()),
            'hidden_valid_abstentions':int(b.action.eq('ABSTAIN').sum()),
            'valid_abstention_difference_pp':100*(a.action.eq('ABSTAIN').mean()-b.action.eq('ABSTAIN').mean()),
            'mean_exposure_difference_pp':100*(a.exposure-b.exposure).mean(),
            'mean_daily_net_log_difference_bps':point,'ci95_lower_bps':lo,'ci95_upper_bps':hi,
            'outcome_scope':'one-step fixed-state diagnostic; no compounded portfolio claim' if fixed else 'recorded closed-loop paths',
            'bootstrap_block_days':30,'bootstrap_resamples':5000,'bootstrap_seed':16062})
    frame = pd.DataFrame(inventory).sort_values('estimand',kind='stable')
    frame.to_csv(OUT/'comparison_inventory.csv',index=False,float_format='%.17g')
    closed = frame[frame.estimand=='closed_loop_recorded_path']
    fixed = frame[frame.estimand=='fixed_state_one_step']
    assert len(closed)==8 and len(fixed)==1
    result = {'closed_loop_comparisons':len(closed),
              'closed_loop_intervals_spanning_zero':int(((closed.ci95_lower_bps<=0)&(closed.ci95_upper_bps>=0)).sum()),
              'closed_loop_mean_valid_abstention_difference_pp':float(closed.valid_abstention_difference_pp.mean()),
              'closed_loop_abstention_difference_range_pp':[float(closed.valid_abstention_difference_pp.min()),float(closed.valid_abstention_difference_pp.max())],
              'closed_loop_mean_exposure_difference_pp':float(closed.mean_exposure_difference_pp.mean()),
              'closed_loop_exposure_difference_range_pp':[float(closed.mean_exposure_difference_pp.min()),float(closed.mean_exposure_difference_pp.max())],
              'fixed_state_valid_abstention_difference_pp':float(fixed.valid_abstention_difference_pp.iloc[0]),
              'fixed_state_mean_exposure_difference_pp':float(fixed.mean_exposure_difference_pp.iloc[0]),
              'all_nine_higher_abstention':bool((frame.valid_abstention_difference_pp>0).all()),
              'all_nine_lower_mean_exposure':bool((frame.mean_exposure_difference_pp<0).all()),
              'maximum_reproduced_point_or_CI_error_bps':largest_error,
              'interpretation':'descriptive comparison averages, not independent replicates or a pooled effect test'}
    (OUT/'comparison_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
