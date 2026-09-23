"""Pinned historical paths, normalized without altering their performance."""
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def verify_run(directory, complete_sha, trace_sha):
    directory=Path(directory).resolve()
    if sha(directory/'RUN_COMPLETE.json')!=complete_sha: raise ValueError('Completion SHA mismatch: '+str(directory))
    complete=json.loads((directory/'RUN_COMPLETE.json').read_text())
    if complete['status']!='PASS': raise ValueError('Source run not PASS')
    for name,expected in complete['artifact_sha256'].items():
        p=(directory/name).resolve()
        if directory not in p.parents or sha(p)!=expected: raise ValueError('Artifact integrity mismatch: '+str(p))
    if sha(directory/'03_DAILY_TRACE.csv')!=trace_sha: raise ValueError('Trace SHA mismatch')
    contract=json.loads((directory/'00_CONTRACT.json').read_text())
    return pd.read_csv(directory/'03_DAILY_TRACE.csv'),contract,dict(path=str(directory),complete_sha256=complete_sha,trace_sha256=trace_sha,artifacts_verified=len(complete['artifact_sha256']))

def normalize_trace(trace, prefix, family, cost=.001):
    rows=[]
    wealth=1.
    for _,v in trace.iterrows():
        nr=float(v[prefix+'_net_return']);x=float(v[prefix+'_exposure']);p=float(v[prefix+'_pretrade_exposure'])
        row=dict(arm=family+'_'+prefix,decision_date=v.decision_date,return_date=v.return_date,hard_regime=v.hard_regime,
            asset_simple_return=float(v.asset_simple_return),pretrade_exposure=p,exposure=x,
            net_return_after_trading_costs=nr,turnover=abs(x-p),cost_fraction=cost*abs(x-p),wealth_before=wealth,wealth_after=wealth*(1+nr),origin='ARCHIVED_REAL_LLAMA_PATH')
        for field in ['action','beta','desired_exposure','latency_seconds','valid','retrieved_memory_count','cited_memory_count','core_desired_exposure']:
            if prefix+'_'+field in v: row[field]=v[prefix+'_'+field]
        rows.append(row);wealth*=1+nr
    return rows

def passive_rows(trace,cost=.001):
    rows=[]
    for arm,target in [('cash',0.),('buy_hold',1.),('static50',.5)]:
        p=0.;w=1.
        for _,v in trace.iterrows():
            x=target;r=float(v.asset_simple_return);turn=abs(x-p);nr=(1-cost*turn)*(1+x*r)-1
            rows.append(dict(arm=arm,decision_date=v.decision_date,return_date=v.return_date,hard_regime=v.hard_regime,asset_simple_return=r,
                pretrade_exposure=p,exposure=x,net_return_after_trading_costs=nr,turnover=turn,cost_fraction=cost*turn,wealth_before=w,wealth_after=w*(1+nr),action='PASSIVE',beta=0.,origin='PASSIVE_ACCOUNTING_REPLAY'))
            p=x*(1+r)/(1+x*r);w*=1+nr
    return rows

def load_archives(root,config,source_config, adaptive_override=None, fixed_override=None):
    root=Path(root)
    ad=Path(adaptive_override) if adaptive_override else root/source_config['adaptive_reference_relative']
    fx=Path(fixed_override) if fixed_override else root/config['fixed_reference_relative']
    a,ac,aa=verify_run(ad,source_config['expected_adaptive_complete_sha256'],source_config['expected_adaptive_trace_sha256'])
    f,fc,fa=verify_run(fx,config['expected_fixed_complete_sha256'],config['expected_fixed_trace_sha256'])
    if len(a)!=source_config['expected_rows'] or len(f)!=len(a): raise ValueError('Wrong historical horizon')
    for key in ['decision_date','return_date','hard_regime']:
        if not a[key].equals(f[key]): raise ValueError('Unmatched historical '+key)
    if not np.allclose(a.asset_simple_return,f.asset_simple_return,atol=1e-14,rtol=0):raise ValueError('Unmatched BTC returns')
    if ac['identity']['provider_identity']!=fc['identity']['provider_identity']: raise ValueError('Archived serving settings differ')
    rows=[]
    for family,tr in [('adaptive',a),('fixed',f)]:
        for prefix in ['memory','no_memory']:rows.extend(normalize_trace(tr,prefix,family))
    rows.extend(passive_rows(f))
    return pd.DataFrame(rows),f,dict(adaptive=aa,fixed=fa),fc['identity']
