#!/usr/bin/env python3
"""One sequential, resumable retrospective component suite. No profit gates."""
import argparse,fcntl,json,platform,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from suite64.archive import sha,load_archives,passive_rows
from suite64.metrics import atomic_csv
from stage63lib.journal import Journal,atomic_json,digest
from stage63lib.legacy import source,ControlledProvider,SYSTEM_PROMPT

PACKAGE=Path(__file__).resolve().parent

def verify_package():
    manifest=PACKAGE/'PACKAGE_MANIFEST.sha256'
    for line in manifest.read_text().splitlines():
        expected,name=line.split(maxsplit=1);p=(PACKAGE/name.lstrip('*')).resolve()
        if PACKAGE not in p.parents or sha(p)!=expected:raise RuntimeError('Package differs: '+str(p))
    return sha(manifest)

def artifact_map(directory):
    return {p.relative_to(directory).as_posix():sha(p) for p in sorted(directory.rglob('*')) if p.is_file() and p != directory/'RUN_COMPLETE.json' and p.name not in {'RUN.lock','SERVER.lock','FAILURE.json'} and not p.name.endswith(('.pending','.tmp'))}

def verify_completed(directory):
    complete=json.loads((directory/'RUN_COMPLETE.json').read_text())
    for name,expected in complete['artifact_sha256'].items():
        p=(directory/name).resolve()
        if directory not in p.parents or sha(p)!=expected:raise RuntimeError('Completed file differs: '+str(p))
    return complete

def stable_record(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise RuntimeError('Source audit changed on resume: '+str(path))
    else:atomic_json(path,value)

def metrics_outputs(out,ledger,cfg):
    from suite64.metrics import write_reports,paired_block_comparison
    write_reports(ledger,out)
    pairs=[('fixed_memory','rule_fixed','PRIMARY'),('adaptive_memory','rule_adaptive','SECONDARY'),
        ('fixed_no_memory','rule_fixed','SECONDARY'),('adaptive_memory','numerical_only','SECONDARY'),
        ('rule_adaptive','numerical_only','SECONDARY'),('adaptive_memory','adaptive_no_memory','PRIOR_CONTRAST'),
        ('fixed_memory','fixed_no_memory','PRIOR_CONTRAST'),('adaptive_memory','fixed_memory','SECONDARY_TRUST')]
    for v in cfg['variants']:
        if v['name'].startswith('llama_'):pairs.append(('adaptive_memory',v['name'],'SECONDARY_COMPONENT'))
        elif v['name'].startswith('numerical_') and v['name']!='numerical_only':pairs.append(('numerical_only',v['name'],'SECONDARY_NUMERICAL_CORE'))
    pairs.extend(('adaptive_memory',b,'SECONDARY_EXTERNAL') for b in ['cash','buy_hold','static50'])
    contrasts=[];counts=[]
    for a,b,role in pairs:
        if a not in set(ledger.arm) or b not in set(ledger.arm):continue
        # Interface adapted below after module integration.
        result=paired_block_comparison(ledger,a,b,block_length=cfg['analysis']['block_days'],n_resamples=cfg['analysis']['bootstrap_resamples'],seed=cfg['analysis']['seed'])
        result['role']=role;contrasts.append(result)
        aa=ledger[ledger.arm==a].set_index('return_date');bb=ledger[ledger.arm==b].set_index('return_date')
        common=aa.index.intersection(bb.index);common=common[common>='2022-01-01'];aa=aa.loc[common];bb=bb.loc[common]
        for regime in ['ALL','bear','bull','mix']:
            mask=np.ones(len(common),dtype=bool) if regime=='ALL' else (aa.hard_regime.to_numpy()==regime)
            rec=dict(left=a,right=b,regime=regime,days=int(mask.sum()))
            for col in ['action','desired_exposure','exposure','net_return_after_trading_costs']:
                if col not in aa or col not in bb:rec[col+'_different_days']=None;continue
                valid=aa[col].notna()&bb[col].notna()
                diff=(aa[col].to_numpy()!=bb[col].to_numpy()) if col=='action' else np.abs(aa[col].to_numpy(float)-bb[col].to_numpy(float))>1e-12
                rec[col+'_different_days']=int((diff&mask&valid.to_numpy()).sum())
            counts.append(rec)
    atomic_json(out/'COMPARISONS.json',dict(primary='fixed_memory_minus_rule_fixed',comparison_scope='POST2021_REUSED_HISTORY',intervals='POINTWISE_95_PERCENT_NOT_MULTIPLICITY_ADJUSTED',secondary_interpretation='DESCRIPTIVE_EXPLORATORY_DO_NOT_SELECT_WINNING_ENDPOINT',contrasts=contrasts))
    atomic_csv(pd.DataFrame(counts),out/'COMPONENT_TRANSMISSION.csv')
    # The multiplicative portfolio identity is checked in the metrics module.
    return contrasts

def manifest_status(cfg,completed):
    items=[dict(component='External cash/B&H/static50',status='COMPLETED_FROM_ARCHIVED_RETURN_STREAM'),
       dict(component='Memory access; adaptive vs fixed beta',status='REUSED_COMPLETED_STAGE62_STAGE63'),
       dict(component='Next-day forecast calibration and fit chronology',status='UNRESOLVED_SOURCE_PROVENANCE_REQUIRED'),
       dict(component='Executable decision/inference/fill clock',status='UNRESOLVED_NEW_CLOCK_CONTRACT_REQUIRED'),
       dict(component='Reference seam correction',status='KNOWN_DIAGNOSTIC_PRESERVED_NOT_REPAIRED'),
       dict(component='T/V/D/ATP deletion',status='STRUCTURAL_ZERO_WEIGHT_INVARIANCE_ONLY_NOT_POLICY_UTILITY'),
       dict(component='All-q-interface and pooled-W training changes',status='NOT_COVERED_ALLOCATOR_ONLY_Q_INTERVENTION'),
       dict(component='Fresh OOS/novelty/ML literature baseline',status='NOT_ESTABLISHED_BY_THIS_SUITE')]
    items.extend(dict(component=v['name'],status='COMPLETE' if v['name'] in completed else 'PENDING',new_llm_required=v['advisor']=='llama') for v in cfg['variants'])
    return items

def execute(args):
    package_sha=verify_package();cfg=json.loads((PACKAGE/'config.json').read_text());sc=json.loads((PACKAGE/'source_config.json').read_text())
    if sc['provider']['model']!='llama3.3:70b' or sc['qwen_used'] is not False:raise ValueError('Model restriction changed')
    out=args.output.expanduser().resolve();root=args.project_root.expanduser().resolve()
    forbidden=[PACKAGE,root/sc['source_result_relative'],root/sc['corrected_run_relative'],root/sc['adaptive_reference_relative'],root/cfg['fixed_reference_relative']]
    for override in [args.adaptive_run,args.fixed_run]:
        if override is not None:forbidden.append(override)
    for old in forbidden:
        old=old.resolve()
        if out==old or old in out.parents:raise ValueError('Output cannot be inside source/code evidence')
    out.mkdir(parents=True,exist_ok=True);lock=(out/'RUN.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another suite uses this output')
    active_contract=False
    try:
        identity=dict(package_manifest_sha256=package_sha,mode=args.mode,demo=args.demo,project_root=str(root),config_sha256=sha(PACKAGE/'config.json'),full_dataset=str(args.full_dataset.resolve()) if args.full_dataset else None,adaptive_run=str(args.adaptive_run.resolve()) if args.adaptive_run else None,fixed_run=str(args.fixed_run.resolve()) if args.fixed_run else None,runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__))
        contract=out/'00_CONTRACT.json'
        if contract.exists():
            if not args.resume:raise ValueError('Use --resume for an existing run')
            if json.loads(contract.read_text())['identity']!=identity:raise ValueError('Resume identity changed')
            if (out/'RUN_COMPLETE.json').exists():
                verify_completed(out);print('RESUME_ACTION=ALREADY_COMPLETE_NO_NEW_CALLS',flush=True);return 0
        elif args.resume:raise ValueError('No contract to resume')
        else:atomic_json(contract,dict(identity=identity,config=cfg,started_utc=datetime.now(timezone.utc).isoformat(),no_yearly_reset=True,pretrained_weights_updated=False,economic_hard_stop=False))
        active_contract=True
        if args.demo:
            from stage63lib.fixture import build_fixture
            from stage63lib.fixed_trust import FixedTrust
            from suite64.engine import run_arm
            period,bc,accounting,risk=build_fixture();provider=ControlledProvider();pid={'kind':'controlled','economic_evidence':False}
            frame=pd.DataFrame(dict(decision_date=period.decision_dates.strftime('%Y-%m-%d'),return_date=period.return_dates.strftime('%Y-%m-%d'),hard_regime=[('bear','bull','mix')[g] for g in np.argmax(period.router_probabilities,axis=1)],asset_simple_return=period.asset_returns))
            rows=passive_rows(frame)
            for fam,trust in [('adaptive','adaptive'),('fixed','fixed')]:
                for memory in ['expanding','none']:
                    name=fam+('_memory' if memory=='expanding' else '_no_memory')
                    spec=dict(name=name,advisor='llama',trust=trust,memory=memory,core='original',risk='standard',fixed_beta=.05)
                    j=Journal(out/'demo_archives'/name,provider,identity)
                    values,_=run_arm(period,sc,bc,accounting,risk,j,out/'demo_archives'/name,spec)
                    rows.extend(values)
            ledger=pd.DataFrame(rows);archive_audit={'synthetic':True};archived_identity={'provider_identity':pid}
        else:
            ledger,frame,archive_audit,archived_identity=load_archives(root,cfg,sc,args.adaptive_run,args.fixed_run)
        stable_record(out/'01_ARCHIVE_AUDIT.json',archive_audit)
        print('ARCHIVED_EVIDENCE_VERIFIED=TRUE',flush=True)
        metrics_outputs(out,ledger,cfg);completed=[]
        atomic_json(out/'TEST_STATUS.json',manifest_status(cfg,completed))
        if args.mode!='archive':
            if not args.demo:
                from stage63lib.inputs import load_inputs
                from stage63lib.scientific_audit import audit_scientific_inputs
                period,bc,accounting,risk,audit=load_inputs(root,sc,args.full_dataset)
                scientific,_=audit_scientific_inputs(root,sc,period,bc,accounting)
                stable_record(out/'02_SOURCE_AUDIT.json',audit);stable_record(out/'03_CLOCK_REFERENCE_AUDIT.json',scientific)
                from suite64.core_variants import build_core_variants
                cores,core_audit,core_trace=build_core_variants(period,bc,Path(audit['corrected_source_run']))
                stable_record(out/'04_CORE_AUDIT.json',core_audit);atomic_csv(core_trace,out/'CORE_DAILY_STATE.csv')
                for key,value in identity['runtime'].items():
                    if value!=archived_identity[key]:raise RuntimeError('Restore original matched runtime: '+key+' expected '+archived_identity[key])
            else:
                cores={'original':np.full(len(frame),.45),'hard_allocator':np.full(len(frame),.55),'uniform_allocator':np.full(len(frame),.5),'frozen_W':np.full(len(frame),.5)}
                atomic_json(out/'03_CLOCK_REFERENCE_AUDIT.json',dict(evidence_kind='CONTROLLED_NON_ECONOMIC'))
            from suite64.engine import run_arm
            for spec in cfg['variants']:
                name=spec['name']
                if args.mode=='numeric' and spec['advisor']=='llama':continue
                armout=out/'arms'/name;armout.mkdir(parents=True,exist_ok=True)
                if (armout/'RUN_COMPLETE.json').exists():
                    done=verify_completed(armout)
                    if done['spec']!=spec or done.get('suite_identity')!=identity:raise RuntimeError('Arm definition or suite identity changed')
                    rows=pd.read_csv(armout/'DAILY_LEDGER.csv').to_dict('records')
                    print('ARM='+name+' STATUS=REUSED_COMPLETE',flush=True)
                else:
                    if spec['advisor']=='llama' and not args.demo:
                        from stage63lib.provider import OllamaProvider
                        provider=OllamaProvider(sc['provider'],sc['agent']);pid=provider.pinned
                        if pid!=archived_identity['provider_identity']:raise RuntimeError('Restore exact historical Llama/Ollama/options identity')
                    elif not args.demo:provider=None;pid={'kind':'deterministic','model_calls':0}
                    journal=Journal(armout,provider,dict(suite=identity,spec=spec,provider_identity=pid))
                    print('ARM='+name+' STATUS=START NEW_LLAMA_REQUIRED='+str(spec['advisor']=='llama'),flush=True)
                    rows,arm=run_arm(period,sc,bc,accounting,risk,journal,armout,spec,core_desired=cores[spec['core']])
                    atomic_csv(pd.DataFrame(rows),armout/'DAILY_LEDGER.csv')
                    valid_rate=float(pd.DataFrame(rows).valid.mean())
                    if spec['advisor']=='llama' and valid_rate<sc['minimum_valid_action_rate']:raise RuntimeError('Technical action-validity rate below declared minimum: '+name)
                    if spec['advisor']=='llama' and hasattr(provider,'assert_identity'):provider.assert_identity()
                    atomic_json(armout/'RUN_COMPLETE.json',dict(status='COMPLETE',suite_identity=identity,spec=spec,rows=len(rows),valid_rate=valid_rate,new_calls=journal.new_calls,reused_calls=journal.reused_calls,artifact_sha256=artifact_map(armout)))
                ledger=pd.concat([ledger,pd.DataFrame(rows)],ignore_index=True);completed.append(name)
                metrics_outputs(out,ledger,cfg)
                atomic_json(out/'TEST_STATUS.json',manifest_status(cfg,completed))
                print('ARM='+name+' STATUS=COMPLETE SUITE_PROGRESS='+str(len(completed)),flush=True)
        exported=pd.read_csv(out/'daily_ledger.csv')
        if len(exported)!=len(ledger) or exported.groupby('arm').size().to_dict()!=ledger.groupby('arm').size().to_dict():raise RuntimeError('Final CSV export row/arm counts differ')
        final=dict(status='COMPLETE',mode=args.mode,evidence_kind='CONTROLLED_NON_ECONOMIC' if args.demo else 'REUSED_HISTORY_LEGACY_CLOCK_DIAGNOSTIC',new_arms_completed=completed,total_ledger_rows=len(ledger),all_component_claims_validated=False,continuous_learning_established=False,forecast_and_executable_fill_fully_verified=False,qwen_used=False,completed_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(out/'FINAL_STATUS.json',final)
        (out/'READ_RESULTS_FIRST.md').write_text('# RAMAS suite results\n\nStandard name: **net return after trading costs**. Log return is a separate additive metric.\n\nRead METRIC_DEFINITIONS.md in the code package and TEST_STATUS.json here. PENDING/UNRESOLVED tests are not passes. The numerical/router controls are internal ablations. Cash, B&H and daily-rebalanced50/50 are external benchmarks.\n\nThe main method is adaptive-memory RAMAS. Fixed-beta results are attribution controls. Primary new contrast is fixed_memory minus rule_fixed; all other contrasts are secondary/exploratory. Intervals are pointwise, not adjusted for selecting among many tests. A larger return alone does not establish improvement in risk-adjusted performance.\n\nThis suite preserves inherited clocks/reference feedback for matched mechanism diagnostics. It does not certify executable fills, original router fitting chronology, fresh OOS, novelty or optimality. Changing the clock requires a separately versioned matched rerun.\n')
        atomic_json(out/'RUN_COMPLETE.json',dict(status='COMPLETE',identity=identity,artifact_sha256=artifact_map(out)))
        print('RAMAS_STAGE6_4_STATUS=COMPLETE\nSCIENCE_CLAIMS_AUTOMATICALLY_PASSED=FALSE\nOUTPUT='+str(out),flush=True)
        return 0
    except Exception as exc:
        if active_contract and not (out/'RUN_COMPLETE.json').exists():
            atomic_json(out/'FAILURE.json',dict(status='STOP_TECHNICAL_OR_SOURCE',error_type=type(exc).__name__,message=str(exc),timestamp_utc=datetime.now(timezone.utc).isoformat(),economic_gate=False))
        raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,default=Path('/home/infonet/wahid/leader_router_fresh'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['archive','numeric','full'],default='full');p.add_argument('--full-dataset',type=Path);p.add_argument('--adaptive-run',type=Path);p.add_argument('--fixed-run',type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--demo',action='store_true');args=p.parse_args()
    try:return execute(args)
    except Exception as exc:
        print('RAMAS_STAGE6_4_STATUS=STOP\nERROR='+type(exc).__name__+': '+str(exc),flush=True)
        # Never mutate an existing/completed run merely because a bad invocation failed.
        traceback.print_exc();return 2
if __name__=='__main__':raise SystemExit(main())
