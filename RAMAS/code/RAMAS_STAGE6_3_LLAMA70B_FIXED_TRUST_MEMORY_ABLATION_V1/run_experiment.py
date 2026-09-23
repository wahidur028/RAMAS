#!/usr/bin/env python3
"""Stage 6.3 matched episodic-memory ablation, with durable restart."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
from stage63lib.legacy import legacy,source,SYSTEM_PROMPT,ControlledProvider,validate_decision
from stage63lib.journal import Journal,atomic_json,digest
from stage63lib.engine import run_pair
from stage63lib.reporting import write_reports
from stage63lib.fixed_trust import FixedTrust

PACKAGE=Path(__file__).resolve().parent
EXPERIMENT='RAMAS_STAGE6_3_LLAMA70B_FIXED_TRUST_MEMORY_ABLATION_V1'

def verify_package():
    manifest=PACKAGE/'PACKAGE_MANIFEST.sha256'
    if not manifest.is_file(): raise RuntimeError('Package manifest missing')
    for line in manifest.read_text().splitlines():
        sha,relative=line.split(maxsplit=1)
        path=(PACKAGE/relative.strip().removeprefix('*')).resolve()
        if PACKAGE not in path.parents or source.sha256_file(path)!=sha:
            raise RuntimeError('Package hash mismatch: '+str(path))
    return source.sha256_file(manifest)

def load_config():
    config=source.load_json(PACKAGE/'config.json')
    if config['experiment_id']!=EXPERIMENT or config['provider']['model']!='llama3.3:70b' or config['qwen_used'] is not False:
        raise ValueError('Experiment/model lock violation')
    old=source.load_json(PACKAGE/'vendor/stage61/config.json')
    if config['agent']!=old['agent'] or config['trust']!=old['trust']:
        raise ValueError('Frozen agent and archived trust metadata changed')
    if (config.get('agent_trust_mode')!='FIXED_POSITIVE' or
            config.get('fixed_beta')!=0.05 or config.get('common_legacy_trust_rule') is not False):
        raise ValueError('Both new arms must use fixed beta 0.05; adaptive beta is disabled')
    FixedTrust(config)
    if config.get('scientific_scope')!='LEGACY_MECHANISM_DIAGNOSTIC_NOT_REALISTIC_EXECUTION_OR_FRESH_OOS_CONFIRMATION':
        raise ValueError('Unresolved clock evidence supports the declared legacy diagnostic only')
    if config['arms']!=['memory','no_memory'] or config['cost_rate']!=.001:
        raise ValueError('Arm/cost contract changed')
    if source.sha256_text(SYSTEM_PROMPT)!=config['expected_vendor_prompt_sha256']:
        raise ValueError('System prompt changed')
    if config['continue_across_years'] is not True or config['no_yearly_economic_hard_stop'] is not True:
        raise ValueError('Continuous experiment contract changed')
    return config

def require_matched_provider(config,provider_identity,scientific_audit):
    old=scientific_audit['adaptive_reference_identity']
    if provider_identity.get('model_digest')!=config['expected_model_digest']:
        raise RuntimeError('Llama model digest differs from the matched Stage 6.2 model; restore the recorded model')
    if provider_identity!=old['provider_identity']:
        raise RuntimeError('Ollama model/version/template/options differ from the adaptive reference; do not combine different serving protocols')
    current=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    for name,value in current.items():
        if value!=old[name]:
            raise RuntimeError(f'Matched runtime differs: {name}={value}, expected={old[name]}; use the original wahid_test environment')

def preflight(journal,config):
    records=[]
    for i,case in enumerate(legacy.semantic_cases()):
        payload=dict(task_type='semantic_contract_test',case_id=case['case_id'],state=case['state'],
            memory=dict(similar_completed_episodes=[],summary='NO_COMPLETED_SIMILAR_EPISODES'),
            safe_exposure_previews={'BTC':.525,'CASH':.475,'ABSTAIN':.5})
        raw,outer,_=journal.complete(f'preflight-{i+1:02d}',payload)
        valid=False
        action='ABSTAIN'
        error=''
        try:
            if outer.get('done_reason')=='length': raise ValueError('Output limit')
            action=validate_decision(raw,config['agent'],set())['action']
            valid=True
        except (ValueError,TypeError) as exc: error=str(exc)
        match=valid and action==case['expected_action']
        records.append(dict(case_id=case['case_id'],valid=valid,action=action,expected=case['expected_action'],match=match,error=error))
        print(f'STAGE63_PREFLIGHT={i+1}/12 VALID={valid} EXPECTED={case["expected_action"]} ACTION={action}',flush=True)
    valid_rate=sum(x['valid'] for x in records)/len(records)
    match_rate=sum(x['match'] for x in records)/len(records)
    passed=(valid_rate>=config['preflight']['minimum_valid_rate'] and match_rate>=config['preflight']['minimum_expected_action_rate']
            and set(config['preflight']['required_actions']).issubset(x['action'] for x in records if x['valid']))
    return dict(passed=passed,valid_rate=valid_rate,expected_action_rate=match_rate,records=records,
                purpose='Schema and instruction comprehension only; not financial skill or memory learning')

def execute(args):
    package_sha=verify_package()
    config=load_config()
    output=args.output.expanduser().resolve()
    if output==PACKAGE or PACKAGE in output.parents:
        raise ValueError('Keep results outside the immutable code package')
    for relative in [config['source_result_relative'],config['corrected_run_relative'],config['adaptive_reference_relative']]:
        old=(args.project_root/relative).expanduser().resolve()
        if output==old or old in output.parents:
            raise ValueError('The result directory cannot overwrite historical source evidence')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'RUN.lock').open('a+')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Another process is using this result directory')
    try:
        existing=(output/'00_CONTRACT.json').exists()
        if args.audit_only and existing:
            raise RuntimeError('Use a fresh output directory for audit-only; do not overwrite a trading contract')
        if existing and not args.resume:
            raise RuntimeError('Result exists: use --resume with unchanged code, or choose a new directory')
        if args.resume and not existing:
            raise RuntimeError('Resume requires an existing 00_CONTRACT.json')
        print('RAMAS_STAGE6_3_INPUT_CHECK=START',flush=True)
        if args.demo:
            from stage63lib.fixture import build_fixture
            from stage63lib.scientific_audit import audit_controlled_fixture
            period,base_config,accounting,risk=build_fixture()
            audit={'kind':'CONTROLLED_NON_ECONOMIC','synthetic_rows':len(period.frame)}
            scientific_audit,adaptive_rows=audit_controlled_fixture(period,base_config,accounting)
            provider=ControlledProvider()
            provider_identity={'kind':'controlled','economic_evidence':False}
            config['evidence_kind']='CONTROLLED_NON_ECONOMIC'
        else:
            from stage63lib.inputs import load_inputs
            period,base_config,accounting,risk,audit=load_inputs(args.project_root,config,args.full_dataset)
            if len(period.frame)!=config['expected_rows']:
                raise RuntimeError('Incorrect experiment horizon')
            if float(base_config['transaction_cost_bps'])/10000!=config['cost_rate']:
                raise RuntimeError('Source cost mismatch')
            from stage63lib.scientific_audit import audit_scientific_inputs
            scientific_audit,adaptive_rows=audit_scientific_inputs(args.project_root,config,period,base_config,accounting)
            config['evidence_kind']='RETROSPECTIVE_LLAMA70B'
        # Source diagnostics are available even if serving checks subsequently fail.
        if not existing:
            atomic_json(output/'01_SOURCE_AUDIT.json',audit)
            atomic_json(output/'01A_SCIENTIFIC_AUDIT.json',scientific_audit)
        elif source.load_json(output/'01A_SCIENTIFIC_AUDIT.json')!=scientific_audit:
            raise RuntimeError('Scientific input audit changed on resume')
        print('SCIENTIFIC_SCOPE='+config['scientific_scope'],flush=True)
        print('FORECAST_AND_EXECUTION_TIMING_FULLY_VERIFIED=FALSE',flush=True)
        if args.audit_only:
            atomic_json(output/'AUDIT_ONLY_COMPLETE.json',dict(status='INPUT_AUDIT_COMPLETE_WITH_DECLARED_LIMITATIONS',
                new_llm_calls=0,scientific_scope=config['scientific_scope'],
                source_audit_sha256=digest(audit),scientific_audit_sha256=digest(scientific_audit)))
            print('RAMAS_STAGE6_3_AUDIT_STATUS=COMPLETE_WITH_DECLARED_LIMITATIONS\nNEW_LLM_CALLS=0\nOUTPUT='+str(output),flush=True)
            return 0
        if not args.demo:
            from stage63lib.provider import OllamaProvider
            provider=OllamaProvider(config['provider'],config['agent'])
            provider_identity=provider.pinned
            require_matched_provider(config,provider_identity,scientific_audit)
        print('RAMAS_STAGE6_3_INPUT_CHECK=PASS',flush=True)
        identity=dict(experiment_id=EXPERIMENT,config_sha256=source.sha256_file(PACKAGE/'config.json'),
            package_manifest_sha256=package_sha,source_audit_sha256=digest(audit),
            scientific_audit_sha256=digest(scientific_audit),
            system_prompt_sha256=source.sha256_text(SYSTEM_PROMPT),provider_identity=provider_identity,
            python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
            evidence_kind=config['evidence_kind'])
        if existing:
            contract=source.load_json(output/'00_CONTRACT.json')
            if contract['identity']!=identity:
                raise RuntimeError('Resume identity changed: source, config, package, runtime or model differs')
            if source.load_json(output/'01_SOURCE_AUDIT.json')!=audit:
                raise RuntimeError('Source audit changed on resume')
        else:
            # Commit the contract last so it never refers to a missing source audit.
            atomic_json(output/'01_SOURCE_AUDIT.json',audit)
            atomic_json(output/'00_CONTRACT.json',dict(identity=identity,config=config,
                started_utc=datetime.now(timezone.utc).isoformat(),initialization='BOTH_FRESH_EMPTY_2021',
                hypothesis='Effect of retrieved episodic evidence at fixed agent beta 0.05',
                agent_trust_mode='FIXED_POSITIVE',fixed_beta=0.05,adaptive_beta_updates_enabled=False,
                pretrained_llama_weights_updated=False,legacy_reference_seam_preserved=True,
                no_yearly_economic_hard_stop=True,primary='POST2021_MEAN_DAILY_LOG_MEMORY_MINUS_NO_MEMORY',
                independent_arm_state=True,original_stage61_actions_reused=False))
        if (output/'RUN_COMPLETE.json').exists():
            complete=source.load_json(output/'RUN_COMPLETE.json')
            for name,sha in complete['artifact_sha256'].items():
                if source.sha256_file(output/name)!=sha:
                    raise RuntimeError('Completed artifact changed: '+name)
            print('RAMAS_STAGE6_3_STATUS='+complete['status'],flush=True)
            print('RESUME_ACTION=ALREADY_COMPLETE_NO_NEW_CALLS',flush=True)
            return 0 if complete['status']=='PASS' else 5
        journal=Journal(output,provider,identity)
        semantic=preflight(journal,config)
        atomic_json(output/'02_PREFLIGHT.json',semantic)
        if not semantic['passed']:
            raise RuntimeError('Semantic preflight failed; no economic replay started')
        rows,arms=run_pair(period,config,base_config,accounting,risk,journal,output)
        if any(abs(row[name+'_beta']-0.05)>1e-15 for row in rows for name in arms):
            raise RuntimeError('Fixed-beta invariant failed')
        if any(arm.trust.events for arm in arms.values()):
            raise RuntimeError('Fixed-trust experiment unexpectedly updated adaptive beta')
        if hasattr(provider,'assert_identity'): provider.assert_identity()
        source.atomic_csv(output/'03_DAILY_TRACE.csv',pd.DataFrame(rows))
        for name,arm in arms.items():
            source.atomic_jsonl(output/'episodes'/(name+'.jsonl'),arm.ledger.episodes)
            atomic_json(output/'trust'/(name+'.json'),dict(beta_final=arm.trust.beta,events=arm.trust.events,
                 mode='FIXED_POSITIVE',fixed_beta=0.05,adaptive_beta_updates_enabled=False,
                 stale_evidence_changes=arm.stale_changes,pretrade_final=arm.pretrade,
                 shadow_pretrade_final=arm.shadow_pretrade,terminal_wealth=arm.wealth))
        summary=write_reports(output,rows,config,adaptive_rows=adaptive_rows or None)
        technical=all(a.valid/len(rows)>=config['minimum_valid_action_rate'] for a in arms.values())
        final=dict(status='PASS' if technical else 'TECHNICAL_VALIDITY_FAILED',
             decision='FIXED_TRUST_MEMORY_EXPERIMENT_COMPLETE_REQUIRES_INTERPRETATION',
             technical_pipeline_valid=technical,evidence_kind=config['evidence_kind'],
             scientific_scope=config['scientific_scope'],fully_verified_forecast_and_execution_timing=False,
             agent_trust_mode='FIXED_POSITIVE',fixed_beta=0.05,adaptive_beta_updates_enabled=False,
             primary_evidence_flag=summary['evidence_flag'],continuous_learning_established=False,
             rows=len(rows),valid_actions={k:a.valid for k,a in arms.items()},
             new_calls_this_process=journal.new_calls,reused_calls_this_process=journal.reused_calls,
             archived_call_count=len(list((output/'calls').glob('*.json'))),
             no_qwen_requested=True,provider_identity=provider_identity,
             post2021_label='REUSED_OOS_DIAGNOSTIC' if not args.demo else 'CONTROLLED_NON_ECONOMIC',
             completed_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(output/'10_FINAL_STATUS.json',final)
        hashed={p.relative_to(output).as_posix():source.sha256_file(p) for p in sorted(output.rglob('*'))
                if p.is_file() and p.name not in ['RUN_COMPLETE.json','RUN.lock','SERVER.lock','FAILURE.json'] and not p.name.endswith('.pending')}
        atomic_json(output/'RUN_COMPLETE.json',dict(status=final['status'],identity=identity,artifact_sha256=hashed))
        for key,value in [('RAMAS_STAGE6_3_STATUS',final['status']),('DECISION',final['decision']),
                          ('PRIMARY_EVIDENCE_FLAG',summary['evidence_flag']),('QWEN_USED','FALSE'),('OUTPUT',str(output))]:
            print(f'{key}={value}',flush=True)
        return 0 if technical else 5
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path('/home/infonet/wahid/leader_router_fresh'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--full-dataset',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--audit-only',action='store_true',help='Audit frozen sources and timing limitations, with no Ollama or Llama calls')
    p.add_argument('--demo',action='store_true',help='Synthetic controlled-provider mechanism check; never economic evidence')
    args=p.parse_args()
    try: return execute(args)
    except Exception as exc:
        print(f'RAMAS_STAGE6_3_STATUS=STOP\nERROR={type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        return 2

if __name__=='__main__': raise SystemExit(main())
