#!/usr/bin/env python3
"""Collect missing existing response provenance. Does not launch models."""
from pathlib import Path
import argparse, hashlib, json, zipfile

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=Path('/home/infonet/wahid/leader_router_fresh'))
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'runs/existing_metadata.zip')
    ap.add_argument('--balanced-run',type=Path,help='Exact recorded balanced-retrieval run, if stored elsewhere')
    args=ap.parse_args()
    branches=args.project/'EXPERIMENT_BRANCHES'
    rerun=branches/'RAMAS_STAGE7_SEQUENTIAL_MEMORY_VALIDATION_V1/artifacts/20260911T053155161422662Z'
    factorial=branches/'RAMAS_FULL_PIPELINE_ISOLATION_V1/artifacts/20260920T084300Z'
    roots=[('rerun',rerun),('factorial',factorial)]
    balanced_parent=branches/'RAMAS_STAGE8_CORRECTED_MEMORY_V1/artifacts'
    balanced=[]
    if balanced_parent.is_dir() or args.balanced_run is not None:
        contracts=[args.balanced_run/'00_RUN_CONTRACT.json'] if args.balanced_run else balanced_parent.glob('*/00_RUN_CONTRACT.json')
        for p in contracts:
            try:
                c=json.loads(p.read_text())
                if (c.get('started_at')=='2026-09-22T01:49:28+00:00' and
                    c.get('identity',{}).get('config_sha256')=='bb43811e1dcfdfcb92123ec5cc4e53d7869980906c9faebba414b2904ce8fbec'):
                    balanced.append(p.parent)
            except (ValueError,OSError):
                continue
    if len(balanced)==1:
        roots.append(('balanced',balanced[0]))
    names={'00_CONTRACT.json','00_RUN_CONTRACT.json','RUN_COMPLETE.json','PREFLIGHT.json',
           'PREFLIGHT_PASS.json','01_PREFLIGHT.json','PROTOCOL_LOCK.json','config.json',
           'PREFLIGHT_REPORT.json','provider_identity.json','STATUS.json','COMPARISONS.json',
           'last_transport_failure.json','last_serving_failure.json'}
    report={'sources':[{'label':label,'path':str(root),'exists':root.is_dir()} for label,root in roots],
            'balanced_matching_runs':len(balanced),'copied_files':[],
            'purpose':'Recover original serving and raw validity provenance; not a new experiment',
            'transport_attempt_logs':'Not assumed complete. Supply separate original retry/error logs if available.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED) as z:
        for label,root in roots:
            if not root.is_dir():
                continue
            for p in sorted(root.rglob('*')):
                if not p.is_file():
                    continue
                relative=p.relative_to(root)
                raw=p.suffix=='.json' and any(x in ('calls','responses','pending','pending_calls') for x in relative.parts)
                if label=='factorial' and 'cells' in relative.parts and 'llama3.3_70b__seed_42' not in relative.parts:
                    continue
                if p.name not in names and not raw:
                    continue
                archive_name=str(Path(label)/relative); z.write(p,archive_name)
                report['copied_files'].append({'relative_path':archive_name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        z.writestr('COLLECTION_REPORT.json',json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'sources':report['sources'],
                      'balanced_matching_runs':len(balanced),'files_copied':len(report['copied_files'])}))

if __name__=='__main__':
    main()
