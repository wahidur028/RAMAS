"""Descriptive response and transmission analysis; no portfolio returns."""
from pathlib import Path
import collections, csv, itertools, statistics
import runner as r

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(''); return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def rate(num, den):
    return num/den if den else None

def pair_metrics(a, b, preview):
    aa, bb = a['decision'], b['decision']
    pa, pb = preview[aa['action']], preview[bb['action']]
    return {'advice_disagreement':int(aa['action'] != bb['action']),
            'desired_disagreement':int(abs(pa['blended_desired_exposure']-pb['blended_desired_exposure'])>r.TOL),
            'execution_disagreement':int(abs(pa['risk_limited_exposure']-pb['risk_limited_exposure'])>r.TOL),
            'abstention_difference':int(aa['action']=='ABSTAIN')-int(bb['action']=='ABSTAIN'),
            'exposure_difference':pa['risk_limited_exposure']-pb['risk_limited_exposure'],
            'confidence_absolute_difference':abs(aa['confidence']-bb['confidence']),
            'reason_set_disagreement':int(set(aa['reason_codes']) != set(bb['reason_codes'])),
            'citation_set_disagreement':int(set(aa['cited_memory_ids']) != set(bb['cited_memory_ids']))}

def aggregate_pairs(rows):
    n = len(rows)
    advice = sum(x['advice_disagreement'] for x in rows)
    desired = sum(x['desired_disagreement'] for x in rows)
    execution = sum(x['execution_disagreement'] for x in rows)
    if execution > advice:
        raise ValueError('Same action gave different deterministic execution')
    return {'valid_pairs':n, 'advice_disagreements':advice, 'desired_disagreements':desired,
            'execution_disagreements':execution, 'advice_disagreement_rate':rate(advice,n),
            'execution_disagreement_rate':rate(execution,n),
            'conditional_transmission_rate':rate(execution,advice),
            'desired_merges':advice-desired, 'projection_merges':desired-execution,
            'abstention_difference_pp':100*sum(x['abstention_difference'] for x in rows)/n if n else None,
            'exposure_difference_pp':100*sum(x['exposure_difference'] for x in rows)/n if n else None}

def analyze(output):
    bank, agent, prompt = r.load_inputs_cached()
    contract, schedule, records = r.load_run(output, bank, agent, prompt)
    out = output/'analysis'; out.mkdir(exist_ok=True)
    indices = contract['indices']; repeats = range(1,contract['repeats']+1)
    lookup = {(x['model'],x['state_index'],x['condition'],x['repeat']):x for x in records}
    by_key = {x['key']:x for x in records}
    calls = []
    for cell in schedule:
        rec = by_key.get(cell['key']); s = bank[cell['state_index']]['request']['state']
        row = {**cell, 'decision_date':s['decision_date'],'return_date':s['target_return_date'],
               'regime':s['hard_regime'], 'recorded':rec is not None,
               'valid':bool(rec and rec['valid']), 'model_digest':contract['identity']['serving']['models'][cell['model']]['digest']}
        if rec:
            outer = rec['provider_response'] or {}
            row.update(error=rec['error'], latency_seconds=rec['latency_seconds'],
                       prompt_tokens=outer.get('prompt_eval_count'), output_tokens=outer.get('eval_count'),
                       stop_reason=outer.get('done_reason'),request_sha256=rec['request_sha256'],
                       response_record_sha256=rec['record_sha256'],
                       transport_outcome_unknown=rec['transport_outcome_unknown'])
            if rec['valid']:
                decision = rec['decision']; preview = bank[cell['state_index']]['request']['safe_exposure_previews'][decision['action']]
                row.update(action=decision['action'],confidence=decision['confidence'],
                           reason_labels='|'.join(decision['reason_codes']),cited_episode_ids='|'.join(decision['cited_memory_ids']),
                           desired_exposure=preview['blended_desired_exposure'],
                           executed_exposure=preview['risk_limited_exposure'])
        calls.append(row)
    write_csv(out/'calls.csv', calls)
    distributions = []
    for model, condition, repeat in itertools.product(r.MODELS,r.CONDITIONS,repeats):
        g = [x for x in calls if (x['model'],x['condition'],x['repeat']) == (model,condition,repeat)]
        valid = [x for x in g if x['valid']]
        dist = {'model':model,'condition':condition,'repeat':repeat, 'planned':len(g),
                'recorded':sum(x['recorded'] for x in g), 'valid':len(valid),
                'invalid_or_missing':len(g)-len(valid)}
        for action in ('BTC','CASH','ABSTAIN'):
            count = sum(x['action']==action for x in valid)
            dist[action+'_count'] = count; dist[action+'_valid_rate'] = rate(count,len(valid))
        latencies = [x['latency_seconds'] for x in g if x.get('latency_seconds') is not None]
        dist['mean_latency_seconds'] = statistics.mean(latencies) if latencies else None
        dist['mean_executed_exposure'] = statistics.mean(x['executed_exposure'] for x in valid) if valid else None
        distributions.append(dist)
    write_csv(out/'validity_and_distributions.csv', distributions)

    paired, aggregates, common_counts, state_summary = [], [], {}, []
    for model in r.MODELS:
        common = {i for i in indices if all((model,i,c,j) in lookup and lookup[model,i,c,j]['valid']
                                           for c,j in itertools.product(r.CONDITIONS,repeats))}
        common_counts[model] = len(common)
        specifications = []
        for c in r.CONDITIONS:
            specifications += [('within_'+c,a,b,c,c) for a,b in itertools.combinations(repeats,2)]
        specifications += [('between_contexts',j,j,'exposed','hidden') for j in repeats]
        for kind, ja, jb, ca, cb in specifications:
            rows = []
            for i in indices:
                a, b = lookup.get((model,i,ca,ja)), lookup.get((model,i,cb,jb))
                if a is None or b is None or not a['valid'] or not b['valid']:
                    continue
                q = bank[i]['request']
                row = {'model':model,'comparison':kind,'repeat_a':ja,'repeat_b':jb,
                       'condition_a':ca,'condition_b':cb,'state_index':i,
                       'decision_date':q['state']['decision_date'],'regime':q['state']['hard_regime'],
                       'all_responses_valid_for_state':i in common,
                       **pair_metrics(a,b,q['safe_exposure_previews'])}
                rows.append(row); paired.append(row)
            for scope in ('available_valid_pairs','all_responses_valid_states'):
                selected = rows if scope=='available_valid_pairs' else [x for x in rows if x['state_index'] in common]
                expected = indices if scope=='available_valid_pairs' else sorted(common)
                for regime in ('all','bear','bull','mix'):
                    g = selected if regime=='all' else [x for x in selected if x['regime']==regime]
                    n = sum(regime=='all' or bank[i]['request']['state']['hard_regime']==regime for i in expected)
                    aggregates.append({'model':model,'comparison':kind,'repeat_a':ja,'repeat_b':jb,
                                       'scope':scope,'regime':regime,'eligible_states':n,
                                       'omitted_invalid_or_missing_pairs':n-len(g),**aggregate_pairs(g)})
        if contract['repeats'] == 3:
            for i in sorted(common):
                p = bank[i]['request']['safe_exposure_previews']
                ex = [lookup[model,i,'exposed',j] for j in repeats]
                hi = [lookup[model,i,'hidden',j] for j in repeats]
                cross = [pair_metrics(a,b,p) for a,b in itertools.product(ex,hi)]
                within_ex = [pair_metrics(a,b,p) for a,b in itertools.combinations(ex,2)]
                within_hi = [pair_metrics(a,b,p) for a,b in itertools.combinations(hi,2)]
                row = {'model':model,'state_index':i,'decision_date':bank[i]['request']['state']['decision_date']}
                for label,g in [('cross_all_nine_pairs',cross),('within_exposed_three_pairs',within_ex),('within_hidden_three_pairs',within_hi)]:
                    for measure in ('advice_disagreement','execution_disagreement'):
                        row[label+'_'+measure] = statistics.mean(x[measure] for x in g)
                for measure in ('advice_disagreement','execution_disagreement'):
                    row['cross_minus_mean_within_'+measure] = row['cross_all_nine_pairs_'+measure] - (
                        row['within_exposed_three_pairs_'+measure]+row['within_hidden_three_pairs_'+measure])/2
                state_summary.append(row)
    write_csv(out/'paired_state_results.csv', paired)
    write_csv(out/'paired_summary.csv', aggregates)
    write_csv(out/'secondary_state_repeat_summary.csv', state_summary)

    overview = []
    for model in r.MODELS:
        entries = [x for x in aggregates if x['model']==model and x['scope']=='all_responses_valid_states' and x['regime']=='all']
        for kind in ('within_exposed','within_hidden','between_contexts'):
            g = [x for x in entries if x['comparison']==kind]
            for metric in ('advice_disagreement_rate','execution_disagreement_rate','conditional_transmission_rate',
                           'abstention_difference_pp','exposure_difference_pp'):
                values = [x[metric] for x in g if x[metric] is not None]
                overview.append({'model':model,'comparison':kind,'metric':metric,'complete_states':common_counts[model],
                                 'defined_repeat_pairs':len(values),'mean':statistics.mean(values) if values else None,
                                 'minimum':min(values) if values else None,'maximum':max(values) if values else None})
    write_csv(out/'repeat_ranges.csv', overview)
    secondary_means = {}
    for model in r.MODELS:
        g = [x for x in state_summary if x['model']==model]
        secondary_means[model] = {k:statistics.mean(x[k] for x in g) for k in g[0] if k not in ('model','state_index','decision_date')} if g else {}
    summary = {'mode':contract['mode'],'planned_calls':len(schedule),'recorded_calls':len(records),
               'valid_calls':sum(x['valid'] for x in records), 'complete_valid_states':common_counts,
               'call_completion':len(records)==len(schedule), 'all_valid':len(records)==len(schedule) and all(x['valid'] for x in records),
               'secondary_equal_state_weighted_means':secondary_means,
               'zero_disagreement_transmission':'undefined, represented as null or an empty CSV cell',
               'independent_observations':'states; repeated calls and pair combinations are dependent',
               'inference':'descriptive repeat ranges only; no significance test or power claim',
               'portfolio_returns_computed':False,'semantic_content_isolated':False,
               'model_family_effect_isolated':False,'prompt_length_matched':False}
    if contract['mode']=='pilot':
        values = [x['mean_latency_seconds'] for x in distributions]
        summary['estimated_main_hours_from_pilot'] = sum(values)*1244*3/3600 if all(x is not None for x in values) else None
        summary['runtime_note'] = 'Approximate sequential runtime; includes pilot latency, excludes pauses and changing server load.'
    r.atomic(out/'summary.json',summary)
    print(r.canonical(summary))
    return summary
