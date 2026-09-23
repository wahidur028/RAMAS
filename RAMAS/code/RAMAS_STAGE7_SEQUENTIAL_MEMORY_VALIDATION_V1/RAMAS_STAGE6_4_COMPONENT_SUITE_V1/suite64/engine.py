"""One independently reconstructed continuous arm of the Stage 6.4 suite.

All outcome advantages deliberately use the immutable legacy numerical-reference
return. Retrieval variants alter only the visible evidence, never the private
trust ledger. This is a matched legacy-mechanism diagnostic, not a clock repair.
"""
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import json
import math
import os

import numpy as np
import pandas as pd

from stage63lib.engine import Arm, eligible_episodes, make_state
from stage63lib.fixed_trust import FixedTrust
from stage63lib.journal import atomic_json, digest
from stage63lib.legacy import (
    EpisodicMemory, RegimeTrust, blend_exposure, legacy, state_vector,
    validate_decision,
)


MEMORY_MODES = {'expanding', 'none', 'frozen2021', 'current_year', 'pooled'}
REGIMES = ('bear', 'bull', 'mix')


class PooledMemory(EpisodicMemory):
    """The original distance/ranking contract with no same-regime prefilter."""
    def retrieve(self, state, maximum):
        if maximum <= 0:
            return []
        query = state_vector(state)
        candidates = [
            (float(np.linalg.norm(query - np.asarray(e['state_vector'], dtype=float))), e)
            for e in self.episodes
        ]
        candidates.sort(key=lambda item: (item[0], item[1]['episode_id']))
        return [dict(
            episode_id=e['episode_id'], decision_date=e['decision_date'],
            hard_regime=e['hard_regime'], action=e['action'],
            confidence=e['confidence'],
            shadow_log_advantage_vs_ramoe=e['shadow_log_advantage_vs_ramoe'],
            asset_return=e['asset_return'], distance=distance,
        ) for distance, e in candidates[:maximum]]


def retrieval_evidence(episodes, state, mode, maximum):
    """Build causal model evidence without mutating the supplied private ledger."""
    if mode not in MEMORY_MODES:
        raise ValueError('Unknown memory mode: ' + str(mode))
    current = date.fromisoformat(state['decision_date'])
    completed = eligible_episodes(episodes, current)
    if mode == 'none':
        maximum = 0
    elif mode == 'frozen2021':
        completed = [e for e in completed if date.fromisoformat(e['return_date']) <= date(2021, 12, 31)]
    elif mode == 'current_year':
        completed = [e for e in completed if date.fromisoformat(e['return_date']).year == current.year]
    memory_class = PooledMemory if mode == 'pooled' else EpisodicMemory
    return memory_class(completed).evidence_summary(state, maximum)


def project_action(action, beta, base_desired, pretrade, scenarios, base_config, risk, risk_mode='standard'):
    if risk_mode == 'standard':
        return legacy.project_action(action, beta, base_desired, pretrade, scenarios, base_config, risk)
    if risk_mode != 'none':
        raise ValueError('Unknown risk mode: ' + str(risk_mode))
    # This ablation removes CVaR projection, exposure-grid rounding and the
    # turnover constraint together. Long-only/no-leverage bounds remain.
    desired = float(blend_exposure(base_desired, action, beta))
    if not math.isfinite(desired):
        raise ValueError('Non-finite desired exposure')
    exposure = float(np.clip(desired, 0., 1.))
    return SimpleNamespace(exposure=exposure, turnover=abs(exposure - pretrade), ambiguity_cvar=None)


def _update_trust(arm, decision_date, config):
    completed = eligible_episodes(arm.ledger.episodes, decision_date)
    if isinstance(arm.trust, FixedTrust):
        return
    before = len(arm.trust.events)
    arm.trust.maybe_update(decision_date, completed)
    # Preserve Stage 6.2's trust-audit annotations as well as its update rule.
    for event in arm.trust.events[before:]:
        regime = event['regime']
        ids = [e['episode_id'] for e in completed
               if e['hard_regime'] == regime and e['action'] != 'ABSTAIN'][-config['trust']['lookback_episodes_per_regime']:]
        old = arm.previous_evidence.get(regime, [])
        event['eligible_episode_ids'] = ids
        event['fresh_evidence_since_prior_update'] = bool(set(ids) - set(old))
        event['actual_latest_evidence_return_date'] = max(
            (e['return_date'] for e in completed if e['episode_id'] in ids), default=None)
        if not event['fresh_evidence_since_prior_update'] and event['old_beta'] != event['new_beta']:
            arm.stale_changes.append(dict(event))
        arm.previous_evidence[regime] = ids


def payload_for(arm, period, i, config, base_config, risk, spec, base_desired=None):
    decision_date = period.decision_dates[i].date()
    _update_trust(arm, decision_date, config)
    if base_desired is None:
        base_desired = float(period.current_trace.iloc[i]['desired_exposure'])
    regime = REGIMES[int(np.argmax(period.router_probabilities[i]))]
    beta = 0. if spec['advisor'] == 'abstain' else arm.trust.value(regime)
    state = make_state(period, i, arm.pretrade, base_desired, beta)
    evidence = retrieval_evidence(arm.ledger.episodes, state, spec['memory'],
                                  config['agent']['maximum_similar_episodes'])
    previews = {}
    for action in config['agent']['actions']:
        projected = project_action(action, beta, base_desired, arm.pretrade,
                                   period.scenarios[i], base_config, risk, spec['risk'])
        previews[action] = dict(
            blended_desired_exposure=blend_exposure(base_desired, action, beta),
            risk_limited_exposure=float(projected.exposure), turnover=float(projected.turnover),
            ambiguity_cvar=None if projected.ambiguity_cvar is None else float(projected.ambiguity_cvar))
    # Preserve every legacy payload key/string for default-memory/default-risk
    # arms. New experiment identity belongs to the journal, not this prompt.
    return dict(
        task_type='historical_development_decision',
        objective='add incremental risk-adjusted value over the pre-agent RAMoE control after costs',
        state=state, memory=evidence, safe_exposure_previews=previews,
        constraints=dict(allowed_actions=['BTC', 'CASH', 'ABSTAIN'], shorting=False,
                         leverage=False, invalid_or_failed_response='ABSTAIN_TO_PRE_AGENT_RAMOE'))


def _validate_spec(spec):
    required = {'name', 'advisor', 'trust', 'memory', 'risk'}
    if required - set(spec):
        raise ValueError('Missing arm fields: ' + str(sorted(required - set(spec))))
    name = spec['name']
    if not isinstance(name, str) or not name or not all(c.isalnum() or c in '-_' for c in name):
        raise ValueError('Invalid arm name')
    for field, allowed in [('advisor', {'llama', 'rule', 'abstain'}),
                           ('trust', {'adaptive', 'fixed'}),
                           ('memory', MEMORY_MODES), ('risk', {'standard', 'none'})]:
        if spec[field] not in allowed:
            raise ValueError('Invalid ' + field + ': ' + str(spec[field]))
    if spec['trust'] == 'fixed':
        beta = spec.get('fixed_beta', .05)
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or beta != .05:
            raise ValueError('Matched fixed-trust arms require fixed_beta exactly 0.05')


def _new_arm(config, spec):
    if spec['trust'] == 'fixed':
        fixed_config = dict(config, fixed_beta=.05, agent_trust_mode='FIXED_POSITIVE',
                            common_legacy_trust_rule=False, arms=['memory', 'no_memory'],
                            continue_across_years=True)
        trust = FixedTrust(fixed_config)
    else:
        trust = RegimeTrust(config['trust'])
    return Arm(spec['name'], trust)


def _checkpoint(output, rows, arm, spec, total):
    output.mkdir(parents=True, exist_ok=True)
    # CSV is a diagnostic reconstruction checkpoint. Durable model responses
    # remain authoritative for recovery and are never resampled on restart.
    destination = output / 'daily_trace.partial.csv'
    temporary = destination.with_name(destination.name + '.pending')
    serial_rows = [{k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()} for row in rows]
    with temporary.open('w', newline='') as handle:
        pd.DataFrame(serial_rows).to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    atomic_json(output / 'PROGRESS.json', dict(
        arm=arm.name, spec=spec, completed_days=len(rows), total_days=total,
        last_return_date=rows[-1]['return_date'] if rows else None,
        wealth=arm.wealth, pretrade=arm.pretrade, shadow_pretrade=arm.shadow_pretrade,
        beta=arm.trust.beta, episodes=len(arm.ledger.episodes), valid=arm.valid,
        invalid_streak=arm.invalid_streak, checkpoint_type='RECONSTRUCT_FROM_DURABLE_CALLS'))


def run_arm(period, config, base_config, accounting, risk, journal, output, spec, core_desired=None):
    """Replay a single independent arm from empty memory/cash and initial trust.

    ``journal`` may be None for deterministic/abstain arms. The caller supplies
    an experiment-identity-guarded Stage 6.3-compatible journal for Llama arms.
    ``core_desired`` alters only the allocator's desired exposure; forecast and
    feature inputs and the outcome-reference stream are deliberately unchanged.
    """
    _validate_spec(spec)
    if spec['advisor'] == 'llama' and journal is None:
        raise ValueError('Llama arm requires a durable journal')
    if set(config['agent']['actions']) != {'BTC', 'CASH', 'ABSTAIN'}:
        raise ValueError('All three legacy actions are required')
    if not math.isfinite(config['cost_rate']) or not 0 <= config['cost_rate'] < 1:
        raise ValueError('Invalid proportional trading cost')
    total = len(period.frame)
    if not total:
        raise ValueError('Cannot run an empty period')
    decisions = [d.date() for d in period.decision_dates]
    if any((b-a).days != 1 for a, b in zip(decisions, decisions[1:])):
        raise ValueError('Continuous replay requires ordered consecutive decision days')
    if core_desired is None:
        core = np.asarray(period.current_trace['desired_exposure'], dtype=float)
    else:
        core = np.asarray(core_desired, dtype=float)
    if core.shape != (total,) or not np.all(np.isfinite(core)) or np.any((core < 0) | (core > 1)):
        raise ValueError('core_desired must contain one finite bounded exposure per day')
    output = Path(output)
    arm = _new_arm(config, spec)
    rows = []
    every = int(config.get('checkpoint_every', 25))
    if every <= 0:
        raise ValueError('checkpoint_every must be positive')
    for i in range(total):
        payload = payload_for(arm, period, i, config, base_config, risk, spec, float(core[i]))
        visible = {e['episode_id'] for e in payload['memory']['similar_completed_episodes']}
        decision = dict(action='ABSTAIN', confidence=0., reason_codes=['INSUFFICIENT_EVIDENCE'], cited_memory_ids=[])
        outer, latency, error = {}, 0., ''
        valid = True
        if spec['advisor'] == 'llama':
            raw, outer, latency = journal.complete(f'{i+1:06d}-{arm.name}', payload)
            try:
                if outer.get('done_reason') == 'length':
                    raise ValueError('Output hit the token limit')
                decision = validate_decision(raw, config['agent'], visible)
            except (ValueError, TypeError) as exc:
                valid = False
                error = f'{type(exc).__name__}: {exc}'
        elif spec['advisor'] == 'rule':
            decision['action'] = legacy.deterministic_action(payload['state'])
            # Confidence is uncalibrated and unused by this rule: keep zero.
            decision['reason_codes'] = (['BULLISH_ROUTER', 'POSITIVE_MOMENTUM'] if decision['action'] == 'BTC'
                                        else ['INSUFFICIENT_EVIDENCE'])
        state, action = payload['state'], decision['action']
        beta, base_desired = state['llama_trust_beta'], state['base_ramoe_desired_exposure']
        projected = project_action(action, beta, base_desired, arm.pretrade,
                                   period.scenarios[i], base_config, risk, spec['risk'])
        shadow = project_action(action, 1. if action != 'ABSTAIN' else 0., base_desired,
                                arm.shadow_pretrade, period.scenarios[i], base_config, risk, spec['risk'])
        asset = float(period.asset_returns[i])
        net = float(accounting.net_return(projected.exposure, arm.pretrade, asset, config['cost_rate']))
        shadow_net = float(accounting.net_return(shadow.exposure, arm.shadow_pretrade, asset, config['cost_rate']))
        reference_return = float(period.current_trace.iloc[i]['portfolio_net_return'])
        if not all(math.isfinite(v) and v > -1 for v in (asset, net, shadow_net, reference_return)):
            raise RuntimeError('Invalid asset/portfolio/reference return')
        if not 0 <= float(projected.exposure) <= 1:
            raise RuntimeError('Unbounded exposure')
        turnover = abs(float(projected.exposure) - arm.pretrade)
        if not math.isclose(turnover, float(projected.turnover), abs_tol=1e-12, rel_tol=0):
            raise RuntimeError('Projector turnover differs from own pretrade holdings')
        decision_date, return_date = state['decision_date'], state['target_return_date']
        visible_episodes = {e['episode_id']: e for e in arm.ledger.episodes if e['episode_id'] in visible}
        ages = [(date.fromisoformat(decision_date) - date.fromisoformat(e['return_date'])).days
                for e in visible_episodes.values()]
        previews = payload['safe_exposure_previews']
        preview_values = [p['risk_limited_exposure'] for p in previews.values()]
        desired = blend_exposure(base_desired, action, beta)
        row = dict(
            index=i, arm=arm.name, decision_date=decision_date, return_date=return_date,
            hard_regime=state['hard_regime'], asset_simple_return=asset,
            pretrade_exposure=arm.pretrade, exposure=float(projected.exposure),
            net_return_after_trading_costs=net, net_log_return_after_trading_costs=math.log1p(net),
            gross_return_before_trading_costs=float(projected.exposure)*asset,
            turnover=turnover, cost_fraction=config['cost_rate']*turnover,
            trading_cost_in_wealth_units=arm.wealth*config['cost_rate']*turnover,
            wealth_before=arm.wealth, wealth_after=arm.wealth*(1+net),
            action=action, beta=beta, core_desired_exposure=base_desired, desired_exposure=desired,
            shadow_pretrade_exposure=arm.shadow_pretrade, shadow_exposure=float(shadow.exposure),
            shadow_net_return=shadow_net, reference_net_return=reference_return,
            shadow_log_advantage_vs_ramoe=math.log1p(shadow_net)-math.log1p(reference_return),
            valid=valid, error=error, confidence=float(decision['confidence']),
            cited_memory_ids=list(decision['cited_memory_ids']),
            cited_memory_count=len(decision['cited_memory_ids']),
            retrieved_memory_ids=sorted(visible), retrieved_memory_count=len(visible),
            memory_total_before=len(arm.ledger.episodes),
            retrieved_memory_cross_year_count=sum(e['return_date'][:4] < decision_date[:4] for e in visible_episodes.values()),
            retrieved_memory_cross_regime_count=sum(e['hard_regime'] != state['hard_regime'] for e in visible_episodes.values()),
            retrieved_memory_max_age_days=max(ages) if ages else None,
            request_sha256=digest(payload), latency_seconds=float(latency),
            input_token_count=int(outer.get('prompt_eval_count', 0) or 0),
            output_token_count=int(outer.get('eval_count', 0) or 0),
            action_headroom=max(preview_values)-min(preview_values) > 1e-12,
            projection_changes_selected_target=abs(float(projected.exposure)-desired) > 1e-12,
            advisor=spec['advisor'], trust_mode=spec['trust'], memory_mode=spec['memory'], risk_mode=spec['risk'])
        for preview_action, preview in previews.items():
            row['preview_' + preview_action.lower() + '_desired_exposure'] = preview['blended_desired_exposure']
            row['preview_' + preview_action.lower() + '_exposure'] = preview['risk_limited_exposure']
        episode = dict(
            episode_id=f'episode-{i+1:06d}', decision_date=decision_date, return_date=return_date,
            hard_regime=state['hard_regime'], state_vector=state_vector(state).tolist(),
            action=action, confidence=float(decision['confidence']),
            shadow_log_advantage_vs_ramoe=row['shadow_log_advantage_vs_ramoe'], asset_return=asset)
        arm.ledger.add_completed(episode)
        arm.pretrade = float(accounting.drifted_exposure(projected.exposure, asset))
        arm.shadow_pretrade = float(accounting.drifted_exposure(shadow.exposure, asset))
        arm.wealth = row['wealth_after']
        arm.valid += int(valid)
        arm.invalid_streak = 0 if valid else arm.invalid_streak + 1
        rows.append(row)
        stop_invalid = arm.invalid_streak >= config.get('maximum_consecutive_invalid', 3)
        if i == 0 or (i+1) % every == 0 or i+1 == total or stop_invalid:
            _checkpoint(output, rows, arm, spec, total)
            print(f'STAGE64_ARM={arm.name} PROGRESS={i+1}/{total} VALID={arm.valid} '
                  f'NEW_CALLS={getattr(journal, "new_calls", 0)} REUSED_CALLS={getattr(journal, "reused_calls", 0)}', flush=True)
        if stop_invalid:
            raise RuntimeError('Consecutive invalid Llama responses; journal retained for investigation')
    atomic_json(output / 'episodes.json', arm.ledger.episodes)
    atomic_json(output / 'trust_events.json', arm.trust.events)
    atomic_json(output / 'ARM_STATE.json', dict(
        arm=arm.name, spec=spec, wealth=arm.wealth, pretrade=arm.pretrade,
        shadow_pretrade=arm.shadow_pretrade, beta=arm.trust.beta, episodes=len(arm.ledger.episodes),
        valid=arm.valid, stale_evidence_trust_changes=arm.stale_changes,
        outcome_reference='UNCHANGED_ARCHIVED_NUMERICAL_REFERENCE'))
    return rows, arm
