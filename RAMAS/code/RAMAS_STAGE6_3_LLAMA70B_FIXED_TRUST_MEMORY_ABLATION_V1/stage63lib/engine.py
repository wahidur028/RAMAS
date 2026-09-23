"""Independent sequential arms; only retrieved episodic evidence is ablated."""
from dataclasses import dataclass,field
from datetime import date
import json
import math
import numpy as np
from .legacy import legacy, EpisodicMemory, state_vector, validate_decision, blend_exposure
from .journal import atomic_json,digest
from .fixed_trust import FixedTrust, validate_fixed_trust_config

FEATURE_COLUMNS=('return_1','return_7','return_30','return_90','realized_vol_30',
                 'drawdown_90','router_entropy','router_confidence','router_transition_l1')

@dataclass
class Arm:
    name:str
    trust:object
    ledger:object=field(default_factory=EpisodicMemory)
    pretrade:float=0.
    shadow_pretrade:float=0.
    wealth:float=1.
    valid:int=0
    invalid_streak:int=0
    stale_changes:list=field(default_factory=list)
    previous_evidence:dict=field(default_factory=dict)

def eligible_episodes(episodes,decision):
    # Date strings are ISO YYYY-MM-DD, parsed before comparisons.
    current=date.fromisoformat(str(decision))
    return [x for x in episodes if date.fromisoformat(x['return_date'])<=current
            and date.fromisoformat(x['decision_date'])<current]

def make_state(period,i,pretrade,base_desired,beta):
    q=np.asarray(period.router_probabilities[i],float)
    state=dict(decision_date=str(period.decision_dates[i].date()),
               target_return_date=str(period.return_dates[i].date()),
               prob_bear=float(q[0]),prob_bull=float(q[1]),prob_mix=float(q[2]),
               hard_regime=('bear','bull','mix')[int(np.argmax(q))],
               pretrade_exposure=float(pretrade),base_ramoe_desired_exposure=float(base_desired),
               llama_trust_beta=float(beta))
    for k in FEATURE_COLUMNS: state[k]=float(period.features.iloc[i][k])
    if (date.fromisoformat(state['target_return_date'])-date.fromisoformat(state['decision_date'])).days!=1:
        raise ValueError('Decisions must precede their return by one day')
    return state

def payload_for(arm,period,i,config,base_config,risk):
    decision_date=period.decision_dates[i].date()
    completed=eligible_episodes(arm.ledger.episodes,decision_date)
    if not isinstance(arm.trust,FixedTrust):
        raise TypeError('Stage 6.3 requires FixedTrust in every arm')
    base_desired=float(period.current_trace.iloc[i]['desired_exposure'])
    regime=('bear','bull','mix')[int(np.argmax(period.router_probabilities[i]))]
    beta=arm.trust.value(regime)
    state=make_state(period,i,arm.pretrade,base_desired,beta)
    visible_memory=EpisodicMemory(completed)
    maximum=config['agent']['maximum_similar_episodes'] if arm.name=='memory' else 0
    evidence=visible_memory.evidence_summary(state,maximum)
    previews={}
    for action in config['agent']['actions']:
        projected=legacy.project_action(action,beta,base_desired,arm.pretrade,period.scenarios[i],base_config,risk)
        previews[action]=dict(blended_desired_exposure=blend_exposure(base_desired,action,beta),
              risk_limited_exposure=float(projected.exposure),turnover=float(projected.turnover),
              ambiguity_cvar=float(projected.ambiguity_cvar))
    payload=dict(task_type='historical_development_decision',
        objective='add incremental risk-adjusted value over the pre-agent RAMoE control after costs',
        state=state,memory=evidence,safe_exposure_previews=previews,
        constraints=dict(allowed_actions=['BTC','CASH','ABSTAIN'],shorting=False,leverage=False,
                         invalid_or_failed_response='ABSTAIN_TO_PRE_AGENT_RAMOE'))
    # The sole memory API path is evidence; outcome/other-arm rows never enter it.
    if arm.name=='no_memory' and (evidence['similar_completed_episodes'] or evidence['summary']!='NO_COMPLETED_SIMILAR_EPISODES'):
        raise RuntimeError('No-memory arm leaked episodic information')
    return payload

def run_pair(period,config,base_config,accounting,risk,journal,output):
    validate_fixed_trust_config(config)
    arms={name:Arm(name,FixedTrust(config)) for name in config['arms']}
    rows=[]
    for i in range(len(period.frame)):
        # Alternate service order to avoid making one arm always go first.
        order=config['arms'] if i%2==0 else config['arms'][::-1]
        row=dict(index=i,decision_date=str(period.decision_dates[i].date()),
                 return_date=str(period.return_dates[i].date()),
                 hard_regime=('bear','bull','mix')[int(np.argmax(period.router_probabilities[i]))],
                 asset_simple_return=float(period.asset_returns[i]))
        for name in order:
            arm=arms[name]
            payload=payload_for(arm,period,i,config,base_config,risk)
            key=f'{i+1:06d}-{name}'
            raw,outer,latency=journal.complete(key,payload)
            visible={e['episode_id'] for e in payload['memory']['similar_completed_episodes']}
            decision=dict(action='ABSTAIN',confidence=0.,reason_codes=['INSUFFICIENT_EVIDENCE'],cited_memory_ids=[])
            valid=False
            error=''
            try:
                if outer.get('done_reason')=='length':
                    raise ValueError('Output hit the token limit')
                decision=validate_decision(raw,config['agent'],visible)
                valid=True
            except (ValueError,TypeError) as exc:
                error=f'{type(exc).__name__}: {exc}'
            state=payload['state']
            action=decision['action']
            beta=state['llama_trust_beta']
            base_desired=state['base_ramoe_desired_exposure']
            projected=legacy.project_action(action,beta,base_desired,arm.pretrade,period.scenarios[i],base_config,risk)
            shadow=legacy.project_action(action,1. if action!='ABSTAIN' else 0.,base_desired,
                                        arm.shadow_pretrade,period.scenarios[i],base_config,risk)
            asset=row['asset_simple_return']
            net=float(accounting.net_return(projected.exposure,arm.pretrade,asset,config['cost_rate']))
            shadow_net=float(accounting.net_return(shadow.exposure,arm.shadow_pretrade,asset,config['cost_rate']))
            base_return=float(period.current_trace.iloc[i]['portfolio_net_return'])
            if not all(math.isfinite(x) and x>-1 for x in [net,shadow_net,base_return]):
                raise RuntimeError('Invalid portfolio return')
            if not 0<=float(projected.exposure)<=1:
                raise RuntimeError('Invalid risk-limited exposure')
            episode=dict(episode_id=f'episode-{i+1:06d}',decision_date=row['decision_date'],return_date=row['return_date'],
                hard_regime=row['hard_regime'],state_vector=state_vector(state).tolist(),action=action,
                confidence=float(decision['confidence']),shadow_log_advantage_vs_ramoe=math.log1p(shadow_net)-math.log1p(base_return),
                asset_return=asset)
            visible_episodes={e['episode_id']:e for e in arm.ledger.episodes if e['episode_id'] in visible}
            ages=[(date.fromisoformat(row['decision_date'])-date.fromisoformat(e['return_date'])).days
                  for e in visible_episodes.values()]
            previews=payload['safe_exposure_previews']
            preview_values=[p['risk_limited_exposure'] for p in previews.values()]
            arm.ledger.add_completed(episode)
            values=dict(net_return=net,exposure=float(projected.exposure),turnover=float(projected.turnover),
                action=action,beta=beta,valid=valid,cited_memory_count=len(decision['cited_memory_ids']),
                retrieved_memory_count=len(visible),memory_total_before=i,
                shadow_net_return=shadow_net,pretrade_exposure=arm.pretrade,shadow_pretrade_exposure=arm.shadow_pretrade,
                desired_exposure=blend_exposure(base_desired,action,beta),shadow_exposure=float(shadow.exposure),
                confidence=float(decision['confidence']),error=error,request_sha256=digest(payload),
                latency_seconds=latency,
                core_desired_exposure=base_desired,
                action_headroom=max(preview_values)-min(preview_values)>1e-12,
                projection_changes_selected_target=abs(float(projected.exposure)-blend_exposure(base_desired,action,beta))>1e-12,
                retrieved_memory_cross_year_count=sum(e['return_date'][:4]<row['decision_date'][:4]
                                                      for e in visible_episodes.values()),
                retrieved_memory_max_age_days=max(ages) if ages else None,
                wealth_before=arm.wealth,wealth_after=arm.wealth*(1+net))
            for preview_action,preview in previews.items():
                values['preview_'+preview_action.lower()+'_desired_exposure']=preview['blended_desired_exposure']
                values['preview_'+preview_action.lower()+'_exposure']=preview['risk_limited_exposure']
            row.update({name+'_'+k:v for k,v in values.items()})
            arm.pretrade=float(accounting.drifted_exposure(projected.exposure,asset))
            arm.shadow_pretrade=float(accounting.drifted_exposure(shadow.exposure,asset))
            arm.wealth*=1+net
            arm.valid+=int(valid)
            arm.invalid_streak=0 if valid else arm.invalid_streak+1
        rows.append(row)
        # These are reconstruction checkpoints, not mutable training seeds.
        atomic_json(output/'PROGRESS.json',dict(completed_days=i+1,total_days=len(period.frame),
            last_return_date=row['return_date'],state={name:dict(wealth=a.wealth,pretrade=a.pretrade,
            shadow_pretrade=a.shadow_pretrade,beta=a.trust.beta,episodes=len(a.ledger.episodes)) for name,a in arms.items()}))
        if i==0 or (i+1)%25==0 or i+1==len(period.frame):
            print(f"STAGE63_PROGRESS={i+1}/{len(period.frame)} MEMORY_VALID={arms['memory'].valid} NO_MEMORY_VALID={arms['no_memory'].valid} NEW_CALLS={journal.new_calls} REUSED_CALLS={journal.reused_calls}",flush=True)
        # Invalid model output is retained as ABSTAIN. An economic loss never stops.
        if any(a.invalid_streak>=config.get('maximum_consecutive_invalid',3) for a in arms.values()):
            raise RuntimeError('Three consecutive invalid model outputs; journal retained for interface investigation')
    return rows,arms
