# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 13:27:58 2022

@author: rintala
"""

import numpy as np

def breslow_baseline_survival_time_estimate_bad(event, time, risk, pred_times):
    time_filter = time >= 0
    
    order = np.argsort(time[time_filter])
    
    risk_score = np.exp(risk[time_filter])
    risk_score_ordered = risk_score[order]
    cum_risk = np.flip(np.cumsum(np.flip(risk_score_ordered)))
    
    times, times_idx, times_n = np.unique(time[time_filter][order], return_inverse = True, return_counts = True)
    
    total_events = np.cumsum(event[time_filter][order])
    unique_time_counts = np.cumsum(times_n)
    unique_time_events_total = total_events[unique_time_counts - 1]
    unique_time_events = np.concatenate([[unique_time_events_total[0]], 
                                         np.diff(unique_time_events_total)], 
                                        axis = 0)
    
    # How many have occurred or been censored up and including this moment
    #n_at_risk_removed = np.concatenate([[0], unique_time_counts], axis = 0)
    # Cumulative sum of reciprocals of at risk cumulative hazards
    cum_risk_reciprocal = np.cumsum(unique_time_events[:-1] / cum_risk[unique_time_counts[:-1]])
    
    base_survival = np.full((pred_times.shape[0],), np.nan)
    
    for i in np.arange(pred_times.shape[0]):
        times_ind = np.argwhere(times >= pred_times[i]).item(0)
        # Cumulative baseline hazard
        cum_bh = cum_risk_reciprocal.item(times_ind)
        # Baseline survival
        base_survival[i] = np.exp(- cum_bh)
    
    return base_survival

def breslow_baseline_survival_time_estimate(event, time, risk, pred_times):
    time_filter = time >= 0
    
    order = np.argsort(time[time_filter])
    
    risk_score = np.exp(risk[time_filter])
    #risk_score = risk_score[order]
    
    # Get counts for times, events and at risk
    
    uniq_times, n_events, n_at_risk, _ = lifetable(event[time_filter], time[time_filter], order)
    
    divisor = np.empty(n_at_risk.shape, dtype=float)
    value = np.sum(risk_score[order])
    divisor[0] = value
    k = 0
    for i in range(1, len(n_at_risk)):
        d = n_at_risk[i - 1] - n_at_risk[i]
        value -= risk_score[order][k:(k + d)].sum()
        k += d
        divisor[i] = value

    assert k == n_at_risk[0] - n_at_risk[-1]

    baseline_hazard = np.cumsum(n_events / divisor)
    baseline_survival = np.full((pred_times.shape[0],), np.nan)
    for i in np.arange(pred_times.shape[0]):
        times_ind = np.argwhere(uniq_times > pred_times[i]).item(0)
        baseline_survival[i] = np.exp(-baseline_hazard[times_ind])
    return baseline_survival

def lifetable(event, time, order):
    n_samples = event.shape[0]

    uniq_times = np.empty(n_samples, dtype=time.dtype)
    uniq_events = np.empty(n_samples, dtype=int)
    uniq_counts = np.empty(n_samples, dtype=int)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    times = np.resize(uniq_times, j)
    n_events = np.resize(uniq_events, j)
    total_count = np.resize(uniq_counts, j)
    n_censored = total_count - n_events

    # offset cumulative sum by one
    total_count = np.r_[0, total_count]
    n_at_risk = n_samples - np.cumsum(total_count)

    return times, n_events, n_at_risk[:-1], n_censored