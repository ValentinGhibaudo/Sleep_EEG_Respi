import numpy as np
import scipy.interpolate

# Tool used to rescale time axis of time-frequency sigma epochs into respiratory cycle phase axis

def deform_to_cycle_template(data, times, cycle_times, nb_point_by_cycle=40, inspi_ratio = 0.4):
    """
    Input:
    data: ND array time axis must always be 0
    times: real timestamps associated to data
    cycle_times: N*2 array columns are inspi and expi times. If expi is "nan", corresponding cycle is skipped
    nb_point_by_cycle: number of respi phase per cycle
    inspi_ratio: relative length of the inspi in a full cycle (between 0 and 1)
    
    Output:
    clipped_times: real times used (when both respi and signal exist)
    times_to_cycles: conversion of clipped_times in respi cycle phase
    cycles: array of cycle indices (rows of cycle_times) used
    cycle_points: respi cycle phases where deformed_data is computed
    deformed_data: data rescaled to have cycle_points as "time" reference
    """
    
    #~ nb_point_inspi = int(nb_point_by_cycle*inspi_ratio)
    #~ nb_point_expi = nb_point_by_cycle - nb_point_inspi
    #~ one_cycle = np.linspace(0,1,nb_point_by_cycle)
    #~ two_cycle = np.linspace(0,2,nb_point_by_cycle*2)
    
    #~ print('cycle_times.shape', cycle_times.shape)
    
    #clip cycles if data/times smaller than cycles
    keep_cycle = (cycle_times[:, 0]>=times[0]) & (cycle_times[:, 1]<times[-1])
    first_cycle = np.where(keep_cycle)[0].min()
    last_cycle = np.where(keep_cycle)[0].max()+1
    if last_cycle==cycle_times.shape[0]:
        #~ print('yep')
        last_cycle -= 1
    #~ print('first_cycle', first_cycle, 'last_cycle', last_cycle)

    #clip times/data if cycle_times smaller than times
    keep = (times>=cycle_times[first_cycle,0]) & (times<cycle_times[last_cycle,0])
    #~ print(keep)
    clipped_times = times[keep]
    clipped_data = data[keep]
    #~ print('clipped_times', clipped_times.shape, clipped_times[0], clipped_times[-1])
    
    # construct cycle_step
    times_to_cycles = np.zeros(clipped_times.shape)*np.nan
    cycles = np.arange(first_cycle, last_cycle)
    t_start = clipped_times[0]
    sr = np.median(np.diff(clipped_times))
    #~ print('t_start', t_start, 'sr', sr)
    for c in cycles:
        #2 segments : inspi + expi
        
        if not np.isnan(cycle_times[c, 1]):
            #no missing cycles
            mask_inspi_times=(clipped_times>=cycle_times[c, 0])&(clipped_times<cycle_times[c, 1])
            mask_expi_times=(clipped_times>=cycle_times[c, 1])&(clipped_times<cycle_times[c+1, 0])
            times_to_cycles[mask_inspi_times]=(clipped_times[mask_inspi_times]-cycle_times[c, 0])/(cycle_times[c, 1]-cycle_times[c, 0])*inspi_ratio+c
            times_to_cycles[mask_expi_times]=(clipped_times[mask_expi_times]-cycle_times[c, 1])/(cycle_times[c+1, 0]-cycle_times[c, 1])*(1-inspi_ratio)+c+inspi_ratio
                    
        else:
            #there is a missing cycle
            mask_cycle_times=(clipped_times>=cycle_times[c, 0])&(clipped_times<cycle_times[c+1, 0])
            times_to_cycles[mask_cycle_times]=(clipped_times[mask_cycle_times]-cycle_times[c, 0])/(cycle_times[c+1, 0]-cycle_times[c, 0])+c
    
    # new clip with cycle
    keep = ~np.isnan(times_to_cycles)
    times_to_cycles = times_to_cycles[keep]
    clipped_times = clipped_times[keep]
    clipped_data = clipped_data[keep]
    
    
    interp = scipy.interpolate.interp1d(times_to_cycles, clipped_data, kind='linear', axis=0, bounds_error=False, fill_value='extrapolate')
    cycle_points = np.arange(first_cycle, last_cycle, 1./nb_point_by_cycle)

    if cycle_points[-1]>times_to_cycles[-1]:
        # it could append that the last bins of the last cycle is out last cycle
        # due to rounding effect so:
        last_cycle = last_cycle-1
        cycles = np.arange(first_cycle, last_cycle)
        cycle_points = np.arange(first_cycle, last_cycle, 1./nb_point_by_cycle)
    
    deformed_data = interp(cycle_points)
    
    #put NaN for missing cycles
    missing_ind,  = np.nonzero(np.isnan(cycle_times[:, 1]))
    #~ print('missing_ind', missing_ind)
    for c in missing_ind:
        #mask = (cycle_points>=c) & (cycle_points<(c+1))
        #due to rounding problem add esp
        esp = 1./nb_point_by_cycle/10.
        mask = (cycle_points>=(c-esp)) & (cycle_points<(c+1-esp))
        deformed_data[mask] = np.nan
    
    
    return clipped_times, times_to_cycles, cycles, cycle_points, deformed_data
    
    
    
    
    
    
    
def time_to_cycle(times, cycle_times,  inspi_ratio = 0.4):
    """
    Map absoulut event time to cycle position.
    Util for spike, events, trigs...
    
    Input:
    times: a times vector
    cycle_times: N*2 array columns are inspi and expi times. If expi is "nan", corresponding cycle is skipped
    
    Output:
    cycles: cycle position for times (same size than times) nan if outside.
    
    """
    
    n = cycle_times.shape[0]
    num_seg_phase = cycle_times.shape[1]
    assert num_seg_phase in (1, 2)
    
    
    cycle_point = np.zeros_like(cycle_times)
    if num_seg_phase ==1:
        cycle_point[:, 0] = np .arange(n)
    elif num_seg_phase ==2:
        cycle_point = np.zeros_like(cycle_times)
        cycle_point[:, 0] = np .arange(n)
        cycle_point[:, 1] = np .arange(n) + inspi_ratio
    
    flat_cycle_times = cycle_times.flatten()
    flat_cycle_point = cycle_point.flatten()
    keep = ~np.isnan(flat_cycle_times)
    flat_cycle_times = flat_cycle_times[keep]
    flat_cycle_point = flat_cycle_point[keep]
    
    interp = scipy.interpolate.interp1d(flat_cycle_times, flat_cycle_point, kind='linear', axis=0, bounds_error=False, fill_value='extrapolate')
    
    
    inside = (times>=cycle_times[0,0]) & (times<cycle_times[-1,0])
    cycles = np.zeros_like(times) * np.nan
    cycles[inside] = interp(times[inside])
    
    # put nan when some times are in missing cycles
    if num_seg_phase == 2:
        ind_missing, = np.nonzero(np.isnan(cycle_times[:, 1]))
        in_missing = np.in1d(np.floor(cycles), ind_missing.astype(cycles.dtype))
        cycles[in_missing] = np.nan
    
    
    return cycles
    

