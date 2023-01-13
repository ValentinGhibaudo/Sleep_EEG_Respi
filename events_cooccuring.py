import numpy as np
import pandas as pd
from params import subjects, spindles_freq_threshold


def cooccuring_sp_sw_df(spindles, slowwaves): 
    
    """
    This function will add columns to the initial dataframe giving information of presence of the spindle inside slowwave or not and in this case, when does it occur in the slowwave
    -------------
    Inputs =
    spindles : dataframe of spindles
    slowwaves : dataframe of slowwaves
    
    Outputs = 
    sp_return : dataframe of spindles labelized according to the co-occurence or not with slowwave
    sw_return : dataframe of slowwaves labelized according to the presence of spindles inside or not
    """
    
    features_cooccuring_sp = []
    sw_with_spindle_inside = []
    for ch in slowwaves['Channel'].unique(): # loop on chans
        for stage in slowwaves['Stage_Letter'].unique(): # loop on stages 
            sw_staged_ch = slowwaves[(slowwaves['Channel'] == ch)&(slowwaves['Stage_Letter'] == stage)]  # mask slowwave of the chan and the stage
            sp_staged_ch = spindles[(spindles['Channel'] == ch)&(spindles['Stage_Letter'] == stage)]  # mask spindles of the chan and the stage

            if not sw_staged_ch.shape[0] == 0: # if masked slowwave df is not empty ...
                for i, row in sw_staged_ch.iterrows(): # ... loop on rows = on slowwaves 

                    start_window = row['Start'] # get start time of the slowwave
                    stop_window = row['End'] # get stop time of the slowwave
                    negpeak = row['NegPeak'] # get time of the neg peak of the slowwave
                    duration = row['Duration'] # get duration of the slowwave

                    co_occuring_spindles = sp_staged_ch[(sp_staged_ch['Peak'] >= start_window) & (sp_staged_ch['Peak'] < stop_window)] # mask spindle present between start and stop time of slowwave
                    
                    if not co_occuring_spindles.shape[0] == 0: # if masked spindle df is not empty ...
                        sw_with_spindle_inside.append(i) # ... add to a list the index of the slowwave 
                        for s, sp in co_occuring_spindles.iterrows(): # loop on spindles of the slowwave

                            t = sp['Peak'] # get peak time of the spindle
                            cooccuring_with_sw = True # set boolean information that spindle is in a slowwave
                            absolute_t_vs_negpeak = t - negpeak # compute time between peak of the spindle and negpeak of the slowwave
                            absolute_t_vs_start = t - start_window  # compute time between peak of the spindle and start of the slowwave
                            relative_t_vs_duration = absolute_t_vs_start / duration # compute relative time of spindle peak in the slowwave duration
                            features_cooccuring_sp.append([s, cooccuring_with_sw, absolute_t_vs_negpeak, absolute_t_vs_start, relative_t_vs_duration]) 
                            
                            
                        
    cooccurors = pd.DataFrame(features_cooccuring_sp, columns = ['index','in_slowwave','t_vs_NegPeak_sw','t_vs_Start_sw','relative_t_in_sw']).set_index('index') # spindle cooccuring df
    
    sp_return = spindles.reindex(columns = list(spindles.columns) + list(cooccurors.columns)) # extend the columns of the initial spindle df
    sp_return.loc[cooccurors.index,cooccurors.columns] = cooccurors # add the cooccuring df to initial spindle df
    sp_return.loc[:,'in_slowwave'] = sp_return['in_slowwave'].fillna(False) # fill na with False
    
    slowwaves_return = slowwaves.copy()
    slowwaves_return['sp_inside'] = np.nan 
    slowwaves_return.loc[sw_with_spindle_inside, 'sp_inside'] = True # add a column in the slowwave df with idea of presence of spindle inside or not
    slowwaves_return.loc[:,'sp_inside'] = slowwaves_return['sp_inside'].fillna(False)
    
    return sp_return, slowwaves_return


for subject in subjects:
    print(subject)
    
    spindles = pd.read_excel(f'../event_detection/{subject}_spindles_reref_yasa.xlsx', index_col = 0)
    slowwaves = pd.read_excel(f'../event_detection/{subject}_slowwaves_reref_yasa.xlsx', index_col = 0)
    
    sp_cooccuring, sw_spindled = cooccuring_sp_sw_df(spindles, slowwaves)
    
    sp_speed = sp_cooccuring.copy()
    sp_speed['Sp_Speed'] = np.nan
    sp_speed.loc[:,'Sp_Speed'] = (sp_speed['Frequency'] >= spindles_freq_threshold[subject]).map({False:'SS',True:'FS'}) # add a column on spindle df setting if the spindle is a slow or a fast spindle according to the set threshold manually chosen for each subject (bimodal distribution of frequency of spindles)
    
    sp_speed.to_excel(f'../event_detection/{subject}_spindles_cooccuring.xlsx')
    sw_spindled.to_excel(f'../event_detection/{subject}_slowwaves_cooccuring.xlsx')
    
    
    