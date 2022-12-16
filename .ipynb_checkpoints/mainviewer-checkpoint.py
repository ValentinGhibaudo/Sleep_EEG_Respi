# -*- coding: utf-8 -*-

from params import subjects, srate

from configuration import base_folder

import xarray as xr
import pandas as pd


import os
import json
import datetime
import dateutil.parser
import re
from pprint import pprint

import numpy as np

import ephyviewer as ev
from ephyviewer import MainViewer, TraceViewer, TimeFreqViewer, EpochViewer, EventList, VideoViewer, DataFrameView, InMemoryAnalogSignalSource

from myqt import QT, DebugDecorator



#~ try:
    #~ import av
    #~ HAVE_AV = True
#~ except:
    #~ HAVE_AV = False



#~ main_index = get_main_index()


def get_viewer_from_run_key(run_key, parent=None, with_video=False):
    settings_name = None
    
    
    prepros_bipol = xr.open_dataarray(base_folder / 'preproc' / f'{run_key}_bipol.nc')
    prepros_reref = xr.open_dataarray(base_folder / 'preproc' / f'{run_key}_reref.nc')
    resp_features = pd.read_excel(base_folder / 'resp_features' / f'{run_key}_resp_features.xlsx')
    hypnos = pd.read_excel(base_folder / 'hypnos'/  f'hypno_{run_key}_yasa.xlsx')
    hypnos['time'] = hypnos.index * 30.
    hypnos['duration'] = 30.
    
    
    spindles = pd.read_excel(base_folder / 'event_detection' / f'{run_key}_spindles_reref_yasa.xlsx')
    slowwaves = pd.read_excel(base_folder / 'event_detection' / f'{run_key}_slowwaves_reref_yasa.xlsx')
    
    
    
    
    
    
    # trigs = detect_odor_trig_job.get(run_key)
    
    # artifacts = detect_artifact_period_job.get(run_key)
    
    # datatime
    #~ datetime0 = annotations.get('rec_datetime', None)
    #~ show_label_datetime = (datetime0 is not None)
    datetime0 = None
    show_label_datetime = False


    # respi
    inspi_index = resp_features['start'].iloc[:-1].values
    expi_index = resp_features['transition'].iloc[:-1].values
    
    scatter_indexes_resp = {0: inspi_index, 1:expi_index}
    scatter_channels_resp = {0: [0], 1: [0]}
    scatter_colors_resp = {0: '#FF0000', 1: '#00FF00'}


    win = MainViewer(show_label_datetime=show_label_datetime, parent=parent, show_global_xsize=True,
            show_auto_scale=True, datetime0=datetime0, settings_name=settings_name, debug=False)

    #~ xsize = 60.

    t_start = 0

    #respi
    sig_resp = prepros_bipol.sel(chan='DEBIT').values[:, None]
    
    view1 = TraceViewer.from_numpy( sig_resp, srate, t_start, 'resp', channel_names=['resp'],
                scatter_indexes=scatter_indexes_resp, scatter_channels=scatter_channels_resp, scatter_colors=scatter_colors_resp)
    win.add_view(view1)
    view1.params['scale_mode'] = 'real_scale'
    #~ view1.params['xsize'] = xsize
    view1.params['display_labels'] = False
    view1.params['display_offset'] = False
    view1.params['antialias'] = True
    #~ view1.params['background_color'] = '#ffffff'
    view1.by_channel_params[ 'ch0' ,'color'] = '#ffc83c'
    
    
    
    ###### viewer2 
    channel_names = prepros_reref.coords['chan'].values
    
    scatter_indexes = {}
    scatter_channels = {}
    scatter_colors = {}
    i = 0
    for chan_name in np.unique(slowwaves['Channel']):
        mask = slowwaves['Channel'] == chan_name
        df = slowwaves[mask]
        chan = list(channel_names).index(chan_name)
        scatter_indexes[i] = (df['NegPeak'] * srate).astype('int64')
        scatter_channels[i] = [chan]
        scatter_colors[i] = 'red'
        i += 1

    for chan_name in np.unique(spindles['Channel']):
        mask = spindles['Channel'] == chan_name
        df = spindles[mask]
        chan = list(channel_names).index(chan_name)
        scatter_indexes[i] = (df['Peak'] * srate).astype('int64')
        scatter_channels[i] = [chan]
        scatter_colors[i] = 'green'
        i += 1
    
    sigs = prepros_reref.values.T
    view2 = TraceViewer.from_numpy(sigs,  srate, t_start, 'reref', channel_names=channel_names, 
                scatter_indexes=scatter_indexes, scatter_channels=scatter_channels, scatter_colors=scatter_colors)
    win.add_view(view2)
    view2.params['display_labels'] = True
    view2.params['scale_mode'] = 'same_for_all'
    view2.by_channel_params[ 'ch15' ,'visible'] = False
    for c, chan_name in enumerate(channel_names):
        view2.by_channel_params[ f'ch{c}' ,'visible'] = c < 11
    
    
    #### viewer 5 
    source = InMemoryAnalogSignalSource(sigs, srate, t_start, channel_names=channel_names)
    #create a time freq viewer conencted to the same source
    view5 = TimeFreqViewer(source=source, name='tfr')
    win.add_view(view5)
    view5.params['show_axis'] = True
    view5.params['timefreq', 'deltafreq'] = 0.5
    view5.params['timefreq', 'f0'] = 3.
    view5.params['timefreq', 'f_start'] = 11.
    view5.params['timefreq', 'f_stop'] = 17.
    for c, chan_name in enumerate(channel_names):
        view5.by_channel_params[ f'ch{c}' ,'visible'] = c < 1




    #### viewer 3
    channel_names = prepros_bipol.coords['chan'].values
    sigs = prepros_bipol.values.T
    view3 = TraceViewer.from_numpy(sigs,  srate, t_start, 'bipol', channel_names=channel_names)
    win.add_view(view3)
    view3.params['display_labels'] = True
    view3.params['scale_mode'] = 'same_for_all'
    for c, chan_name in enumerate(channel_names):
        view3.by_channel_params[ f'ch{c}' ,'visible'] = c < 9

    
    
    #~ for i in range(all_sigs.shape):
        
    #### viewer 4 
    periods = []
    for k in ('W', 'N1', 'N2', 'N3', 'R'):
        mask = hypnos['str'] == k
        d = {
            'time' : hypnos.loc[mask, 'time'].values,
            'duration' : hypnos.loc[mask, 'duration'].values,
            'label': hypnos.loc[mask, 'str'].values,
            'name': k
        }
        periods.append(d)

    view4 = EpochViewer.from_numpy(periods, 'hypno')
    win.add_view(view4)

    
    #####
    events = []
    events.append({
        'time' : slowwaves['NegPeak'].values,
        'duration' : np.zeros(slowwaves.shape[0]),
        'label': slowwaves['Channel'].astype(str),
        'name': 'Slow Waves ',
        }
    )
    events.append({
        'time' : spindles['Peak'].values,
        'duration' : np.zeros(spindles.shape[0]),
        'label': slowwaves['Channel'].astype(str),
        'name': 'spindles',
        }
    )

    
    
    view6 = EventList.from_numpy(events, 'slowwaves')
    win.add_view(view6, location='right',  orientation='horizontal')


    win.set_xsize(60.)

    win.auto_scale()
    
        
    
    #~ sig_duration = sig_resp.shape[0] / sr
    #~ win.navigation_toolbar.set_start_stop(t_start, t_start+sig_duration)
    
    

    return win


def test_get_viewer():
    
    run_key = 'S1'

    app = ev.mkQApp()
    win = get_viewer_from_run_key(run_key)
    
    
    win.show()
    app.exec_()


if __name__ == '__main__':
    #~ test_find_avi_file()
    
    test_get_viewer()
    
    
    
