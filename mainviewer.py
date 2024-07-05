# -*- coding: utf-8 -*-

import netCDF4

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

from preproc_staging import preproc_job, hypnogram_job
from rsp_detection import resp_features_job
from detect_sleep_events import spindles_detect_job, slowwaves_detect_job
# Just a viewer of raw or preprocessed data + detections to check for right functionning of processes

### PARAMS
mono_chans = ['Fz','Fp1','Fp2','C4','C3','Cz','Pz']

###




def get_viewer_from_run_key(run_key, parent=None, with_video=False):
    settings_name = None
    
    prepros_reref = preproc_job.get(run_key)['preproc']
    srate = prepros_reref.attrs['srate']
    resp_features = resp_features_job.get(run_key).to_dataframe()
    hypnos = hypnogram_job.get(run_key).to_dataframe()
    
    
    spindles = spindles_detect_job.get(run_key).to_dataframe()
    slowwaves = slowwaves_detect_job.get(run_key).to_dataframe()
    
    
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



    t_start = 0

    #respi
    sig_resp = prepros_reref.sel(chan='DEBIT').values[:, None]
    
    view1 = TraceViewer.from_numpy( sig_resp, srate, t_start, 'resp', channel_names=['resp'],
                scatter_indexes=scatter_indexes_resp, scatter_channels=scatter_channels_resp, scatter_colors=scatter_colors_resp)
    win.add_view(view1)
    view1.params['scale_mode'] = 'real_scale'
    view1.params['display_labels'] = False
    view1.params['display_offset'] = False
    view1.params['antialias'] = True
    view1.by_channel_params[ 'ch0' ,'color'] = 'ffc83c'
    
    
    
    ###### viewer2 
    channel_names = mono_chans
    
    scatter_indexes = {}
    scatter_channels = {}
    scatter_colors = {}
    i = 0

    # SLOWWAVES
    # for chan_name in mono_chans:
    #     mask = slowwaves['Channel'] == chan_name
    #     df = slowwaves[mask]
    #     chan = list(channel_names).index(chan_name)
    #     scatter_indexes[i] = (df['Start'] * srate).astype('int64')
    #     scatter_channels[i] = [chan]
    #     scatter_colors[i] = '00F404'
    #     i += 1

    for chan_name in mono_chans:
        mask = slowwaves['Channel'] == chan_name
        df = slowwaves[mask]
        chan = list(channel_names).index(chan_name)
        scatter_indexes[i] = (df['NegPeak'] * srate).astype('int64')
        scatter_channels[i] = [chan]
        scatter_colors[i] = 'f3ff33'
        i += 1

    # for chan_name in mono_chans:
    #     mask = slowwaves['Channel'] == chan_name
    #     df = slowwaves[mask]
    #     chan = list(channel_names).index(chan_name)
    #     scatter_indexes[i] = (df['End'] * srate).astype('int64')
    #     scatter_channels[i] = [chan]
    #     scatter_colors[i] = '000BED'
    #     i += 1

    # SPINDLES
    for chan_name in mono_chans:
        mask = spindles['Channel'] == chan_name
        df = spindles[mask]
        chan = list(channel_names).index(chan_name)
        scatter_indexes[i] = (df['Peak'] * srate).astype('int64')
        scatter_channels[i] = [chan]
        scatter_colors[i] = 'ff33ca'
        i += 1

    for chan_name in mono_chans:
        mask = spindles['Channel'] == chan_name
        df = spindles[mask]
        chan = list(channel_names).index(chan_name)
        scatter_indexes[i] = (df['Start'] * srate).astype('int64')
        scatter_channels[i] = [chan]
        scatter_colors[i] = 'ffffff'
        i += 1

    for chan_name in mono_chans:
        mask = spindles['Channel'] == chan_name
        df = spindles[mask]
        chan = list(channel_names).index(chan_name)
        scatter_indexes[i] = (df['End'] * srate).astype('int64')
        scatter_channels[i] = [chan]
        scatter_colors[i] = 'ff0000'
        i += 1
    
    
    sigs = prepros_reref.sel(chan=mono_chans).values.T
    print(sigs.dtype)
    view2 = TraceViewer.from_numpy(sigs,  srate, t_start, 'reref', channel_names=channel_names, 
                scatter_indexes=scatter_indexes, scatter_channels=scatter_channels, scatter_colors=scatter_colors)
    win.add_view(view2)
    view2.params['display_labels'] = True
    view2.params['scale_mode'] = 'same_for_all'

    #### viewer 5 
    source = InMemoryAnalogSignalSource(sigs, srate, t_start, channel_names=channel_names)
    #create a time freq viewer conencted to the same source
    view5 = TimeFreqViewer(source=source, name='tfr')
    win.add_view(view5)
    view5.params['show_axis'] = True
    view5.params['timefreq', 'deltafreq'] = 0.2
    view5.params['timefreq', 'f0'] = 3.
    view5.params['timefreq', 'f_start'] = 10.
    view5.params['timefreq', 'f_stop'] = 20.
    for c, chan_name in enumerate(channel_names):
        view5.by_channel_params[ f'ch{c}' ,'visible'] = c < 1

        
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

    win.set_xsize(30.)

    win.auto_scale()
    
        
    


    return win


def test_get_viewer():
    
    run_key = 'S20'

    app = ev.mkQApp()
    win = get_viewer_from_run_key(run_key)
    
    
    win.show()
    app.exec_()


if __name__ == '__main__':
    test_get_viewer()
    
    
    
