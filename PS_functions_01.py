

# PS Function Collection
"""
This script contains functions to preprocess MEG data, perform ICA, segment data, extract blink-related events,
and compute and save ERP data for multiple subjects. 
Written on 19.06.2024.

Functions Included:
1. load_raw_data(subj, run, data_path, dict_filtering):
    - Purpose: Load raw MEG data for a specific subject and run, and apply bandpass filtering.
    - Inputs:
        - subj: Subject identifier (string).
        - run: Run number (integer).
        - data_path: Path to the data directory (string).
        - dict_filtering: Dictionary with filtering parameters, specifically the frequency band (dictionary).
    - Outputs:
        - raw: Loaded and filtered raw MEG data (MNE Raw object).
        - events: Events found in the raw data (numpy array).

2. perform_ICA_and_find_badsEOG(raw, subj, save_path):
    - Purpose: Perform ICA on the raw data, identify and exclude ECG and EOG components, and save the ICA object.
    - Inputs:
        - raw: Raw MEG data (MNE Raw object).
        - subj: Subject identifier (string).
        - save_path: Path to save the ICA object and components (string).
    - Outputs:
        - ICA: Fitted ICA object (MNE ICA object).
        - rraw: Raw data with ICA components excluded (MNE Raw object).
        - blinkIC: Index of the blink component (integer).
        - blinkIC_data: Data of the blink component (numpy array).

3. segment_raw_data(raw, events, dict_trigger_orig, events_of_interest):
    - Purpose: Segment raw data based on events of interest and return run information.
    - Inputs:
        - raw: Raw MEG data (MNE Raw object).
        - events: Events found in the raw data (numpy array).
        - dict_trigger_orig: Dictionary mapping trigger names to event IDs (dictionary).
        - events_of_interest: List of event IDs of interest (list).
    - Outputs:
        - runs_per_sub_info: List of dictionaries containing information about each run (list of dictionaries).

4. extract_blinks_in_sections(run_data, sfreq, section_duration=60):
    - Purpose: Extract blink timestamps in sections of the run data.
    - Inputs:
        - run_data: Data for a specific run (numpy array).
        - sfreq: Sampling frequency of the data (float).
        - section_duration: Duration of each section to analyze in seconds (integer, default is 60).
    - Outputs:
        - blink_timestamps: Timestamps of detected blinks (numpy array).

5. extract_erps(runs_info, blinkIC_data, rraw, selected_channels, subj, data_path):
    - Purpose: Extract ERPs for each condition of interest.
    - Inputs:
        - runs_info: List of dictionaries containing information about each run (list of dictionaries).
        - blinkIC_data: Data of the blink component (numpy array).
        - rraw: Raw data with ICA components excluded (MNE Raw object).
        - selected_channels: List of channel names to include (list).
        - subj: Subject identifier (string).
        - data_path: Path to the data directory (string).
    - Outputs: None. ERPs are saved using the save_erp_data function.

6. save_erp_data(data_path, subj, run, condition, ERP):
    - Purpose: Save ERP data and metadata to a pickle file.
    - Inputs:
        - data_path: Path to the data directory (string).
        - subj: Subject identifier (string).
        - run: Run number (integer).
        - condition: Condition name (string).
        - ERP: ERP data (MNE Evoked object).
    - Outputs: None. ERP data is saved to a pickle file.

7. load_and_plot_erps(data_path, conditions_to_plot, brainregion='parietal', tmin=-0.8, tmax=1.2)
   - Purpose: Load ERP data for all subjects, sort and collect ERP data by condition,
    average the ERPs for each condition, and plot the averaged ERPs for specified conditions.

    - Inputs:
    - data_path: str, path to the data directory containing subject data.
    - conditions_to_plot: list of str, conditions to include in the plots.
    - brainregion: str, specify the brain region for the plot title (default: 'parietal').
    - tmin: float, start time for ERP plot window (default: -0.8s).
    - tmax: float, end time for ERP plot window (default: 1.2s).


Dependencies:
- os
- mne
- numpy
- pickle
- neurokit2
- matplotlib
- sklearn
- warnings
- PyQt5
- pandas
"""

import os
import mne
#from networkx import non_randomness
import numpy as np
import pickle
import neurokit2 as nk
import matplotlib.pyplot as plt
import sklearn
import warnings
import PyQt5
import pandas as pd
import neurokit2 as nk
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs 
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*does not conform to MNE naming conventions.*")
#%matplotlib qt

# Define Data Path and Subject List
# data_path = os.path.join(r"C:\Users\pemas\Nextcloud\Master Thesis\Data\derivatives\analyse_now")    # Data path on my Laptop
#data_path = os.path.join(r"Q:\ZA320330\Group430\Transfer\KL\cogni_affect\meg\PS\raw")                 # Data on Hiwi Computer



# 1. Function to load raw data for a subject and run
def load_raw_data(subj, run, data_path, dict_filtering):
    file_name = os.path.join(data_path, subj, 'run_0' + str(run), 'raw_fs_100Hz_grad+mag.fif')
    raw = mne.io.read_raw_fif(file_name, allow_maxshield=False, preload=True, verbose=False)
    raw.pick(picks=["mag", "stim", "eog"])
    #print(raw.first_samp) # for debugging

    # get events
    events = mne.find_events(raw, stim_channel='STI101', initial_event=True, consecutive=True, shortest_event=1, verbose=False)
    
    # Korrektur Problem in 'sub-ER04WE13'   : in run 2 gibt es 2x einen trigger 17
    # vielleicht ist 13025, 21, 17 gemeint?
    if subj == 'sub-ER04WE13' and run == 2:
            # Find the event to be corrected
            idx_to_be_fixed = np.where((events[:, 0] == 13025) & (events[:, 2] == 17))[0]
            if len(idx_to_be_fixed) > 0:
                events[idx_to_be_fixed[0], :] = [13025, 21, 2]
                print(f"Note: Corrected event trigger at index for subject {subj}")


    #events[:, 0] -= raw.first_samp

    # filtering
    iir_params = dict(order=4, ftype='butter', output='sos')
    iir_params = mne.filter.construct_iir_filter(iir_params, dict_filtering['ffreq'], None,
                                                 raw.info['sfreq'], 'bandpass', return_copy=False, verbose=False)
    raw = raw.filter(l_freq=dict_filtering['ffreq'][0], h_freq=dict_filtering['ffreq'][1], method='iir',
                     iir_params=iir_params, verbose=False)

    return raw, events





# 2. Function to segment raw data based on events of interest
def segment_raw_data(raw, run_nr, events, dict_trigger_orig, events_of_interest):
    """ Segment raw data for the input run/condition based on the start of the experimental condition.

    Parameters:
    -----------
    raw : mne.io.Raw object
        The raw MEG data for the subject.
    run_nr : int
        The run number of the current data segment / run.
    events : numpy array
        Array of shape (n_events, 3) containing event information (sample, previous, event_id).
    dict_trigger_orig : dict
        Dictionary mapping event IDs to condition names.
    events_of_interest : list
        List of event IDs that we defined as events of interest.

    Returns:
    --------
    list
        A list of dictionaries containing information about each segmented run,
        including run number, condition name, start and end sample indices, start
        and end times, and event ID.

    Notes:
    ------
    This function segments the raw MEG data into a continuous data segment based on events of interest identified by their event IDs.
    It extracts the segment of data from the start of the first event of interest to the end of raw data the given run
    (so it basically only cuts away the first segment before the condition starts)."""

    # Find the first event of interest
    event = events[np.isin(events[:, 2], events_of_interest)][0]
    
    # Determine start and end sample indices
    start_sample = event[0]
    end_sample = events[np.isin(events[:, 2], events_of_interest)][1, 0] if len(events[np.isin(events[:, 2], events_of_interest)]) > 1 else len(raw.times)
    
    # Calculate start and end times in seconds
    start_time = raw.times[start_sample] if start_sample < len(raw.times) else 0.0
    end_time = raw.times[end_sample - 1] if end_sample <= len(raw.times) else raw.times[-1]
    
    # Determine event ID and condition name
    # Create a reverse lookup dictionary for event_id to condition_name
    event_id_to_condition = {v: k for k, v in dict_trigger_orig.items()}

    # Determine event ID and condition name
    event_id = event[2]
    condition_name = event_id_to_condition.get(event_id, f'Unknown_trigger_{event_id}')
    
    # Return run information as a list containing a single dictionary
    return [{
        'run': run_nr,
        'condition': condition_name,
        'start_sample': start_sample,
        'start_time': start_time,
        'end_sample': end_sample,
        'end_time': end_time,
        'event_id': event_id
    }]






# 3. Perform ICA or load existing ICA
def perform_or_load_ICA(raw, subj, save_path):
    """ Perform ICA or load existing ICA instance and exclude ECG components.
    Parameters:
    -----------------------------------------
        raw : mne.io.Raw object
            The raw MEG data for the subject.
        subj : str
            Subject identifier.
        save_path : str
            Path to save the ICA object.

    Returns:
    -----------------------------------------
        ICA : mne.preprocessing.ICA
            The fitted ICA object.
        rraw : mne.io.Raw object
            Raw data after applying the ICA."""
    
    # First check if ICA object already exists for the subject
    ICA_filename = os.path.join(save_path, f'{subj}_ICA.pkl')
    if os.path.exists(ICA_filename):
        # Load existing ICA object
        with open(ICA_filename, 'rb') as f:
            ICA = pickle.load(f)
        print(f"Loaded the already existing ICA object for subject: {subj}")
    else:
        # If not: initialize new ICA object
        print("No ICA performed for this subject. Performing now... ")
        ICA = mne.preprocessing.ICA(n_components=0.99, random_state=97, method='infomax', 
                                    fit_params=dict(extended=True, max_iter=500), verbose=False)
        ICA.fit(raw)
        print(f"\nICA completed for subject: {subj}")

        # Save ICA object
        with open(ICA_filename, 'wb') as f:
            pickle.dump(ICA, f)
        print(f"ICA object pickled for subject: {subj}")

    # Exclude ECG components
    ICA.exclude = []
    ecg_indices, ecg_scores = ICA.find_bads_ecg(raw, method="correlation", threshold=0.99, verbose=False)
    ICA.exclude = ecg_indices[:3]  # Exclude the 3 components with strongest correlation with ECG

    # Apply ICA to raw data after excluding ECG components
    rraw = raw.copy()
    ICA.apply(rraw, verbose=False)

    return ICA, rraw





















# 4. find_bads_EOG
def find_badsEOG_and_get_blink_IC(ICA, rraw, subj, save_path, noisy_VEOG_channel):
    """ Find bad EOG components and get the blink IC (- data).
    Parameters:
    --------------------------------------------------------------
        ICA : mne.preprocessing.ICA
            The fitted ICA object.
        rraw : mne.io.Raw object
            Raw data after applying the ICA.
        subj : str
            Subject identifier.
        save_path : str
            Path to save the ICA results.
        noisy_VEOG_channel : list
            List of subjects with noisy VEOG channels.

    Returns:
    -------------------------------------------------------------
        ICA : mne.preprocessing.ICA
            The updated ICA object with excluded EOG components.
        rraw : mne.io.Raw object
            Raw data after excluding EOG components.
        blinkIC : int
            Index of the identified blink IC.
        blinkIC_data : numpy array
            Data of the identified blink IC.    """


    # ----------------------- create aEOG channel ----------------------------------------------------------------------
    frontal_channels = ['MEG0511', 'MEG0921']
    valid_channel_indices = [rraw.ch_names.index(ch_name) for ch_name in frontal_channels]
    raw_abs = rraw.copy()
    raw_abs._data = abs(rraw._data)
    groups = {'aEOG': valid_channel_indices}
    aEOG = mne.channels.combine_channels(raw_abs, groups, method='mean', keep_stim=False, drop_bad=False,  verbose=False)
    rraw.add_channels([aEOG], force_update_info=True)


    # ----------------------- find bads EOG ------------------------------------------------------------------------------
    
    eog_channels = ["VEOG", "aEOG"]        # Channels to try
    thresholds = [0.5, 0.45, 0.4]          # Thresholds to try

    if subj in noisy_VEOG_channel:
        eog_channels = ["aEOG"]       # Use only aEOG for subjects in noisy_VEOG_channel
        #thresholds = [0.5]           # Use only 0.5 threshold initially for noisy_VEOG_channel
        thresholds = [0.5, 0.45, 0.4]          

    eog_components = []
    lst_eog_scores = []

    for threshold in thresholds:
        for ch_name in eog_channels:
            components, eog_scores = ICA.find_bads_eog(
                inst=rraw,
                ch_name=ch_name,
                measure="correlation",
                threshold=threshold,
                verbose=False
            )
            if components:
                eog_components.extend(components)
                lst_eog_scores.append(eog_scores[components[0]])
                print(f'Source {ch_name} with IC {components[0]} and Score {np.round(eog_scores[components[0]], 3)}')
                break  # Stop searching if an EOG component is found
            else:
                print(f"No EOG component found for {ch_name} with the applied threshold of {threshold}.")

        if eog_components:
            break  # Stop searching if an EOG component is found

    if not eog_components:
        print(f"💥 💥 💥 No EOG component found by find_bads_eog for subject {subj} 💥 💥 💥")
        return None, None, None, None


    blinkIC = eog_components[lst_eog_scores.index(max(lst_eog_scores))] # identify the blink component (the one with the highest score)

    # get the blink IC data out of the rraw instance for the respective run/condition
#    blinkIC_data = ICA.get_sources(rraw)._data[eog_components[lst_eog_scores.index(max(lst_eog_scores))]]
    blinkIC_data = ICA.get_sources(rraw)._data[blinkIC]

    chosen_channel = eog_channels[lst_eog_scores.index(max(lst_eog_scores))]
    print(f'Continuing with {chosen_channel} channel.\n')

    # Check if the chosen channel is 'aEOG' and update the counter                              # note: finish this another time (not essential)
    # aEOG_counter = 0
    # if chosen_channel == 'aEOG':
    #     aEOG_counter += 1
    
    # Add blinkIC data as new channel to rraw instance
    info = mne.create_info(ch_names=['Blink_IC'], sfreq=rraw.info['sfreq'], ch_types='eog', verbose=False) # add blinkIC data as new channel to rraw inst.
    blink_raw = mne.io.RawArray(blinkIC_data[np.newaxis, :], info, verbose=False)
    rraw.add_channels([blink_raw], force_update_info=True)
    
    # exclude EOG Components 
    ICA.exclude = eog_components
    ICA.apply(rraw, verbose=False)

    # Save ICA results and other information
    ica_results_path = os.path.join(save_path, f'{subj}_ICA_results.pkl')
    data_to_save = {
        'subject': subj,
        'eog_channels': eog_channels,
        'eog_components': eog_components,
        'correlation_scores': lst_eog_scores,
        'blinkIC': blinkIC,
        'chosen_channel': chosen_channel
    }

    with open(ica_results_path, 'wb') as file:
        pickle.dump(data_to_save, file)

    return ICA, rraw, blinkIC, blinkIC_data











# 5. Function to extract ERPs for each condition of interest (equalise event count)
def extract_ERPs(runs_info, blinkIC_data, rraw, subj, save_path, blinkIC, tmin, tmax, section_duration, reject):
#def extract_ERPs(runs_info, blinkIC_data, rraw, selected_channels, subj, save_path, blinkIC, selected_channel_set):
    
    sfreq = rraw.info['sfreq']

    # Initialize dictionary to store epochs for each condition
    all_epochs = {condition: [] for condition in set(run_info['condition'] for run_info in runs_info)}
    
    all_blink_counts_per_subj = []  # Initialize a list to store blink counts for all runs

    for run_info in runs_info:
        run = run_info['run']
        start_sample = run_info['start_sample']
        end_sample = run_info['end_sample']
        event_id = run_info['event_id']
        condition = run_info['condition']

        #picks = mne.pick_channels(rraw.ch_names, include=selected_channels, ordered=True)         # Pick the channels from the raw data

        # Extract the relevant segment of blink IC data for the selected run
        blink_ic_data_run = blinkIC_data[start_sample:end_sample]

        # Extract blinks in sections (function call)
        blink_timestamps, blink_counts_per_run, total_blinks_per_run = extract_blinks_in_sections(blink_ic_data_run, sfreq, section_duration, subj, condition)    # FUNCTION CALL: EXTRACT BLINKS
        all_blink_counts_per_subj.append(blink_counts_per_run)          # Append the blink counts for this run to the overall list
        print(f"RUN run {condition}: total blinks found for subject {subj} = {total_blinks_per_run}")

        # Convert to a numpy array
        blink_timestamps = np.array(blink_timestamps)

        # Create events array with the blink timestamps
        blink_events = np.c_[blink_timestamps, np.zeros_like(blink_timestamps), np.full_like(blink_timestamps, event_id, dtype=int)]



        # Correct for first sample 📛📛📛
        print(f'\nFirst blink is found at sample {blink_events[0,0]} for raw with first sample at {rraw.first_samp}')
        blink_events[:, 0] = blink_events[:, 0] + rraw.first_samp # correcting
        print(f'First blink is at sample {blink_events[0,0]} after first sample correction')


        
        # Create epochs around these events for each run 
        epochs = mne.Epochs(rraw, events=blink_events, event_id=None, tmin=tmin, tmax=tmax, baseline=None, preload=True, reject=reject, verbose=False, picks='all')  # explicitly include all channels
        print(f"Number of epochs found: {len(epochs)}")

        # Inspect the drop log to see why epochs were rejected - for Debugging 
        drop_log = epochs.drop_log
        for i, reason in enumerate(drop_log):
            if reason:  # if the drop log entry is not empty
                print(f"\nEpoch {i} was rejected due to: {reason}")
        #fig = mne.viz.plot_drop_log(drop_log, threshold=0, n_max_plot=20, subject='subj', width=0.8, ignore=('IGNORED',), show=True)

        # Append epochs to the corresponding condition
        all_epochs[condition].append(epochs)


    # Concatenate epochs for each condition
    for condition in all_epochs:
        if len(all_epochs[condition]) > 1:
            all_epochs[condition] = mne.concatenate_epochs(all_epochs[condition])
        else:
            all_epochs[condition] = all_epochs[condition][0]

    # Ensure concatenated_epochs is a list of Epochs objects
    concatenated_epochs = list(all_epochs.values())

    # Equalize event counts across all conditions                                                                     # 🟣 EQUALIZE EPOCH COUNT HERE
    try:
        mne.epochs.equalize_epoch_counts(concatenated_epochs, method='mintime') # note: change method here later (idea: fit to shape of P300)  
    except Exception as e:                                                      #       room for improvement here, only few events are left
        print(f"Error during equalize_event_counts: {e}")


    # Update all_epochs with the equalized epochs
    all_epochs = dict(zip(all_epochs.keys(), concatenated_epochs)) 
    nr_of_epochs = len(next(iter(all_epochs.values())))
    print(f'\nNow we are left with {nr_of_epochs} epochs for each of the conditions.\n')


    # Apply baseline correction and compute ERPs for each condition # note: should I do baseline correction here?
    dict_all_ERPs_all_conditions = {} # dict, will store the averaged Event-Related Potentials (ERPs) for each condition
    for condition, epochs in all_epochs.items():
        epochs.apply_baseline(baseline=(-0.8, -0.5), verbose=False)
        ERP = epochs.average() # compute ERP for the current conditions
        dict_all_ERPs_all_conditions[condition] = ERP

        # Save the ERP data and metadata to  pickle file for each run/conditions
        save_erp_data(save_path, subj, condition, ERP, blinkIC, all_epochs, all_blink_counts_per_subj, total_blinks_per_run)  # Call save function
        print(f'ERP data extracted and saved for subject {subj}, condition {condition}.')

    print(f'ERP data extracted and saved for subject {subj}.')


    return nr_of_epochs















# 5.1 Function to extract a segment of data sections of one run and append them
def extract_blinks_in_sections(blink_ic_data_run, sfreq, section_duration, subj, condition):
    
    # if subj == 'sub-ER04WE13':
    #     blink_ic_data_run = -blink_ic_data_run          # flip the data of this subject 
    
    n_samples = len(blink_ic_data_run)
    section_samples = int(section_duration * sfreq)
    blink_timestamps = []
    total_blinks_per_run = 0  # to keep track of the total number of blinks for each run/condition

    # Dictionary to store blink counts
    blink_counts_per_run = {
        "subject": subj,
        "condition": condition,
        "section_duration": section_duration,
        "sections": []
    }

    #for start in range(0, n_samples, section_samples):
    for start in range(0, n_samples, section_samples):      # corrected for first sample intead of 0
        end = min(start + section_samples, n_samples)
        section_data = blink_ic_data_run[start:end]
        #section_data = np.abs(blink_ic_data_run[start:end]) # absolute values

        
        # Skip very short sections at the end of run
        if len(section_data) < sfreq:  # less than 1 second of data
            continue
        
        # Neurokit Toolbox to find blinks in this section
        try:
            eog_signals, info = nk.eog_process(section_data, sampling_rate=sfreq, method='neurokit')                                        # NEUROKIT 
            
            if len(info['EOG_Blinks']) == 0:
                # If no blinks are found in this section, skip it
                print(f"\n💥Zero blinks found in this {section_duration}s section")
                continue

            # Append blink timestamps (adjusted by start sample of the section, this works reliably) (in sample so blink timestamp at 1138 corresponds to 11.38s)
            blink_timestamps.extend(start + info['EOG_Blinks'])
           
            # Store blink count for this section
            blink_count = len(info['EOG_Blinks'])
            total_blinks_per_run += blink_count             # update the total blink count
            blink_counts_per_run["sections"].append({"section_start": start / sfreq, "section_end": end / sfreq, "blink_count": blink_count})
           
            #print("Blinks found in this section =", blink_count)

            # Plot the section data and detected blinks (uncomment to inspect which peaks are detected as blinks per section)
            # time = np.arange(len(section_data)) / sfreq
            # plt.figure(figsize=(20, 4))
            # plt.plot(time, section_data, label='EOG Signal')
            # plt.scatter(info['EOG_Blinks'] / sfreq, section_data[info['EOG_Blinks']], color='red', label='Detected Blinks')
            # plt.title(f'Section {start/sfreq:.2f}s to {end/sfreq:.2f}s')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Amplitude')
            # plt.legend()
            # plt.show()


        
        except Exception as e:
            print(f"Error processing section {start}-{end}: {e}")
    
    return np.array(blink_timestamps), blink_counts_per_run, total_blinks_per_run








# 5.2 Function to save ERP data and metadata
def save_erp_data(save_path, subj, condition, ERP, blinkIC, all_epochs, all_blink_counts_per_subj, total_blinks_per_run):
    os.makedirs(save_path, exist_ok=True)

    file_name = f'{subj}_{condition}_ERP.pkl'           # e.g. "sub-AN05OC04_HW_negative_ERP.pkl"
    file_path = os.path.join(save_path, file_name)

    # Saving ERP data to pickle file
    subj_data_to_save = {
        'subject': subj,
        'condition': condition,
        'ERP': ERP,
        'blinkIC': blinkIC,
        #'selected_channels': selected_channel_set,
        'all_epochs': all_epochs,    # dict, all epochs over all conditions for each subject
        'all_blink_counts_per_subj': all_blink_counts_per_subj, 
        'total_blinks_per_run': total_blinks_per_run
    }

    with open(file_path, 'wb') as file:
        pickle.dump(subj_data_to_save, file)











# 6. Data Visualisation (optional function call)

def plot_data_visualization(subj, rraw, blinkIC_data, start_times, duration, picks, sfreq=None):
    num_plots = len(start_times) * 2  # Two rows per timeframe: one for channel data, one for IC data with detected blinks

    fig, axs = plt.subplots(num_plots, 1, figsize=(15, 3 * num_plots))

    data, times = rraw[picks, :]

    for i, start in enumerate(start_times):
        start_sample = int(start * rraw.info['sfreq'])
        end_sample = int((start + duration) * rraw.info['sfreq'])

        # Plot the selected channel data
        for j, channel in enumerate(picks):
            axs[i * 2].plot(times[start_sample:end_sample], data[j, start_sample:end_sample], label=channel, alpha=0.8)
        axs[i * 2].set_title(f'Subject {subj}: VEOG and HEOG Channels: Time Segment: {start} to {start + duration} seconds')
        axs[i * 2].set_xlabel('Time (s)')
        axs[i * 2].set_ylabel('Amplitude')
        axs[i * 2].legend(loc='upper right')

        # Plot the blinkIC_data and detected blinks
        section_data = blinkIC_data[start_sample:end_sample]

        if sfreq is None:
            sfreq = rraw.info['sfreq']  # use the sampling frequency from raw data if not provided

        eog_signals, info = nk.eog_process(section_data, sampling_rate=sfreq, method='neurokit')
        time = np.arange(len(section_data)) / sfreq

        axs[i * 2 + 1].plot(time, section_data, label='Blink IC Data', color='black')
        axs[i * 2 + 1].scatter(info['EOG_Blinks'] / sfreq, section_data[info['EOG_Blinks']], color='red', label='Detected Blinks')
        axs[i * 2 + 1].set_title(f'Blink IC Data with Detected Blinks: Time Segment: {start} to {start + duration} seconds')
        axs[i * 2 + 1].set_xlabel('Time (s)')
        axs[i * 2 + 1].set_ylabel('Amplitude')
        axs[i * 2 + 1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

