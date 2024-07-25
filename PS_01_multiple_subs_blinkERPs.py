

# ------------------------------------------------------------------------- Main Script ----------------------------------------------------------------------------------------------------

import os
#from time import pthread_getcpuclockid
import mne
import numpy as np
import pickle
import time  
import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import neurokit2 as nk

import PS_functions_01 as fPS
#_______________________________________________________________________________________________________________________________________________________________________________________________________


#                                                   NOTE: everywhere where you can see this symbol 🚩, there is a decision to be made !
# git config --global user.name "pemasherpa1"
# git config --global user.email pema.sherpa@web.de
#_______________________________________________________________________________________________________________________________________________________________________________________________________

dict_trigger_orig = {
    'Block_start': 2,
    'Baseline': 3,
    'Condition': 4,

    'Resting_state': 5,
    'Fixation_cross': 6,
    
    'Questionnaire_Mode': 7,
    'Q_Effort': 8,
    'Q_Frustration': 9,
    'Q_SAM_Valence': 10,
    'Q_SAM_Arousal': 11,
    'Q_Audio': 12, 
    'Q_SAM_Valence_Audio': 13,
    'Q_SAM_Arousal_Audio': 14, 
    'Q_Distraction' : 21,

    'LW_neutral' : 15,
    'LW_positive' : 16,
    'LW_negative' : 17,
    'HW_neutral' : 18,
    'HW_positive' : 19,
    'HW_negative' : 20,
    
    'Control_trigger_1': 22, 
    'Control_trigger_2': 23, 
    'Control_trigger_3': 24, 
    'Control_trigger_4': 25, 
    'Control_trigger_5': 26, 
    'Control_trigger_6': 27, 
    'Control_trigger_7': 28, 
    'Control_trigger_8': 29, 
    'Control_trigger_9': 30,
    'Control_trigger_10': 31
    }

# Define Data Path and Subject List 🚩
#data_path = os.path.join(r"C:\Users\pemas\Nextcloud\Master Thesis\Data\derivatives\analyse_now")    # Data path on my Laptop
#data_path = os.path.join(r"Q:\ZA320330\Group430\Transfer\KL\cogni_affect\meg")               # Data on Hiwi Computer 
#data_path = os.path.join(r"Q:\ZA320330\Group430\Transfer\KL\cogni_affect\meg\analyse_now")          # Data on Hiwi Computer - one specific subject
#data_path = os.path.join(r"Q:\ZA320330\Group430\Transfer\KL\cogni_affect\meg_equalized")             # Data on Hiwi Computer - all subjects ()
data_path = os.path.join(r"C:\Users\sherpa\PS_MA_DATA_LOCAL\meg_backup")

no_eog_component_subjects = []  # List to keep track of subjects with no EOG components found

subj_list = sorted([subj for subj in os.listdir(data_path) if subj[:3] == 'sub']) # sorted alphabetically
total_subjects = len(subj_list)
print(f"\nNumber of subject file folders in the data path: {total_subjects}\n")


# Define Filtering Parameters
dict_filtering = {'ffreq': [0.1, 40.0]}  # Define frequency band for filtering 🚩


# Select the desired set of channels by name 🚩
channel_sets = {
    'parietal':     ['MEG2011', 'MEG2021', 'MEG2031', 'MEG2041', 'MEG2111', 'MEG1911', 'MEG2311'],
    'Cz':           ['MEG0711', 'MEG0741', 'MEG0721', 'MEG0731'],
    'left_temp':    ['MEG0121', 'MEG0111', 'MEG0131', 'MEG0141', 'MEG0211'],
    'right_temp':   ['MEG1411', 'MEG1421', 'MEG1441', 'MEG1431', 'MEG1321'],
    'frontal':      ['MEG0511', 'MEG0921', 'MEG0521', 'MEG0531', 'MEG0811', 'MEG0821', 'MEG0911', 'MEG0941'],
    'occipital':    ['MEG1931', 'MEG2141', 'MEG2121', 'MEG2131', 'MEG2331']
    }

# selected_channel_set = 'parietal'                            # ❗ choose channels here ❗                ⬅ SELECT CHANNELS COMBINATION HERE 🚩
# selected_channels = channel_sets[selected_channel_set]

# Adjust epoch time window 🚩
tmin, tmax = -0.8, 1.2  



# Define events of interest based on trigger IDs 🚩
events_of_interest = [dict_trigger_orig['LW_neutral'], dict_trigger_orig['LW_positive'], 
                      dict_trigger_orig['LW_negative'], dict_trigger_orig['HW_neutral'], 
                      dict_trigger_orig['HW_positive'], dict_trigger_orig['HW_negative']]


#_______________________________________________________________________________________________________________________________________________________________________________________________________

#-------------------------------- Main loop to iterate over subjects and runs--------------------------------------------------------------------------------------------------------------------------
#_______________________________________________________________________________________________________________________________________________________________________________________________________

no_eog_component_subjects = []

artefact_biased_subj = ['sub-ER06US02', 'sub-GE05EL05', 'sub-NG05NK23', 'sub-TZ07AN02', 'sub-KE06ED05', 'sub-IA06NS24'] # 6 subjs, epochs below 30 in any condition

noisy_VEOG_channel =   ['sub-CK06RD16', 'sub-ER06US02', 'sub-GE05EL05', 'sub-IA06NS24', 'sub-KE06ED05', 'sub-NN04AS13'] # 6 subjs, repetitive artifact in VEOG channel

with_lower_thresh_noisy_VEOG =   ['sub-CK06RD16', 'sub-IA06NS24', 'sub-NN04AS13']                                       # 4 subjs, with lower threshold >30 epochs can be found


start_time = time.time() # Record the start time
nr_of_all_epochs = []

# Loop through each subject
for idx, subj in enumerate(subj_list, start=1):

    print(f"\n▶ Starting with Subject: {subj}, {idx} out of {total_subjects} subjects--------------------------------------------------------------------\n")
    

    # Notify if subject has noisy VEOG channel
    if subj in noisy_VEOG_channel:
        print(f"⚡ Subject {subj} has a noisy VEOG channel. Will use lower theshold and only aEOG channel for find_bads_eog⚡")


    # excluding artefact biased subjects
    if subj in artefact_biased_subj:
        print(f"📛 Subject {subj}: insufficient clean epochs (artefact biased). Will still try...")          # shouldnt be less than 30 approx.
        #continue


    try:
        # Create path to save ICA plot and ERP data for each subject
        save_path = os.path.join(data_path, subj, 'combined') # Q:\\ZA320330\\Group430\\Transfer\\KL\\cogni_affect\\meg\\sub-AN05OC04\\combined'
        os.makedirs(save_path, exist_ok=True)

        all_runs_info = []

        #-------------- Load raw instance and get events for each run / condition and filter -----------------------------------------------------------------------------
        for run_nr in range(1, 7):  # Adjust based on number of runs per subject
            raw, events = fPS.load_raw_data(subj, run_nr, data_path, dict_filtering)

            # Adjust event sample numbers relative to raw data
            #events[:, 0] -= raw.first_samp

            # Segment the raw data based on events of interest
            run_info = fPS.segment_raw_data(raw, run_nr, events, dict_trigger_orig, events_of_interest)
            all_runs_info.extend(run_info)        

        # maybe here we should save "all_runs_info" for the respective subject?

     # Now `all_runs_info` contains segmented information for all runs without concatenating raw data
        print(f"\nNumber of segmented runs: {len(all_runs_info)}")


        #--------- Perform ICA once on the concatenated data-----------------------------------------------------------------------------------------------------------------------------------------
        
        # Perform ICA or load already exsisting ICA object from file
        ICA, rraw = fPS.perform_or_load_ICA(raw, subj, save_path)

        # Find bads EOG and update data         # Note: if subj has noisy VEOG channel: use only aEOG channel + threshold will start looping at 0.4 (otherwise 0.5) 🚩
        ICA, rraw, blinkIC, blinkIC_data = fPS.find_badsEOG_and_get_blink_IC(ICA, rraw, subj, save_path, noisy_VEOG_channel) 

       
        # if no EOG component found with any of the thresholds:
        if ICA is None:
            no_eog_component_subjects.append(subj)
            continue  # Skip the rest of the loop if no EOG components were found



        # ______________ VISUALISATION FOR CHECK UP 👁‍🗨👁‍🗨 (optional, uncomment if not needed) _________________________________________________________________________________________________________________________________________
        start_times = [200, 350, 500] 
        duration = 30.0  
        #picks = ['MEG0511', 'MEG0921', 'VEOG', 'HEOG'] 
        picks = ['HEOG', 'VEOG']  
        #fPS.plot_data_visualization(subj, rraw, blinkIC_data, start_times, duration, picks) # for each subj: will plot 3 timeframes (choose above) out of the whole raw and BlinkIC data



        # --------- Extract ERPs and save data ----------------------------------- and equalize epoch cound here ---------------------------------------------------------
        # (function calls: "extract_blinks_in_sections"; "save_erp_data")
        
        section_duration = 60        # Choose how long the sections should be (for "extract_blinks_in_sections" )  🚩
        reject = dict(mag=4e-12)     # Modify rejection criteria (0.0000000000039) 🚩
        #reject = dict(mag=4e-5)      #0.00003

        nr_of_epochs = fPS.extract_ERPs(all_runs_info, blinkIC_data, rraw, subj, save_path, blinkIC, tmin, tmax, section_duration, reject)          # 🚩        # NEUROKIT inside here
        nr_of_all_epochs.append({'subject': subj, 'nr_of_epochs_left': nr_of_epochs})


        print("\n▶ Completely done with Subject:", subj,"---------------------------------------------------------------------------------------------------\n")

    except Exception as e:
        print(f"\n▶▶▶  💥 💥 💥  Error processing subject {subj}: {e}  💥 💥 💥 ▶▶▶\n")
        continue

#_______________________________________________________________________________________________________________________________________________________________________________________________________

# Analyze nr_of_all_epochs to find subjects with fewer than 30 epochs per condition
subjects_with_few_epochs = []
for subj_data in nr_of_all_epochs:
    subj = subj_data['subject']
    nr_of_epochs = subj_data['nr_of_epochs_left']
    if nr_of_epochs < 30:
        subjects_with_few_epochs.append(subj)


# Record the end time
end_time = time.time()
duration = (end_time - start_time)/60

print(f"\n 📛 No EOG component found for {len(no_eog_component_subjects)} out of {len(subj_list)} subjects: {no_eog_component_subjects}")
print(f"\n ⛔ Subjects with fewer than 30 epochs for any condition: {len(subjects_with_few_epochs)}:\n\n{subjects_with_few_epochs}\n")
print(f"\n 🕒 Total time taken: {duration:.2f} minutes")

#_______________________________________________________________________________________________________________________________________________________________________________________________________













#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





# after concatenating runs
        # Korrektur Problem in 'sub-ER04WE13'   : in run 2 gibt es 2x einen trigger 17
        # if subj == 'sub-ER04WE13':
        #     # Issue in Baseline due to Parallel Port Sending [81625, 21, 17] needs to be corrected to [81625, 21, 2]
        #     idx_to_be_fixed = np.where(events[:, 0] == [81625])[0][0]
        #     assert np.all(events[idx_to_be_fixed] == [81625, 21, 17])
        #     events[idx_to_be_fixed] = np.array([81625, 21, 2])
        #     assert np.all(events[idx_to_be_fixed] == [81625, 21, 2])