import os
import pickle
import time
import numpy as np
import mne
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


# # Script to load ERP data from pickle files and analyse them for all subjects 

# '''
#     Load ERP data for all subjects, sort and collect ERP data by condition,
#     average the ERPs for each condition, and plot the averaged ERPs for specified conditions.

#     Parameters:
#     - data_path: str, path to the data directory containing subject data.
#     - conditions_to_plot: list of str, conditions to include in the plots.
#     - brainregion: str, specify the brain region for the plot title (default: 'parietal').
#     - tmin: float, start time for ERP plot window (default: -0.8s).
#     - tmax: float, end time for ERP plot window (default: 1.2s).
    
#     written by PS on 23.06.24


# reminder to self
# - add counter how many clean epochs go into the plots -done
# - add baseline correction - done
# - zwischenspeichern? - done
# - equalize counts - done


#----------------------------------------- selections -----------------------------------------------------------------------------------------------------------------
tmin = -0.8
tmax = 1.2


# Choose Brain Region / Channels 
channel_sets = {
    'parietal':     ['MEG2011', 'MEG2021', 'MEG2031', 'MEG2041', 'MEG2111', 'MEG1911', 'MEG2311'],
    'Cz':           ['MEG0711', 'MEG0741', 'MEG0721', 'MEG0731'],
    'left_temp':    ['MEG0121', 'MEG0111', 'MEG0131', 'MEG0141', 'MEG0211'],
    'right_temp':   ['MEG1411', 'MEG1421', 'MEG1441', 'MEG1431', 'MEG1321'],
    'frontal':      ['MEG0511', 'MEG0921', 'MEG0521', 'MEG0531', 'MEG0811', 'MEG0821', 'MEG0911', 'MEG0941'],
    'occipital':    ['MEG1931', 'MEG2141', 'MEG2121', 'MEG2131', 'MEG2331']
    }


#brainregion = 'parietal'                              # ❗ choose channels here ❗                ⬅ SELECT CHANNELS COMBINATION HERE ❗
#selected_channels = channel_sets[brainregion]

# or whole brain:
brainregion = 'whole brain'


# Example of calling the function
start_time = time.time() # Record the start time

#data_path = r"Q:\ZA320330\Group430\Transfer\KL\cogni_affect\meg"
#data_path = os.path.join(r"Q:\ZA320330\Group430\Transfer\KL\cogni_affect\meg01")                     # Data on Hiwi Computer - all subjects ()
#data_path = os.path.join(r"C:\Users\sherpa\PS_MA_DATA_LOCAL\meg01")                                  # Local Data on Hiwi Computer - all subjects ()
#data_path = os.path.join(r"Q:\ZA320330\Group430\Transfer\KL\cogni_affect\meg01")                      # Data on Hiwi Computer - all subjects ()
#data_path = os.path.join(r"Q:\ZA320330\Group430\Transfer\KL\cogni_affect\meg_equalized")                     # Data on Hiwi Computer - all subjects ()
data_path = os.path.join(r"C:\Users\sherpa\PS_MA_DATA_LOCAL\meg_backup")

results_folder = os.path.join(r"C:\Users\sherpa\PS_MA_DATA_LOCAL\ERP_results_corrected_1st_sample")                    # store locally on Hiwi Computer (for now because faster)

#conditions_to_plot = ['LW_positive', 'HW_negative']
conditions_to_plot = ['LW_neutral', 'LW_positive', 'LW_negative', 'HW_neutral', 'HW_positive', 'HW_negative'] # all


#---------------------------------------------------------------------------------------------------------------------------------------------------------

def load_and_plot_erps(data_path, conditions_to_plot, brainregion, results_folder, tmin=-0.8, tmax=1.2):    # all channels (seperate later on in script03)
    
    # Set MNE verbosity to suppress unnecessary output
    original_verbose = mne.get_config('MNE_LOGGING_LEVEL')
    mne.set_log_level('ERROR')

    # List all subjects in the data directory, sorted alphabetically
    subj_list = sorted([subj for subj in os.listdir(data_path) if subj[:3] == 'sub'])

    # Initialize dictionary to store ERP data by condition
    condition_erps = {}
    condition_counts = {}  # to keep track of the number of subjects per condition
    epochs_counts = {}  # to keep track of the number of clean epochs per condition

    # Function to load each subjects ERP file from results folder
    def load_erp_file(erp_fname):
        try:
            with open(erp_fname, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading file {erp_fname}: {e}")
            return None


    # Function to extract ERPs from files for each condition for one subject at a time
    def process_subject(subj):
        print(f"\n▶ Starting with Subject: {subj}, {idx} out of {len(subj_list)} subjects in total")
        save_path = os.path.join(data_path, subj, 'combined')

        erp_files = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith('_ERP.pkl') and f.startswith(subj)] 
        if not erp_files:
            print(f"💥Files not found for {subj}💥")
            return

        with ThreadPoolExecutor() as executor:                          # ThreadPoolExecutor to load ERP files for all conditions for one subj in parallel
            results = list(executor.map(load_erp_file, erp_files))      # Function Call 

        for ERP_data_subj in filter(None, results):
            condition = ERP_data_subj['condition']
            ERP = ERP_data_subj['ERP']

            # Check the channel names in the loaded ERP data (debugging)
            # ch_names = ERP.ch_names
            # amount_ch = len(ch_names)
            # print(f"Subject: {subj}, Condition: {condition}: {amount_ch} channels")



            if condition in conditions_to_plot:
                if condition not in condition_erps:
                    condition_erps[condition] = []
                    condition_counts[condition] = 0
                    epochs_counts[condition] = 0
                condition_erps[condition].append(ERP)
                condition_counts[condition] += 1


    # Process all subjects using the "process_subjects" function
    for idx, subj in enumerate(subj_list, start=1):
        process_subject(subj)   # Function Call 





    # # Equalize the number of epochs across all subjects for each condition
    # for condition in condition_erps:
    #     mne.epochs.equalize_epoch_counts(condition_erps[condition], method='mintime')
    #     print(f"Equalized epochs for condition: {condition}")





    # ....... Function to average ERPs using MNE's grand_average ...........
    def f_calc_avg_ERPs(ERPs):
        avg_erp = mne.grand_average(ERPs)
        print(f"Calculating grand average from {len(avg_erp.ch_names)} channels")
        return avg_erp

    #......... Function to compute standard error ...........................
    def f_compute_SE(data):
        """Compute standard error of the mean."""
        return np.std(data, axis=0) / np.sqrt(data.shape[0])

    #...........Function to save infos for later .............................
    def save_results(results_folder, grand_averages, conditions_to_plot):

        results_data = {
            'grand_averages': grand_averages,
            'conditions_to_plot': conditions_to_plot
        }
        try:
            os.makedirs(results_folder, exist_ok=True)
            print(f"Directory '{results_folder}' created successfully or already exists.")
        except OSError as e:
            print(f"Error creating directory '{results_folder}': {e}")

        save_file = os.path.join(results_folder, 'grand_avg_ERPs.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(results_data, f)
        print(f"Results saved to {save_file}")


    # -------------- Fig. 1: Plot averaged ERPs for each condition --------------------------------
    fig, axes = plt.subplots(len(condition_erps), 1, figsize=(6, 4 * len(condition_erps)))
    fig.suptitle(f'Averaged ERPs by Condition for brain region: {brainregion}', fontsize=16)

    grand_averages = {}

    for i, (condition, ERPs) in enumerate(condition_erps.items()):
        ax = axes[i] if len(condition_erps) > 1 else axes
        averaged_ERPs = f_calc_avg_ERPs(ERPs) # Function Call
        count = condition_counts[condition]
        n_clean_epochs = epochs_counts.get(condition, 0)

        # Store the grand average ERP for later use
        grand_averages[condition] = averaged_ERPs

        # Topomap (?)
        #averaged_ERPs.plot_topomap(times='auto', ch_type="mag")

        # Plot the averaged ERP 
        averaged_ERPs.plot(axes=ax, show=False, spatial_colors=True, time_unit='s')
        #averaged_ERPs.plot(picks=selected_channels, axes=ax, show=False, spatial_colors=True, time_unit='s')

        ax.set_xlim(tmin, tmax)
        ax.set_title(f'Condition: {condition} for {count} subjects ({n_clean_epochs} clean epochs)')

    plt.subplots_adjust(hspace=0.9)
    plt.show()

    # ------------- Fig. 2: Plot grand averages with SE -----------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.title(f'Grand Averages by Condition with SE for brain region: {brainregion}')
    times = grand_averages[conditions_to_plot[0]].times

    for condition, grand_average in grand_averages.items():
        if condition in conditions_to_plot:
            data = np.array([erp.data.mean(axis=0) for erp in condition_erps[condition]])
            mean_data = data.mean(axis=0)
            se_data = f_compute_SE(data)

            # Plot the grand average with SE
            plt.plot(times, mean_data, label=condition)
            plt.fill_between(times, mean_data - se_data, mean_data + se_data, alpha=0.1)

    plt.xlim(tmin, tmax)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.legend()
    plt.show()


    #............ save grand averages .........................................................

    save_results(results_folder, grand_averages, conditions_to_plot)

    # Restore the original verbosity level
    mne.set_log_level(original_verbose)


    return



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Function Call
load_and_plot_erps(data_path, conditions_to_plot, brainregion, results_folder, tmin, tmax)


end_time = time.time()
duration = (end_time - start_time)/60
print(f"\n 🕒 Total time taken: {duration:.2f} minutes")




# Notes 11.07
# sub-NG05NK23 9 epochs
# sub-NG05NZ23 20 epochs
# sub-NN05EN15 21 epochs
# sub-SS06ER02 181 epochs
# sub-VA04OR20 26 epochs
# sub-YE07US06 21 epochs

# 🚩 No EOG component found for 9 out of 47 subjects: ['sub-EN03ER05', 'sub-ER06CH12', 'sub-ER08IN09', 
# 'sub-GE05EL05', 'sub-IL04EN13', 'sub-NN06NO14', 'sub-NN07WE18', 'sub-RT05ND31', 'sub-TZ08AY05']