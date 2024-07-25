import os
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt

# ///////////////////////////////////////// SELECTIONS ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# 1. Choose Data Paths
data_path = os.path.join(r"C:\Users\sherpa\PS_MA_DATA_LOCAL\meg_backup")                                  # Local Data on Hiwi Computer - all subjects ()


#results_folder = os.path.join(r"C:\Users\sherpa\PS_MA_DATA_LOCAL\ERP_results")                             # Local Results on Hiwi Computer - all subjects () - not equalized
#results_folder = os.path.join(r"C:\Users\sherpa\PS_MA_DATA_LOCAL\ERP_results_equalized")                  # Local Results on Hiwi Computer - all subjects () - equalized
results_folder = os.path.join(r"C:\Users\sherpa\PS_MA_DATA_LOCAL\ERP_results_corrected_1st_sample")                    # store locally on Hiwi Computer (for now because faster)


# 2. Choose the conditions to compare
#conditions_to_plot = ['LW_positive', 'HW_negative']                                                                 # Extremes
# conditions_to_plot = ['LW_neutral', 'LW_positive', 'LW_negative']                                                   # Low Workload Comparison
# conditions_to_plot = ['HW_neutral', 'HW_positive', 'HW_negative']                                                   # High Workload Comparison
conditions_to_plot = ['LW_neutral', 'LW_positive', 'LW_negative', 'HW_neutral', 'HW_positive', 'HW_negative']      # All Conditions


# 3. Choose Brain Region / Channels 
channel_sets = {
    'parietal':     ['MEG2011', 'MEG2021', 'MEG2031', 'MEG2041', 'MEG2111', 'MEG1911', 'MEG2311'],
    'Cz':           ['MEG0711', 'MEG0741', 'MEG0721', 'MEG0731'],
    'left_temp':    ['MEG0121', 'MEG0111', 'MEG0131', 'MEG0141', 'MEG0211'],
    'right_temp':   ['MEG1411', 'MEG1421', 'MEG1441', 'MEG1431', 'MEG1321'],
    'frontal':      ['MEG0511', 'MEG0921', 'MEG0521', 'MEG0531', 'MEG0811', 'MEG0821', 'MEG0911', 'MEG0941'],
    'occipital':    ['MEG1931', 'MEG2141', 'MEG2121', 'MEG2131', 'MEG2331']
    }


brainregion = 'parietal'                              # ❗ choose channels here ❗                ⬅ SELECT CHANNELS COMBINATION HERE ❗
selected_channels = channel_sets[brainregion]
    # type "None" in the function call for WHOLE brain analysis

#brainregion = 'whole brain'





# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def plot_grand_avg(data_path, conditions_to_plot, brainregion, results_folder, selected_channels=None, tmin=-0.8, tmax=1.2, times_to_plot=None):
    """
    Load precomputed grand averages of ERPs, filter them by specified conditions and channels, and plot the results.

    Parameters
    ----------
    data_path : str
        Path to the directory containing subject data folders.
    conditions_to_plot : list of str
        List of conditions to plot (e.g., ['LW_positive', 'HW_negative']).
    brainregion : str
        Name of the brain region (or specific set of channels) to be plotted.
    results_folder : str
        Path to the folder where the grand averages are saved and will be loaded from.
    selected_channels : list of str, optional
        List of channel names to include in the plots. If None, all channels are included.
    tmin : float, optional
        Start time of the epoch in seconds. Default is -0.8.
    tmax : float, optional
        End time of the epoch in seconds. Default is 1.2.
    times_to_plot : list of float, optional
        Specific times at which to plot topomaps. If None, a default set of times is used.

    Returns
    -------
    None
        This function does not return any values. 
        It displays plots of the ERPs for the specified conditions and channels.

    Notes
    -----
    - This function assumes that the grand averages of the ERPs have been precomputed and saved using another process.
    - It does not compute the grand averages from raw data; it only loads and plots them.
    - The grand averages are filtered to include only the specified conditions and channels before plotting.    """
    
    # Set MNE verbosity to suppress unnecessary output
    original_verbose = mne.get_config('MNE_LOGGING_LEVEL')
    mne.set_log_level('ERROR')

    # Load saved results (Function Call)
    grand_averages, _ = load_results(results_folder)

    # Function to filter ERPs for specific channels
    def filter_erps_for_channels(erps, selected_channels):
        if selected_channels:
            filtered_erps = {condition: erp.copy().pick_channels(selected_channels) for condition, erp in erps.items()}
        else:
            filtered_erps = erps
        return filtered_erps

    # Filter grand averages for selected channels
    grand_averages = filter_erps_for_channels(grand_averages, selected_channels)

    # Filter grand averages to include only specified conditions
    grand_averages = {condition: erp for condition, erp in grand_averages.items() if condition in conditions_to_plot}

    # -------------- Plot averaged ERPs for each condition --------------------------------
    fig, axes = plt.subplots(len(grand_averages), 1, figsize=(6, 4 * len(grand_averages)))
    fig.suptitle(f'Averaged ERPs by Condition for brain region: {brainregion}', fontsize=16)

    for i, (condition, averaged_ERPs) in enumerate(grand_averages.items()):
        ax = axes[i] if len(grand_averages) > 1 else axes
        count = averaged_ERPs.nave

        # Plot the averaged ERP
        averaged_ERPs.plot(picks="all", axes=ax, show=False, spatial_colors=True, time_unit='s')
        ax.set_xlim(tmin, tmax)
        ax.set_title(f'Condition: {condition} for {count} subjects')

    plt.subplots_adjust(hspace=0.9)
    plt.show()

    # Plot grand averages with SE
    plt.figure(figsize=(20, 6))
    plt.title(f'Grand Averages by Condition with SE for brain region: {brainregion}')
    times = next(iter(grand_averages.values())).times

    for condition, grand_average in grand_averages.items():
        data = grand_average.data.mean(axis=0)
        se_data = np.std(grand_average.data, axis=0) / np.sqrt(grand_average.data.shape[0]) # Standard Error

        # Plot the grand average with SE
        plt.plot(times, data, label=condition)
        plt.fill_between(times, data - se_data, data + se_data, alpha=0.1)

    plt.xlim(tmin, tmax)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.legend()
    plt.show()


    # --------- Plot topomaps ---------
    # if times_to_plot is None:
    #     times_to_plot = np.linspace(tmin, tmax, num=10)  # Default to 10 evenly spaced time points

    # for condition, averaged_ERPs in grand_averages.items():
    #     fig, axes = plt.subplots(1, len(times_to_plot), figsize=(15, 5))
    #     fig.suptitle(f'Topomaps for Condition: {condition} for brain region: {brainregion}', fontsize=16) # one for each condition 

    #     for ax, time_point in zip(axes, times_to_plot):
    #         time_idx = averaged_ERPs.time_as_index(time_point)
    #         data = averaged_ERPs.data[:, time_idx].flatten()
    #         mne.viz.plot_topomap(data, averaged_ERPs.info, axes=ax, show=False)
    #         ax.set_title(f'Time: {time_point:.2f}s')

    #     plt.show()


    # --------- Plot averaged ERPs over all conditions ---------
    all_erp_data = np.mean([erp.data for erp in grand_averages.values()], axis=0)

    plt.figure(figsize=(15, 6))
    plt.title(f'Grand Average ERP across All Conditions for brain region: {brainregion}')
    plt.plot(times, all_erp_data.mean(axis=0), label='All Conditions')
    se_all_data = np.std(all_erp_data, axis=0) / np.sqrt(all_erp_data.shape[0]) # Standard Error
    plt.fill_between(times, all_erp_data.mean(axis=0) - se_all_data, all_erp_data.mean(axis=0) + se_all_data, alpha=0.1)
    plt.xlim(tmin, tmax)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.legend()
    plt.show()

    # Restore the original verbosity level
    mne.set_log_level(original_verbose)

    return



# .................. LOAD RESULTS FUNCTION ........................

def load_results(results_folder):
    # Load the saved results
    save_file = os.path.join(results_folder, 'grand_avg_ERPs.pkl')
    with open(save_file, 'rb') as f:
        results_data = pickle.load(f)

    grand_averages = results_data['grand_averages']
    conditions_to_plot = results_data['conditions_to_plot']

    return grand_averages, conditions_to_plot



# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#......................................... FUNCTION CALL .................................................................................................................................................................

plot_grand_avg(data_path, conditions_to_plot, brainregion, results_folder, selected_channels=selected_channels, tmin=-0.5, tmax=0.5)    # Function Call for specific brain regions
#plot_grand_avg(data_path, conditions_to_plot, brainregion, results_folder, selected_channels=None,  tmin=-0.8, tmax=1.2)               # Function Call for ERPs of whole brain

