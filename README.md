MEG Blink Analysis

Overview
This project focuses on the analysis of Magnetoencephalography (MEG) data, specifically aimed at identifying and processing blink artifacts. The analysis involves preprocessing of the MEG data, detecting blinks using Independent Component Analysis (ICA), and visualizing the data to ensure the accuracy of blink detection.

Features
Data Preprocessing: Filtering and cleaning of raw MEG data.
Blink Detection: Identification of blink artifacts using ICA and validation with ocular channels.
Data Visualization: Plotting of raw data, blink components, and detected blinks for quality assurance.



To run the analysis, you need Python with the following libraries installed:

MNE (for MEG data processing)
Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., ... & Hämäläinen, M. S. (2014). MNE software for processing MEG and EEG data. Neuroimage, 86, 446-460.

Neurokit2 (for blink detection)
Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H., ... & Schölzel, C. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. Behavior Research Methods, 1-8.

Matplotlib (for plotting)

