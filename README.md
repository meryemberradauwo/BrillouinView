# BrillouinView
An Advanced Tool for Calibration, Visualization, and Modeling of Brillouin Spectroscopy Data.

BrillouinView is a Python-based graphical software tool designed for the analysis and visualization of Brillouin zone structures in condensed matter physics. It integrates various scientific libraries such as NumPy, SciPy, and Matplotlib to provide interactive plots and tools for analyzing phonon, electronic, and thermodynamic properties. The software utilizes Tkinter for a user-friendly interface, allowing users to load and manipulate data, visualize complex models, and perform calculations.

BrillouinView also uses the BurnMan package for advanced material modeling, so users should ensure that BurnMan is installed before running the software. Key features include data fitting, peak finding, and baseline correction, making it an essential tool for researchers in solid-state physics and materials science.

The .exe file is accessible here:
https://drive.google.com/file/d/1QOGOviSJANr7SuXHBLJq4nptf2gQ0qH-/view?usp=sharing  


# Instructions
## Calibration Tab Instructions
Enter Calibration Settings or Load File: You can manually input the calibration settings in the provided fields or load an existing calibration file.

Identify the 3 Strongest Dips: Select the 3 strongest dips in your data. These dips are what the software will attempt to fit, so ensure that they are clearly visible.

Select the Appropriate Fitting Model: For each dip, choose the fitting model from the dropdown menu. Available models include Gaussian, Lorentzian, Voigt, and others depending on your data. 

Enter Initial Fitting Parameters: Provide initial guesses for the fitting parameters, such as amplitude, mean, sigma, gamma, and fraction. Make sure these initial guesses are close to the actual peak values to improve fitting accuracy.

Adjust Parameters if Necessary: If the fitting routine fails to find a peak, try adjusting the initial parameter guesses. Ensure that the fitting parameters are realistic and within the expected range for your data.

Note that the fitting routine looks for dips in the data, not peaks. If your data contains peaks instead of dips, the software may not be able to locate them. In this case, either reformat the data to contain dips or adjust your fitting parameters.

Save the Fit for Each Dip: Once you are satisfied with the fit for a dip, save the fit and proceed to the next dip.

Save Calibration Settings: After fitting all 3 dips, save the calibration settings. These will be used for later steps, such as data scaling.

Repeat for Multiple Calibration Files: You can repeat the calibration process with multiple files to obtain an average of the calibration values, which helps improve accuracy.

## Data Tab Instructions
Apply Calibration: Click 'Apply Calibration' to rescale your loaded data file using the saved calibration settings.
Background Options: Choose to hide the background or smooth the data as per your preferences. This can help to improve the clarity of the peaks for fitting.

Fit Data Peaks: Click on any peak in the data to fit it. Make sure the fitting parameters are close to the peak values for accurate results.
Observe the Results: The fit results will be displayed in the inset plot, along with the corresponding values in the accompanying table.

Export or Transfer Data: Once your analysis is complete, you can export the results or transfer them to the Stack tab for further comparison.

## Stack Tab Instructions

Load Data Files for Comparison: Load multiple data files for comparison and analysis. This is useful when you need to overlay data from different measurements.

Adjust Vertical and Horizontal Offsets: Adjust each file’s vertical and horizontal offset to align them properly for overlay comparison.

Visualize and Compare Data: Use the stacked plots to visualize and compare the data, helping identify trends or anomalies across multiple measurements.

## Elastic Properties Tab Instructions
Select Crystal Structure and Input Cij Elements: Choose the crystal structure of your sample and input the corresponding Cij elements, which represent the stiffness matrix for your material.

Alternatively, Load a Cij Matrix File: If you have a Cij matrix file in the correct format, you can load it directly into the software.
Forward Modeling: For forward modeling, input the UVZ coordinates of interest or load them from a file.

Run Forward Modeling Analysis: Click 'Forward Modeling' to proceed with the analysis. The software will calculate and display the fitted Cij matrix, as well as the bulk modulus (K), shear modulus (μ), universal elastic anisotropy, and Poisson’s ratio for your material.


For any issues or further assistance, please feel free to contact me at berrada@hawaii.edu. I'm happy to help!
