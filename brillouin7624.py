#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 22:40:33 2023

@author: meryem
"""
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from IPython.display import display, clear_output, HTML
import pandas as pd
import math
from tkinter import ttk
from scipy.special import wofz
from scipy.signal import find_peaks
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, diags
from scipy.optimize import curve_fit
import os
from matplotlib.text import OffsetFrom
from pybaselines import Baseline
import burnman
from burnman.classes import anisotropy
from pyquaternion import Quaternion
import tkinter.messagebox
import burnman.utils.unitcell
import scipy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import random
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from tkinter import colorchooser
from matplotlib.font_manager import FontProperties



root = tk.Tk()
root.geometry("800x1200")
root.title("Brillouin View")

notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

tabs = ["Instructions", "Calibration", "Data", "Stack", "Elastic Properties"]
frames = {}

for tab in tabs:
    frame = ttk.Frame(notebook)
    frames[tab] = frame
    notebook.add(frame, text=tab)

calibration_parameters = {}
data_parameters = {}
calibration_file_path = []
data_file_path = []
calibration = None 
scattering_data = None

elements = []  
checked_states = [] 
current_filename = ""  

SPEED_OF_LIGHT = 299792458
para_theta = 50 #degrees
para_lambda = 532 #nm
para_PS = 5  # Example value, angstrom
para_Cal_channel = 468.75  #nm
para_chi = 10 #degrees

model = None 
# -------------------- FITTING

current_dip_index = 0
initial_amp_guess = -17000
initial_mean_guess = 12.0
initial_stddev_guess = 2.0
initial_gamma_guess = 1.0
initial_fraction_guess = 0.5

initial_amp_guessP2 = -12000
initial_mean_guessP2 = 22.0
initial_stddev_guessP2 = 5.0
initial_gamma_guessP2 = 1.0
initial_fraction_guessP2 = 0.5


saved_means = {f'Dip {i + 1}': None for i in range(3)}
dip = []
params = []  
mean = None  

current_peak_index = 0
initial_amp_guess2 = 2.0
initial_mean_guess2 = -10000.0
initial_stddev_guess2 = 10.0
initial_gamma_guess2 = 1.0
initial_fraction_guess2 = 0.5

initial_amp2_guessP2 = 2
initial_mean2_guessP2 = -8000
initial_stddev2_guessP2 = 10.0
initial_gamma2_guessP2 = 1.0
initial_fraction2_guessP2 = 0.5

peak = []
params2 = []  
params2OPP = []  
velocity_x = None
velocity_y = None

def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-(x - mean)**2 / (2 * stddev**2))

def lorentzian(x, amp, mean, gamma):
    return amp * (gamma**2) / ((x - mean)**2 + gamma**2)

def voigt(x, amp, mean, stddev, gamma):
    z = (x - mean + 1j * gamma) / (stddev * np.sqrt(2))
    denominator = stddev * np.sqrt(2 * np.pi)
    if np.any(np.abs(denominator) < 1e-10):
        print("Division by zero or small denominator encountered.")
        voigt_profile = np.zeros_like(x)
    else:
        wofz_result = wofz(z)
        voigt_profile = amp * np.real(wofz_result / denominator)
        
    return voigt_profile

def pseudo_voigt(x, amp, mean, stddev, fraction):
    return fraction * lorentzian(x, amp, mean, stddev) + (1 - fraction) * gaussian(x, amp, mean, stddev)

def _2gaussian(x, amp1, mean1, stddev1, amp2, mean2, stddev2):
    return amp1 * np.exp(-(x - mean1)**2 / (2 * stddev1**2)) + amp2 * np.exp(-(x - mean2)**2 / (2 * stddev2**2)) 

def _2lorentzian(x, amp1, mean1, gamma1, amp2, mean2, gamma2):
    return amp1 * (gamma1**2) / ((x - mean1)**2 + gamma1**2) + amp2 * (gamma2**2) / ((x - mean2)**2 + gamma2**2)

def _2voigt(x, amp1, mean1, stddev1, gamma1, amp2, mean2, stddev2, gamma2):
    z1 = (x - mean1 + 1j * gamma1) / (stddev1 * np.sqrt(2))
    z2 = (x - mean2 + 1j * gamma2) / (stddev2 * np.sqrt(2))
    denominator1 = stddev1 * np.sqrt(2 * np.pi)
    denominator2 = stddev2 * np.sqrt(2 * np.pi)

    if np.any(np.abs(denominator1) < 1e-10) or np.any(np.abs(denominator2) < 1e-10):
        print("Division by zero or small denominator encountered.")
        voigt_profile = np.zeros_like(x)
    else:
        wofz_result1 = wofz(z1)
        wofz_result2 = wofz(z2)
        voigt_profile = (amp1 * np.real(wofz_result1 / denominator1) +
                         amp2 * np.real(wofz_result2 / denominator2))
        
    return voigt_profile

def _2pseudovoigt(x, amp1, mean1, stddev1, fraction1, amp2, mean2, stddev2, fraction2):
    return fraction1 * lorentzian(x, amp1, mean1, stddev1) + (1 - fraction1) * gaussian(x, amp1, mean1, stddev1) + fraction2 * lorentzian(x, amp2, mean2, stddev2) + (1 - fraction2) * gaussian(x, amp2, mean2, stddev2)



#------------------------------------------- CALIBRATION TAB

Tcal_paned_window = tk.PanedWindow(frames["Calibration"], orient=tk.HORIZONTAL)
Tcal_paned_window.pack(fill=tk.BOTH, expand=False)
calcenter_frame = tk.Frame(frames["Calibration"], highlightbackground="black", highlightthickness=1)
calcenter_frame.pack(fill=tk.BOTH, expand=True)
Bcal_paned_window = tk.PanedWindow(frames["Calibration"], orient=tk.HORIZONTAL)
Bcal_paned_window.pack(fill=tk.BOTH, expand=False)

Tcalleft_frame = tk.Frame(Tcal_paned_window, highlightbackground="black", highlightthickness=1)
Tcalcenter_frame = tk.Frame(Tcal_paned_window, highlightbackground="black", highlightthickness=1)
Tcalright_frame = tk.Frame(Tcal_paned_window, highlightbackground="black", highlightthickness=1)
Tcal_paned_window.add(Tcalleft_frame, width=200, height=150)
Tcal_paned_window.add(Tcalcenter_frame, width=350, height=150)
Tcal_paned_window.add(Tcalright_frame, width=200, height=150)

Bcalleft_frame = tk.Frame(Bcal_paned_window, highlightbackground="black", highlightthickness=1)
Bcalcenter_frame = tk.Frame(Bcal_paned_window, highlightbackground="black", highlightthickness=1)
Bcalright_frame = tk.Frame(Bcal_paned_window, highlightbackground="black", highlightthickness=1)
Bcal_paned_window.add(Bcalleft_frame, width=200, height=160)
Bcal_paned_window.add(Bcalcenter_frame, width=180, height=160)
Bcal_paned_window.add(Bcalright_frame, width=400, height=160)

def fit_selected_model(event=None):
    global calibration
    global dip
    global current_dip_index
    global params
    global mean
    
    if calibration is not None:
        selected_model = fit_model_var.get()
        x = list(range(len(calibration)))
        y = calibration

        ax1.clear()
        
        amp1 = float(amp_var.get())
        mean1 = float(mean_var.get())
        stddev1 = float(stddev_var.get())
        gamma1 = float(gamma_var.get())
        fraction1 = float(fraction_var.get())
        amp2 = float(amp_varP2.get())
        mean2 = float(mean_varP2.get())
        stddev2 = float(stddev_varP2.get())
        gamma2 = float(gamma_varP2.get())
        fraction2 = float(fraction_varP2.get())

        if not dip:
            dip_indices = np.argpartition(calibration, 3)[:3]
            dip_indices = np.sort(dip_indices)
            dip = dip_indices.tolist()

        dip_index = dip[current_dip_index]
        dip_x = x[dip_index]
        dip_y = y[dip_index]

        baseline_fitter = Baseline(x_data=x)
        
        maxhalfwindowCAL_value = int(float(maxhalfwindowCAL_var.get()))
        smoothhalfwindowCAL_value = float(smoothhalfwindowCAL_var.get())

        negated_y = [-val for val in y] 
        background = - baseline_fitter.asls(negated_y, lam=maxhalfwindowCAL_value, p=smoothhalfwindowCAL_value)[0]

        ax1.plot(x, y, 'c-', linewidth=1, label='spectrum')
        ax1.plot(x, background, 'r--', linewidth=1, label='baseline')

        
        if selected_model == '2 peak Gaussian':
            params,_ = curve_fit(_2gaussian, x, y - background, p0=[amp1, mean1, stddev1, amp2, mean2, stddev2])
            fitted_dip1 = gaussian(x, *params[:3])  
            fitted_dip2 = gaussian(x, *params[3:])
            ax1.plot(x, fitted_dip1 + background, 'k--', linewidth=1, label='peak 1')
            ax1.plot(x, fitted_dip2 + background, 'b--', linewidth=1, label='peak 2')
            fwhm = 2 * np.sqrt(2 * np.log(2)) * params[2]
            mean = params[1]
            area = np.trapz(fitted_dip1, x)
        elif selected_model == '2 peak Lorentzian':
            params, _ = curve_fit(_2lorentzian, x, y - background, p0=[amp1, mean1, stddev1, amp2, mean2, stddev2])
            fitted_dip1 = lorentzian(x, *params[:3])
            fitted_dip2 = lorentzian(x, *params[3:])
            ax1.plot(x, fitted_dip1 + background, 'k--', linewidth=1, label='peak 1')
            ax1.plot(x, fitted_dip2 + background, 'b--', linewidth=1, label='peak 2')
            fwhm = 2 * np.sqrt(2 * np.log(2)) * params[2]
            mean = params[1]
            area = np.trapz(fitted_dip1, x)
        elif selected_model == '2 peak Voigt':
            params, _ = curve_fit(_2voigt, x, y - background, p0=[amp1, mean1, stddev1, gamma1, amp2, mean2, stddev2, gamma2])
            fitted_dip1 = voigt(x, *params[:4])
            fitted_dip2 = voigt(x, *params[4:])
            ax1.plot(x, fitted_dip1 + background, 'k--', linewidth=1, label='peak 1')
            ax1.plot(x, fitted_dip2 + background, 'b--', linewidth=1, label='peak 2')  # Plot both peaks on ax1
            fwhm = 2 * np.sqrt(2 * np.log(2)) * params[2]
            mean = params[1]
            area = np.trapz(fitted_dip1, x)
        elif selected_model == '2 peak Pseudo-Voigt':
            params, _ = curve_fit(_2pseudovoigt, x, y - background, p0=[amp1, mean1, stddev1, fraction1, amp2, mean2, stddev2, fraction2])
            fitted_dip1 = pseudo_voigt(x, *params[:4])
            fitted_dip2 = pseudo_voigt(x, *params[4:])
            ax1.plot(x, fitted_dip1 + background, 'k--', linewidth=1, label='peak 1')
            ax1.plot(x, fitted_dip2 + background, 'b--', linewidth=1, label='peak 2')  # Plot both peaks on ax1
            fwhm = 2 * np.sqrt(2 * np.log(2)) * params[2]
            mean = params[1]
            area = np.trapz(fitted_dip1, x)
        elif selected_model == 'Voigt':
            params, _ = curve_fit(voigt, x, y - background, p0=[amp1, mean1, stddev1, gamma1])
            fitted_dip1 = voigt(x, *params)
            ax1.plot(x, fitted_dip1 + background, 'k--', linewidth=1, label='peak 1')
            fwhm = 2 * np.sqrt(2 * np.log(2)) * params[2]
            mean = params[1]
            area = np.trapz(fitted_dip1, x)
        elif selected_model == 'Pseudo-Voigt':
            params, _ = curve_fit(pseudo_voigt, x, y - background, p0=[amp1, mean1, stddev1, fraction1])
            fitted_dip1 = pseudo_voigt(x, *params)
            ax1.plot(x, fitted_dip1 + background, 'k--', linewidth=1, label='peak 1')
            fwhm = 2 * np.sqrt(2 * np.log(2)) * params[2]
            mean = params[1]
            area = np.trapz(fitted_dip1, x)
        elif selected_model == 'Gaussian':
            params, _ = curve_fit(gaussian, x, y - background, p0=[amp1, mean1, stddev1])
            fitted_dip1 = gaussian(x, *params)
            ax1.plot(x, fitted_dip1 + background, 'k--', linewidth=1, label='peak 1')
            fwhm = 2 * np.sqrt(2 * np.log(2)) * params[2]
            mean = params[1]
            area = np.trapz(fitted_dip1, x)
        elif selected_model == 'Lorentzian':
            params, _ = curve_fit(lorentzian, x, y - background, p0=[amp1, mean1, stddev1])
            fitted_dip1 = lorentzian(x, *params)
            ax1.plot(x, fitted_dip1 + background, 'k--', linewidth=1, label='peak 1')
            fwhm = 2 * np.sqrt(2 * np.log(2)) * params[2]
            mean = params[1]
            area = np.trapz(fitted_dip1, x)

        ax1.plot(dip_x, dip_y, 'ro', markersize=1, label='dip')
        ax1.set_xlabel('Channel Number', fontsize=8)
        ax1.set_ylabel('Intensity [a.u.]', fontsize=8)
        #ax1.set_title("Calibration", loc='center', fontsize=8)
        ax1.legend(loc='upper left', fontsize=4)
        ax1.tick_params(axis='x', labelsize=8)
        ax1.tick_params(axis='y', labelsize=8)
        canvas1.draw() 
    
    else:
        print("No calibration data available. Please load a calibration file.")    

def next_dip():
    global current_dip_index
    if dip and 0 <= current_dip_index < len(dip):
        current_dip_index = (current_dip_index + 1) % len(dip)
        fit_selected_model()
        
def update_meantable():
    for item in table.get_children():
        table.delete(item)
    for i, dip_index in enumerate(dip):
        mean = saved_means.get(f'Dip {i + 1}', None)  # Get the mean for this dip index
        table.insert("", "end", values=(dip_index, mean))
        
def save_current_fit(selected_model):
    global params
    global mean
    global current_dip_index
    dip_number = current_dip_index + 1
    saved_means[f'Dip {dip_number}'] = mean
    update_meantable()
    
def calculate_shift():
    if len(saved_means) >= 3:
        mean_values = [saved_means.get(f'Dip {i}', None) for i in range(1, 4)]
        shift = sum(mean_values[i] - mean_values[i - 1] for i in range(1, 3))
        calibration_file_path = load_calibration_text.get()
        calibration_file_name = os.path.basename(calibration_file_path)
        calibration_treeview.insert("", "end", values=(calibration_file_name, shift))
        
def clear_table():
    for item in calibration_treeview.get_children():
        calibration_treeview.delete(item)

def update_plot1():
    global calibration 
    calibration_file_path = filedialog.askopenfilename(title="Calibration")
    if calibration_file_path:
        calibration = []
        with open(calibration_file_path, 'r') as file: 
            for line in file:
                if ':' not in line:
                    try:
                        calibration.append(float(line.strip()))
                    except ValueError:
                        pass
                if 'Wavelength:' in line:
                    wavelength = float(line.split(':')[-1].strip())
                elif 'Mirror sp. :' in line:
                    mirror_spacing = float(line.split(':')[-1].strip())
                elif 'Ch. duration :' in line:
                    channel_duration = float(line.split(':')[-1].strip())
                elif 'Scan amplitude :' in line:
                    scan_amplitude = float(line.split(':')[-1].strip())
        ax1.clear()
        ax1.plot(calibration,'c-', linewidth=1)
        ax1.set_xlabel('Channel Number', fontsize=8)
        ax1.set_ylabel('Intensity [a.u.]', fontsize=8)
        #ax1.set_title("Calibration", loc='center', fontsize=8)
        canvas1.draw()
        load_calibration_text.delete(0, 'end')
        load_calibration_text.insert(0, calibration_file_path)

                
def update_table1():
    calibration_file_path = load_calibration_text.get()  
    if calibration_file_path:
        calibration = []  
        with open(calibration_file_path, 'r') as file:
            for line in file:
                if ':' not in line:
                    try:
                        calibration.append(float(line.strip()))
                    except ValueError:
                        pass
        calibration_treeview.insert("", "end", values=(calibration_file_path, BS_shift))
        average_BS_shift = sum(float(calibration_treeview.item(child)["values"][1]) for child in calibration_treeview.get_children()) / len(calibration_treeview.get_children())
        average_BS_shift_label.config(text=f"Average Calibration: {average_BS_shift:.2f}")

def export_calibration():
    filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if filename:
        with open(filename, 'w') as file:
            file.write(f"para_Cal_channel: {para_Cal_channel_entry.get()}\n")
            file.write(f"para_PS: {para_PS_entry.get()}\n")
            file.write(f"para_lambda: {para_lambda_entry.get()}\n")
            file.write(f"para_theta: {para_theta_entry.get()}\n")
            average_BS_shift = sum(float(calibration_treeview.item(child)["values"][1]) for child in calibration_treeview.get_children()) / len(calibration_treeview.get_children())
            file.write(f"average_BS_shift: {average_BS_shift}\n")
        #tkinter.messagebox.showinfo("Export Calibration", "Calibration data exported successfully.")

def load_calibration():
    filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if filename:
        with open(filename, 'r') as file:
            calibration_data = {}
            for line in file:
                key, value = line.strip().split(": ")
                calibration_data[key] = value
                
            para_Cal_channel_entry.delete(0, tk.END)
            para_PS_entry.delete(0, tk.END)
            para_lambda_entry.delete(0, tk.END)
            para_theta_entry.delete(0, tk.END)

            para_PS_entry.insert(0, calibration_data.get('para_PS', ''))
            para_lambda_entry.insert(0, calibration_data.get('para_lambda', ''))
            para_theta_entry.insert(0, calibration_data.get('para_theta', ''))

            if 'average_BS_shift' in calibration_data:
                para_Cal_channel_entry.insert(0, calibration_data['average_BS_shift'])
            else:
                para_Cal_channel_entry.insert(0, calibration_data.get('para_Cal_channel', ''))
        #tkinter.messagebox.showinfo("Load Calibration", "Calibration data loaded successfully.")

    
load_calibration_button = tk.Button(Tcalcenter_frame, text="Load Calibration", command=update_plot1)
load_calibration_button.pack()
load_calibration_text = tk.Entry(Tcalcenter_frame)
load_calibration_text.pack()

fig1, ax1 = plt.subplots(figsize=(3,1.7))
canvas1 = FigureCanvasTkAgg(fig1, calcenter_frame)
canvas1_widget = canvas1.get_tk_widget()
canvas1_widget.pack(fill=tk.BOTH, expand=True)
ax1.set_xlabel('Channel Number', fontsize=8)
ax1.set_ylabel('Intensity [a.u.]', fontsize=8)
#ax1.set_title("Calibration", loc='center', fontsize=8)
toolbar1 = NavigationToolbar2Tk(canvas1, calcenter_frame)
toolbar1.update()
toolbar1.pack()
plt.tight_layout() 

load_calibration_button = tk.Button(Tcalleft_frame, text="Load Settings", command=load_calibration)
load_calibration_button.pack()

para_theta_frame = tk.Frame(Tcalleft_frame)
para_theta_frame.pack(side=tk.TOP) 
para_theta_label = tk.Label(para_theta_frame, text="Scattering angle [°]:")
para_theta_label.pack(side=tk.LEFT)
para_theta_entry = tk.Entry(para_theta_frame)
para_theta_entry.insert(0, str(para_theta))  # Set the default value
para_theta_entry.pack(side=tk.LEFT)

para_lambda_frame = tk.Frame(Tcalleft_frame)
para_lambda_frame.pack(side=tk.TOP)
para_lambda_label = tk.Label(para_lambda_frame, text="Lambda [nm]:")
para_lambda_label.pack(side=tk.LEFT)
para_lambda_entry = tk.Entry(para_lambda_frame)
para_lambda_entry.insert(0, str(para_lambda))  # Set the default value
para_lambda_entry.pack(side=tk.LEFT)

para_PS_frame = tk.Frame(Tcalleft_frame)
para_PS_frame.pack(side=tk.TOP)
para_PS_label = tk.Label(para_PS_frame, text="Spacing [Å]:")
para_PS_label.pack(side=tk.LEFT)
para_PS_entry = tk.Entry(para_PS_frame)
para_PS_entry.insert(0, str(para_PS))  # Set the default value
para_PS_entry.pack(side=tk.LEFT)

para_Cal_frame = tk.Frame(Tcalleft_frame)
para_Cal_frame.pack(side=tk.TOP)
para_Cal_channel_label = tk.Label(para_Cal_frame, text="Calibration #:")
para_Cal_channel_label.pack(side=tk.LEFT)
para_Cal_channel_entry = tk.Entry(para_Cal_frame)
para_Cal_channel_entry.insert(0, str(para_Cal_channel))  # Set the default value
para_Cal_channel_entry.pack(side=tk.LEFT)

calibration_treeview = ttk.Treeview(Tcalcenter_frame, columns=("Calibration File", "Calibration"), show="headings")
calibration_treeview.heading("#1", text="Calibration File")
calibration_treeview.heading("#2", text="Calibration")
calibration_treeview.pack()
calibration_treeview.column("#1", width=150)  
calibration_treeview.column("#2", width=80)
calibration_treeview.configure(height=3)

amp_var = tk.StringVar(value=str(initial_amp_guess))
mean_var = tk.StringVar(value=str(initial_mean_guess))
stddev_var = tk.StringVar(value=str(initial_stddev_guess))
gamma_var = tk.StringVar(value=str(initial_gamma_guess))
fraction_var = tk.StringVar(value=str(initial_fraction_guess))

amp_varP2 = tk.StringVar(value=str(initial_amp_guessP2))
mean_varP2 = tk.StringVar(value=str(initial_mean_guessP2))
stddev_varP2 = tk.StringVar(value=str(initial_stddev_guessP2))
gamma_varP2 = tk.StringVar(value=str(initial_gamma_guessP2))
fraction_varP2 = tk.StringVar(value=str(initial_fraction_guessP2))

peak1_label = tk.Label(Bcalcenter_frame, text='Peak 1')
peak1_label.place(relx=0.4, rely=0)
peak2_label = tk.Label(Bcalcenter_frame, text='Peak 2')
peak2_label.place(relx=0.7, rely=0)

amp_label = tk.Label(Bcalcenter_frame, text='Amplitude:')
amp_label.place(relx=0, rely=0.15)
amp_entry = ttk.Entry(Bcalcenter_frame, width=4, textvariable=amp_var)
amp_entry.place(relx=0.4, rely=0.15)
amp_entryP2 = ttk.Entry(Bcalcenter_frame, width=4, textvariable=amp_varP2)
amp_entryP2.place(relx=0.7, rely=0.15)

mean_label = tk.Label(Bcalcenter_frame, text='Mean:')
mean_label.place(relx=0, rely=0.3)
mean_entry = ttk.Entry(Bcalcenter_frame, width=4, textvariable=mean_var)
mean_entry.place(relx=0.4, rely=0.3)
mean_entryP2 = ttk.Entry(Bcalcenter_frame, width=4, textvariable=mean_varP2)
mean_entryP2.place(relx=0.7, rely=0.3)

stddev_label = tk.Label(Bcalcenter_frame, text='Sigma:')
stddev_label.place(relx=0, rely=0.45)
stddev_entry = ttk.Entry(Bcalcenter_frame, width=4, textvariable=stddev_var)
stddev_entry.place(relx=0.4, rely=0.45)
stddev_entryP2 = ttk.Entry(Bcalcenter_frame, width=4, textvariable=stddev_varP2)
stddev_entryP2.place(relx=0.7, rely=0.45)

gamma_label = tk.Label(Bcalcenter_frame, text='Gamma:')
gamma_label.place(relx=0, rely=0.6)
gamma_entry = ttk.Entry(Bcalcenter_frame, width=4, textvariable=gamma_var)
gamma_entry.place(relx=0.4, rely=0.6)
gamma_entryP2 = ttk.Entry(Bcalcenter_frame, width=4, textvariable=gamma_varP2)
gamma_entryP2.place(relx=0.7, rely=0.6)

fraction_label = tk.Label(Bcalcenter_frame, text='Fraction:')
fraction_label.place(relx=0, rely=0.75)
fraction_entry = ttk.Entry(Bcalcenter_frame, width=4, textvariable=fraction_var)
fraction_entry.place(relx=0.4, rely=0.75)
fraction_entryP2 = ttk.Entry(Bcalcenter_frame, width=4, textvariable=fraction_varP2)
fraction_entryP2.place(relx=0.7, rely=0.75)

bg_frame = tk.Frame(Bcalleft_frame)
bg_frame.pack(side=tk.TOP)
bg_label = tk.Label(bg_frame, text='Baseline:')
bg_label.pack(side=tk.LEFT)

maxhalfwindowCAL_var = tk.StringVar(value=str(1e7))
maxhalfwindowCAL_entry = ttk.Entry(bg_frame, width=4, textvariable=maxhalfwindowCAL_var)
maxhalfwindowCAL_entry.pack(side=tk.LEFT)
maxhalfwindowCAL_entry.bind("<<Modified>>", fit_selected_model)

smoothhalfwindowCAL_var= tk.StringVar(value=str(0.02))
smoothhalfwindowCAL_entry = ttk.Entry(bg_frame, width=4, textvariable=smoothhalfwindowCAL_var)
smoothhalfwindowCAL_entry.pack(side=tk.LEFT)
smoothhalfwindowCAL_entry.bind("<<Modified>>", fit_selected_model)

fit_model_options = ['Gaussian','2 peak Gaussian', 'Lorentzian', '2 peak Lorentzian', 'Voigt', '2 peak Voigt', 'Pseudo-Voigt', '2 peak Pseudo-Voigt']
fit_model_var = tk.StringVar(value='2 peak Gaussian')  # Set the default option
fit_model_combobox = ttk.Combobox(Bcalleft_frame, textvariable=fit_model_var, values=fit_model_options, width=12)
fit_model_combobox.pack()

fit_next_button = tk.Button(Bcalleft_frame, text="Fit", command=fit_selected_model)
fit_next_button.pack()

move_next_button = tk.Button(Bcalleft_frame, text="Next", command=next_dip)
move_next_button.pack()

save_fit_button = tk.Button(Bcalright_frame, text="Save Peak 1 Fit", command=lambda: save_current_fit(fit_model_var.get()))
save_fit_button.pack()

table = ttk.Treeview(Bcalright_frame, columns=("Dip", "Mean"), show="headings")
table.heading("#1", text="Dip")
table.heading("#2", text="Mean")
table.column("#1", width=50)  
table.column("#2", width=100)
table.pack()
table.configure(height=3)

calculate_shift_button = tk.Button(Bcalright_frame, text="Save Calibration", command=calculate_shift)
calculate_shift_button.pack()

export_calibration_button = tk.Button(Tcalright_frame, text="Export Settings", command=export_calibration)
export_calibration_button.pack()

reset_button = tk.Button(Tcalright_frame, text="Reset", command=clear_table)
reset_button.pack()
       
        
        
        
        
        
        
        
        
# ------------------------------------------- DATA TAB 

Tdat_paned_window = tk.PanedWindow(frames["Data"], orient=tk.HORIZONTAL)
Tdat_paned_window.pack(fill=tk.BOTH, expand=False)
datcenter_frame = tk.Frame(frames["Data"], highlightbackground="black", highlightthickness=1)
datcenter_frame.pack(fill=tk.BOTH, expand=True)
Bdat_paned_window = tk.PanedWindow(frames["Data"], orient=tk.HORIZONTAL)
Bdat_paned_window.pack(fill=tk.BOTH, expand=False)

Tdatleft_frame = tk.Frame(Tdat_paned_window, highlightbackground="black", highlightthickness=1)
Tdatcenter_frame = tk.Frame(Tdat_paned_window, highlightbackground="black", highlightthickness=1)
Tdatright_frame = tk.Frame(Tdat_paned_window, highlightbackground="black", highlightthickness=1)
Tdat_paned_window.add(Tdatleft_frame, width=300, height=30)
Tdat_paned_window.add(Tdatcenter_frame, width=200, height=30)
Tdat_paned_window.add(Tdatright_frame, width=200, height=30)

Bdatleft_frame = tk.Frame(Bdat_paned_window, highlightbackground="black", highlightthickness=1)
Bdatcenter_frame = tk.Frame(Bdat_paned_window, highlightbackground="black", highlightthickness=1)
Bdatright_frame = tk.Frame(Bdat_paned_window, highlightbackground="black", highlightthickness=1)
Bdat_paned_window.add(Bdatleft_frame, width=200, height=160)
Bdat_paned_window.add(Bdatcenter_frame, width=180, height=160)
Bdat_paned_window.add(Bdatright_frame, width=400, height=160)

def fit_selected_model2(event=None):
    global velocity_x, velocity_y, peak, current_dip_index2, params2, mean, elements, checked_states
    
    if velocity_x is not None: 
        selected_model = fit_model_var2.get()
        x = velocity_x
        y = velocity_y
        
        
        initial_amp_guess = float(amp_var2.get())
        initial_mean_guess = float(mean_var2.get())
        initial_stddev_guess = float(stddev_var2.get())
        initial_gamma_guess = float(gamma_var2.get())
        initial_fraction_guess = float(fraction_var2.get())
        
        amp1 = float(amp_var2.get())
        mean1 = float(mean_var2.get())
        stddev1 = float(stddev_var2.get())
        gamma1 = float(gamma_var2.get())
        fraction1 = float(fraction_var2.get())
        amp2 = float(amp2_varP2.get())
        mean2 = float(mean2_varP2.get())
        stddev2 = float(stddev2_varP2.get())
        gamma2 = float(gamma2_varP2.get())
        fraction2 = float(fraction2_varP2.get())
        
        peak_x_entry_value = float(peak_x_entry.get())
        closest_peak_index = np.argmin(np.abs(x - peak_x_entry_value))
        opposite_x_position = - peak_x_entry_value
        closest_peak_indexOPP = np.argmin(np.abs(x - opposite_x_position))

        range_half_width = 3 * initial_stddev_guess
        start_index = max(0, closest_peak_index - int(range_half_width))
        end_index = min(len(x), closest_peak_index + int(range_half_width))
        new_x = x[start_index:end_index]
        new_y = y[start_index:end_index]
        
        start_indexOPP = max(0, closest_peak_indexOPP - int(range_half_width))
        end_indexOPP = min(len(x), closest_peak_indexOPP + int(range_half_width))
        new_xOPP = x[start_indexOPP:end_indexOPP]
        new_yOPP = y[start_indexOPP:end_indexOPP]
        
        baseline_fitter = Baseline(x_data=new_x)
        maxhalfwindow_value = int(maxhalfwindow_var.get())
        smoothhalfwindow_value = int(smoothhalfwindow_var.get())
        background = baseline_fitter.snip(new_y, max_half_window=maxhalfwindow_value, decreasing=True, smooth_half_window=smoothhalfwindow_value)[0]
        
        baseline_fitterOPP = Baseline(x_data=new_xOPP)
        backgroundOPP = baseline_fitterOPP.snip(new_yOPP, max_half_window=maxhalfwindow_value, decreasing=True, smooth_half_window=smoothhalfwindow_value)[0]
        
                
        params2_values = ["", "", "", "", "", "", "", ""]
        params2OPP_values = ["", "", "", "", "", "", "", ""]
        
        param_names = []
        model_param_names = {
            'Gaussian': ['Amplitude', 'Mean', 'Sigma'],
            'Lorentzian': ['Amplitude', 'Mean', 'Gamma'],
            'Voigt': ['Amplitude', 'Mean', 'Sigma', 'Gamma'],
            'Pseudo-Voigt': ['Amplitude', 'Mean', 'Sigma', 'Fraction'],
            '2 peak Gaussian': ['Amplitude', 'Mean', 'Sigma'],
            '2 peak Lorentzian': ['Amplitude', 'Mean', 'Gamma'],
            '2 peak Voigt': ['Amplitude', 'Mean', 'Sigma', 'Gamma'],
            '2 peak Pseudo-Voigt': ['Amplitude', 'Mean', 'Sigma', 'Fraction']
        }
        
        canvasINSET.draw_idle()
        axINSET.clear()
        ax2.clear()
        
        if selected_model == 'Gaussian':
            params2, _ = curve_fit(gaussian, new_x, new_y - background, p0=[initial_amp_guess, peak_x_entry_value, initial_stddev_guess])
            fitted_peak = gaussian(new_x, *params2)
            fwhm = 2 * np.sqrt(2 * np.log(2)) * params2[2]
            mean = params2[1]
            area = np.trapz(fitted_peak, new_x)
            
            params2OPP, _ = curve_fit(gaussian, new_xOPP, new_yOPP - backgroundOPP, p0=[initial_amp_guess, opposite_x_position, initial_stddev_guess])
            fitted_peakOPP = gaussian(new_xOPP, *params2OPP)
            fwhmOPP = 2 * np.sqrt(2 * np.log(2)) * params2OPP[2]
            meanOPP = params2OPP[1]
            areaOPP = np.trapz(fitted_peakOPP, new_xOPP)
            
            axINSET.plot(new_x, fitted_peak + background, 'k--', linewidth=1, label='fitted trend')
            ax2.plot(new_x, fitted_peak + background, 'k--', linewidth=1, label='fitted trend')
            ax2.plot(new_xOPP, fitted_peakOPP + backgroundOPP, 'k--', linewidth=1)
            
            larger_peak = fitted_peak
            x1, x2 = min(new_x), max(new_x)
            y1, y2 = min(larger_peak + background), 1.1 * max(larger_peak + background)

        elif selected_model == 'Lorentzian':
            params2, _ = curve_fit(lorentzian, new_x, new_y - background, p0=[initial_amp_guess, peak_x_entry_value, initial_gamma_guess])
            fitted_peak = lorentzian(new_x, *params2)
            fwhm = 2 * np.sqrt(2 * np.log(2)) * params2[2]
            mean = params2[1]
            area = np.trapz(fitted_peak, new_x)
            
            params2OPP, _ = curve_fit(lorentzian, new_xOPP, new_yOPP - backgroundOPP, p0=[initial_amp_guess, opposite_x_position, initial_gamma_guess])
            fitted_peakOPP = lorentzian(new_xOPP, *params2OPP)
            fwhmOPP = 2 * np.sqrt(2 * np.log(2)) * params2OPP[2]
            meanOPP = params2OPP[1]
            areaOPP = np.trapz(fitted_peakOPP, new_xOPP)
            
            axINSET.plot(new_x, fitted_peak + background, 'k--', linewidth=1, label='fitted trend')
            ax2.plot(new_x, fitted_peak + background, 'k--', linewidth=1, label='fitted trend')
            ax2.plot(new_xOPP, fitted_peakOPP + backgroundOPP, 'k--', linewidth=1)
            
            larger_peak = fitted_peak
            x1, x2 = min(new_x), max(new_x)
            y1, y2 = min(larger_peak + background), 1.1 * max(larger_peak + background)

            
        elif selected_model == 'Voigt':
            params2, _ = curve_fit(voigt, new_x, new_y - background, p0=[initial_amp_guess, peak_x_entry_value, initial_stddev_guess, initial_gamma_guess])
            fitted_peak = voigt(new_x, *params2)
            fwhm = 2 * np.sqrt(2 * np.log(2)) * params2[2]
            mean = params2[1]
            area = np.trapz(fitted_peak, new_x)
            
            params2OPP, _ = curve_fit(voigt, new_xOPP, new_yOPP - backgroundOPP, p0=[initial_amp_guess, opposite_x_position, initial_stddev_guess, initial_gamma_guess])
            fitted_peakOPP = voigt(new_xOPP, *params2OPP)
            fwhmOPP = 2 * np.sqrt(2 * np.log(2)) * params2OPP[2]
            meanOPP = params2OPP[1]
            areaOPP = np.trapz(fitted_peakOPP, new_xOPP)
            
            axINSET.plot(new_x, fitted_peak + background, 'k--', linewidth=1, label='fitted trend')
            ax2.plot(new_x, fitted_peak + background, 'k--', linewidth=1, label='fitted trend')
            ax2.plot(new_xOPP, fitted_peakOPP + backgroundOPP, 'k--', linewidth=1)
            
            larger_peak = fitted_peak
            x1, x2 = min(new_x), max(new_x)
            y1, y2 = min(larger_peak + background), 1.1 * max(larger_peak + background)

            
        elif selected_model == 'Pseudo-Voigt':
            params2, _ = curve_fit(pseudo_voigt, new_x, new_y - background, p0=[initial_amp_guess, peak_x_entry_value, initial_stddev_guess, initial_fraction_guess])
            fitted_peak = pseudo_voigt(new_x, *params2)
            fwhm = 2 * np.sqrt(2 * np.log(2)) * params2[2]
            mean = params2[1]
            area = np.trapz(fitted_peak, new_x)
            
            params2OPP, _ = curve_fit(pseudo_voigt, new_xOPP, new_yOPP - backgroundOPP, p0=[initial_amp_guess, opposite_x_position, initial_stddev_guess, initial_fraction_guess])
            fitted_peakOPP = pseudo_voigt(new_xOPP, *params2OPP)
            fwhmOPP = 2 * np.sqrt(2 * np.log(2)) * params2OPP[2]
            meanOPP = params2OPP[1]
            areaOPP = np.trapz(fitted_peakOPP, new_xOPP)
            
            axINSET.plot(new_x, fitted_peak + background, 'k--', linewidth=1, label='fitted trend')
            ax2.plot(new_x, fitted_peak + background, 'k--', linewidth=1, label='fitted trend')
            ax2.plot(new_xOPP, fitted_peakOPP + backgroundOPP, 'k--', linewidth=1)
            
            larger_peak = fitted_peak
            x1, x2 = min(new_x), max(new_x)
            y1, y2 = min(larger_peak + background), 1.1 * max(larger_peak + background)

        
        elif selected_model == '2 peak Gaussian':
            params2,_ = curve_fit(_2gaussian, new_x, new_y - background, p0=[amp1, mean1, stddev1, amp2, mean2, stddev2])
            fitted_peak = gaussian(new_x, *params2[:3])  
            fitted_peak2 = gaussian(new_x, *params2[3:])
            ax2.plot(new_x, fitted_peak + background, 'k--', linewidth=1, label='peak 1')
            ax2.plot(new_x, fitted_peak2 + background, 'b--', linewidth=1, label='peak 2')
            axINSET.plot(new_x, fitted_peak + background, 'k--', linewidth=1, label='peak 1')
            axINSET.plot(new_x, fitted_peak2 + background, 'b--', linewidth=1, label='peak 2')
            
            fwhm = 2 * np.sqrt(2 * np.log(2)) * params2[2]
            mean = params2[1]
            area = np.trapz(fitted_peak, new_x)
            
            params2OPP,_ = curve_fit(_2gaussian, new_xOPP, new_yOPP - backgroundOPP, p0=[amp1, -mean1, stddev1, amp2, -mean2, stddev2])
            fitted_peakOPP = gaussian(new_xOPP, *params2OPP[:3])  
            fitted_peak2OPP = gaussian(new_xOPP, *params2OPP[3:])
            ax2.plot(new_xOPP, fitted_peakOPP + backgroundOPP, 'k--', linewidth=1)
            ax2.plot(new_xOPP, fitted_peak2OPP + backgroundOPP, 'b--', linewidth=1)
            axINSET.plot(new_xOPP, fitted_peakOPP + backgroundOPP, 'k--', linewidth=1)
            axINSET.plot(new_xOPP, fitted_peak2OPP + backgroundOPP, 'b--', linewidth=1)
            
            fwhmOPP = 2 * np.sqrt(2 * np.log(2)) * params2OPP[2]
            meanOPP = params2OPP[1]
            areaOPP = np.trapz(fitted_peakOPP, new_xOPP)
            
            if max(fitted_peak2) > max(fitted_peak):
                larger_peak = fitted_peak2
            else:
                larger_peak = fitted_peak
            x1, x2 = min(new_x), max(new_x)
            y1, y2 = min(larger_peak + background), 1.1 * max(larger_peak + background)

        elif selected_model == '2 peak Lorentzian':
            params2, _ = curve_fit(_2lorentzian, new_x, new_y - background, p0=[amp1, mean1, stddev1, amp2, mean2, stddev2])
            fitted_peak = lorentzian(new_x, *params2[:3])
            fitted_peak2 = lorentzian(new_x, *params2[3:])
            ax2.plot(new_x, fitted_peak + background, 'k--', linewidth=1, label='peak 1')
            ax2.plot(new_x, fitted_peak2 + background, 'b--', linewidth=1, label='peak 2')
            axINSET.plot(new_x, fitted_peak + background, 'k--', linewidth=1, label='peak 1')
            axINSET.plot(new_x, fitted_peak2 + background, 'b--', linewidth=1, label='peak 2')
            
            fwhm = 2 * np.sqrt(2 * np.log(2)) * params2[2]
            mean = params2[1]
            area = np.trapz(fitted_peak, new_x)
            
            params2OPP, _ = curve_fit(_2lorentzian, new_xOPP, new_yOPP - backgroundOPP, p0=[amp1, -mean1, stddev1, amp2, -mean2, stddev2])
            fitted_peakOPP = lorentzian(new_xOPP, *params2OPP[:3])
            fitted_peak2OPP = lorentzian(new_xOPP, *params2OPP[3:])
            ax2.plot(new_xOPP, fitted_peakOPP + backgroundOPP, 'k--', linewidth=1)
            ax2.plot(new_xOPP, fitted_peak2OPP + backgroundOPP, 'b--', linewidth=1)
            axINSET.plot(new_xOPP, fitted_peakOPP + backgroundOPP, 'k--', linewidth=1)
            axINSET.plot(new_xOPP, fitted_peak2OPP + backgroundOPP, 'b--', linewidth=1)
            
            fwhmOPP = 2 * np.sqrt(2 * np.log(2)) * params2OPP[2]
            meanOPP = params2OPP[1]
            areaOPP = np.trapz(fitted_peakOPP, new_xOPP)
            
            if max(fitted_peak2) > max(fitted_peak):
                larger_peak = fitted_peak2
            else:
                larger_peak = fitted_peak
            x1, x2 = min(new_x), max(new_x)
            y1, y2 = min(larger_peak + background), 1.1 * max(larger_peak + background)

        elif selected_model == '2 peak Voigt':
            params2, _ = curve_fit(_2voigt, new_x, new_y - background, p0=[amp1, mean1, stddev1, gamma1, amp2, mean2, stddev2, gamma2])
            fitted_peak = voigt(new_x, *params2[:4])
            fitted_peak2 = voigt(new_x, *params2[4:])
            ax2.plot(new_x, fitted_peak + background, 'k--', linewidth=1, label='peak 1')
            ax2.plot(new_x, fitted_peak2 + background, 'b--', linewidth=1, label='peak 2')
            axINSET.plot(new_x, fitted_peak + background, 'k--', linewidth=1, label='peak 1')
            axINSET.plot(new_x, fitted_peak2 + background, 'b--', linewidth=1, label='peak 2')
            
            fwhm = 2 * np.sqrt(2 * np.log(2)) * params2[2]
            mean = params2[1]
            area = np.trapz(fitted_peak, new_x)
            
            params2OPP, _ = curve_fit(_2voigt, new_xOPP, new_yOPP - backgroundOPP, p0=[amp1, -mean1, stddev1, gamma1, amp2, -mean2, stddev2, gamma2])
            fitted_peakOPP = voigt(new_xOPP, *params2OPP[:4])
            fitted_peak2OPP = voigt(new_xOPP, *params2OPP[4:])
            ax2.plot(new_xOPP, fitted_peakOPP + backgroundOPP, 'k--', linewidth=1)
            ax2.plot(new_xOPP, fitted_peak2OPP + backgroundOPP, 'b--', linewidth=1)
            axINSET.plot(new_xOPP, fitted_peakOPP + backgroundOPP, 'k--', linewidth=1)
            axINSET.plot(new_xOPP, fitted_peak2OPP + backgroundOPP, 'b--', linewidth=1)
            
            fwhmOPP = 2 * np.sqrt(2 * np.log(2)) * params2OPP[2]
            meanOPP = params2OPP[1]
            areaOPP = np.trapz(fitted_peakOPP, new_xOPP)
            
            if max(fitted_peak2) > max(fitted_peak):
                larger_peak = fitted_peak2
            else:
                larger_peak = fitted_peak
            x1, x2 = min(new_x), max(new_x)
            y1, y2 = min(larger_peak + background), 1.1 * max(larger_peak + background)

        elif selected_model == '2 peak Pseudo-Voigt':
            params2, _ = curve_fit(_2pseudovoigt, new_x, new_y - background, p0=[amp1, mean1, stddev1, fraction1, amp2, mean2, stddev2, fraction2])
            fitted_peak = pseudo_voigt(new_x, *params2[:4])
            fitted_peak2 = pseudo_voigt(new_x, *params2[4:])
            ax2.plot(new_x, fitted_peak + background, 'k--', linewidth=1, label='peak 1')
            ax2.plot(new_x, fitted_peak2 + background, 'b--', linewidth=1, label='peak 2')
            axINSET.plot(new_x, fitted_peak + background, 'k--', linewidth=1, label='peak 1')
            axINSET.plot(new_x, fitted_peak2 + background, 'b--', linewidth=1, label='peak 2')
            
            fwhm = 2 * np.sqrt(2 * np.log(2)) * params2[2]
            mean = params2[1]
            area = np.trapz(fitted_peak, new_x)
            
            params2OPP, _ = curve_fit(_2pseudovoigt, new_xOPP, new_yOPP - backgroundOPP, p0=[amp1, -mean1, stddev1, fraction1, amp2, -mean2, stddev2, fraction2])
            fitted_peakOPP = pseudo_voigt(new_xOPP, *params2OPP[:4])
            fitted_peak2OPP = pseudo_voigt(new_xOPP, *params2OPP[4:])
            ax2.plot(new_xOPP, fitted_peakOPP + backgroundOPP, 'k--', linewidth=1)
            ax2.plot(new_xOPP, fitted_peak2OPP + backgroundOPP, 'b--', linewidth=1)
            axINSET.plot(new_xOPP, fitted_peakOPP + backgroundOPP, 'k--', linewidth=1)
            axINSET.plot(new_xOPP, fitted_peak2OPP + backgroundOPP, 'b--', linewidth=1)
            
            fwhmOPP = 2 * np.sqrt(2 * np.log(2)) * params2OPP[2]
            meanOPP = params2OPP[1]
            areaOPP = np.trapz(fitted_peakOPP, new_xOPP)
            
            if max(fitted_peak2) > max(fitted_peak):
                larger_peak = fitted_peak2
            else:
                larger_peak = fitted_peak
            x1, x2 = min(new_x), max(new_x)
            y1, y2 = min(larger_peak + background), 1.1 * max(larger_peak + background)

        
        if selected_model in model_param_names:
            param_names = model_param_names[selected_model]
            params2_formatted = [f"{params2[i]:.3f}" for i in range(len(param_names))]
            params2OPP_formatted = [f"{params2OPP[i]:.3f}" for i in range(len(param_names))]

            # Check for parameter names and assign values accordingly
            if 'Amplitude' in param_names:
                params2_values[1] = params2_formatted[param_names.index('Amplitude')]
                params2OPP_values[1] = params2OPP_formatted[param_names.index('Amplitude')]
            if 'Mean' in param_names:
                params2_values[2] = params2_formatted[param_names.index('Mean')]
                params2OPP_values[2] = params2OPP_formatted[param_names.index('Mean')]
            if 'Sigma' in param_names:
                params2_values[3] = params2_formatted[param_names.index('Sigma')]
                params2OPP_values[3] = params2OPP_formatted[param_names.index('Sigma')]
            if 'Gamma' in param_names:
                params2_values[4] = params2_formatted[param_names.index('Gamma')]
                params2OPP_values[4] = params2OPP_formatted[param_names.index('Gamma')]
            if 'Fraction' in param_names:
                params2_values[5] = params2_formatted[param_names.index('Fraction')]
                params2OPP_values[5] = params2OPP_formatted[param_names.index('Fraction')]
                
        # Set FWHM, Height, and Area values
        params2_values[6] = f"{fwhm:.3f}"
        params2_values[7] = f"{area:.3f}"
        params2OPP_values[6] = f"{fwhm:.3f}"
        params2OPP_values[7] = f"{area:.3f}"
        
        # Insert values into the table
        tableDATA.delete(*tableDATA.get_children())
        tableDATA.insert("", "end", values=(params2_values[1], params2_values[2], params2_values[3], params2_values[4], params2_values[5], params2_values[6], params2_values[7]))
        tableDATA.insert("", "end", values=(params2OPP_values[1], params2OPP_values[2], params2OPP_values[3], params2OPP_values[4], params2OPP_values[5], params2OPP_values[6], params2OPP_values[7]))
        
        axINSET.plot(x, y, 'c-', linewidth=1, label='spectrum')
        axINSET.plot(new_xOPP, fitted_peakOPP + backgroundOPP, 'k--', linewidth=1)
        axINSET.plot(new_x, background, 'r--', linewidth=1, label='baseline')
        axINSET.plot(new_xOPP, backgroundOPP, 'r--', linewidth=1)
        axINSET.tick_params(axis='both', labelsize=6)
        axINSET.set_xlim(x1, x2)
        axINSET.set_ylim(y1, y2)
        para_chi_value = para_chi_entry.get()
        axINSET.annotate(f'Chi: {para_chi_value}°', xy=(0.95, 0.95), xycoords='axes fraction',fontsize=6, ha='right', va='top')
        canvasINSET.draw()
        
        ax2.plot(x, y, 'c-', linewidth=1, label='spectrum')
        ax2.plot(new_x, background, 'r--', linewidth=1, label='baseline')
        ax2.plot(x[closest_peak_index], y[closest_peak_index], 'ro', markersize=1, label='peak')
        ax2.plot(new_xOPP, backgroundOPP, 'r--', linewidth=1)
        ax2.plot(x[closest_peak_indexOPP], y[closest_peak_indexOPP], 'ro', markersize=1)
        ax2.set_xlabel('Velocity [m/s]', fontsize=8)
        ax2.set_ylabel('Intensity [a.u.]', fontsize=8)
        ax2.legend(loc='upper left', fontsize=4)
        ax2.tick_params(axis='x', labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)
        update_axis_bounds()
        canvas2.draw() 
    else:
        print("No data available. Please load a new file.")

def subtract_background():
    global velocity_y, original_scattering_dataSB, background_subtraction_active, elements
    
    if original_scattering_dataSB is None:
        return

    if background_subtraction_active:
        velocity_y = np.copy(original_scattering_dataSB)
        background_subtraction_active = False
    else:
        if original_scattering_dataSB is None:
            original_scattering_dataSB = np.copy(velocity_y)

        baseline_fitter = Baseline(x_data=velocity_x)
        maxhalfwindow_value = int(maxhalfwindow_var.get())
        smoothhalfwindow_value = int(smoothhalfwindow_var.get())
        background = baseline_fitter.snip(velocity_y, max_half_window=maxhalfwindow_value, decreasing=True, smooth_half_window=smoothhalfwindow_value)[0]
        
        velocity_y = velocity_y - background
        background_subtraction_active = True


    axINSET.clear()
    canvasINSET.draw_idle()
    ax2.clear()
    #plot_checkbox_data()
    ax2.plot(velocity_x, velocity_y, 'c-', linewidth=1)
    ax2.set_xlabel('Velocity [m/s]', fontsize=8)
    ax2.set_ylabel('Intensity [a.u.]', fontsize=8)
    #ax2.set_title('Data', loc='center', fontsize=8)
    ax2.legend(loc='upper left', fontsize=4)
    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    update_axis_bounds()
    canvas2.draw()

def smooth_data():
    global velocity_y,  original_scattering_dataSD, smooth_active, elements
    
    if original_scattering_dataSD is None:
        return
    
    if not smooth_active:  # Check if already smoothed
        original_scattering_dataSD = np.copy(velocity_y)  # Store original data

        smooth_window_size = 3  
        velocity_y_smoothed = np.convolve(velocity_y, np.ones(smooth_window_size) / smooth_window_size, mode='same')
        velocity_y_smoothed[np.isnan(velocity_y_smoothed)] = 0
        
        velocity_y = velocity_y_smoothed
        smooth_active = True
    else:
        # Restore original data
        velocity_y = np.copy(original_scattering_dataSD)
        smooth_active = False
    
    axINSET.clear()
    canvasINSET.draw_idle()
    ax2.clear()
    #plot_checkbox_data()
    ax2.plot(velocity_x, velocity_y, 'c-', linewidth=1)
    ax2.set_xlabel('Velocity [m/s]', fontsize=8)
    ax2.set_ylabel('Intensity [a.u.]', fontsize=8)
    #ax2.set_title('Data', loc='center', fontsize=8)
    ax2.legend(loc='upper left', fontsize=4)
    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    update_axis_bounds()
    canvas2.draw()
    
def update_initial_amp2(*args):
    try:
        initial_amp_guess2 = float(amp_var2.get())
        fit_selected_model2()
    except ValueError:
        pass

def update_initial_mean2(*args):
    try:
        initial_mean_guess2 = float(mean_var2.get())
        fit_selected_model2()
    except ValueError:
        pass

def update_initial_stddev2(*args):
    try:
        initial_stddev_guess2 = float(stddev_var2.get())
        fit_selected_model2()
    except ValueError:
        pass

def update_initial_gamma2(*args):
    try:
        initial_gamma_guess2 = float(gamma_var2.get())
        fit_selected_model2()
    except ValueError:
        pass

def update_initial_fraction2(*args):
    try:
        initial_fraction_guess2 = float(fraction_var2.get())
        fit_selected_model2()
    except ValueError:
        pass
        
def update_axis_bounds(*args):
    x_min = float(x_lower_bound_entry.get())
    x_max = float(x_upper_bound_entry.get())
    y_min = float(y_lower_bound_entry.get())
    y_max = float(y_upper_bound_entry.get())
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlim(x_min, x_max)
    canvas2.draw()

def export_data():
    filename = filedialog.asksaveasfilename(defaultextension=".txt")
    if filename:
        with open(filename, 'w') as file:
            for x, y in zip(velocity_x, velocity_y):
                file.write(f"{x}\t{y}\n")
                
def on_click(event):
    if event.inaxes == ax2:
        x_clicked = event.xdata
        peak_x_entry.delete(0, tk.END)
        peak_x_entry.insert(0, f"{x_clicked:.2f}")
        fit_selected_model2()

def applycalibration():
    global scattering_data, elements, calibration, velocity_x, velocity_y, original_scattering_dataSD, original_scattering_dataSB

    para_theta = float(para_theta_entry.get())
    
    total_channel = len(scattering_data)
    channel_number = np.linspace(0.5 - total_channel/2, total_channel/2 - 0.5, total_channel)
    
    if calibration_treeview.get_children():  # Check if there are entries in calibration_treeview
        bs_shift_values = [float(calibration_treeview.item(child)["values"][1]) for child in calibration_treeview.get_children()]
        average_bs_shift = sum(bs_shift_values) / len(bs_shift_values)
        para_Cal_channel = average_bs_shift
    else:
        para_Cal_channel = float(para_Cal_channel_entry.get())
    
    BS_shift = SPEED_OF_LIGHT * channel_number / (2 * para_PS / 1000 * para_Cal_channel) * 1e-6
    velocity_y = para_lambda * scattering_data * BS_shift * 10**-9 / (2 * np.sin(math.radians(para_theta / 2)))

    velocity_y = np.abs(velocity_y)
    velocity_x = channel_number * 100
    
    axINSET.clear()
    canvasINSET.draw_idle()
    ax2.clear()
    ax2.plot(velocity_x, velocity_y, 'c-', linewidth=1)
    ax2.set_xlabel('Velocity [m/s]', fontsize=8)
    ax2.set_ylabel('Intensity [a.u.]', fontsize=8)
    #ax2.set_title('Data', loc='center', fontsize=8)
    update_axis_bounds()
    canvas2.draw()
    original_scattering_dataSD = velocity_y.copy()
    original_scattering_dataSB = velocity_y.copy()
    subtract_background_button.configure(state="normal")
    smooth_button.configure(state="normal")
    #overlay_button.configure(state="normal")

def update_plot2():
    global scattering_data, current_filename
    data_file_path = filedialog.askopenfilename(title="Data")
    if data_file_path:
        current_filename = os.path.basename(data_file_path)
        scattering_data = []
        with open(data_file_path, 'r') as file:
            for line in file:
                if ':' not in line:
                    try:
                        scattering_data.append(float(line.strip()))
                    except ValueError:
                        pass
        scattering_data = np.array(scattering_data)
        ax2.clear()
        ax2.plot(scattering_data, 'c-', linewidth=1)
        ax2.set_xlabel('Channel Number', fontsize=8)
        ax2.set_ylabel('Intensity [a.u.]', fontsize=8)
        ax2.tick_params(axis='x', labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)
        canvas2.draw()
        load_data_text.delete(0, 'end')
        load_data_text.insert(0, data_file_path)

def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

def on_mousewheel(event):
    if event.delta > 0:
        canvas.yview_scroll(-1, "units") 
    elif event.delta < 0:
        canvas.yview_scroll(1, "units")

def use_in_stack():
    global datasets
    # Generate a unique dataset name
    dataset_name = f"Dataset_{len(datasets) + 1}"
    # Assuming velocity_x and velocity_y are accessible here
    datasets[dataset_name] = {"x": velocity_x, "y": velocity_y}
    calibrated_flags[dataset_name] = False
    offset_vars[dataset_name] = tk.DoubleVar(value=0) 
    horizontal_offset_vars[dataset_name] = tk.DoubleVar(value=0) 
    plot_colors[dataset_name] = 'blue'  # Default color
    create_controls_for_data(dataset_name)
    update_plotSTACK()
    update_dataset_menu()

loaddata_frame = tk.Frame(Tdatleft_frame)
loaddata_frame.pack(side=tk.TOP)
load_data_button = tk.Button(loaddata_frame, text="Load Data", command=update_plot2)
load_data_button.pack(side=tk.LEFT)
load_data_text = tk.Entry(loaddata_frame)
load_data_text.pack(side=tk.LEFT)

fig2, ax2 = plt.subplots(figsize=(3,2))
canvas2 = FigureCanvasTkAgg(fig2, datcenter_frame)
canvas2_widget = canvas2.get_tk_widget()
canvas2_widget.pack(fill=tk.BOTH, expand=True)
ax2.set_xlabel('Channel Number', fontsize=8)
ax2.set_ylabel('Intensity [a.u.]', fontsize=8)
toolbar2 = NavigationToolbar2Tk(canvas2, datcenter_frame)
toolbar2.update()
toolbar2.pack()
fig2.canvas.mpl_connect('button_press_event', on_click)
plt.tight_layout()

class CustomToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas_, parent_):
        self.toolitems = [t for t in NavigationToolbar2Tk.toolitems if t[0] == 'Save']
        super().__init__(canvas_, parent_)

figINSET, axINSET = plt.subplots(figsize=(2,2))
canvasINSET = FigureCanvasTkAgg(figINSET, datcenter_frame)
canvasINSET_widget = canvasINSET.get_tk_widget()
canvasINSET_widget.place(relx=0.85, rely=0.185, relwidth=0.3, relheight=0.35, anchor=tk.CENTER)
axINSET.tick_params(axis='both', labelsize=8)
toolbarINSET = CustomToolbar(canvasINSET, datcenter_frame)
toolbarINSET.update()
toolbarINSET.place(relx=0.35, rely=0.965, anchor=tk.CENTER)  # Adjust position as needed
plt.tight_layout()

x_lower_bound_var = tk.StringVar(value="-15000")
x_upper_bound_var = tk.StringVar(value="15000")
y_lower_bound_var = tk.StringVar(value="0")
y_upper_bound_var = tk.StringVar(value="2")

bounds_frame = tk.Frame(Bdatright_frame)
bounds_frame.pack(side=tk.TOP)
x_bound_label = tk.Label(bounds_frame, text="X Bounds:")
x_bound_label.pack(side=tk.LEFT)
x_lower_bound_entry = ttk.Entry(bounds_frame, textvariable=x_lower_bound_var, width=5)
x_lower_bound_entry.pack(side=tk.LEFT)
x_upper_bound_entry = ttk.Entry(bounds_frame, textvariable=x_upper_bound_var, width=5)
x_upper_bound_entry.pack(side=tk.LEFT)

y_bound_label = tk.Label(bounds_frame, text="Y Bounds:")
y_bound_label.pack(side=tk.LEFT)
y_lower_bound_entry = ttk.Entry(bounds_frame, textvariable=y_lower_bound_var, width=4)
y_lower_bound_entry.pack(side=tk.LEFT)
y_upper_bound_entry = ttk.Entry(bounds_frame, textvariable=y_upper_bound_var, width=4)
y_upper_bound_entry.pack(side=tk.LEFT)

x_lower_bound_entry.bind("<FocusOut>", lambda event: update_axis_bounds())
x_upper_bound_entry.bind("<FocusOut>", lambda event: update_axis_bounds())
y_lower_bound_entry.bind("<FocusOut>", lambda event: update_axis_bounds())
y_upper_bound_entry.bind("<FocusOut>", lambda event: update_axis_bounds())

apply_calibration_button = tk.Button(Tdatcenter_frame, text="Apply Calibration", command=applycalibration)
apply_calibration_button.pack()

parachi_frame = tk.Frame(Bdatleft_frame)
parachi_frame.pack(side=tk.TOP)
para_chi_label = tk.Label(parachi_frame, text="Chi [°]:")
para_chi_label.pack(side=tk.LEFT)
para_chi_entry = tk.Entry(parachi_frame)
para_chi_entry.insert(0, str(para_theta))  
para_chi_entry.pack(side=tk.LEFT)

amp_var2 = tk.StringVar(value=str(initial_amp_guess2))
mean_var2 = tk.StringVar(value=str(initial_mean_guess2))
stddev_var2 = tk.StringVar(value=str(initial_stddev_guess2))
gamma_var2 = tk.StringVar(value=str(initial_gamma_guess2))
fraction_var2 = tk.StringVar(value=str(initial_fraction_guess2))

amp2_varP2 = tk.StringVar(value=str(initial_amp2_guessP2))
mean2_varP2 = tk.StringVar(value=str(initial_mean2_guessP2))
stddev2_varP2 = tk.StringVar(value=str(initial_stddev2_guessP2))
gamma2_varP2 = tk.StringVar(value=str(initial_gamma2_guessP2))
fraction2_varP2 = tk.StringVar(value=str(initial_fraction2_guessP2))

peak1_label = tk.Label(Bdatcenter_frame, text='Peak 1')
peak1_label.place(relx=0.4, rely=0)
peak2_label = tk.Label(Bdatcenter_frame, text='Peak 2')
peak2_label.place(relx=0.7, rely=0)

amp_label = tk.Label(Bdatcenter_frame, text='Amplitude:')
amp_label.place(relx=0, rely=0.15)
amp_entry = ttk.Entry(Bdatcenter_frame, width=4, textvariable=amp_var2)
amp_entry.place(relx=0.4, rely=0.15)
amp_entryP2 = ttk.Entry(Bdatcenter_frame, width=4, textvariable=amp2_varP2)
amp_entryP2.place(relx=0.7, rely=0.15)

mean_label = tk.Label(Bdatcenter_frame, text='Mean:')
mean_label.place(relx=0, rely=0.3)
mean_entry = ttk.Entry(Bdatcenter_frame, width=4, textvariable=mean_var2)
mean_entry.place(relx=0.4, rely=0.3)
mean_entryP2 = ttk.Entry(Bdatcenter_frame, width=4, textvariable=mean2_varP2)
mean_entryP2.place(relx=0.7, rely=0.3)

stddev_label = tk.Label(Bdatcenter_frame, text='Sigma:')
stddev_label.place(relx=0, rely=0.45)
stddev_entry = ttk.Entry(Bdatcenter_frame, width=4, textvariable=stddev_var2)
stddev_entry.place(relx=0.4, rely=0.45)
stddev_entryP2 = ttk.Entry(Bdatcenter_frame, width=4, textvariable=stddev2_varP2)
stddev_entryP2.place(relx=0.7, rely=0.45)

gamma_label = tk.Label(Bdatcenter_frame, text='Gamma:')
gamma_label.place(relx=0, rely=0.6)
gamma_entry = ttk.Entry(Bdatcenter_frame, width=4, textvariable=gamma_var2)
gamma_entry.place(relx=0.4, rely=0.6)
gamma_entryP2 = ttk.Entry(Bdatcenter_frame, width=4, textvariable=gamma2_varP2)
gamma_entryP2.place(relx=0.7, rely=0.6)

fraction_label = tk.Label(Bdatcenter_frame, text='Fraction:')
fraction_label.place(relx=0, rely=0.75)
fraction_entry = ttk.Entry(Bdatcenter_frame, width=4, textvariable=fraction_var2)
fraction_entry.place(relx=0.4, rely=0.75)
fraction_entryP2 = ttk.Entry(Bdatcenter_frame, width=4, textvariable=fraction2_varP2)
fraction_entryP2.place(relx=0.7, rely=0.75)

amp_var2.trace_add("write", update_initial_amp2)
mean_var2.trace_add("write", update_initial_mean2)
stddev_var2.trace_add("write", update_initial_stddev2)
gamma_var2.trace_add("write", update_initial_gamma2)
fraction_var2.trace_add("write", update_initial_fraction2)

fitpeak_frame = tk.Frame(Bdatleft_frame)
fitpeak_frame.pack(side=tk.TOP)
peak_x_label = tk.Label(fitpeak_frame, text="Fit peak at:")
peak_x_label.pack(side=tk.LEFT)
peak_x_entry = tk.Entry(fitpeak_frame, width=5)
peak_x_entry.pack(side=tk.LEFT)

fit_model_options2 = ['Gaussian','2 peak Gaussian', 'Lorentzian', '2 peak Lorentzian', 'Voigt', '2 peak Voigt', 'Pseudo-Voigt', '2 peak Pseudo-Voigt']
fit_model_var2 = tk.StringVar(value='Lorentzian')  # Set the default option
fit_model_combobox2 = ttk.Combobox(Bdatleft_frame, textvariable=fit_model_var2, values=fit_model_options2, width=10)
fit_model_combobox2.pack()
fit_model_combobox2.bind("<<ComboboxSelected>>", fit_selected_model2)

fit_button = tk.Button(Bdatleft_frame, text="Fit", command=fit_selected_model2)
fit_button.pack()

bgsm_frame = tk.Frame(Tdatright_frame)
bgsm_frame.pack(side=tk.TOP)
original_scattering_dataSB = None 
subtract_background_button = tk.Button(bgsm_frame, text="Bg", command=subtract_background)
subtract_background_button.pack(side=tk.LEFT)
subtract_background_button.configure(state="disabled", anchor="center") 
background_subtraction_active = False

maxhalfwindow_var = tk.StringVar(value=str(10))
maxhalfwindow_entry = ttk.Entry(bgsm_frame, width=2, textvariable=maxhalfwindow_var)
maxhalfwindow_entry.pack(side=tk.LEFT)
maxhalfwindow_entry.bind("<<Modified>>", fit_selected_model2)

smoothhalfwindow_var= tk.StringVar(value=str(3))
smoothhalfwindow_entry = ttk.Entry(bgsm_frame, width=2, textvariable=smoothhalfwindow_var)
smoothhalfwindow_entry.pack(side=tk.LEFT)
smoothhalfwindow_entry.bind("<<Modified>>", fit_selected_model2)

original_scattering_dataSD = None 
smooth_button = tk.Button(bgsm_frame, text="Smooth", command=smooth_data)
smooth_button.pack(side=tk.LEFT)
smooth_button.configure(state="disabled", anchor="center") 
smooth_active = False

tableDATA = ttk.Treeview(Bdatright_frame, show='headings', height=2)
tableDATA["columns"] = ("column1", "column2", "column3", "column4", "column5", "column6", "column7")
tableDATA.heading("column1", text="Amp.", anchor="w")
tableDATA.heading("column2", text="Mean", anchor="w")
tableDATA.heading("column3", text="Sigma", anchor="w")
tableDATA.heading("column4", text="Gamma", anchor="w")
tableDATA.heading("column5", text="Frac.", anchor="w")
tableDATA.heading("column6", text="FWHM", anchor="w")
tableDATA.heading("column7", text="Area", anchor="w")
column_widths = [len(tableDATA.heading(column)["text"]) for column in tableDATA["columns"]]
for i, column in enumerate(tableDATA["columns"]):
    tableDATA.column(column, width=column_widths[i] * 12)
tableDATA.pack()

buttons_frame = tk.Frame(Bdatright_frame)
buttons_frame.pack(side=tk.TOP)
export_button = tk.Button(buttons_frame, text="Export Data", command=export_data)
export_button.pack(side=tk.LEFT)
use_in_stack_button = tk.Button(buttons_frame, text="Use in Stack", command=use_in_stack)
use_in_stack_button.pack(side=tk.LEFT)








#------------- ELASTIC PROPERTIES

elastic_paned_window = tk.PanedWindow(frames["Elastic Properties"], orient=tk.HORIZONTAL)
elastic_paned_window.pack(fill=tk.BOTH, expand=False)
bottom_frame = tk.Frame(frames["Elastic Properties"], highlightbackground="black", highlightthickness=1)
left_frame = tk.Frame(elastic_paned_window, highlightbackground="black", highlightthickness=1)
center_frame = tk.Frame(elastic_paned_window, highlightbackground="black", highlightthickness=1)
right_frame = tk.Frame(elastic_paned_window, highlightbackground="black", highlightthickness=1)
bottom_frame.pack(fill=tk.BOTH, expand=True)
elastic_paned_window.add(left_frame, width=340, height=230)
elastic_paned_window.add(center_frame, width=180, height=230)
elastic_paned_window.add(right_frame, width=200, height=230)

label_x = [tk.Label(left_frame, text=str(i+1)) for i in range(6)]
label_y = [tk.Label(left_frame, text=str(i+1)) for i in range(6)]

for i, (label_x_i, label_y_i) in enumerate(zip(label_x, label_y)):
    label_x_i.place(relx=0.15 + i * 0.15, rely=0.13) 
    label_y_i.place(relx=0.02, rely=0.23 + i * 0.13)   

cij_elements = {
    "Isotropic": ["C11","C44"],
    "Cubic": ["C11", "C12", "C44"],
    "Triclinic": ["C11", "C12", "C13", "C14", "C15", "C16", "C22", "C23", "C24", "C25", "C26", "C33", "C34", "C35", "C36", "C44", "C45", "C46", "C55", "C56", "C66"],
    "Monoclinic": ["C11", "C12", "C13","C16","C22","C23","C26" "C33", "C36","C44","C45", "C55", "C66"],
    "Orthorhombic": ["C11", "C12", "C13", "C22", "C23", "C33", "C44", "C55", "C66"],
    "Hexagonal": ["C11", "C12", "C14","C33", "C44"],
    "Tetragonal": ["C11", "C12", "C13", "C33", "C44", "C66" ],
    "Rhombohedral": ["C11", "C12", "C13", "C14", "C33", "C44"],
}

initial_values = {
    "Cubic": [309.61,226.27,189.49],  # Random initial values for Cubic structure
    "Triclinic": [219.83, 59.66, -4.82, -0.82, -33.87, -1.04, 216.38, -3.67, 1.79, -16.51, -0.62, 48.89, 4.12, -15.52, -3.59, 26.54, -3.6, -6.41, 22.85, -1.67, 78.29],  # Talc
    "Tetragonal": [455, 199, 192, 762, 258, 321],  # Stishovite
}

cij_values = [[0.0 for _ in range(6)] for _ in range(6)]
entries = [[ttk.Entry(left_frame, width=4)
            for _ in range(6)] for _ in range(6)]

for i in range(6):  # Vertical lines
    tk.Frame(left_frame, width=1, height=230, bg="black").place(relx=0.105 + i * 0.15, rely=0.13)
for j in range(6):  # Horizontal lines
    tk.Frame(left_frame, width=340, height=1, bg="black").place(relx=0.02, rely=0.23 + j * 0.13)

# Global data storage
model_data = {
    'Data Vp': [],
    'Data Vs1': [],
    'Data Vs2': [],
    'Fit Vp': [],
    'Fit Vs1': [],
    'Fit Vs2': [],
    'True Vp': [],
    'True Vs1': [],
    'True Vs2': []
}

def update_cij_value(i, j, event=None):
    try:
        new_value = float(entries[i][j].get())
        cij_values[i][j] = new_value
        entry_modified[i][j] = True
    except ValueError:
        pass

        
def make_update_function(i, j):
    """Create a lambda function for updating cij_values at the given indices."""
    return lambda event: update_cij_value(i, j, event)


for i in range(6):
    for j in range(6):
        update_func = make_update_function(i, j)
        entries[i][j].bind("<Return>", update_func)
        entries[i][j].bind("<FocusOut>", update_func)
        
def on_structure_selected(event):
    selected_value = selected_structure.get()
    highlight_cells(selected_value)

entry_modified = [[False for _ in range(6)] for _ in range(6)]

def highlight_cells(selected_value):
    structure_elements = cij_elements[selected_value]

    # Insert or reset values based on whether they are part of the structure elements
    initial_vals = initial_values.get(selected_value, [0.0] * 36)
    k = 0
    for i in range(6):
        for j in range(6):
            cij_index = f"C{i+1}{j+1}"
            if cij_index in structure_elements:
                if not entry_modified[i][j]:
                    value = initial_vals[k] if k < len(initial_vals) else 0.0
                    cij_values[i][j] = value
                    entries[i][j].delete(0, tk.END)
                    entries[i][j].insert(0, str(value))
                entries[i][j].place(relx=0.105 + j * 0.15, rely=0.23 + i * 0.13)
                k += 1
            else:
                # Remove the entry for non-highlighted cells
                entries[i][j].place_forget()
                cij_values[i][j] = 0.0
                entry_modified[i][j] = False



def compute_anisotropic_vel(mineral,angles = None,plane_normal = None,direction = None,offset = 0,angle_range = None,angle_count = None,):
    if angles is None:
        angles = np.linspace(offset, offset+angle_range, angle_count)

    direction_prime = []
    angle = []

    for a in angles:
        q = Quaternion(axis = plane_normal, degrees = a)
        d_prime = q.rotate(direction)
        direction_prime.append(d_prime)
        angle.append(q.degrees)

    vps = []
    vs1s = []
    vs2s = []
    betas = []
    Es = []
    
    for i, d in enumerate(direction_prime):
        velocities = mineral.wave_velocities(d)
        vps.append(velocities[0][0])
        vs1s.append(velocities[0][1])
        vs2s.append(velocities[0][2])

    return (vps, vs1s, vs2s)

def on_fitmodel_clicked():
    selected_value = selected_structure.get()
    mineral_type, adjusted_matrix = model_structure(selected_value)

    if mineral_type is not None and adjusted_matrix is not None:
        model_velocity(mineral_type, adjusted_matrix)
    else:
        # Handle the error case
        print("Model structure returned an error.")


def model_structure(selected_value):
    try:
        on_structure_selected(selected_value)
        density_value = density_var.get()
        adjusted_matrix = [cij_values[i][j] * 1e9 for i in range(6) for j in range(6) if cij_values[i][j] != 0.0]
        mineral_type = None
        
        if selected_value == "Cubic":
            if len(adjusted_matrix) != 3:
                raise ValueError("Incorrect number of elements in Cij matrix for Cubic material")
            mineral_type = anisotropy.CubicMaterial
        elif selected_value == "Triclinic":
            if len(adjusted_matrix) != 21:
                raise ValueError("Incorrect number of elements in Cij matrix for Triclinic material")
            mineral_type = anisotropy.TriclinicMaterial
        elif selected_value == "Monoclinic":
            if len(adjusted_matrix) != 13:
                raise ValueError("Incorrect number of elements in Cij matrix for Monoclinic material")
            mineral_type = anisotropy.MonoclinicMaterial
        elif selected_value == "Orthorhombic":
            if len(adjusted_matrix) != 9:
                raise ValueError("Incorrect number of elements in Cij matrix for Orthorhombic material")
            mineral_type = anisotropy.OrthorhombicMaterial
        elif selected_value == "Rhombohedral":
            if len(adjusted_matrix) != 6:
                raise ValueError("Incorrect number of elements in Cij matrix for Rhombohedral material")
            mineral_type = anisotropy.RhombohedralMaterial
        elif selected_value == "Hexagonal":
            if len(adjusted_matrix) != 5:
                raise ValueError("Incorrect number of elements in Cij matrix for Hexagonal material")
            mineral_type = anisotropy.HexagonalMaterial
        elif selected_value == "Tetragonal":
            if len(adjusted_matrix) != 6:
                raise ValueError("Incorrect number of elements in Cij matrix for Tetragonal material")
            mineral_type = anisotropy.TetragonalMaterial

        #if mineral_type:
            #model_velocity(mineral_type, adjusted_matrix)
        else:
            raise ValueError(f"Issue with Cij matrix")
        return mineral_type, adjusted_matrix

    except ValueError as e:
        tkinter.messagebox.showerror("Error", str(e))
        return None, None
    
def calc_elastic_properties(mineral_type, Cijs, density_value):
    mineral = mineral_type(density_value, Cijs)
    bulk_modulus = mineral.isentropic_bulk_modulus_vrh / 1e9
    bulk_error = np.std([mineral.isentropic_bulk_modulus_reuss, mineral.isentropic_bulk_modulus_vrh, mineral.isentropic_bulk_modulus_voigt]) / 1e9
    shear_modulus = mineral.isentropic_shear_modulus_reuss / 1e9
    shear_error = np.std([mineral.isentropic_shear_modulus_reuss, mineral.isentropic_shear_modulus_vrh, mineral.isentropic_shear_modulus_voigt]) / 1e9
    elastic_anisotropy = mineral.isentropic_universal_elastic_anisotropy
    #poisson_ratio = mineral.isentropic_isotropic_poisson_ratio #this calculation seems like it is missing factor of 2
    poisson_ratio = (3 * bulk_modulus - 2 * shear_modulus) / (2 * (3 * bulk_modulus + shear_modulus))

    # Update the entry fields with the calculated values
    bulk_modulus_var.set(f"{bulk_modulus:.4f}")
    bulk_error_var.set(f"{bulk_error:.4f}")
    shear_modulus_var.set(f"{shear_modulus:.4f}")
    shear_error_var.set(f"{shear_error:.4f}")
    elastic_anisotropy_var.set(f"{elastic_anisotropy:.4f}")
    poisson_ratio_var.set(f"{poisson_ratio:.4f}")


def generate_sparse_test_data(mineral_type,Cijs,offset,start_angle,end_angle,density_value, hkl = None,xyz_norm = None,n_data = 15,noise_mean = 1,noise_sigma = 0.1,error_est = False):
    mineral = mineral_type(density_value, Cijs)

    if xyz_norm is None and hkl is None:
        print("Please specify start direction in either cartesian coordinates (xyz_norm=) or Miller indices (hkl_norm)")
        return -1
    if hkl is not None:
        plane_normal = hkl
    else:
        plane_normal = xyz_norm
    angle_range = -(start_angle - end_angle)
    angles = np.linspace(start_angle, end_angle, n_data)
    
    if plane_normal[0] == 0.0 and plane_normal[1] == 0.0:
        direction = [1,0,0]
    else:
        direction = np.array([-plane_normal[1], plane_normal[0], 0])
        direction = direction / np.linalg.norm(direction) 
    (vps, vs1s, vs2s) = compute_anisotropic_vel(mineral = mineral,angle_range = angle_range,angle_count = n_data, plane_normal = plane_normal,direction = direction,offset = offset)

    vps = np.array(vps)
    vs1s = np.array(vs1s)
    vs2s = np.array(vs2s)

    vps += np.random.normal(noise_mean, noise_sigma, len(vps))
    vs1s += np.random.normal(noise_mean, noise_sigma,len(vps))
    vs2s += np.random.normal(noise_mean, noise_sigma,len(vps))

    if error_est:
        vps_err *= 0.1
        vs1s_err *= 0.1
        vs2s_err *= 0.1

        return angles, vps, vs1s, vs2s, vps_err, vs1s_err, vs2s_err
    return angles, vps, vs1s, vs2s

def fit_Cij_lsq(x_vps, vps, x_vs1s, vs1s, x_vs2s, vs2s, theta_0, mineral_type, density_value, result_text, selected_value, xyz_norm=None, hkl_norm=None):
    if xyz_norm is None and hkl_norm is None:
        return -1

    if hkl_norm is not None:
        if isinstance(hkl_norm, list):
            hkl_norm = np.vstack(hkl_norm)
        if len(hkl_norm.shape) > 1:
            plane_normals = hkl_norm
        else:
            plane_normals = hkl_norm
    else:
            plane_normals = xyz_norm
            plane_normals = np.array(plane_normals) 

    num_directions = plane_normals.shape[0]
    plane_normals = np.vstack([pn/np.linalg.norm(pn) for pn in plane_normals])
    
    if isinstance(x_vs1s, list):
        x_vs1s = np.vstack(x_vs1s)
    if isinstance(x_vps, list):
        x_vps = np.vstack(x_vps)
    if isinstance(x_vs2s, list):
        x_vs2s = np.vstack(x_vs2s)
    if isinstance(vps, list):
        vps = np.vstack(vps)
    if isinstance(vs1s, list):
        vs1s = np.vstack(vs1s)
    if isinstance(vs2s, list):
        vs2s = np.vstack(vs2s)

    if len(x_vps.shape) == 1:
        x_vps = np.expand_dims(x_vps, axis = 0)
        x_vs1s = np.expand_dims(x_vs1s, axis = 0)
        x_vs2s = np.expand_dims(x_vs2s, axis = 0)
        vps = np.expand_dims(vps, axis = 0)
        vs1s = np.expand_dims(vs1s, axis = 0)
        vs2s = np.expand_dims(vs2s, axis = 0)

    xs = list(zip(x_vps, x_vs1s, x_vs2s))
    ys = list(zip(vps, vs1s, vs2s))

    ys_flat = [np.hstack(i) for i in ys]

    start_directions = []
    for plane_normal in plane_normals:
        if plane_normal[0] == 0.0 and plane_normal[1] == 0.0:
            direction = [1, 0, 0]
        else:
            direction = np.array([-plane_normal[1], plane_normal[0], 0])
            direction = direction / np.linalg.norm(direction)
        start_directions.append(direction)

    def obj_fun(params,xs = xs, ys = ys_flat, mineral_type = mineral_type, direction = start_directions, plane_normal = plane_normals):
        offset = params[-1]
        Cijs = params[:-1]
        
        mineral = mineral_type(density_value, Cijs)
        ymodel = []

        (vps, _, _) = compute_anisotropic_vel(mineral = mineral,
                                                      angles = xs[0] + offset,
                                                      plane_normal = plane_normal,
                                                      direction = direction,
                                                      offset = offset)
        (_, vs1s, _) = compute_anisotropic_vel(mineral = mineral,
                                                      angles = xs[1] + offset,
                                                      plane_normal = plane_normal,
                                                      direction = direction,
                                                      offset = offset)
        (_, _, vs2s) = compute_anisotropic_vel(mineral = mineral,
                                                      angles = xs[2] + offset,
                                                      plane_normal = plane_normal,
                                                      direction = direction,
                                                      offset = offset)
        ymodel = np.array(vps + vs1s + vs2s)
        return ymodel - ys

    def obj_fun_complete(params, xs = xs, ys = ys_flat, mineral_type = mineral_type, direction = start_directions, plane_normals = plane_normals):
        Cij_0 = params[:-num_directions]
        offsets = params[-num_directions:]
        resids = []
        #print("Cij_0", Cij_0)

        for (plane_normal, start_dir, x, y, offset) in zip(plane_normals, start_directions, xs, ys_flat, offsets):
                temp_params = np.append(Cij_0, offset)
                resids.append(obj_fun(params = temp_params,
                                     xs = x,
                                     ys = y,
                                     mineral_type=mineral_type,
                                     direction = start_dir,
                                     plane_normal = plane_normal))
                out = np.hstack(resids)
        return out.flatten() 

    low_bound = np.ones_like(theta_0[:-num_directions]) * - np.inf
    high_bound = -low_bound

    low_bound = low_bound.tolist() + [-180.] * num_directions
    high_bound = high_bound.tolist() + [180.] * num_directions
    

    #print("theta_0", theta_0)
    result = scipy.optimize.least_squares(fun = obj_fun_complete,x0 = theta_0,bounds = (low_bound, high_bound),verbose = 1,max_nfev=1e15,ftol = 1e-6, xtol = 1e-6)
    #print("results computed")
    J = result.jac
    
    try:
        # Attempt to calculate the covariance matrix
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        # Handle the singularity case here
        #print("Warning: Covariance matrix calculation failed due to singularity.")
        cov = None
    
    if cov is not None:
        std_deviation = np.sqrt(np.diagonal(cov))
    else:
        std_deviation = None 

    #print("Fit parameters: ", result.x)
    #print("Standard deviation: ", std_deviation)
    result_values = result.x / 1e9

    zero_positions = [i for i, val in enumerate(initial_values) if val == 0]
    for position in zero_positions:
        result_values = np.insert(result_values, position, 0)

    result_values = result_values[:-1]

    # Define a function to create a 6x6 matrix using cij_elements and result_values
    def create_cij_matrix(structure_elements, values):
        cij_matrix = np.zeros((6, 6))

        for i in range(6):
            for j in range(6):
                cij_index = f"C{i + 1}{j + 1}"
                if cij_index in structure_elements:
                    cij_matrix[i][j] = values[structure_elements.index(cij_index)]

        return cij_matrix
    structure_elements = cij_elements[selected_value]
    updated_values_padded = create_cij_matrix(structure_elements, result_values)
    np.set_printoptions(precision=2, suppress=True)
    result_text.delete("1.0", tk.END)
    result_text.insert("1.0", np.array2string(updated_values_padded))
    result_text.config(font=("Helvetica", 10))
    result_text.config(highlightthickness=0)
    rows, cols = updated_values_padded.shape
    result_text.config(height=rows, width=cols * 7)

    return (result.x, std_deviation)

def model_velocity(mineral_type, cij_values):
    Cijs = [value * 1 for value in cij_values]
    #print("Cijs",Cijs)
    selected_value = selected_structure.get()
    start_angle = float(start_entry.get())
    end_angle = float(end_entry.get())
    density_value = float(density_entry.get())
    
    all_angles_vps = []
    all_vps = []
    all_angles_vs1s = []
    all_vs1s = []
    all_angles_vs2s = []
    all_vs2s = []
    all_hkl_norm = []
    fit_results = []
    errors = []

    valid_item_count = 0
    offset_values = []  # Initialize a list to store offset values

    # Create empty lists to collect data for each hkl combination
    all_angles_fit = []
    all_vps_fit = []
    all_vs1s_fit = []
    all_vs2s_fit = []

    all_angles_true = []
    all_vps_true = []
    all_vs1s_true = []
    all_vs2s_true = []

    all_hkl_labels = []  # Store hkl labels for legend

    for item in hkl_tree.get_children():
        values = hkl_tree.item(item, "values")
        values = [float(value) if value.strip() != "" else None for value in values]

        if not any(val is None for val in values):
            valid_item_count += 1
            u, v, z = values

            # Generate a random offset between 5 and 50
            offset = random.uniform(5, 100)
            offset_values.append(offset)  # Store the offset value
            
            angles_vps, vps, _, _ = generate_sparse_test_data(mineral_type=mineral_type, Cijs=Cijs, offset=offset, start_angle=start_angle, end_angle=end_angle, hkl=[u, v, z], density_value=density_value, n_data=100, noise_mean=0, noise_sigma=0)
            angles_vs1s, _, vs1s, _ = generate_sparse_test_data(mineral_type=mineral_type, Cijs=Cijs, offset=offset, start_angle=start_angle, end_angle=end_angle, hkl=[u, v, z], density_value=density_value, n_data=100, noise_mean=0, noise_sigma=0)
            angles_vs2s, _, _, vs2s = generate_sparse_test_data(mineral_type=mineral_type, Cijs=Cijs, offset=offset, start_angle=start_angle, end_angle=end_angle, hkl=[u, v, z], density_value=density_value, n_data=100, noise_mean=0, noise_sigma=0)

            all_angles_vps.append(angles_vps)
            all_vps.append(vps)
            all_angles_vs1s.append(angles_vs1s)
            all_vs1s.append(vs1s)
            all_angles_vs2s.append(angles_vs2s)
            all_vs2s.append(vs2s)
            all_hkl_norm.append([u, v, z])

            all_hkl_labels.append(f"hkl={values}")  # Store hkl labels for legend

            # Generate random offsets for Cij_0
            Cij_0 = np.array(Cijs) #* np.random.uniform(0.9, 1.1, len(Cijs))
            theta_0 = np.append(Cij_0, offset_values)
            #print("Cij_0", Cij_0)
            #print("offset_values", offset_values)
            #print("theta_0", theta_0)
            #print("all_hkl_norm", all_hkl_norm)
            
            fit_result, err = fit_Cij_lsq(x_vps=all_angles_vps, vps=all_vps, x_vs1s=all_angles_vs1s, vs1s=all_vs1s, x_vs2s=all_angles_vs2s, vs2s=all_vs2s, theta_0=theta_0, mineral_type=mineral_type, density_value=density_value, hkl_norm=all_hkl_norm, result_text=result_text, selected_value=selected_value)
            fit_results.append(fit_result)
            errors.append(err)
            
            num_params = len(fit_results[0])

            # Generate data using fit results for the current hkl
            angles_fit, vps_fit, vs1s_fit, vs2s_fit = generate_sparse_test_data(mineral_type=mineral_type, Cijs=fit_result[:num_params - 1], offset=fit_result[num_params - 1], start_angle=start_angle, end_angle=end_angle, hkl=[u, v, z],density_value=density_value,n_data=1000, noise_mean=0, noise_sigma=0)

            # Append the data for the current hkl combination to the respective lists
            all_angles_fit.append(angles_fit)
            all_vps_fit.append(vps_fit)
            all_vs1s_fit.append(vs1s_fit)
            all_vs2s_fit.append(vs2s_fit)
            
            # Generate synthetic data for the true values (using original Cijs and fixed offset)
            angles_true, vps_true, vs1s_true, vs2s_true = generate_sparse_test_data(mineral_type=mineral_type, Cijs=Cijs, offset=offset, start_angle=start_angle, end_angle=end_angle, hkl=[u, v, z], density_value=density_value, n_data=1000, noise_mean=0, noise_sigma=0)

            # Append the data for the true values for the current hkl combination to the respective lists
            all_angles_true.append(angles_true)
            all_vps_true.append(vps_true)
            all_vs1s_true.append(vs1s_true)
            all_vs2s_true.append(vs2s_true)
    
    fig3.clf()
    ax_vp, ax_vs1, ax_vs2 = fig3.subplots(3, 1, sharex=True)

    lines, labels = [], []
    for i, (fit_result, hkl) in enumerate(zip(fit_results, all_hkl_labels)):
        Cij_fit = fit_result[:num_params - 1]
        offset_fit = fit_result[num_params - 1]
        hkl_value = hkl.split('=')[1]
        
        """ 
        # Plot the data for the current hkl combination
        ax_vp.plot(all_angles_vps[i], all_vps[i], '.', markersize=3)
        lines += ax_vp.lines[-1:]
        labels += [f"Data {hkl_value}"]
        ax_vs1.plot(all_angles_vs1s[i], all_vs1s[i], '.', markersize=3)
        ax_vs2.plot(all_angles_vs2s[i], all_vs2s[i], '.', markersize=3)
        """

        # Plot the fit results for the current hkl combination
        ax_vp.plot(all_angles_fit[i], all_vps_fit[i], '-', linewidth=1)
        lines += ax_vp.lines[-1:]
        labels += [f"Fit {hkl_value}"]
        ax_vs1.plot(all_angles_fit[i], all_vs1s_fit[i], '-', linewidth=1)
        ax_vs2.plot(all_angles_fit[i], all_vs2s_fit[i], '-', linewidth=1)

        """
        # Plot the true values for the current hkl combination
        ax_vp.plot(all_angles_true[i], all_vps_true[i], '-', linewidth=1)
        lines += ax_vp.lines[-1:]
        labels += [f"True {hkl_value}"]
        ax_vs1.plot(all_angles_true[i], all_vs1s_true[i], '-', linewidth=1)
        ax_vs2.plot(all_angles_true[i], all_vs2s_true[i], '-', linewidth=1)
        """
        
        # Appending data for each hkl combination
        #model_data['Data Vp'].append({'hkl': hkl_value, 'x': all_angles_vps[i], 'y': all_vps[i]})
        #model_data['Data Vs1'].append({'hkl': hkl_value, 'x': all_angles_vs1s[i], 'y': all_vs1s[i]})
        #model_data['Data Vs2'].append({'hkl': hkl_value, 'x': all_angles_vs2s[i], 'y': all_vs2s[i]})
        model_data['Fit Vp'].append({'hkl': hkl_value, 'x': all_angles_fit[i], 'y': all_vps_fit[i]})
        model_data['Fit Vs1'].append({'hkl': hkl_value, 'x': all_angles_fit[i], 'y': all_vs1s_fit[i]})
        model_data['Fit Vs2'].append({'hkl': hkl_value, 'x': all_angles_fit[i], 'y': all_vs2s_fit[i]})
        #model_data['True Vp'].append({'hkl': hkl_value, 'x': all_angles_true[i], 'y': all_vps_true[i]})
        #model_data['True Vs1'].append({'hkl': hkl_value, 'x': all_angles_true[i], 'y': all_vs1s_true[i]})
        #model_data['True Vs2'].append({'hkl': hkl_value, 'x': all_angles_true[i], 'y': all_vs2s_true[i]})


    ax_vp.set_ylabel('Vp [m/s]', fontsize=6)
    ax_vs1.set_ylabel('Vs1 [m/s]', fontsize=6)
    ax_vs2.set_ylabel('Vs2 [m/s]', fontsize=6)
    ax_vs2.set_xlabel('Chi [°]', fontsize=6)

    plt.setp(ax_vp.get_xticklabels(), fontsize=5)  # Set x-axis tick label font size
    plt.setp(ax_vp.get_yticklabels(), fontsize=5)  # Set y-axis tick label font size
    plt.setp(ax_vs1.get_xticklabels(), fontsize=5)  # Set x-axis tick label font size
    plt.setp(ax_vs1.get_yticklabels(), fontsize=5)  # Set y-axis tick label font size
    plt.setp(ax_vs2.get_xticklabels(), fontsize=5)  # Set x-axis tick label font size
    plt.setp(ax_vs2.get_yticklabels(), fontsize=5)  # Set y-axis tick label font size


    canvas3.draw()
    legend_font = FontProperties(size=5)  # FontProperties object to set font size
    fig3.legend(lines, labels, loc='upper center', ncol=4, prop=legend_font)
    calc_elastic_properties(mineral_type=mineral_type, Cijs=fit_result[:num_params - 1], density_value=density_value)
    

def on_single_click(event):
    item = hkl_tree.focus()
    column = hkl_tree.identify_column(event.x)
    column_no = int(column.split("#")[-1]) - 1  
    x, y, width, height = hkl_tree.bbox(item, column)
    x += hkl_tree.winfo_x()
    y += hkl_tree.winfo_y()
    large_font = ('Verdana', 12) 
    entry = tk.Entry(center_frame, width=width, font=large_font)
    entry.place(x=x, y=y, width=width, height=height)
    entry.insert(0, hkl_tree.item(item, 'values')[column_no])

    def confirm_edit(event=None):
        hkl_tree.item(item, values=tuple(
            entry.get() if col == column_no else val 
            for col, val in enumerate(hkl_tree.item(item, 'values'))
        ))
        entry.destroy()

    entry.bind("<Return>", confirm_edit)  
    entry.bind("<FocusOut>", confirm_edit) 
    entry.focus()
    
def load_hkl():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, "r") as file:
            for item in hkl_tree.get_children():
                hkl_tree.delete(item)

            for line in file:
                hkl_values = line.strip().split(',')
                if len(hkl_values) == 3:
                    try:
                        hkl_values = [int(val) for val in hkl_values]
                        hkl_tree.insert("", "end", values=hkl_values)
                    except ValueError:
                        print(f"Invalid line in file: {line}")

def on_inverse_clicked():
    selected_value = selected_structure.get()
    mineral_type, adjusted_matrix = model_structure(selected_value)

    if mineral_type is not None and adjusted_matrix is not None:
        reverse_modeling(mineral_type, adjusted_matrix)
    else:
        # Handle the error case
        print("Model structure returned an error.")
    
def reverse_modeling(mineral_type, cij_values):
    Cijs = [value * 1 for value in cij_values]
    #print("Cijs",Cijs)
    selected_value = selected_structure.get()
    start_angle = float(start_entry.get())
    end_angle = float(end_entry.get())
    density_value = float(density_entry.get())
    
    # Open a file dialog to select a file
    file_path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
    if not file_path:
        print("No file selected.")
        return

    try:
        # Reading the data from the selected file
        if file_path.endswith('.csv') or file_path.endswith('.txt'):
            data = pd.read_csv(file_path, comment='#')
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        else:
            print("Unsupported file format. Please use CSV, TXT, or Excel.")
            return
    except Exception as e:
        print("Error reading file:", e)
        return

    expected_columns = ['Chi', 'Vp', 'Vs1', 'Vs2', 'uvz']
    if not all(column in data.columns for column in expected_columns):
        print("File does not contain the required columns.")
        return

    # Extracting unique 'uvz' combinations from the data
    data['uvz_tuples'] = data['uvz'].apply(lambda uvz: tuple(map(float, uvz.split(','))))
    unique_uvz_tuples = data['uvz_tuples'].unique()

    # Initialize the lists to collect the processed data
    all_angles_vps = []
    all_vps = []
    all_angles_vs1s = []
    all_vs1s = []
    all_angles_vs2s = []
    all_vs2s = []
    all_hkl_norm = []
    offset_values = []
    fit_results = []
    errors = []

    all_angles_fit = []
    all_vps_fit = []
    all_vs1s_fit = []
    all_vs2s_fit = []

    all_angles_true = []
    all_vps_true = []
    all_vs1s_true = []
    all_vs2s_true = []

    all_hkl_labels = []
    
    for uvz_tuple in unique_uvz_tuples:
        offset = random.uniform(5, 100)
        #offset = 86
        offset_values.append(offset)

        subset = data[data['uvz_tuples'] == uvz_tuple]

        # Filter out rows with zero velocities for each velocity type separately
        non_zero_vp_subset = subset[subset['Vp'] > 0]
        non_zero_vs1_subset = subset[subset['Vs1'] > 0]
        non_zero_vs2_subset = subset[subset['Vs2'] > 0]

        # Check non-empty subsets and append their data to the lists
        if not non_zero_vp_subset.empty:
            angles_vps = non_zero_vp_subset['Chi'].to_numpy()
            vps = non_zero_vp_subset['Vp'].to_numpy() * 1e3

        if not non_zero_vs1_subset.empty:
            angles_vs1s = non_zero_vs1_subset['Chi'].to_numpy()
            vs1s = non_zero_vs1_subset['Vs1'].to_numpy() * 1e3

        if not non_zero_vs2_subset.empty:
            angles_vs2s = non_zero_vs2_subset['Chi'].to_numpy()
            vs2s = non_zero_vs2_subset['Vs2'].to_numpy() * 1e3
        
        all_angles_vps.append(angles_vps)
        all_vps.append(vps)
        all_angles_vs1s.append(angles_vs2s)
        all_vs1s.append(vs2s)
        all_angles_vs2s.append(angles_vs1s)
        all_vs2s.append(vs1s)
            
        all_hkl_norm.append(uvz_tuple)
        all_hkl_labels.append(f"hkl={uvz_tuple}")
    
        Cij_0 = np.array(Cijs) #* np.random.uniform(0.1, 1.9, len(Cijs))
        theta_0 = np.append(Cij_0, offset_values)
        #print("Cij_0", Cij_0)
        #print("offset_values", offset_values)
        #print("theta_0", theta_0)
        
        #print("all_angles_vs1s", all_angles_vs1s)
        #print("all_angles_vs2s", all_angles_vs2s)
        #print("all_vs1s",all_vs1s)
        #print("all_vs2s",all_vs2s)
        #normalized_uvz = [list(tup) for tup in unique_uvz_tuples]
        #print("unique_uvz_tuples", unique_uvz_tuples)
        #print("all_hkl_norm",all_hkl_norm)
        
        fit_result, err = fit_Cij_lsq(x_vps=all_angles_vps, vps=all_vps, x_vs1s=all_angles_vs1s, vs1s=all_vs1s, x_vs2s=all_angles_vs2s, vs2s=all_vs2s, theta_0=theta_0, mineral_type=mineral_type, density_value=density_value, xyz_norm=all_hkl_norm, result_text=result_text, selected_value=selected_value)
        fit_results.append(fit_result)
        errors.append(err)

        num_params = len(fit_results[0])
        offset_fit = fit_result[num_params - 1]
        #print("offset", offset)
        Cijs_fit = fit_result[:num_params - 1]
        
        angles_fit, vps_fit, vs1s_fit, vs2s_fit = generate_sparse_test_data(mineral_type = mineral_type, Cijs=Cijs_fit, offset=offset_fit, start_angle = start_angle,end_angle = end_angle, xyz_norm = uvz_tuple,density_value=density_value,n_data = 1000,noise_mean = 0,noise_sigma = 0)

        all_angles_fit.append(angles_fit)
        all_vps_fit.append(vps_fit)
        all_vs1s_fit.append(vs1s_fit)
        all_vs2s_fit.append(vs2s_fit)

        angles_true, vps_true, vs1s_true, vs2s_true = generate_sparse_test_data(mineral_type=mineral_type, Cijs = Cijs, offset = offset, start_angle = start_angle,end_angle=end_angle, xyz_norm = uvz_tuple,density_value=density_value,n_data = 1000,noise_mean = 0,noise_sigma = 0)

        all_angles_true.append(angles_true)
        all_vps_true.append(vps_true)
        all_vs1s_true.append(vs1s_true)
        all_vs2s_true.append(vs2s_true)

    fig3.clf()
    ax_vp, ax_vs1, ax_vs2 = fig3.subplots(3, 1, sharex=True)

    lines, labels = [], []
    for i, (fit_result, hkl) in enumerate(zip(fit_results, all_hkl_labels)):
        Cij_fit = fit_result[:num_params - 1]
        offset_fit = fit_result[num_params - 1]
        hkl_value = hkl.split('=')[1]
        
        # Plot the data for the current hkl combination
        ax_vp.plot(all_angles_vps[i], all_vps[i], '.', markersize=3)
        lines += ax_vp.lines[-1:]
        labels += [f"Data {hkl_value}"]
        ax_vs1.plot(all_angles_vs1s[i], all_vs1s[i], '.', markersize=3) #switch these two for some reason ?
        ax_vs2.plot(all_angles_vs2s[i], all_vs2s[i], '.', markersize=3)

        # Plot the fit results for the current hkl combination
        ax_vp.plot(all_angles_fit[i], all_vps_fit[i], '-', linewidth=1)
        lines += ax_vp.lines[-1:]
        labels += [f"Fit {hkl_value}"]
        ax_vs1.plot(all_angles_fit[i], all_vs1s_fit[i], '-', linewidth=1)
        ax_vs2.plot(all_angles_fit[i], all_vs2s_fit[i], '-', linewidth=1)

        """
        # Plot the true values for the current hkl combination
        ax_vp.plot(all_angles_true[i], all_vps_true[i], '-', linewidth=1)
        lines += ax_vp.lines[-1:]
        labels += [f"True {hkl_value}"]
        ax_vs1.plot(all_angles_true[i], all_vs1s_true[i], '-', linewidth=1)
        ax_vs2.plot(all_angles_true[i], all_vs2s_true[i], '-', linewidth=1)
        """

        # Appending data for each hkl combination
        model_data['Data Vp'].append({'hkl': hkl_value, 'x': all_angles_vps[i], 'y': all_vps[i]})
        model_data['Data Vs1'].append({'hkl': hkl_value, 'x': all_angles_vs1s[i], 'y': all_vs1s[i]})
        model_data['Data Vs2'].append({'hkl': hkl_value, 'x': all_angles_vs2s[i], 'y': all_vs2s[i]})
        model_data['Fit Vp'].append({'hkl': hkl_value, 'x': all_angles_fit[i], 'y': all_vps_fit[i]})
        model_data['Fit Vs1'].append({'hkl': hkl_value, 'x': all_angles_fit[i], 'y': all_vs1s_fit[i]})
        model_data['Fit Vs2'].append({'hkl': hkl_value, 'x': all_angles_fit[i], 'y': all_vs2s_fit[i]})
        #model_data['True Vp'].append({'hkl': hkl_value, 'x': all_angles_true[i], 'y': all_vps_true[i]})
        #model_data['True Vs1'].append({'hkl': hkl_value, 'x': all_angles_true[i], 'y': all_vs1s_true[i]})
        #model_data['True Vs2'].append({'hkl': hkl_value, 'x': all_angles_true[i], 'y': all_vs2s_true[i]})

    ax_vp.set_ylabel('Vp [m/s]', fontsize=6)
    ax_vs1.set_ylabel('Vs1 [m/s]', fontsize=6)
    ax_vs2.set_ylabel('Vs2 [m/s]', fontsize=6)
    ax_vs2.set_xlabel('Chi [°]', fontsize=6)

    plt.setp(ax_vp.get_xticklabels(), fontsize=5)  # Set x-axis tick label font size
    plt.setp(ax_vp.get_yticklabels(), fontsize=5)  # Set y-axis tick label font size
    plt.setp(ax_vs1.get_xticklabels(), fontsize=5)  # Set x-axis tick label font size
    plt.setp(ax_vs1.get_yticklabels(), fontsize=5)  # Set y-axis tick label font size
    plt.setp(ax_vs2.get_xticklabels(), fontsize=5)  # Set x-axis tick label font size
    plt.setp(ax_vs2.get_yticklabels(), fontsize=5)  # Set y-axis tick label font size

    canvas3.draw()
    legend_font = FontProperties(size=5)  # FontProperties object to set font size
    fig3.legend(lines, labels, loc='upper center', ncol=4, prop=legend_font)
    calc_elastic_properties(mineral_type=mineral_type, Cijs=fit_result[:num_params - 1], density_value=density_value)

    

def save_model():
    save_dir = filedialog.askdirectory(title="Select Directory to Save Model")
    if not save_dir:
        print("No directory selected, saving cancelled.")
        return

    # Get the selected structure value
    selected_structure_value = selected_structure.get()

    # Save the Cij matrix with the selected structure
    cij_matrix_text = result_text.get("1.0", tk.END)
    cij_matrix_filename = os.path.join(save_dir, "cij_matrix.txt")
    with open(cij_matrix_filename, "w") as file:
        file.write(f"Selected Structure: {selected_structure_value}\n\n")
        file.write(cij_matrix_text)
    
    # Save the xy data
    xy_data_filename = os.path.join(save_dir, "all_xy_data.txt")
    with open(xy_data_filename, "w") as file:
        for key, datasets in model_data.items():
            for dataset in datasets:
                file.write(f"# {key} - hkl: {dataset['hkl']}\n")
                for x, y in zip(dataset['x'], dataset['y']):
                    file.write(f"{x}\t{y}\n")
                file.write("\n")

def load_cij():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not file_path:
        print("No file selected.")
        return

    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract the selected structure
    selected_structure_line = lines[0].strip()
    structure_name = selected_structure_line.split(":")[1].strip()
    selected_structure.set(structure_name)

    # Parse the Cij matrix
    cij_matrix = []
    for line in lines[2:]:  # Skipping the first two lines
        if line.strip():
            row = [float(value) for value in line.strip()[1:-1].split() if value != '0.']
            cij_matrix.append(row)

    # Populate the entries with the Cij matrix values
    for i in range(6):
        for j in range(6):
            entries[i][j].delete(0, tk.END)
            if i < len(cij_matrix) and j < len(cij_matrix[i]):
                entries[i][j].insert(0, str(cij_matrix[i][j]))
            else:
                entries[i][j].insert(0, "0")
    
    # Call the function to update the UI based on the selected structure
    on_structure_selected(None)


fig3, (ax_vp, ax_vs1, ax_vs2) = plt.subplots(3, 1, sharex=True, figsize=(4,2))
for ax in (ax_vp, ax_vs1, ax_vs2):
    ax.tick_params(axis='both', which='major', labelsize=5)
canvas3 = FigureCanvasTkAgg(fig3, bottom_frame)
canvas3_widget = canvas3.get_tk_widget()
canvas3_widget.pack(fill=tk.BOTH, expand=True)
ax_vp.set_ylabel('Vp [m/s]', fontsize=6)
ax_vs1.set_ylabel('Vs1 [m/s]', fontsize=6)
ax_vs2.set_xlabel('Chi [°]', fontsize=6)
ax_vs2.set_ylabel('Vs2 [m/s]', fontsize=6)
toolbar3 = NavigationToolbar2Tk(canvas3, bottom_frame)
toolbar3.update()
toolbar3.pack()
plt.tight_layout()

matrix_label= tk.Label(left_frame, text="Cij Matrix [GPa]")
matrix_label.place(relx=0, rely=0.015)

load_button = tk.Button(left_frame, text="Load Matrix", command=load_cij)
load_button.place(relx=0.3, rely=0)

selected_structure = tk.StringVar()
structures = ["Cubic", "Hexagonal", "Tetragonal", "Rhombohedral", "Orthorhombic", "Monoclinic", "Triclinic"]
structure_dropdown = ttk.Combobox(left_frame, textvariable=selected_structure, values=structures, width=10)
structure_dropdown.set("Cubic")
structure_dropdown.place(relx=0.63, rely=0.015)
structure_dropdown.bind("<<ComboboxSelected>>", on_structure_selected)

density_frame = tk.Frame(center_frame)
density_frame.pack(side=tk.TOP)  
density_var = tk.DoubleVar(value=2.735e3)
density_label= tk.Label(density_frame, text="Density [kg/m³]:")
density_label.pack(side=tk.LEFT)
density_entry = ttk.Entry(density_frame, textvariable=density_var, width=5)
density_entry.pack(side=tk.LEFT)

start_frame = tk.Frame(center_frame)
start_frame.pack(side=tk.TOP)
start_label= tk.Label(start_frame, text="Chi Bounds[°]:")
start_label.pack(side=tk.LEFT)
start_var = tk.DoubleVar(value=0)
start_entry = ttk.Entry(start_frame, textvariable=start_var, width=3)
start_entry.pack(side=tk.LEFT)
end_var = tk.DoubleVar(value=360)
end_entry = ttk.Entry(start_frame, textvariable=end_var, width=3)
end_entry.pack(side=tk.LEFT)

hkl_tree = ttk.Treeview(center_frame, columns=("U", "V", "Z"), show="headings", height=3)
hkl_tree.heading("U", text="U")
hkl_tree.heading("V", text="V")
hkl_tree.heading("Z", text="Z")
hkl_tree.column("U", width=30)
hkl_tree.column("V", width=30)
hkl_tree.column("Z", width=30)
hkl_tree.insert("", "end", values=(1, 0, -1))
for _ in range(25):
    hkl_tree.insert("", "end", values=("", "", ""))
hkl_tree.pack()
hkl_tree.bind("<Button-1>", on_single_click)

load_button = tk.Button(center_frame, text="Load UVZ", command=load_hkl)
load_button.pack()

fitmodel = tk.Button(center_frame, text="Model Velocity", command=on_fitmodel_clicked)
fitmodel.pack()

loadvelocitybutton = tk.Button(center_frame, text="Model Cij", command=on_inverse_clicked) 
loadvelocitybutton.pack()

result_text = tk.Text(right_frame, width=30, height=5)
result_text.insert("1.0", "Cij fit: ")  
result_text.pack()

bulk_modulus_var = tk.StringVar()
bulk_error_var = tk.StringVar()
shear_modulus_var = tk.StringVar()
shear_error_var = tk.StringVar()
elastic_anisotropy_var = tk.StringVar()
poisson_ratio_var = tk.StringVar()

bulk_frame = tk.Frame(right_frame)
bulk_frame.pack(side=tk.TOP)
bulk_label= tk.Label(bulk_frame, text="K [GPa]:")
bulk_label.pack(side=tk.LEFT)
bulk_entry = ttk.Entry(bulk_frame, textvariable=bulk_modulus_var, width=4)
bulk_entry.pack(side=tk.LEFT)
bulkerror_entry = ttk.Entry(bulk_frame, textvariable=bulk_error_var, width=3)
bulkerror_entry.pack(side=tk.LEFT)

shear_frame = tk.Frame(right_frame)
shear_frame.pack(side=tk.TOP)
shear_label= tk.Label(shear_frame, text="\u03BC [GPa]:")
shear_label.pack(side=tk.LEFT)
shear_entry = ttk.Entry(shear_frame, textvariable=shear_modulus_var, width=4)
shear_entry.pack(side=tk.LEFT)
shearerror_entry = ttk.Entry(shear_frame, textvariable=shear_error_var, width=3)
shearerror_entry.pack(side=tk.LEFT)

ElasticAnisotropy_frame = tk.Frame(right_frame)
ElasticAnisotropy_frame.pack(side=tk.TOP)
ElasticAnisotropy_label= tk.Label(ElasticAnisotropy_frame, text="Elastic Anisotropy:")
ElasticAnisotropy_label.pack(side=tk.LEFT)
ElasticAnisotropy_entry = ttk.Entry(ElasticAnisotropy_frame, textvariable=elastic_anisotropy_var, width=3)
ElasticAnisotropy_entry.pack(side=tk.LEFT)

poisson_frame = tk.Frame(right_frame)
poisson_frame.pack(side=tk.TOP)
poisson_label= tk.Label(poisson_frame, text="Poisson's Ratio:")
poisson_label.pack(side=tk.LEFT)
poisson_entry = ttk.Entry(poisson_frame, textvariable=poisson_ratio_var, width=3)
poisson_entry.pack(side=tk.LEFT)

exportmodelbutton = tk.Button(right_frame, text="Save Model", command=save_model) 
exportmodelbutton.pack()



#-------------------- STACK TAB

datasets = {}
checkbox_vars = {}
offset_vars = {}
horizontal_offset_vars = {}
plot_colors = {} 
calibrated_flags = {}

stack_paned_window = tk.PanedWindow(frames["Stack"], orient=tk.HORIZONTAL)
stack_paned_window.pack(fill=tk.BOTH, expand=True)

config_frame = tk.Frame(stack_paned_window, highlightbackground="black", highlightthickness=1)
graph_frame = tk.Frame(stack_paned_window, highlightbackground="black", highlightthickness=1)  

stack_paned_window.add(config_frame, minsize=150)
stack_paned_window.add(graph_frame)

selected_dataset = tk.StringVar()
selected_dataset.set("") 

def update_dataset_menu():
    dataset_menu['menu'].delete(0, 'end')
    for name in datasets:
        dataset_menu['menu'].add_command(label=name, command=lambda name=name: selected_dataset.set(name))

def delete_dataset(dataset_name):
    if dataset_name and dataset_name in datasets:
        datasets.pop(dataset_name, None)
        calibrated_flags.pop(dataset_name, None)
        offset_vars.pop(dataset_name, None)
        horizontal_offset_vars.pop(dataset_name, None)
        plot_colors.pop(dataset_name, None)

        checkbox_info = checkbox_vars.get(dataset_name)
        if checkbox_info and checkbox_info["widget"].winfo_exists():
            checkbox_info["widget"].destroy()
        checkbox_vars.pop(dataset_name, None)

        update_plotSTACK()
        update_dataset_menu()

def on_checkbox_frame_configure(event):
    checkbox_canvas.configure(scrollregion=checkbox_canvas.bbox("all"))

def load_dataSTACK():
    global datasets, calibrated_flags, offset_vars, horizontal_offset_vars
    filepaths = filedialog.askopenfilenames(title="Select Data Files")
    for filepath in filepaths:
        with open(filepath, 'r') as file:
            file_data = []
            for line in file:
                if ':' not in line:
                    try:
                        file_data.append(float(line.strip()))
                    except ValueError:
                        pass

        dataset_name = os.path.basename(filepath)
        datasets[dataset_name] = file_data
        calibrated_flags[dataset_name] = False
        offset_vars[dataset_name] = tk.DoubleVar(value=0) 
        horizontal_offset_vars[dataset_name] = tk.DoubleVar(value=0) 

        dataset_menu['menu'].delete(0, 'end')
        for name in datasets:
            dataset_menu['menu'].add_command(label=name, command=lambda name=name: selected_dataset.set(name))

        create_controls_for_data(dataset_name)
        update_plotSTACK()

def change_color():
    dataset_name = selected_dataset.get()
    if dataset_name:
        color_code = colorchooser.askcolor(title="Choose color")[1]
        if color_code:
            plot_colors[dataset_name] = color_code
            update_plotSTACK()

def update_offset_and_plot():
    dataset_name = selected_dataset.get()
    if dataset_name in offset_vars:
        offset_vars[dataset_name].set(shared_offset_var.get())
    update_plotSTACK()

def update_horizontal_offset_and_plot():
    dataset_name = selected_dataset.get()
    if dataset_name in horizontal_offset_vars:
        horizontal_offset_vars[dataset_name].set(shared_horizontal_offset_var.get())
    update_plotSTACK()
    
def update_plotSTACK():
    global checkbox_vars, plot_colors, calibrated_flags, offset_vars, horizontal_offset_vars
    ax4.clear()

    is_calibrated_data_present = False

    for dataset_name, checkbox_info in checkbox_vars.items():
        if checkbox_info["var"].get():
            dataset = datasets[dataset_name]
            color = plot_colors.get(dataset_name, 'blue')  # Set default color

            if isinstance(dataset, dict) and 'x' in dataset and 'y' in dataset:
                x_values = dataset['x']
                y_values = dataset['y']
                vertical_offset = offset_vars[dataset_name].get()
                horizontal_offset = horizontal_offset_vars[dataset_name].get()

                adjusted_y_values = [y + vertical_offset for y in y_values]
                adjusted_x_values = [x + horizontal_offset for x in x_values]

                ax4.plot(adjusted_x_values, adjusted_y_values, label=dataset_name, color=color)

                if calibrated_flags.get(dataset_name, False):
                    is_calibrated_data_present = True
            else:
                data = dataset
                offset = offset_vars[dataset_name].get()
                adjusted_data = [d + offset for d in data]

                ax4.plot(adjusted_data, label=dataset_name, color=color)

    ax4.legend(fontsize=6)

    if is_calibrated_data_present:
        ax4.set_xlabel('Velocity [m/s]', fontsize=8)
        ax4.set_ylabel('Intensity [a.u.]', fontsize=8)
    else:
        ax4.set_xlabel('Channel Number', fontsize=8)
        ax4.set_ylabel('Intensity [a.u.]', fontsize=8)

    ax4.tick_params(axis='x', labelsize=8)
    ax4.tick_params(axis='y', labelsize=8)
    canvas4.draw()
    update_axis_boundsSTACK()

def update_axis_boundsSTACK(*args):
    x_min = float(x_lower_bound_entrySTACK.get())
    x_max = float(x_upper_bound_entrySTACK.get())
    y_min = float(y_lower_bound_entrySTACK.get())
    y_max = float(y_upper_bound_entrySTACK.get())
    ax4.set_ylim(y_min, y_max)
    ax4.set_xlim(x_min, x_max)
    canvas4.draw()

def create_controls_for_data(dataset_name):
    var = tk.BooleanVar(value=True)
    checkbox = tk.Checkbutton(master=checkbox_frame, text=dataset_name, variable=var, command=update_plotSTACK)
    checkbox.pack(side="top", anchor="w")  # Align to the left (west)
    checkbox_vars[dataset_name] = {"var": var, "widget": checkbox}
    offset_vars[dataset_name] = tk.DoubleVar(value=0)
    horizontal_offset_vars[dataset_name] = tk.DoubleVar(value=0)


def applycalibrationSTACK():
    global checkbox_vars, plot_colors, calibration, datasets, calibrated_flags
    current_dataset_name = selected_dataset.get()
    checkbox_info = checkbox_vars.get(current_dataset_name)
    if checkbox_info and checkbox_info["var"].get() and not calibrated_flags.get(current_dataset_name, False):
        y_data = datasets[current_dataset_name]
        y_data = np.array(y_data)
        para_theta = float(para_theta_entry.get())
        total_channel = len(y_data)
        channel_number = np.linspace(0.5 - total_channel/2, total_channel/2 - 0.5, total_channel)
        
        if calibration_treeview.get_children():
            bs_shift_values = [float(calibration_treeview.item(child)["values"][1]) for child in calibration_treeview.get_children()]
            average_bs_shift = sum(bs_shift_values) / len(bs_shift_values)
            para_Cal_channel = average_bs_shift
        else:
            para_Cal_channel = float(para_Cal_channel_entry.get())

        BS_shift = SPEED_OF_LIGHT * channel_number/ (2 * para_PS / 1000 * para_Cal_channel) * 1e-6
        calibrated_y_data = para_lambda * y_data * BS_shift * 10**-9 / (2 * np.sin(math.radians(para_theta / 2)))
        calibrated_y_data = np.abs(calibrated_y_data)
        datasets[current_dataset_name] = {"x": channel_number * 100, "y": calibrated_y_data}
        calibrated_flags[current_dataset_name] = True
        
        update_plotSTACK()




load_buttonSTACK = tk.Button(master=config_frame, text="Load Data", command=load_dataSTACK)
load_buttonSTACK.pack()

dataset_menu = tk.OptionMenu(config_frame, selected_dataset, "Select Data", *datasets.keys())
dataset_menu.pack()

calibrate_buttonSTACK = tk.Button(master=config_frame, text="Apply calibration", command=applycalibrationSTACK)
calibrate_buttonSTACK.pack()

color_button = tk.Button(master=config_frame, text="Change Color", command=change_color)
color_button.pack()

shared_offset_var = tk.DoubleVar(value=0)
shared_slider = tk.Scale(master=config_frame, from_=-10, to=10, orient="horizontal", label="Y Offset:", variable=shared_offset_var, command=lambda event: update_offset_and_plot(), resolution=0.1)
shared_slider.pack()

shared_horizontal_offset_var = tk.DoubleVar(value=0)
shared_horizontal_slider = tk.Scale(master=config_frame, from_=-10000, to=10000, orient="horizontal", label="X Offset:", variable=shared_horizontal_offset_var, command=lambda event: update_horizontal_offset_and_plot())
shared_horizontal_slider.pack()

delete_button = tk.Button(master=config_frame, text="Delete", command=lambda: delete_dataset(selected_dataset.get()))
delete_button.pack()

fig4, ax4 = plt.subplots(figsize=(2.5,3.2))
canvas4 = FigureCanvasTkAgg(fig4, master=graph_frame)
canvas4_widget = canvas4.get_tk_widget()
canvas4_widget.pack(fill='both', expand=True)
toolbarSTACK = NavigationToolbar2Tk(canvas4, graph_frame)
toolbarSTACK.update()
toolbarSTACK.pack(side=tk.TOP, fill=tk.X)
plt.tight_layout()

x_lower_bound_varSTACK = tk.StringVar(value="-15000")
x_upper_bound_varSTACK = tk.StringVar(value="15000")
y_lower_bound_varSTACK = tk.StringVar(value="0")
y_upper_bound_varSTACK = tk.StringVar(value="2")

x_bound_label = tk.Label(config_frame, text="X Bounds:")
x_bound_label.pack()
x_bounds_frame = tk.Frame(config_frame)
x_bounds_frame.pack()  
x_lower_bound_entrySTACK = ttk.Entry(x_bounds_frame, textvariable=x_lower_bound_varSTACK, width=5)
x_lower_bound_entrySTACK.pack(side="left")  
x_upper_bound_entrySTACK = ttk.Entry(x_bounds_frame, textvariable=x_upper_bound_varSTACK, width=5)
x_upper_bound_entrySTACK.pack(side="left")  

y_bound_label = tk.Label(config_frame, text="Y Bounds:")
y_bound_label.pack()
y_bounds_frame = tk.Frame(config_frame)
y_bounds_frame.pack()  
y_lower_bound_entrySTACK = ttk.Entry(y_bounds_frame, textvariable=y_lower_bound_varSTACK, width=5)
y_lower_bound_entrySTACK.pack(side="left")  
y_upper_bound_entrySTACK = ttk.Entry(y_bounds_frame, textvariable=y_upper_bound_varSTACK, width=5)
y_upper_bound_entrySTACK.pack(side="left")  

x_lower_bound_entrySTACK.bind("<FocusOut>", lambda event: update_axis_boundsSTACK())
x_upper_bound_entrySTACK.bind("<FocusOut>", lambda event: update_axis_boundsSTACK())
y_lower_bound_entrySTACK.bind("<FocusOut>", lambda event: update_axis_boundsSTACK())
y_upper_bound_entrySTACK.bind("<FocusOut>", lambda event: update_axis_boundsSTACK())

checkbox_canvas = tk.Canvas(config_frame, width=150)
checkbox_scrollbar = tk.Scrollbar(config_frame, orient="vertical", command=checkbox_canvas.yview)
checkbox_canvas.configure(yscrollcommand=checkbox_scrollbar.set)
checkbox_scrollbar.pack(side="right", fill="y")
checkbox_canvas.pack(side="left", fill="both", expand=False)
checkbox_frame = tk.Frame(checkbox_canvas)
checkbox_canvas.create_window((0, 0), window=checkbox_frame, anchor="nw")
checkbox_frame.bind("<Configure>", on_checkbox_frame_configure)

#-----------Instruction tab
# Function to add instructions
def add_instructions(parent_frame, title, text):
    title_label = tk.Label(parent_frame, text=title, font=('Arial', 16, 'bold'), anchor='w')
    title_label.pack(side="top", anchor='w')
    text_label = tk.Label(parent_frame, text=text, justify=tk.LEFT, wraplength=700, anchor='w')
    text_label.pack(side="top", fill="both", expand=True, anchor='w')

# Calibration Tab Instructions
calibration_instructions = (
    "  - Enter Calibration Settings or Load File\n"
    "  - Manually input calibration settings in the provided fields or load file.\n"
    "  - Identify the 3 strongest dips in your data.\n"
    "  - For each peak, select the appropriate fitting model from the dropdown menu.\n"
    "  - Enter the required fitting parameters for the selected model.\n"
    "  - Once satisfied with the fit for a peak, save the fit and proceed to the next peak.\n"
    "  - After fitting all 3 peaks, save the calibration settings.\n"
    "  - Repeat this process for multiple calibration files to get an average of the calibration values.\n"
)
add_instructions(frames["Instructions"], 'Calibration Tab', calibration_instructions)

# Data Tab Instructions
data_instructions = (
    "  - Click 'Apply Calibration' to rescale your loaded data file using the saved calibration settings.\n"
    "  - Choose to hide the background or smooth the data as per your preference.\n"
    "  - Click on any peak in the data to fit it.\n"
    "  - Observe the fit results in the inset plot and the accompanying table.\n"
    "  - For further analysis, export or transfer your data to the Stack tab.\n"
)
add_instructions(frames["Instructions"], 'Data Tab', data_instructions)

# Stack Tab Instructions
stack_instructions = (
    "  - Load several data files for comparison and analysis.\n"
    "  - Adjust each file's vertical and horizontal offset.\n"
    "  - Overlay files to visualize and compare data.\n"
)
add_instructions(frames["Instructions"], 'Stack Tab', stack_instructions)

# Elastic Properties Tab Instructions
elastic_instructions = (
    "  - Choose the crystal structure of your sample and input the corresponding cij elements.\n"
    "  - Alternatively, load a cij matrix file of the correct format.\n"
    "  - For forward modeling, input the uvz coordinates of interest or load them from a file.\n"
    "  - Click 'Forward Modeling' to proceed with the analysis.\n"
    "  - The software will calculate and display the fitted Cij matrix, bulk modulus (K), shear modulus (\u03BC), universal elastic anisotropy, and Poisson's ratio.\n"
)
add_instructions(frames["Instructions"], 'Elastic Properties Tab', elastic_instructions)


notebook.select(frames["Instructions"])
root.mainloop()