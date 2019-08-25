### Version 15.07.2017
### works on ### Python 3.6.1 -- Matplotlib 2.0.0 -- Numpy 1.12.1 -- Scipy 0.19.0
################################
### Alexander Riss & Mathias Pörtner : Data loading and header parsing
### Mathias Pörtner : Adding graph abilities and plotting spectral line 'maps', GUI
### Domenik Zimmermann    : Documentation and dependency testing
################################
# Edit the topographic image you want to show with gwyddion and save it as ASCII data matrix (.txt) with the option "Add informational header"
# Important: Do not trim images, and record them in the same size as u did the spectra one, otherwise the positions will not match
# Start the script and choose the directory of your data
################################
### Chose image color and spectra color ###
contrast_spec='afmhot'
contrast_topo='afmhot'
### Choose scalebar length in nm ###
scalebar_length=5
### Average # points of spectra ###
average_specs=5
################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import re
import glob
import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.optimize import curve_fit as cv
from tkinter import filedialog
import os as os

fontna=16
fontnu=12

fig = plt.figure(figsize=(10,5), dpi=100, tight_layout=True)
aplot = fig.add_subplot(121)
bplot = fig.add_subplot(122)

root = tk.Tk()
root.withdraw()
path = filedialog.askdirectory()
root.destroy()
os.chdir(path)

class Screen(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        container = tk.Frame(self)
        container.pack(side = 'top', fill = 'both', expand = True)
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)


        self.frames={}

        frame = Page(container,self)
        self.title("Plot line Spectra")
        self.frames[Page] = frame
        frame.grid(row = 0, column = 0, sticky = 'nsew')


class Page(tk.Frame):
    def __init__(self, parent, controller):
        # Initialize GUI for choosing files
        tk.Frame.__init__(self, parent)

        self.widget = None
        self.toolbar = None

        # Choose topo file (.txt)
        labe1 = tk.Label(self, text = 'Image-file').grid(row = 0,column = 0,sticky = 'w')
        choicesIma = glob.glob('*.txt')
        choicesIma.sort()
        self.variableIma = tk.StringVar(self)
        self.variableIma.set(choicesIma[0])
        wIma = ttk.Combobox(self, textvariable = self.variableIma, values = choicesIma, width = 25)
        wIma.grid(row = 0, column = 1, sticky = 'w')

        # Choose file of spectra along line (.L*.VERT) (only first spectrum is displayed
        labe2=tk.Label(self, text = 'Spectra-file').grid(row = 1, column = 0, sticky = 'w')
        choicesSpec = glob.glob('*L0001.VERT')
        choicesSpec.sort()
        self.variableSpec = tk.StringVar(self)
        self.variableSpec.set(choicesSpec[0])
        wSpec = ttk.Combobox(self, textvariable = self.variableSpec, values = choicesSpec, width = 25)
        wSpec.grid(row = 1, column = 1, sticky = 'w')

        # Load data button
        loadbuttonIma=tk.Button(self,text = 'Load',command = self.plotimage, width = 12)
        loadbuttonIma.grid(row = 0, column = 2)

        # Quit button
        button = tk.Button(self, text = 'Quit', command = self._quit, width = 12)
        button.grid(row = 0,column = 3)

        self.ima_min = tk.DoubleVar(self)
        self.ima_max = tk.DoubleVar(self)
        self.sp_min = tk.DoubleVar(self)
        self.sp_max = tk.DoubleVar(self)

    def _quit(self):
        self.quit()
        self.destroy()

    def string_simplify(self, str):
        # simplifies a string (i.e. removes replaces space for "_", and makes it lowercase
        return str.replace(' ', '_').lower()

    def ladenSTS(self, data):
        # Reads .dat file for spectral information, returns data-array for Voltage, dIdV and spectrum position
        headersp = {}
        f = open(data, encoding='utf-8', errors='ignore')
        headersp_ended = False
        caption = re.compile(':*:')
        key = ''
        contents = ''
        line=f.readline()
        U = []
        dIdU = []
        if line[:10] == 'Experiment':
            # Data from Nanonis
            while not headersp_ended:
                # load header
                line = f.readline()
                if line == '\n':
                    headersp_ended = True
                else:
                    parts = line.split('    ')
                    if len(parts) != 3: continue
                    key, contents, throw = parts
                    line = line.strip()
                    key = self.string_simplify(key)
                    headersp[key] = contents.strip()

            while headersp_ended:
                # load data
                line = f.readline()
                if not line: break
                else:
                    U.append(float(line.split()[2].replace(',', '.')))
                    dIdU.append(float(line.split()[9].replace(',', '.')))
            f.close()

            # get positions of spectrum
            posi=[float(headersp['x_(m)']), float(headersp['y_(m)'])]

            U = np.array(U, float)
            dIdU = np.array(dIdU, float)

            return U, dIdU, posi

        elif line[:10] == '[Parameter':
            # Data from Createc 2.0
            while not headersp_ended:
                # load header
                line = f.readline()
                if not line: break
                if line[0:4] == "    ":
                    parts = line.split()
                    posi = np.array([float(parts[-2]), float(parts[-1])],float)
                    headersp_ended = True
                else:
                    parts = line.split('=')
                    if len(parts) != 2: continue
                    key, contents = parts
                    line = line.strip()
                    key = self.string_simplify(key)
                    headersp[key] = contents.strip()

            while headersp_ended:
                # load data
                line = f.readline()
                if not line: break
                else:
                    U.append(float(line.split()[3].replace(',','.')))
                    dIdU.append(float(line.split()[2].replace(',','.')))
            f.close()

            dacstep = np.array([float(headersp['delta_x_/_delta_x_[dac]']), float(headersp['delta_y_/_delta_y_[dac]'])])
            pixelsize = np.array([float(headersp['num.x_/_num.x']), float(headersp['num.y_/_num.y'])])
            imagesize = np.array([float(headersp['length_x[a]']), float(headersp['length_y[a]'])])

            # match position with image size
            posi /= dacstep
            posi[0] = (pixelsize[0] / 2.0 + posi[0]) * imagesize[0] / pixelsize[0] / 10
            posi[1] = (pixelsize[1] - posi[1]) * imagesize[1] / pixelsize[1] / 10

            if int(headersp['vertspecback']) == 1:
                # Average back and foreward scan of spectrum
                Uneu = []
                dIdUneu = []
                for i in range(int(len(U) / 2)):
                    Uneu.append((U[i] + U[len(U) - 1 - i]) / 2)
                    dIdUneu.append((dIdU[i] + dIdU[len(U) - 1 - i]) / 2)
                U = np.array(Uneu,float)
                dIdU = np.array(dIdUneu,float)

            return U, dIdU, posi

        elif line[:10]=='[ParVERT30':
            # Data from Createc 3.1
            while not headersp_ended:
                line = f.readline()
                if not line: break
                if line[0:4] == "    ":
                    parts=line.split()
                    posi=np.array([float(parts[-3]),float(parts[-2])],float)
                    headersp_ended = True
                else:
                    parts = line.split('=')
                    if len(parts)!=2: continue
                    key, contents = parts
                    line = line.strip()
                    key = self.string_simplify(key)
                    headersp[key] = contents.strip()

            while headersp_ended:
                line = f.readline()
                if not line: break
                else:
                    U.append(float(line.split()[1].replace(',','.')))
                    dIdU.append(float(line.split()[4].replace(',','.')))
            f.close()

            dacstep=np.array([float(headersp['delta_x_/_delta_x_[dac]']),float(headersp['delta_y_/_delta_y_[dac]'])],float)
            self.pixelsize=np.array([float(headersp['num.x_/_num.x']),float(headersp['num.y_/_num.y'])],float)
            imagesize=np.array([float(headersp['length_x[a]']),float(headersp['length_y[a]'])],float)

            # match position with image size
            posi /= dacstep
            posi[0] = (pixelsize[0] / 2.0 + posi[0]) * imagesize[0] / pixelsize[0] / 10
            posi[1] = (pixelsize[1] - posi[1]) * imagesize[1] / pixelsize[1] / 10

            if int(headersp['vertspecback']) == 1:
                # Average back and foreward scan of spectrum
                Uneu = []
                dIdUneu = []
                for i in range(int(len(U) / 2)):
                    Uneu.append((U[i] + U[len(U) - 1 - i]) / 2)
                    dIdUneu.append((dIdU[i] + dIdU[len(U) - 1 - i]) / 2)
                U = np.array(Uneu, float)
                dIdU = np.array(dIdUneu, float)

            return U, dIdU, posi

    def laden_image(self,data):
        #Reads .txt file from Gwyddion for topographic information, returns 2D data-array for topography and the size of the image in nm
        header = {}
        f = open(data, encoding='utf-8', errors='ignore')
        header_ended = False
        caption = re.compile(':*:')
        key = ''
        contents = ''
        while not header_ended:
            # load header
            line = f.readline()
            if not line: break
            if line[0] != "#":
                header_ended = True
            else:
                parts = line.split(':')
                if len(parts) != 2: continue
                key, contents = parts
                line = line.strip()
                key = self.string_simplify(key[2:])
                header[key] = contents[:-4].strip()
        f.close()

        # get image size
        ext=np.array([float(header['width']), float(header['height'])], float)

        # get data in form of 2D array in nm
        X=np.loadtxt(data) * 1e10

        return(X,ext)

    def minmax(self):
        #calculation of the minimal and maximal values of the 2D arrays (for setting pf the contrast)
        self.ima_min0 = np.amin(self.ima)
        self.ima_max0 = np.amax(self.ima)
        self.mitte_ima0 = np.mean([self.ima_max0, self.ima_min0])
        self.sp_min0 = np.amin(self.matrixy)
        self.sp_max0 = np.amax(self.matrixy)
        self.mitte_sp0 = np.mean([self.sp_max0, self.sp_min0])

    def update_ima_min(self, val):
        self.ima_min = val
        self.imagepl.set_clim(vmin = val)
        self.canvas.draw()

    def update_ima_max(self, val):
        self.ima_max = val
        self.imagepl.set_clim(vmax = val)
        self.canvas.draw()

    def update_sp_min(self, val):
        self.sp_min = val
        self.specpl.set_clim(vmin = val)
        self.canvas.draw()

    def update_sp_max(self, val):
        self.sp_max = val
        self.specpl.set_clim(vmax = val)
        self.canvas.draw()

    def reset(self):
        self.imagepl.set_clim(vmin = self.ima_min0, vmax = self.ima_max0)
        self.specpl.set_clim(vmin = self.sp_min0, vmax = self.sp_max0)
        self.canvas.draw()
        print('Reset Contrast')

    def save_con(self):
        untenima = float(self.ima_min)
        obenima = float(self.ima_max)
        untensp = float(self.sp_min)
        obensp = float(self.sp_max)
        np.savetxt(self.data_name + '.contrast.csv', [untenima, obenima, untensp, obensp], delimiter=',')
        print('Contrast saved to: ' + self.data_name + '.contrast.csv')

    def save_ima(self):
        fig.savefig(self.data_name + '.png')
        print('Image saved to: ' + self.data_name + '.png')

    def old_con(self):
        con = np.loadtxt(self.data_name+'.contrast.csv',delimiter=',')

        self.imagepl.set_clim(vmin = con[0], vmax = con[1])
        self.specpl.set_clim(vmin = con[2], vmax = con[3])
        self.canvas.draw()
        print('Load old contrast from: ' + self.data_name+'.contrast.csv')

    def average_spec(self, matrixx, matrixy, ave):
        #average spectra
        matrixyneu = []
        matrixxneu = []
        for n, i in enumerate(matrixy):
            matrixyneu.append([])
            matrixxneu.append([])
            for m, j in enumerate(i):
                if m % ave == (ave - 1):
                    matrixyneu[-1].append(sum(matrixy[n][m - (ave - 1) : m]) / ave)
                    matrixxneu[-1].append(sum(matrixx[n][m - (ave - 1) : m]) / ave)
        matrixyneu = np.array(matrixyneu, float)
        matrixxneu = np.array(matrixxneu, float)
        return(matrixxneu, matrixyneu)

    def plotimage(self):
        #Plot image and set up final GUI
        s = self.variableSpec.get()
        spec = glob.glob(s[:-9] + '*.VERT')
        spec.sort()
        self.data_name = self.variableIma.get()

        self.ima, self.imagesize = self.laden_image(self.data_name)

        ext = []
        matrixx = []
        matrixy = []
        spec_posi = []
        for i in spec:
            x, y, posi = self.ladenSTS(i)
            matrixy.append(y)
            matrixx.append(x)
            spec_posi.append(posi)
            if i == spec[0] or i == spec[-1]:
                ext.append(posi)
        self.matrixx = np.array(matrixx, float)
        self.matrixy = np.array(matrixy, float)

        line_length = np.linalg.norm(ext[1] - ext[0])

        self.matrixx, self.matrixy = self.average_spec(self.matrixx, self.matrixy, average_specs)

        if self.widget:
            self.widget.destroy()
        if self.toolbar:
            self.toolbar.destroy()
        aplot.clear()
        bplot.clear()

        self.minmax()

        #Plot Topography
        self.imagepl = aplot.imshow(self.ima, cmap = contrast_topo, extent = [0, self.imagesize[0], 0, self.imagesize[1]], vmin = self.ima_min0, vmax = self.ima_max0)
        aplot.set_xticks([])
        aplot.set_yticks([])
        for pos in spec_posi:
            aplot.plot(pos[0], pos[1], '.', color='white', ms=1)
        aplot.plot([1, 1 + scalebar_length], [1, 1], color='white', lw=2)
        aplot.text(1 + scalebar_length / 2, 1 + self.imagesize[1] * 0.01, '{:d}nm'.format(int(scalebar_length)), va='bottom', ha='center', color='white', fontsize=16)

        #Plot spectra map
        self.specpl=bplot.imshow(self.matrixy.T, cmap=contrast_spec, extent = [0, line_length, min(self.matrixx[0]), max(self.matrixx[0])], aspect='auto', vmin = self.sp_min0, vmax = self.sp_max0)
        bplot.set_xlabel('Distance x [nm]', fontsize = fontna)
        bplot.set_ylabel('Bias voltage [mV]', fontsize = fontna)

        #Put Plot in GUI
        self.canvas = FigureCanvasTkAgg(fig, self)
        self.canvas.draw()
        self.widget = self.canvas.get_tk_widget()
        self.widget.grid(row = 2, columnspan = 4)

        #Sliders for contrast
        sl_min_ima = tk.Scale(self, from_= self.ima_min0 - (self.mitte_ima0 - self.ima_min0), to = self.ima_max0, resolution = 0.01, variable = self.ima_min, command = self.update_ima_min, orient = tk.HORIZONTAL, length = 300, width = 15)
        sl_min_ima.set(self.ima_min0)
        sl_min_ima.grid(row = 3, column = 1)

        label_i_min = tk.Label(self, text = 'Lower value image')
        label_i_min.grid(row = 3, column = 0)

        sl_max_ima = tk.Scale(self, from_ = self.ima_min0, to = self.ima_max0 + (self.ima_max0 - self.mitte_ima0), resolution = 0.01, variable = self.ima_max, command = self.update_ima_max, orient = tk.HORIZONTAL, length = 300, width = 15)
        sl_max_ima.set(self.ima_max0)
        sl_max_ima.grid(row = 4, column = 1)

        label_i_max = tk.Label(self, text = 'Upper value image')
        label_i_max.grid(row = 4, column = 0)

        sl_min_sp = tk.Scale(self, from_ = self.sp_min0 - (self.mitte_sp0 - self.sp_min0), to = self.sp_max0, resolution = 0.01, variable = self.sp_min, command = self.update_sp_min, orient = tk.HORIZONTAL, length = 300, width = 15)
        sl_min_sp.set(self.sp_min0)
        sl_min_sp.grid(row = 3, column = 3)

        label_s_min=tk.Label(self, text = 'Lower value spectra')
        label_s_min.grid(row = 3, column = 2)

        sl_max_sp=tk.Scale(self, from_ = self.sp_min0, to = self.sp_max0 + (self.sp_max0 - self.mitte_sp0), resolution = 0., variable = self.sp_max, command = self.update_sp_max, orient = tk.HORIZONTAL, length = 300, width = 15)
        sl_max_sp.set(self.sp_max0)
        sl_max_sp.grid(row = 4, column = 3)

        label_s_max=tk.Label(self, text = 'Upper value spectra')
        label_s_max.grid(row = 4, column = 2)

        #Save contrast button
        buttonsav = tk.Button(self, text = "Save contrast", command = self.save_con, width = 12)
        buttonsav.grid(row = 3, column = 8)

        #Old contrast button
        buttonoc = tk.Button(self, text = "Old contrast", command = self.old_con, width = 12)
        buttonoc.grid(row = 4, column = 8)

        #Save image button
        buttonsavima = tk.Button(self, text = "Save image", command = self.save_ima, width = 12)
        buttonsavima.grid(row = 1, column = 2)

        #Reset contrast button
        buttonres = tk.Button(self, text = 'Reset', command = self.reset, width = 12)
        buttonres.grid(row = 1, column = 3)

app = Screen()
app.mainloop()
