#from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import math
import pims
import trackpy as tp
# change the following to %matplotlib notebook for interactive plotting

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

"""
@pims.pipeline
def gray(image):
    return image[:, :, 1]  # Take just the green channel


def acquire_video(path):
	return gray(pims.open(path))
	
def annotate_video(v, t2):
	for i in range(len(v)):
		plt.figure()
		plt.title("Image:" + str(i))
		tp.annotate(t2[t2['frame'] == i], v[i])

def plots(v):
	f = tp.locate(v[0], 5, invert=False )
	#tp.annotate(f, v[10])

	fig, ax = plt.subplots()
	ax.hist(f['mass'], bins=20)
	plt.show()
	# Optionally, label the axes.
	ax.set(xlabel='mass', ylabel='count');
	tp.subpx_bias(f)
	tp.subpx_bias(tp.locate(v[0], 5, invert=False, minmass=20));	
	plt.show()

def mainflow(
		video_path, size=5, memory=3,
		min_mass=25, max_size=2.6, max_ecc=0.3,
        invertColor=True, fractionExistInFrames=1//8,
		annotate=True, plots=False
	):
	v = acquire_video(video_path)
	print(v)
	#size = 5 # estimated size of the features in pixels (must be an odd integer)
	f = tp.batch(v[:], 5, minmass=20, invert=invertColor, processes=1)	

	maximum_displacement_between_frames = 5 # the larger, the slower the computation
	memory = 3 # the amount of frames that a particle can "disappear" between appearances
	t = tp.link(f, maximum_displacement_between_frames, memory=memory) # f, displacement in pixels, memory is number of frames backwards incase particle slips out of sight

	t.head()

	t1 = tp.filter_stubs(t, len(v)*fractionExistInFrames)
	print('Before:', t['particle'].nunique())
	print('After:', t1['particle'].nunique())

	if plots:
		plt.figure()
		tp.mass_size(t1.groupby('particle').mean()); # convenience function -- just plots size vs. mass

	#min_mass = 25
	#max_size = 2.6
	#max_ecc = 0.3

	t2 = t1[
			((t1['mass'] > min_mass) & 
			(t1['size'] < max_size) &
			(t1['ecc'] < max_ecc))
			]

	if annotate:
		frame = 1
		plt.figure()
		plt.title("Image:" + str(frame))
		tp.annotate(t2[t2['frame'] == frame], v[frame])


	d = tp.compute_drift(t2)
	tm = tp.subtract_drift(t2.copy(), d)

	if plots:
		d.plot()
		ax = tp.plot_traj(tm)

	#im = tp.imsd(tm, 100/285., 24)  # microns per pixel = 100/285., frames per second = 24
	v_fps = pims.open(video_path)
	fps = v_fps.frame_rate

	mpp = 10000/2048 # from pco camera calculation from a conversation with Dr. Tanner 
	im = tp.imsd(tm, mpp, fps)  # microns per pixel, frames per seconds
	if plots:

		fig, ax = plt.subplots()
		ax.plot(im.index, im, 'k-', alpha=0.1)  # black lines, semitransparent
		ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
			xlabel='lag time $t$')
		ax.set_xscale('log')
		ax.set_yscale('log')
		#plt.show()


	em = tp.emsd(tm, mpp, fps) # microns per pixel = 100/285., frames per second = 24
	tp.utils.fit_powerlaw(em)  # performs linear best fit in log space, plots]
	if plots:

		fig, ax = plt.subplots()
		ax.plot(em.index, em, 'o')
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
			xlabel='lag time $t$')
		ax.set(ylim=(1e-2, 10));
		plt.show()

		plt.figure()
		plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
		plt.xlabel('lag time $t$');
		
	# this is single plot if plot=False
	df = tp.utils.fit_powerlaw(em)
	n = df.iloc[0,0]
	A = df.iloc[0,1]
	print('\npower-law exponent, n:',n, 'A:', A, 'um^2$/s', '\n')

	# in water, a viscous material, the expected power-law exponent n=1. 
	# The value of A = 4D, where D is the particles' diffusivity. 
	# D = k_b*T/(6 * math.PI() * viscosity * radius) # Stokes-Einstein equation
	D = A / 4
	print("Diffusivity =", D, 'um^2/s = ', D*1E-12, 'm^s/s\n')
	return v, t2, n, A, D


if __name__ == '__main__':
	v, t2, n, A, D = mainflow(
			video_path=r'trackingvid.mp4', size=5, memory=10,
			min_mass=20, max_size=2.6, max_ecc=0.3,
			invertColor=False, fractionExistInFrames=1//8,
			annotate=True, plots=False
	)




	# Original Tests
	mpp = 10000/2048
	# second_col/(4*lag time)
	# second_col = microns^2
	data = tp.motion.imsd(t2, mpp, 0.983)
	#print(data)

	particle_number = 9
	microns_sq = np.vstack(data[particle_number])[:,0]
	#print(microns_sq)
	#data heads are the particle number assigned above

	#print(data.columns.values)
	#print(data.head())

	data[particle_number].to_csv("test.csv")
	array = np.genfromtxt('test.csv', skip_header=1, delimiter=',')
	#print(array)
	lag_time = array[:,0]
	microns_sq = array[:,1]
	#print(microns_sq)
	#print(lag_time)

	dif_co = microns_sq[-1] / (4*lag_time[-1])

	dif_co_m_sq_per_s = dif_co * (1E-6**2)
	print(dif_co_m_sq_per_s)
"""

# change the following to %matplotlib notebook for interactive plotting
@pims.pipeline
def gray(image):
    image=image[:, :, 1]
    image= zero_image(image,150)
    return image
"""
def gray(image):
    return image[:, :, 1]
"""


def zero_image(image, threshold):
    for row in range(len(image)):
        row1 = image[row,:]   
        row1[row1 < threshold] =0
        image[row,:] = row1
    
    return image
def acquire_video(path):
	return gray(pims.open(path))
	
def annotate_video(v, t2):
    for i in range(len(v)):
        plt.figure()
        plt.title("Image:" + str(i))
        tp.annotate(t2[t2['frame'] == i], v[i])

def plots(v):
	f = tp.locate(v[0], 5, invert=False )
	#tp.annotate(f, v[10])

	fig, ax = plt.subplots()
	ax.hist(f['mass'], bins=20)
	plt.show()
	# Optionally, label the axes.
	ax.set(xlabel='mass', ylabel='count');
	tp.subpx_bias(f)
	tp.subpx_bias(tp.locate(v[0], 5, invert=False, minmass=20));	
	plt.show()

def mainflow(
		video_path, size=5, memory=3,
		min_mass=25, max_size=2.6, max_ecc=0.3,
        invertColor=True, fractionExistInFrames=1//8,
        max_lagtime=100, maximum_displacement_between_frames = 5,
		annotate=True, plots=False
	):
    v = acquire_video(video_path)
    #for n, i in enumerate(v):
        #v[n] = zero_image(i,220)
    #print(v)
    #print(len(v))
    frame_cnt = len(v)-3
    frame_cnt = 18
    
    
    #size = 5 # estimated size of the features in pixels (must be an odd integer)
    f = tp.batch(v[:frame_cnt], 5, minmass=min_mass, invert=invertColor, processes=1)	

    #maximum_displacement_between_frames = 5 # the larger, the slower the computation
    memory = 3 # the amount of frames that a particle can "disappear" between appearances
    t = tp.link(f, maximum_displacement_between_frames, memory=memory) # f, displacement in pixels, memory is number of frames backwards incase particle slips out of sight
    
    t.head()
    
    t1 = tp.filter_stubs(t, len(v)*fractionExistInFrames)
    print('Before:', t['particle'].nunique())
    print('After:', t1['particle'].nunique())

    if plots:
        plt.figure()
        tp.mass_size(t1.groupby('particle').mean()); # convenience function -- just plots size vs. mass

    #min_mass = 25
    #max_size = 2.6
    #max_ecc = 0.3

    t2 = t1[
            ((t1['mass'] > min_mass) & 
            (t1['size'] < max_size) &
            (t1['ecc'] < max_ecc))
            ]
    #print("t2:", t2)
    if annotate:
        frame = 1
        plt.figure()
        plt.title("Image:" + str(frame))
        tp.annotate(t2[t2['frame'] == frame], v[frame])


    d = tp.compute_drift(t2)
    tm = tp.subtract_drift(t2.copy(), d)
    #print("tm", tm)
    if plots:
        d.plot()
        ax = tp.plot_traj(tm)

    #im = tp.imsd(tm, 100/285., 24)  # microns per pixel = 100/285., frames per second = 24
    v_fps = pims.open(video_path)
    fps = v_fps.frame_rate

    mpp = 10000/2048 # from pco camera calculation from a conversation with Dr. Tanner 
    #im = tp.imsd(tm, mpp, fps, max_lagtime=max_lagtime)  # microns per pixel, frames per seconds
    im = tp.msd(tm, mpp, fps, max_lagtime=max_lagtime)  # microns per pixel, frames per seconds
    #print("im:", im.head(15))
    if plots:

        fig, ax = plt.subplots()
        ax.plot(im.index, im, 'k-', alpha=0.1)  # black lines, semitransparent
        ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
            xlabel='lag time $t$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        #plt.show()


    em = tp.emsd(tm, mpp, fps) # microns per pixel = 100/285., frames per second = 24
    print("em:", em)
    tp.utils.fit_powerlaw(em)  # performs linear best fit in log space, plots]
    if plots:

        fig, ax = plt.subplots()
        ax.plot(em.index, em, 'o')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
            xlabel='lag time $t$')
        ax.set(ylim=(1e-2, 10));
        plt.show()

        plt.figure()
        plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
        plt.xlabel('lag time $t$');
        
    # this is single plot if plot=False
    df = tp.utils.fit_powerlaw(em)
    n = df.iloc[0,0]
    A = df.iloc[0,1]
    print('\npower-law exponent, n:',n, 'A:', A, 'um^2$/s', '\n')

    # in water, a viscous material, the expected power-law exponent n=1. 
    # The value of A = 4D, where D is the particles' diffusivity. 
    # D = k_b*T/(6 * math.PI() * viscosity * radius) # Stokes-Einstein equation
    D = A / 4
    print("Diffusivity =", D, 'um^2/s = ', D*1E-12, 'm^s/s\n')
    return v, t2, n, A, D

"""
	# Original Tests
	mpp = 10000/2048
	# second_col/(4*lag time)
	# second_col = microns^2
	data = tp.motion.imsd(t2, mpp, 0.983)
	#print(data)

	particle_number = 9
	microns_sq = np.vstack(data[particle_number])[:,0]
	#print(microns_sq)
	#data heads are the particle number assigned above

	#print(data.columns.values)
	#print(data.head())

	data[particle_number].to_csv("test.csv")
	array = np.genfromtxt('test.csv', skip_header=1, delimiter=',')
	#print(array)
	lag_time = array[:,0]
	microns_sq = array[:,1]
	#print(microns_sq)
	#print(lag_time)

	dif_co = microns_sq[-1] / (4*lag_time[-1])

	dif_co_m_sq_per_s = dif_co * (1E-6**2)
	print(dif_co_m_sq_per_s)
"""

if __name__ == "__main__":

	v, t2, n, A, D = mainflow(
			video_path=r'ChitosanTPP&water.mp4', size=3, memory=10,
			#video_path=r'trackingvid.mp4', size=3, memory=10,
			min_mass=20, max_size=7, max_ecc=0.3,
			invertColor=False, fractionExistInFrames=1//8,
			max_lagtime=4, maximum_displacement_between_frames = 15,
			annotate=False, plots=False
	)
# need to determine which rows in y are removed and do appropriate for x