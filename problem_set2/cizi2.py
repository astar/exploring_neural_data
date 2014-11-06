#
#  NAME
#    problem_set2_solutions.py
#
#  DESCRIPTION
#    Open, view, and analyze action potentials recorded during a behavioral
#    task.  In Problem Set 2, you will write create and test your own code to
#    create tuning curves.
#

#Helper code to import some functions we will use
import numpy as np
import matplotlib.pylab as plt
import matplotlib.mlab as mlab
from scipy import optimize
from scipy import stats


def load_experiment(filename):
    """
    load_experiment takes the file name and reads in the data.  It returns a
    two-dimensional array, with the first column containing the direction of
    motion for the trial, and the second column giving you the time the
    animal began movement during thaht trial.
    """
    data = np.load(filename)[()];
    return np.array(data)

def load_neuraldata(filename):
    """
    load_neuraldata takes the file name and reads in the data for that neuron.
    It returns an arary of spike times.
    """
    data = np.load(filename)[()];
    return np.array(data)
    
def bin_spikes(trials, spk_times, time_bin):
    """
    bin_spikes takes the trials array (with directions and times) and the spk_times
    array with spike times and returns the average firing rate for each of the
    eight directions of motion, as calculated within a time_bin before and after
    the trial time (time_bin should be given in seconds).  For example,
    time_bin = .1 will count the spikes from 100ms before to 100ms after the 
    trial began.
    
    dir_rates should be an 8x2 array with the first column containing the directions
    (in degrees from 0-360) and the second column containing the average firing rate
    for each direction
    
    """
    # create array to store (sum of spike counts per direction)
    column0 = np.arange(0,360,45)
    column1 = np.zeros(8)
    dir_spike_count = np.column_stack((column0, column1))
   
    # iterate through all the trials
    for i in range(0, len(trials)):
        # get direction and determine time window to count spikes
        direction = trials[i,0]
        start_time = trials[i,1] - time_bin
        stop_time = trials[i,1] + time_bin
        
        # check all spike times to see if they are in the time window
        for j in range(0, len(spk_times)):
            if (start_time <= spk_times[j] <= stop_time):
                
                # if spike is in time window count it!
                # iterate through all 8 directions to 
                # correctly increment the correct direction
                for k in range(0, len(dir_spike_count)):
                    if (dir_spike_count[k,0] == direction):
                        dir_spike_count[k,1] = dir_spike_count[k,1] + 1
         
    # create output data structure (firing rate)
    column0 = np.arange(0,360,45)
    column1 = np.zeros(8)
    dir_rates = np.column_stack((column0, column1))
  
    # Call function to calculate number of trials per direction
    # (17 each direction in this data)
    dir_trial_count = count_trials(trials)
    
    # convert counts to average firing rate (spikes/s)  
    for i in range(0, len(dir_spike_count)):
        dir_rates[i,1] = dir_spike_count[i,1]/(2*time_bin)
        dir_rates[i,1] = dir_rates[i,1]/dir_trial_count[i,1] 
    
    return dir_rates
    
def count_trials(trials):
    """
    count_trials calculates total number of trials per direction
    input:  2-dimenstional array (direction, time)
    output:  2-dimenstional array (direction, total_trials)
    """
    # count total number of trials per direction

    column0 = np.arange(0,360,45)
    column1 = np.zeros(8)
    dir_trial_count = np.column_stack((column0, column1))

    for i in range(0, len(trials)):
        direction = trials[i,0]
        
        for k in range(0, len(dir_trial_count)):
            if (dir_trial_count[k,0] == direction):
                dir_trial_count[k,1] = dir_trial_count[k,1] + 1    
       
    return dir_trial_count   

def plot_tuning_curves(direction_rates, title):
    """
    This function takes the x-values and the y-values  in units of spikes/s 
    (found in the two columns of direction_rates) and plots a histogram and 
    polar representation of the tuning curve. It adds the given title.
    """
    # create a new figure
    plt.figure()    
    
    # create two plots in one figure (1 row, 2 cols)    
    plt.subplot(2,2,1)
    
    # histogram
    plt.bar(direction_rates[:,0],direction_rates[:,1], width=45,align='center')     
    plt.xlabel('Direction of Motion (degrees)')
    plt.ylabel('Firing Rate (spike/s)')
    plt.title(title)
    plt.axis([0,360,0,max(direction_rates[:,1])+1])
    # formatting for plot x-axis    
    plt.xticks(np.arange(0,360,45))
    plt.xlim(-22.5, 337.5)
    
#   # polar plot
    plt.subplot(2,2,2,polar=True)
    
    # copy the array to a new array
    polar_data = direction_rates*1
    
#   # convert degrees to radians 
    for i in range(0, len(polar_data)):
        polar_data[i,0] = np.deg2rad(polar_data[i,0])
    
    theta = polar_data[:,0]
    r = polar_data[:,1]
    # append data to connect 315 to 360 
    r2 = np.append(r,r[0])
    theta2 = np.append(theta,theta[0])
    
    plt.polar(theta2, r2, label='Firing Rate (spikes/s)')
    plt.legend(loc=0) # 0 - best, 8- lower center
    plt.title(title)
       
def roll_axes(direction_rates):
    """
    roll_axes takes the x-values (directions) and y-values (direction_rates)
    and return new x and y values that have been "rolled" to put the maximum
    direction_rate in the center of the curve. The first and last y-value in the
    returned list should be set to be the same. (See problem set directions)
    Hint: Use np.roll()
    """
    # make a copy of input data array
    # and place into x and y rrays
    dataIn = direction_rates*1
    new_xs = dataIn[:,0]
    new_ys = dataIn[:,1]
    
    # calculate number of shifts to center max y value
    #   np.argmax() determines the array index of the max value
    shifts = 4 - np.argmax(new_ys)    
    new_ys = np.roll(new_ys, shifts)
    new_ys = np.append(new_ys, new_ys[0])
    
    # rolled degrees = shifts * 45 degrees    
    roll_degrees = shifts * 45 
    
    # roll x-axis by subtracting the "roll degrees"  
    for i in range(0,len(new_xs)):
        new_xs[i] = new_xs[i] - roll_degrees
   
    # add 45 to rightmost x[n] and append it
    new_xs = np.append(new_xs, new_xs[7] + 45)
        
    return new_xs, new_ys, roll_degrees    
    

def normal_fit(x,mu, sigma, A):
    """
    This creates a normal curve over the values in x with mean mu and
    variance sigma.  It is scaled up to height A.
    """  
    n = A*mlab.normpdf(x,mu,sigma)
    return n

def fit_tuning_curve(centered_x,centered_y):
    """
    This takes our rolled curve, generates the guesses for the fit function,
    and runs the fit.  It returns the parameters to generate the curve.
    """
    # What is the biggest y-value? (estimates the amplitude of the curve, A)
    max_y = np.amax(centered_y)      

    # Where is the biggest y-value? (estimates the mean of the curve, mu)
    max_x = centered_x[np.argmax(centered_y)]
    
    # Approximating one standard deviation of our normal distribution
    #(which should be around the width of 2 bars).
    sigma = 90              

    p, cov = optimize.curve_fit(normal_fit,centered_x, centered_y, 
                                p0=[max_x, sigma, max_y])

    return p
    

def plot_fits(direction_rates,fit_curve,title):
    """
    This function takes the x-values and the y-values  in units of spikes/s 
    (found in the two columns of direction_rates and fit_curve) and plots the 
    actual values with circles, and the curves as lines in both linear and 
    polar plots.
    """
    #plt.figure()
    # create two plots in one figure (1 row, 2 cols)    
    plt.subplot(2,2,3)
    
    plt.plot(direction_rates[:,0], direction_rates[:,1], 'bo')
    plt.xlabel('Direction of Motion (degrees)')
    plt.ylabel('Firing Rate (spike/s)')
    plt.title(title)
    plt.xlim(-5, 365)

    plt.plot(fit_curve[:,0], fit_curve[:,1], color='g')
    
    # polar plot
    plt.subplot(2,2,4,polar=True)
    
    # copy the array to a new array
    polar_data = direction_rates*1
    
    # convert degrees to radians 
    for i in range(0, len(polar_data)):
        polar_data[i,0] = np.deg2rad(polar_data[i,0])
    
    theta = polar_data[:,0]
    r = polar_data[:,1]
    # append data to connect 315 to 360 
    r2 = np.append(r,r[0])
    theta2 = np.append(theta,theta[0])
    
    plt.polar(theta2, r2, 'bo')
      
    # plot fitted curve data
    polar_fdata = fit_curve*1
    
    # convert degrees to radians 
    for i in range(0, len(polar_fdata)):
        polar_fdata[i,0] = np.deg2rad(polar_fdata[i,0])
    
    theta = polar_fdata[:,0]
    r = polar_fdata[:,1]
    
    # append data to connect 315 to 360 
    r2 = np.append(r,r[0])
    theta2 = np.append(theta,theta[0])
    
    plt.polar(theta2, r2, color='g', label='Firing Rate (spikes/s)')
    
    plt.legend(loc=0)  # 0 - best, 8- lower center
    plt.title(title)  
    

def von_mises_fitfunc(x, A, kappa, l, s):
    """
    This creates a scaled Von Mises distrubition.
    """
    return A*stats.vonmises.pdf(x, kappa, loc=l, scale=s)

    
def preferred_direction(fit_curve):
    """
    The function takes a 2-dimensional array with the x-values of the fit curve
    in the first column and the y-values of the fit curve in the second.  
    It returns the preferred direction of the neuron (in degrees).
    """
    # What is the biggest y-value?
    max_y = np.amax(fit_curve[:,1])      
    #print ('max y =' + str(max_y))

    # Where is the biggest y-value?
    pd = fit_curve[np.argmax(fit_curve[:,1])] 
    #print ('preferred direction =' + str(pd[0]))
    
    return pd
 
def roll_fit_unroll(direction_rates):
    """
    A function that does the rolling, fitting, and rolling back
    operations used to fit the data to a normal distribution curve
    
    Input - array of average firing rate for each of 8 direction. 
    Output - array of normal fitted firing rates for each direction 
            360 degrees
    """    
    # roll axes to center histogram
    # remember roll_degrees to unroll after fitting process
    new_xs, new_ys, roll_degrees = roll_axes(direction_rates)
    
    # fit rolled curve to normal distribution
    p = fit_tuning_curve(new_xs, new_ys)
    
    # create smooth curve with many x-axis pointswith default step = 1
    curve_xs = np.arange(new_xs[0],new_xs[-1])
    
    # create the fitted normal curve with calculated p parameters    
    curve_fit_ys = normal_fit(curve_xs,p[0],p[1],p[2])
    
    # unroll fitted curve to original axes
    unrolled_ys = np.roll(curve_fit_ys, -(roll_degrees))    
    unrolled_xs = curve_xs + roll_degrees
    
    # package fitted curve for plotting    
    fit_curve = np.column_stack((unrolled_xs, unrolled_ys))
    
    return fit_curve

def run_analysis(trial_file, spikes_file, title_name):
    '''
    This function runs the complete analysis and plots the results.
    Allows you to batch up multiple runs in main program.    
    
    Input - 
        trials = direction and time when the monkey moved the joystick
        spk_times = processed data of detected neuron spike times 
    Output - 
        histogram and polar plot of data and fitted curve
    '''
    # load data files
    trials = load_experiment(trial_file)   
    spk_times = load_neuraldata(spikes_file) 
      
    # run analysis and create histogram
    direction_rates = bin_spikes(trials,spk_times,0.1)
    plot_tuning_curves(direction_rates, title_name)
    
    # fit data to a normal distribution
    fit_curve = roll_fit_unroll(direction_rates)
       
    # call plotting function
    plot_fits(direction_rates,fit_curve,
              title=title_name + ' - Fit')
    
    # determine preferred direction of neuron
    pd = preferred_direction(fit_curve)
    print ('preferred direction = ' + str(pd[0]))   
    
    
##########################
#You can put the code that calls the above functions down here    
if __name__ == "__main__":
#    trials = load_experiment('trials.npy')   
#    spk_times = load_neuraldata('example_spikes.npy') 
#      
#    # run analysis and create histogram
#    direction_rates = bin_spikes(trials,spk_times,0.1)
#    plot_tuning_curves(direction_rates, 'Example Neuron Tuning Curve')
#    
#    # fit data to a normal distribution
#    fit_curve = roll_fit_unroll(direction_rates)
#       
#    # call plotting function
#    plot_fits(direction_rates,fit_curve,
#              title='Example Neuron Tuning Curve - Fit')
#    
#    pd = preferred_direction(fit_curve)
#    print ('preferred direction = ' + str(pd[0]))

    run_analysis('trials.npy','example_spikes.npy',
                 'Example Neuron Tuning Curve')
    
    run_analysis('trials.npy','neuron1.npy',
                 'Neuron 1 Tuning Curve')
                 
    run_analysis('trials.npy','neuron2.npy',
                 'Neuron 2 Tuning Curve')
    
    run_analysis('trials.npy','neuron3.npy',
                 'Neuron 3 Tuning Curve')  