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
import pandas as pd

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

    df = pd.DataFrame(trials, columns=['angle', 'time'])
    df['cnt'] = df.time.apply(lambda x: get_cnt_in_bin(spk_times, x, time_bin))
    gb = df.groupby('angle')
    agg = gb.aggregate(lambda x: np.average(x)/(2*time_bin))
    agg['angle'] = agg.index

    return agg[['angle', 'cnt']].values

def get_cnt_in_bin(spk_times, time, time_bin):
    return len(spk_times[(spk_times >= time - time_bin) & (spk_times<= time + time_bin)])
    
def plot_tuning_curves(direction_rates, title):
    """
    This function takes the x-values and the y-values  in units of spikes/s 
    (found in the two columns of direction_rates) and plots a histogram and 
    polar representation of the tuning curve. It adds the given title.
    """

    ax1 = plt.subplot(2,2,1)
    ax1.set_title(title)

    x = direction_rates[:, 0]
    y = direction_rates[:, 1]

    ax1.bar(x, y, width=45 )
    ax1.axis([0, 360, 0, max(y) + max(y)*0.1])
    ax1.set_ylabel('Firing Rate (spikes/s)')
    ax1.set_xlabel('Direction of Motions (degrees)')


    ax2 = plt.subplot(2, 2, 2, polar=True)
    ax2.set_title(title)
    spikescount = np.append(y, y[0])
    theta = np.arange(0, 361, 45)*np.pi/180
    ax2.plot(theta, spikescount, label='Firing Rate (spikes/s)')
    ax2.legend(loc=8)

    
def roll_axes(direction_rates):
    """
    roll_axes takes the x-values (directions) and y-values (direction_rates)
    and return new x and y values that have been "rolled" to put the maximum
    direction_rate in the center of the curve. The first and last y-value in the
    returned list should be set to be the same. (See problem set directions)
    Hint: Use np.roll()
    """

    degrees = 45
    
    x = direction_rates[:, 0]
    y = direction_rates[:, 1]

    shift = 4 - np.argmax(y)
    new_ys = np.roll(y, shift)
    new_ys = np.append(new_ys, new_ys[0])

    roll_degrees = shift * degrees
    new_xs = x - roll_degrees
    new_xs = np.append(new_xs, new_xs[7] + degrees)

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
    max_y = np.amax(centered_y)
    max_x = centered_x[np.argmax(centered_y)]
    sigma = 90

    p, cov = optimize.curve_fit(normal_fit, centered_x, centered_y,
                                p0=[max_x, sigma, max_y])

    return p
    


def plot_fits(direction_rates,fit_curve,title):
    """
    This function takes the x-values and the y-values  in units of spikes/s 
    (found in the two columns of direction_rates and fit_curve) and plots the 
    actual values with circles, and the curves as lines in both linear and 
    polar plots.
    """
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
  
    return pd
    

def make_analysis(trials, spikes, title):
    trials = load_experiment(trials)   
    spk_times = load_neuraldata(spikes) 
    direction_rates =  bin_spikes(trials, spk_times, time_bin=0.1)
    plot_tuning_curves(direction_rates, title)


    new_xs, new_ys, roll_degrees = roll_axes(direction_rates)
    p = fit_tuning_curve(new_xs, new_ys)
    curve_xs = np.arange(new_xs[0], new_xs[-1])
    curve_fit_ys = normal_fit(curve_xs, p[0], p[1], p[2])
    unrolled_ys = np.roll(curve_fit_ys, -(roll_degrees))
    unrolled_xs = curve_xs + roll_degrees
    fit_curve = np.column_stack((unrolled_xs, unrolled_ys))
    plot_fits(direction_rates, fit_curve,
              title=title + ' - Fit')
    plt.tight_layout()
    plt.show()

##########################
#You can put the code that calls the above functions down here    
if __name__ == "__main__":
    make_analysis('trials.npy','neuron1.npy',
                 'Neuron 1 Tuning Curve')

    make_analysis('trials.npy','neuron2.npy',
                 'Neuron 2 Tuning Curve')

    make_analysis('trials.npy','neuron3.npy',
                 'Neuron 3 Tuning Curve')


