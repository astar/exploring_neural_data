#!/usr/bin/python
#
#  NAME
#    problem_set1.py
#
#  DESCRIPTION
#    Open, view, and analyze raw extracellular data
#    In Problem Set 1, you will write create and test your own spike detector.
#

import numpy as np
import matplotlib.pylab as plt

def load_data(filename):
    """
    load_data takes the file name and reads in the data.  It returns two 
    arrays of data, the first containing the time stamps for when they data
    were recorded (in units of seconds), and the second containing the 
    corresponding voltages recorded (in units of microvolts - uV)
    """
    data = np.load(filename)[()];
    return np.array(data['time']), np.array(data['voltage'])
    
def bad_AP_finder(time,voltage):
    """
    This function takes the following input:
        time - vector where each element is a time in seconds
        voltage - vector where each element is a voltage at a different time
        
        We are assuming that the two vectors are in correspondance (meaning
        that at a given index, the time in one corresponds to the voltage in
        the other). The vectors must be the same size or the code
        won't run
    
    This function returns the following output:
        APTimes - all the times where a spike (action potential) was detected
         
    This function is bad at detecting spikes!!! 
        But it's formated to get you started!
    """
    
    #Let's make sure the input looks at least reasonable
    if (len(voltage) != len(time)):
        print "Can't run - the vectors aren't the same length!"
        APTimes = []
        return APTimes
    
    numAPs = np.random.randint(0,len(time))//10000 #and this is why it's bad!!
 
    # Now just pick 'numAPs' random indices between 0 and len(time)
    APindices = np.random.randint(0,len(time),numAPs)

    # By indexing the time array with these indices, we select those times
    APTimes = time[APindices]

    # Sort the times
    APTimes = np.sort(APTimes)

    return APTimes


def good_AP_finder(time,voltage):
    """
    This function takes the following input:
        time - vector where each element is a time in seconds
        voltage - vector where each element is a voltage at a different time
        
        We are assuming that the two vectors are in correspondance (meaning
        that at a given index, the time in one corresponds to the voltage in
        the other). The vectors must be the same size or the code
        won't run
   
    This function returns the following output:
        APTimes - all the times where a spike (action potential) was detected
    """

    #APTimes = []

    #Let's make sure the input looks at least reasonable
    if (len(voltage) != len(time)):
        print "Can't run - the vectors aren't the same length!"
        return APTimes


    idxs = (voltage - np.mean(voltage))/np.std(voltage) > 3
    potencials = time[idxs]
    JITTER = 0.001
    tmp = []
    for p in potencials:
        local_idxs = (time > p - JITTER) & (time < p + JITTER)
        local_time = time[local_idxs]
        local_voltage = voltage[local_idxs]
        tmp.append(local_time[local_voltage == max(local_voltage)][0] + 0.00024)

    APTimes = np.unique(tmp)

    return APTimes


def get_actual_times(dataset):
    """
    Load answers from dataset
    This function takes the following input:
        dataset - name of the dataset to get answers for

    This function returns the following output:
        APTimes - spike times
    """
    return np.load(dataset)


def detector_tester(APTimes, actualTimes):
    """
    returns percentTrueSpikes (% correct detected) and falseSpikeRate
    (extra APs per second of data)
    compares actual spikes times with detected spike times
    This only works if we give you the answers!
    """
    
    JITTER = 0.025 #2 ms of jitter allowed
    
    #first match the two sets of spike times. Anything within JITTER_MS
    #is considered a match (but only one per time frame!)
    
    #order the lists
    detected = np.sort(APTimes)
    actual = np.sort(actualTimes)
    
    #remove spikes with the same times (these are false APs)
    temp = np.append(detected, -1)
    detected = detected[plt.find(plt.diff(temp) != 0)]
 
    #find matching action potentials and mark as matched (trueDetects)
    trueDetects = [];
    for sp in actual:
        z = plt.find((detected >= sp-JITTER) & (detected <= sp+JITTER))
        if len(z)>0:
            for i in z:
                zz = plt.find(trueDetects == detected[i])
                if len(zz) == 0:
                    trueDetects = np.append(trueDetects, detected[i])
                    break;
    percentTrueSpikes = 100.0*len(trueDetects)/len(actualTimes)
    
    #everything else is a false alarm
    totalTime = (actual[len(actual)-1]-actual[0])
    falseSpikeRate = (len(APTimes) - len(actualTimes))/totalTime
    
    print 'Action Potential Detector Performance performance: '
    print '     Correct number of action potentials = ' + str(len(actualTimes))
    print '     Percent True Spikes = ' + str(percentTrueSpikes)
    print '     False Spike Rate = ' + str(falseSpikeRate) + ' spikes/s'
    print 
    return {'Percent True Spikes':percentTrueSpikes, 'False Spike Rate':falseSpikeRate}
    
    
def plot_spikes(time,voltage,APTimes,titlestr):
    """
    plot_spikes takes four arguments - the recording time array, the voltage
    array, the time of the detected action potentials, and the title of your
    plot.  The function creates a labeled plot showing the raw voltage signal
    and indicating the location of detected spikes with red tick marks (|)
    """
    plt.figure()
    plt.title(titlestr)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (uV)")
    plt.plot(time, voltage)
    #plt.axhline(np.mean(voltage), color='green')
    #plt.axhline(3*np.std(voltage), color='green')
    #plt.axhline(-3*np.std(voltage), color='green')

    for x in APTimes:
        plt.axvline(x=x, ymin=0.9, linewidth=1, color='red')

    plt.show()
    
def plot_waveforms(time,voltage,APTimes,titlestr):
    """
    plot_waveforms takes four arguments - the recording time array, the voltage
    array, the time of the detected action potentials, and the title of your
    plot.  The function creates a labeled plot showing the waveforms for each
    detected action potential
    """
    APRange = 3e-3

    plt.figure()
    plt.title(titlestr)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (uV)")
    
    for APTime in APTimes:

        APRange = 3e-3
        start = APTime - APRange 
        stop = APTime + APRange

        y = voltage[(time > start) & (time < stop)]
        x = np.linspace(-APRange, APRange, len(y))
        plt.plot(x, y)

    plt.show()
    

        
##########################
#You can put the code that calls the above functions down here    
if __name__ == "__main__":
    t,v = load_data('spikes_easy_test.npy')    
    #actualTimes = get_actual_times('spikes_easy_test_answers.npy')
    APTime = good_AP_finder(t,v)
    plot_spikes(t,v,APTime,'Action Potencials in Raw signal from Easy Test Data Set')
    plot_waveforms(t,v,APTime,'Waveforms from Easy Test Data Set')
    #detector_tester(APTime,actualTimes)


