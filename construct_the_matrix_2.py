from obspy import read # obspy is the package used to access the seismic data (.mseed files)
from obspy import *
import matplotlib.pyplot as plt # to plot everything
from scipy.stats import kurtosis, skew  # for statistical calculations

df = pd.DataFrame(feat_matrix, columns = features, index=dates)

dir = '/Users/Jose/Desktop/Big_Data/Seismic_2010_2020'
df2 = df.copy()

for i in os.listdir(dir):
    try:
        st = read('{}/{}'.format(dir,i))
    except:
        pass
    else:
        st.detrend('linear')
        st.detrend('demean')  
    
        for j in range(len(st)):
            if st[j].stats.channel == 'BHE':
                tr = st[j]        
                date = str(tr.stats.starttime)[0:10]
                name = str(tr.stats.station)        
                for i in range(8,13):
                    filt = tr.copy()
                    filt.filter(type='bandpass',freqmin=i, freqmax=i+1)
                    #
                    col_kurt = 'kurtosis_' + name + '_Band{}-{}Hz'.format(i,i+1)
                    df2.loc[str(date),str(col_kurt)] = kurtosis(filt.data)
                    #
                    col_var = 'var_' + name + '_Band{}-{}Hz'.format(i,i+1)
                    df2.loc[str(date),str(col_var)] = np.var(filt.data)
                    #
                    col_vr = 'ValRan_' + name + '_Band{}-{}Hz'.format(i,i+1)
                    df2.loc[str(date),str(col_vr)] = np.ptp(filt.data)
                    #
                    col_sk = 'skew_' + name + '_Band{}-{}Hz'.format(i,i+1)
                    df2.loc[str(date),str(col_sk)] = skew(filt.data) 
                    
# Now that we finished to fill up the matrix with features values
# we need to do the same with GPS data