
# coding: utf-8

# In[134]:

import copy
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.basemap import Basemap, solar
from pyhdf.SD import SD, SDC 
from scipy import stats

'''
THESE ARE FUNCTIONS WHICH ARE USEFUL FOR THE NAVIGATION
OF HDF FILES USING PYHDF.

These functions have been last tested with just two types of files: 
    MYD08_D3
    MYD09CMG

Use getSDS to open the files and access the SDS's, then use binavg to
create a 2 x 2 degree table of binned averages of these values. This can
be exported as an ascii table using ascii.write().
'''

def getSDS(path,sds_list):
    
    '''
    Extracts desired Scientific Datasets (SDS's) from HDF file.
    
    Inputs: path - path to a single HDF file
            sds_list - array containing names of the SDS's you would like
    Outputs: hdf - the opened hdf file
             sds - array containing all of the SDS's you would like
    '''

    hdf = SD(path, SDC.READ)   #open hdf file
    
    sdslen = len(sds_list)     #find how many SDS's to open
    sds = []
    
    for i in range(sdslen):
        sel = hdf.select(sds_list[i])
        fill = hdf.select(sds_list[i]).attributes()['_FillValue']
        scale = hdf.select(sds_list[i]).attributes()['scale_factor']
        
        if scale < 1:
            arr = np.array(sel[:,:]) * scale    # Multiply by scale value to obtain useful values,
            arr[arr == fill*scale] = None       # replace fill values with None, or np.nan.
            
        elif scale > 1:                         # If scale value is meant to be divided by,
            arr = np.array(sel[:,:]) / scale    # divide by scale value to obtain useful values,
            arr[arr == fill/scale] = None       # replace fill values with None, or np.nan.
        
        sds.append(arr)       
        
    return hdf, sds 

def avg2(arr, res):
    '''
    Averages every two values in each row of an array. Written for the next function, binavg.
    
    input: arr (np.array) - array of size (N,M)
           res (float) -  resolution of data, used to get proper multiplier to make data into 2x2 deg res.
               ex: if data is res = 1, then mult will be 2
               Only works with data of res x res resolution
               
    output: array of size (N, M/mult)
    ''' 
    height = np.shape(arr)[0]   # find height and length of arr
    length = np.shape(arr)[1]
    
    mult = int(2 / res)   # mult is what must be multiplied to reach resolution of 2 x 2
    
    bins = []
    
    for i in np.arange(0,height,1):    # looks at each individual row
        
        binned = []                    # and creates an empty list to create bins from

        for j in np.arange(0,length,mult):   # looks at columns whose interval is given by mult
            
            catch = []                       # and creates an empty list to add each value in these intervals to
            
            for k in np.arange(j,j+mult,1):
                catch.append(arr[i][k])      
                             
            binned.append(np.nanmean(catch))   # finds mean of all values within a bin

        bins.append(binned)   # and puts this mean in a list of all of the bins. This is our new row
    
    bins = np.array(bins)
    
    return bins

def binavg(arr, res):
    '''
    This function takes a 2d-array, reshapes to 90 x 180, and averages the values
    which fall in these larger bins.
    
    input: arr - 2d-array of size (M,N)
           res (float) -  resolution of data, used to get proper multiplier to make data into 2x2 deg res.
               ex: if data is res = 1, then mult will be 2
               Only works with data of Y x Y resolution
           
    output: 2d-array of size (M/mult, N/mult), where mult = 2/res

    '''
    
    bins1 = avg2(arr, res)
    bins2 = avg2(bins1.T, res).T
    
    return bins2

def submatrx_mode(arr,res):
    
    '''
    Rebins a 2-D array to specified resolutuoin, finds the mode of the values which all within 
    each bin, and fills each bin with this value.
    
    inputs: arr (np.array) - 2D array of size (M,N)
            res (float) -  resolution of data, used to get proper multiplier to make data into 2x2 deg res.
            
    outputs: new_array - 2D array of size (M/mult,N/mult) containing modes of the values within each bin
    
    NOTE - when stats.mode() finds more than 1 mode, it returns the smallest value of mode
    '''

    mult = int(2/res)

    height = np.shape(arr)[0]   #find length and height of array
    length = np.shape(arr)[1]

    rowmarks = np.arange(0,height+mult,mult)   #set new bin edges
    colmarks = np.arange(0,length+mult,mult)
    
    # this is where the values are rebinned and the mode is taken - edit: added list comprehension
    new_array = [[stats.mode(
                             arr[np.ix_(
                                        np.arange( rowmarks[i] , rowmarks[i+1] ), np.arange( colmarks[j] , colmarks[j+1] )
                                         )], axis=None)[0][0]
                  
                                  for j in range(len(colmarks) - 1)]
                                  for i in range(len(rowmarks) - 1)]
        
    return new_array

def modismap(oldarr):
    
    '''
    Takes a 2D array of integers, maps them to a new set of integers
    
    New values follow this integer naming convention:
    
    0, NaN - no data (-9999 from MODIS)
    1 - ocean   (0 from MODIS)
    2 - forest     (1, 2, 3, 4, 5, 6, 11, 14 from MODIS)
    3 - grass    (7, 8, 9, 10, 12 from MODIS)
    4 - sand    (13, 16 from MODIS)
    5 - ice       (15 from MODIS)  
    
    inputs: oldarr - 2d array of old values (90x180)
            
    outputs: arr - array with updated values
    '''
    
    arr = copy.deepcopy(oldarr)   # create copy of original array
    
    for i in range(len(arr)):   # loop through each of the 90 rows
        
        for j, val in enumerate(arr[i]):   # loop through each of the 180 values in each row
            
            if val == 0:     # ocean
                
                arr[i][j] = 1
            
            if ((val >= 1 and val <=6) or val == 11 or val == 14):    # forest
            
                arr[i][j] = 2
                
            if ((val >= 7 and val <= 10) or val == 12):    # grass
                
                arr[i][j] = 3
                
            if (val == 13 or val == 16):    # sand
                
                arr[i][j] = 4
                
            if val == 15:    # ice
                
                arr[i][j] = 5
    
    return arr

def terminator(date):
    
    '''
    inputs: date - solar.datetime object
    
    outputs: numpy masked array of size 90 x 180. Two values: night = 1, day = NaN
    '''

    lons2, lats2, daynight = solar.daynight_grid(date, 2, -180, 179)   # gets grid of day/night values, night = 1, day = NaN

    daynight = np.delete(daynight, 90, 0)   # gets daynight to be 90x180
    
    daynight = daynight[::-1]               # corrects lat/lon to same as MODIS data
    
    return daynight


def rebindat(data, bins):
    
    '''
    rebins your data grid - keeping shape, but changing values
    
    inputs: data (np.array) - numpy array of NxN
            bins (list) - list of bin edges used for rebinning data values
            
    outputs: new_data - numpy array of NxN, but with new values
    '''
    
    new_data = copy.deepcopy(data)
    
    for i in range(len(new_data)):
        for j in range(len(new_data[i])):
            
            if np.isnan(new_data[i][j]) != True:
                
                num = min(bins, key=lambda x:abs(x-data[i][j]))    # finds closest bin edge to value

                new_data[i][j] = num   # replaces original value with this bin edge value in array
            
    return new_data


def landtypes(landfile, snowfile, landsds_name, snowsds_name, daynight, in_res = 0.05):
    
    '''
    inputs: landfile (str) - path to HDF file containing MODIS land cover data (MCD12C1)
            snowfile (str) - path to HDF file containing MODIS snow cover data (MYD10C1)
            landsds_name (str) - name of desired SDS from MODIS land cover file
            snowsds_name (str) - name of desired SDS from MODIS snow cover file
            daynight - numpy masked array (output of local function func.terminator)
            in_res - resolution of input data in degrees
            
    output - landcover - 90 x 180 numpy array of integers showing land type (from func.modismap)
    '''
    
    itdat = range(90)      # used for iterating through cot and ctp data
    itdat2 = range(180)
    

    #----------SNOW DATA-----------------#
    snowhdf = SD(snowfile, SDC.READ) 
    snowsds = snowhdf.select(snowsds_name)
    snowvals = np.array(snowsds.get())


    #----------LAND DATA-----------------#
    landhdf = SD(landfile, SDC.READ) 
    landsds = landhdf.select(landsds_name)
    landvals = np.array(landsds.get())

    landmodes = submatrx_mode(landvals, in_res)   # rebins yearly data to 2x2
    snowmodes = submatrx_mode(snowvals, in_res)   # rebins daily data to 2x2

    landcover = modismap(landmodes)   # remaps integer values the to 5 values of land cover in our model

    for i in range(len(snowmodes[0:74])):
        for j in range(len(snowmodes[i])):                          # this is the world above lat -60 deg.
            if (snowmodes[i][j] > 0) and (snowmodes[i][j] <= 100):

                landcover[i][j] = 5   # this changes snow values to value corresponding to snow in land data

    for i in range(len(daynight)):
        for j in range(len(daynight[i])):
            if daynight[i][j] == 1.0:
                landcover[i][j] = np.nan    # this changes values of unlit parts of earth to NaN

    west_edge = range(0,26)
    east_edge = range(115,180)    # these are the edges of the Earth's disk as seen from the moon
                                  # on the night of observation at 21:20 UT

    # makes values None outside of lon range

    for i in itdat:
        for j in west_edge:
            landcover[i][j] = np.nan    # this changes values of unseen parts of earth to NaN

    for i in itdat:
        for j in east_edge:
            landcover[i][j] = np.nan    # this changes values of unseen parts of earth to NaN
            
    return landcover


def cot_ctp(cloudfile, sdslist, ctpbins, cotbins, daynight, is_ctp = True):
    
    '''
    Use this function to extract data from MYD08_D3 files - particularly cloud top pressure (CTP)
    and cloud optical thickness (COT) data. This function can be used to extract SDS's, but is 
    geared toward these two since these are the primary SDS's this Earthshine project uses.
    Produces 2 x 2 degree numpy arrays of global COT and CTP for which one can choose how the 
    values are binned. This array can then be plotted or saved as an ascii table by the user    
    
    INPUTS: cloudfile: path to HDF file containing MODIS cloud profile data
            sdslist: name of desired SDS from MODIS cloud profile file - [cot, ctp]
            ctpbins: list of bin edges for values of CTP
            cotbins: list of bin edges for values of COT
            daynight: numpy masked array (output of local function func.terminator)
            is_ctp: if True, then converts units of ctp from hPa to bar. Set to False if getting
                    data for an SDS which is not CTP
            
    OUTPUTS: cot: 90 x 180 numpy array containing rebinned COT data
             ctp: 90 x 180 numpy array containing rebinned CTP data
    '''
    
    # opens hdf file, gets the SDS's
    hdf, sds = getSDS(cloudfile, sdslist)

    cot = sds[0]   # cloud optical thickness
    ctp = sds[1]   # cloud top pressure
    
    itdat = range(90)   # used for iterating through cot and ctp data
    itdat2 = range(180)
    
    # rebins array shape of both to 2 x 2 degrees
    cot = binavg(cot,1) 
    ctp = binavg(ctp,1)   

    if is_ctp == True:    
        ctp *= 0.001   # converts from hPa to bars
    
    cot = rebindat(cot, cotbins)
    ctp = rebindat(ctp, ctpbins)   # rebins ctp values to desired bins    
    
    west_edge = range(0,26)     # west edge of earth's visible disk from moon
    east_edge = range(115,180)  # east edge of earth's visible disk from moon

    # makes values None outside of lon range

    for i in itdat:
        for j in west_edge:
            cot[i][j] = np.nan
            ctp[i][j] = np.nan    # this changes values of unseen parts of earth to NaN

    for i in itdat:
        for j in east_edge:
            cot[i][j] = np.nan
            ctp[i][j] = np.nan    # this changes values of unseen parts of earth to NaN
    
    for i in range(len(daynight)):
        for j in range(len(daynight[i])):
            if daynight[i][j] == 1.0:
                cot[i][j] = np.nan
                ctp[i][j] = np.nan    # this changes values of unlit parts of earth to NaN
                
    return cot, ctp


def unique_combos(arr1, arr2, arr3):
    
    '''
    Takes 3 arrays of the same shape, returns an array containing lists of each unique 
    combination of the three arrays at each place in the arrays.
    
    INPUTS: arr1, arr2, arr3 - 2D numpy arrays 
    
    OUTPUTS: combos - list containing lists (each of length 3) of each unique combination
    '''
    
    itdat = range(len(arr1))
    itdat2 = range(len(arr1[0]))      # these are the lengths of array dimensions, used for looping
    
    arr1_new = copy.deepcopy(arr1)    # create copies of arrays to manipulate
    arr2_new = copy.deepcopy(arr2)
    arr3_new = copy.deepcopy(arr3)
    
    for i in itdat:                   # This loop changes NaN values to -9999.0, since np.nan != np.nan
        for j in itdat2:                         # This is necessary for finding unique combinations.
            if np.isnan(arr1_new[i][j]) == True:
                arr1_new[i][j] = -9999.0
            if np.isnan(arr2_new[i][j]) == True:
                arr2_new[i][j] = -9999.0
            if np.isnan(arr3_new[i][j]) == True:
                arr3_new[i][j] = -9999.0
    
    combo_list = [[arr1_new[i][j], arr2_new[i][j], arr3_new[i][j]]
                                for i in itdat
                                for j in itdat2]    # array of same shape as input arrays, but where each value is
                                                    # a list containing the value of each input array at this point.
    
    combos = []
    
    for i in combo_list:
        if i not in combos:
            combos.append(i)    # only adds unique combos of each three arrays at one point to this list.

    return combos