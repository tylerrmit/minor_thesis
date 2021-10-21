'''
Created on 06 Sep 2021

@author: Tyler Saxton
'''

import os
import numpy as np
import pandas as pd
import google_streetview.api

from tqdm.notebook import tqdm

from datetime import datetime
import shutil


class gsv_loader(object):
    '''
    A class used to load and cache Google Street View images based on a batch file of required locations
    '''

    def __init__(self, apikey_filename, download_directory):
        '''
        Parameters
        ----------
        apikey_filename : str
            The name of a file containing the API key for connecting to Google Street View
            Either a full path, or assume it's relative to the current working directory
            
        download_directory : str
            Path to the directory where Google Street View images will be downloaded/cached
        '''
        self.download_directory = download_directory

        with open(apikey_filename) as key_file:
            self.api_key = key_file.readline()
            key_file.close
            
        self.cache_hits = 0
        self.cache_miss = 0


    def write_batch_file(self, batch_filename, points, limit=0):
        '''
        Given a list of points, write them to a batch file
        
        Parameters
        ----------
        
        batch_filename : str
            Path to the output batch file
            
        points : list of lists
            List of points to write to the batch file in the required format
            
        limit : int
            Set a maximum number of records that will be written, to ensure that
            the batch file is not so big that we cannot afford Google's fees to
            download all the images
        '''
        
        # Backup any existing version of the output file
        if os.path.exists(batch_filename):
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = batch_filename.replace('.csv', backup_timestamp + '.csv')
            print('Backup {0:s} to {1:s}'.format(batch_filename, backup_filename))
            shutil.copyfile(batch_filename, backup_filename)
                 
        # Create the new batch file
        print('Write {0:s}'.format(batch_filename))
        with open(batch_filename, 'w') as csv_file:
            csv_file.write('lat,lon,bearing,image_path,way_start_id,way_id,node_id,offset_id,way_name\n')

            for idx, point in enumerate(points):
                if limit > 0 and idx >= limit:
                    break
                
                # Each "point" is actually a list, parse the fields based on their expected order
                # Because we did not bother to make a friendlier custom data structure for this
                lat          = point[0]
                lon          = point[1]
                bearing      = int(round(point[2])) # Force bearings to be integers, fractional headings don't make enough difference to be useful
                offset_id    = point[3]
                way_start_id = point[4]
                way_id       = point[5]
                node_id      = point[6]
                way_name     = point[7]

                # Make sure the bearing is between 0 and 360, for consistency when storing/retrieving with the cache
                if bearing < 0:
                    bearing = int(round(bearing + 360))

                # Calculate the expected output path for the image
                # This will be checked to see if we already have it, before actually downloading from Google
                image_path = os.path.join(
                    self.download_directory,
                    '{0:.6f}'.format(lat),
                    '{0:.6f}'.format(lon),
                    str(int(bearing)),
                    'gsv_0.jpg'
                )
                
                # Write the output record to the batch
                csv_file.write('{0:.6f},{1:.6f},{2:d},{3:s},{4:s},{5:s},{6:s},{7:d},{8:s}\n'.format(lat, lon, bearing, image_path, way_start_id, way_id, node_id, offset_id, way_name))

            # Save and close the batch file
            # We process it as a separate step, to give us a chance to assess the batch file
            # and estimate the costs involved in actually downloading it, before we proceed
            csv_file.close()
            

    def process_batch_file(self, batch_filename, progress=False, verbose=False):
        '''
        Take a batch file, and actually download any images we need from Google Street View,
        where we do not already have a copy in our cache
        
        Parameters
        ----------
        batch_filename : str
            The name of a CSV file containing co-ordinates and bearings to download from Google Street View.
            
        progress : boolean, optional
            Specify whether a progress bar should be displayed in a Jupyter Notebook as we work our way through the list
            
        verbose : boolean, optional
            Specify whether debug messages should be written to STDOUT
        '''
        
        # Keep statistics on how many images we found in the cache, and how many we had to actually download
        self.cache_hits = 0
        self.cache_miss = 0
        
        # Load the batch file into a pandas dataframe
        df = pd.read_csv(batch_filename)

        # Iterate over every requested location in the batch
        if progress:
            tqdm.pandas()   

            df.progress_apply(lambda row: self.load_coordinates(
                row['lat'],
                row['lon'],
                row['bearing'],
                verbose=verbose),
                axis=1
            )
        else:
            for index, row in df.iterrows():
                self.load_coordinates(
                    row['lat'],
                    row['lon'],
                    row['bearing'],
                    verbose=verbose
                )

        print('GSV Cache Hits: {0:10d} Misses: {1:10d}'.format(self.cache_hits, self.cache_miss))


    def load_coordinates(
            self,
            lat,
            lon,
            bearing,
            heading_offsets = [0,90,180,270],
            fov             = 90,
            pitch           = -20,
            verbose         = False
    ): 
        '''
        Given a single image request from the batch file, check the cache and then
        download it if we do not already have it
        
        Parameters
        ----------
        lat : float
            Latitude to be downloaded
        
        lon : float
            Longitude to be downloaded
            
        bearing : int
            "Forward" heading for the route at the location
        
        heading_offsets : list, optional
            List of offsets to be added to the bearing to download a series of images for a 360 degree viewitems
            
        fov : int, optional
            Field of view (in degrees) to request from the Google Street View API
            
        pitch : int, optional
            Angle of the camera relative to a horizontal plane, in degrees.  A negative number will point
            the camera more towards the ground.
            
        verbose : boolean, optional
            Whether debug messages should be written to STDOUT
        '''
        
        # Construct the location field for the request from the latitude and longitude
        location = str(lat) + ', ' + str(lon)

        # Make a request for each offset in heading_offset, to get 360 degree coverage
        for heading_offset in heading_offsets:
            heading = int(round(bearing + heading_offset))
            if heading > 360:
                heading = heading - 360

            # Construct the parameters for the Google Street View API request
            params = [{
                'key'      : self.api_key,
                'size'     : '640x640',
                'location' : location,
                'fov'      : str(fov),
                'pitch'    : str(pitch),
                'heading'  : str(heading)
            }]
        
            # Check if we have already downloaded anything
            full_download_directory = os.path.join(
                self.download_directory,
                '{0:.6f}'.format(lat),
                '{0:.6f}'.format(lon),
                str(int(heading))
            )

            if os.path.isdir(full_download_directory):
                # We already have it in our cache
                self.cache_hits += 1
                if verbose:
                    print('{0:9s} {1:3s} {2:3d}: Using cached data'.format(str(lat), str(lon), int(heading)))
            else:
                # We do not have the image already, download it from Google
                self.cache_miss += 1
                results = google_streetview.api.results(params)
                results.download_links(full_download_directory)
                if verbose:
                    print('{0:9s} {1:3s} {2:3d}: Downloaded'.format(str(lat), str(lon), int(heading)))      
          