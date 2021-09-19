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
    A class used to load and cache Google Street View images in batch
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
        # Backup any existing version of the file
        if os.path.exists(batch_filename):
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = batch_filename.replace('.csv', backup_timestamp + '.csv')
            print('Backup {0:s} to {1:s}'.format(batch_filename, backup_filename))
            shutil.copyfile(batch_filename, backup_filename)
                 
        # Create the new batch file
        print('Write {0:s}'.format(batch_filename))
        with open(batch_filename, 'w') as csv_file:
            csv_file.write('lat,lon,bearing,image_path,way_start_id,way_id,node_id,offset_id\n')

            for idx, point in enumerate(points):
                if limit > 0 and idx >= limit:
                    break
                    
                lat          = point[0]
                lon          = point[1]
                bearing      = int(round(point[2]))
                offset_id    = point[3]
                way_start_id = point[4]
                way_id       = point[5]
                node_id      = point[6]

                if bearing < 0:
                    bearing = int(round(bearing + 360))

                image_path = os.path.join(
                    self.download_directory,
                    '{0:.6f}'.format(lat),
                    '{0:.6f}'.format(lon),
                    str(int(bearing)),
                    'gsv_0.jpg'
                )
                csv_file.write('{0:.6f},{1:.6f},{2:d},{3:s},{4:s},{5:s},{6:s},{7:d}\n'.format(lat, lon, bearing, image_path, way_start_id, way_id, node_id, offset_id))

            csv_file.close()


    def process_batch_file(self, batch_filename, progress=False, verbose=False):
        '''
        Parameters
        ----------
        batch_filename : str
            The name of a CSV file containing co-ordinates and bearings to download from Google Street View.
            If the file has already been downloaded, it is skipped, to save costs.
        '''
        self.cache_hits = 0
        self.cache_miss = 0
        
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
        location = str(lat) + ', ' + str(lon)

        for heading_offset in heading_offsets:
            heading = int(round(bearing + heading_offset))
            if heading > 360:
                heading = heading - 360

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
                self.cache_hits += 1
                if verbose:
                    print('{0:9s} {1:3s} {2:3d}: Using cached data'.format(str(lat), str(lon), int(heading)))
            else:
                self.cache_miss += 1
                results = google_streetview.api.results(params)
                results.download_links(full_download_directory)
                if verbose:
                    print('{0:9s} {1:3s} {2:3d}: Downloaded'.format(str(lat), str(lon), int(heading)))      
          