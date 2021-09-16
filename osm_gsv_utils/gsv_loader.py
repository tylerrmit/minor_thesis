'''
Created on 06 Sep 2021

@author: Tyler Saxton
'''

import os
import numpy as np
import pandas as pd
import google_streetview.api

from tqdm.notebook import tqdm


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


    def save_batch(self, batch_filename, points):
        with open(batch_filename, 'w') as csv_file:
            csv_file.write('node_id,offset_id,lat,lon,bearing\n')

            for point in points:
                node_id   = point[5]
                offset_id = point[3]
                lat       = point[0]
                lon       = point[1]
                bearing   = point[2]

                if bearing < 0:
                    bearing = bearing + 360

                line = str(node_id) + ',' + str(offset_id) + ',' + str(lat) + ',' + str(lon) + ',' + str(bearing) + '\n'

                csv_file.write(line)

            csv_file.close()


    def load_batch(self, batch_filename, verbose=False):
        '''
        Parameters
        ----------
        batch_filename : str
            The name of a CSV file containing co-ordinates and bearings to download from Google Street View.
            If the file has already been downloaded, it is skipped, to save costs.
        '''
        df = pd.read_csv(batch_filename)

        # Iterate over every requested location in the batch
        for index, row in df.iterrows():
            #print(str(int(row['node_id'])))
            self.load_coordinates(
                str(int(row['node_id'])),
                str(int(row['offset_id'])),
                row['lat'],
                row['lon'],
                row['bearing'],
                verbose=verbose
            )


    def load_batch_progress(self, batch_filename, verbose=False):
        df = pd.read_csv(batch_filename)

        tqdm.pandas()

        df.progress_apply(lambda row: self.load_coordinates(
            str(int(row['node_id'])),
            str(int(row['offset_id'])),
            row['lat'],
            row['lon'],
            row['bearing'],
            verbose=verbose),
            axis=1
        )


    def load_coordinates(
            self,
            node_id,
            offset_id,
            lat,
            lon,
            bearing,
            heading_offsets = [0,90,180,270],
            fov             = 90,
            pitch           = -20,
            verbose=False
    ):
        location = str(lat) + ', ' + str(lon)

        for heading_offset in heading_offsets:
            heading = round(bearing + heading_offset)
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
            full_download_directory = os.path.join(self.download_directory, str(node_id), str(offset_id), str(int(heading)))

            if os.path.isdir(full_download_directory):
                if verbose:
                    print('{0:9s} {1:3s} {2:3d}: Using cached data'.format(str(node_id), str(offset_id), int(heading)))
            else:
                results = google_streetview.api.results(params)
                results.download_links(full_download_directory)
                if verbose:
                    print('{0:9s} {1:3s} {2:3d}: Downloaded'.format(str(node_id), str(offset_id), int(heading)))


