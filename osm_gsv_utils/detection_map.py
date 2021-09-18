'''
Created on 06 Sep 2021

@author: Tyler Saxton
'''

import json
import os
import sys

import pandas as pd

from tqdm.notebook import tqdm


class detection_map(object):
    '''
    TODO class description
    '''

    def __init__(self, detection_log):
        '''
        Parameters
        ----------
        detection_log : str
            The path to a CSV recording the detections
        '''

        self.detection_log  = detection_log
        
        self.df = pd.read_csv(detection_log)
        
        
    def write_geojson(self, output_geojson, mode, progress=False):
        self.features = []
        
        if mode == 'Point':
            if progress:
                tqdm.pandas()   

                self.df.progress_apply(lambda row: self.write_point(
                    row['lat'],
                    row['lon']),
                    axis=1
                )
            else:
                for index, row in self.df.iterrows():
                    self.write_point(
                        row['lat'],
                        row['lon']
                    )
        else:
            print('Unrecognised mode: ' + str(mode))
            return
        
        featurecollection = {
            'type':      'FeatureCollection',
            'features': self.features
        }
        
        # Write output file
        print('Writing to: ' + output_geojson)
        with open(output_geojson, 'w') as outfile:
            json.dump(featurecollection, outfile, indent=4)
            
            outfile.close()
                
 
    def write_point(self, lat, lon):
        feature = {
            'type': 'Feature',
            'geometry': {
                'type':        'Point',
                'coordinates': [lat, lon]
            }
        }
        
        self.features.append(feature)