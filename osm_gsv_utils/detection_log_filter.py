'''
Created on 06 Sep 2021

@author: Tyler Saxton
'''

import json
import os
import sys

import pandas as pd

from geopy.distance import geodesic

from tqdm.notebook import tqdm, trange



class detection_log_filter(object):
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
        
        # Build list of hit coordinates from the detection log
        self.hit_coords = []
        for index, row in self.df.iterrows():
            self.hit_coords.append((row['lat'], row['lon']))
    
    
    def count_hits_within_range(self, this_lat, this_lon, min_range_from_hit, max_range_from_hit):
        hits_in_range_count = 0
        
        for index, comparison_row in self.df.iterrows():
            distance = geodesic((this_lat, this_lon), (comparison_row['lat'], comparison_row['lon'])).m
            
            if distance >= min_range_from_hit and distance <= max_range_from_hit:
                hits_in_range_count += 1
        
        return hits_in_range_count
        
        
    def apply_filter(self, output_file, min_hits_in_range=2, min_range_from_hit=10, max_range_from_hit=50):
        with open(output_file, 'w') as f:
            f.write('lat,lon,bearing,heading,way_id_start,way_id,node_id,offset_id,score,bbox_0,bbox_1,bbox_2,bbox_3,orig_filename\n')
            
            for index in trange(len(self.df.index)):
                row = self.df.iloc[[index]]
                
                lat           = row['lat'].item()
                lon           = row['lon'].item()
                bearing       = row['bearing'].item()
                heading       = row['heading'].item()
                way_id_start  = row['way_id_start'].item()
                way_id        = row['way_id'].item()
                node_id       = row['node_id'].item()
                offset_id     = row['offset_id'].item()
                score         = row['score'].item()
                bbox_0        = row['bbox_0'].item()
                bbox_1        = row['bbox_1'].item()
                bbox_2        = row['bbox_2'].item()
                bbox_3        = row['bbox_3'].item()
                orig_filename = row['orig_filename'].item()
                
                hits_in_range_count = self.count_hits_within_range(lat, lon, min_range_from_hit, max_range_from_hit)
                
                if hits_in_range_count >= min_hits_in_range:
                    f.write('{0:.6f},{1:.6f},{2:d},{3:d},{4:d},{5:d},{6:d},{7:f},{8:f},{9:f},{10:f},{11:f},{12:f},{13:s}\n'.format(
                        lat,
                        lon,
                        bearing,
                        heading,
                        way_id_start,
                        way_id_start,
                        node_id,
                        offset_id,
                        score,
                        bbox_0,
                        bbox_1,
                        bbox_2,
                        bbox_3,
                        orig_filename
                    ))
      
            f.close()  