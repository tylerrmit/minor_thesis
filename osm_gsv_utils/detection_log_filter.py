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
    Class to filter a "detection_log.csv" to remove lonely outliers.
    If we detect a bicycle lane marking at a single point -- perhaps a single frame
    from the dash camera footage -- but there are no other detections nearby,
    we might want to filter it out to remove it as a suspected false positive.
    '''

    def __init__(self, detection_log):
        '''
        Parameters
        ----------
        detection_log : str
            The path to a CSV recording the original detections
        '''

        # Load the original "detection_log.csv" into memory
        self.detection_log  = detection_log
        
        self.df = pd.read_csv(detection_log)
        
        # Build list of hit coordinates from the detection log
        self.hit_coords = []
        for index, row in self.df.iterrows():
            self.hit_coords.append((row['lat'], row['lon']))
    
    
    def count_hits_within_range(self, this_lat, this_lon, min_range_from_hit, max_range_from_hit):
        '''
        For a given "hit" location (latitude/longitude), how many "supporting" hits are found
        nearby?
        
        To provide support, a nearby hit must be a maximum of max_range_from_hit metres away
        from the original coordinates.  But it must also be a minimum of min_range_from_hit
        metres away, to exclude multiple consecutive frames where the camera is stationary,
        perhaps stopped at traffic lights right when the false positive occurs.
        
        Parameters
        ----------
        this_lat : float
            The latitude of the point being assessed for adequate support
            
        this_lon : float
            The longitude of the point being assessed for adequate support
            
        min_range_from_hit : float
            The mimimum distance away from the original point that another "hit" must be to be considered support
            
        max_range_from_hit : float
            The maximum distance away from the original point that another "hit" can be to be considered support
            
        Returns:  the number of other "hits" that provided support
        '''
        hits_in_range_count = 0
        
        for index, comparison_row in self.df.iterrows():
            distance = geodesic((this_lat, this_lon), (comparison_row['lat'], comparison_row['lon'])).m
            
            if distance >= min_range_from_hit and distance <= max_range_from_hit:
                hits_in_range_count += 1
        
        return hits_in_range_count
        
        
    def apply_filter(self, output_file, min_hits_in_range=2, min_range_from_hit=10, max_range_from_hit=50):
        '''
        Filter the hits in the original "detection_log.csv" down to those with adequate support,
        based on the supplied parameters and using the count_hits_within_range method
        
        Parameters
        ----------
        
        output_file : str
            Path to where the output "filtered" version of "detection_log.csv" should be written
            
        min_hits_in_range : int, optional
            The minimum number of supporting hits required for a detection point to be included
            in the output
            
        min_range_from_hit : int, optional
            The minimum distance away from the original point that another "hit" must be to be considered support
            
        max_range_from_hit : int, optional
            The maximum distance away from the original point that another "hit" can be to be considered support
        '''
        
        # Open output file and write a header line
        with open(output_file, 'w') as f:
            f.write('lat,lon,bearing,heading,way_id_start,way_id,node_id,offset_id,score,bbox_0,bbox_1,bbox_2,bbox_3,orig_filename\n')
            
            # Process every record in the original "detection_log.csv" and determine whether it has
            # sufficient support to be included in the output
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
                
                # How much support did this "hit" have?
                hits_in_range_count = self.count_hits_within_range(lat, lon, min_range_from_hit, max_range_from_hit)
                
                if hits_in_range_count >= min_hits_in_range:
                    # This "hit" had adequate support, include it in the output
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
            
    @staticmethod       
    def convert_metadata_to_route_log(input_file, output_file):
        '''
        Convert a "metadata.csv" file (that was created when splitting a dash camera video into images)
        into the "detection_log.csv" format.
        
        This is useful because there is other code in the osm_walker module to take points in a "detection_log.csv"
        and align them to known points in the OpenStreetMap data, to compare and measure routes.
        
        So if we are dealing with data in "metadata.csv" format -- e.g. from a process where rather than
        running the images through a TensorFlow 2 detection model we have tried to detect paved shoulders
        with Canny/Hough -- then we can convert that output into "detection_log.csv" format, align to the
        OpenStreetMap database, and draw some maps.
        
        Parameters
        ----------
        input_file : str
            Path to the input "metadata.csv" file
            
        output_file : str
            Path to the output "detection_log.csv" file   
        '''
        
        # Read the input file into a pandas dataframe
        df = pd.read_csv(input_file)
        
        # Open the output file and write a header
        with open(output_file, 'w') as f:
            f.write('lat,lon,bearing,heading,way_id_start,way_id,node_id,offset_id,score,bbox_0,bbox_1,bbox_2,bbox_3,orig_filename\n')
            
            for index in trange(len(df.index)):
                row = df.iloc[[index]]
                
                lat           = row['lat'].item()
                lon           = row['lon'].item()
                bearing       = int(row['heading'].item())
                heading       = int(row['heading'].item())
                way_id_start  = 0
                way_id        = 0
                node_id       = 0
                offset_id     = row['frame_num'].item()
                score         = 1.0
                bbox_0        = 0.5
                bbox_1        = 0.5
                bbox_2        = 0.5
                bbox_3        = 0.5
                orig_filename = row['filename'].item()
                
                # Convert each record into the required output format, including placeholder values where required
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