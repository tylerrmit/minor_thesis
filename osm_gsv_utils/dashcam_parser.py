'''
Created on 06 Sep 2021

@author: Tyler Saxton
'''

import os
import sys
import numpy as np
import pandas as pd

from pynmeagps import NMEAReader

from datetime import datetime, date

import cv2
import math

from pathlib import Path

from tqdm.notebook  import tqdm, trange


class dashcam_parser(object):
    '''
    A class used to load video and corresponding NMEA data from a dashcam
    '''

    def __init__(self, source_fps):
        '''
        blah
        '''
        self.source_fps  = source_fps

        self.ts_coords   = {}
        self.ts_altitude = {}
        self.ts_heading  = {}
        
    
    def split_videos(self, dir, output_dir, output_fps=5, suffix='MP4', verbose=False):
        filenames = os.listdir(dir)
        filenames_filtered = [x for x in filenames if x.endswith('.' + suffix)]
        
        for idx in trange(len(filenames_filtered)):
            filename = filenames_filtered[idx]
            
            prefix = os.path.splitext(filename)[0]
            if verbose:
                print(prefix)
            self.split_video(dir, prefix, output_dir, output_fps=output_fps, suffix=suffix)
    
    
    def split_video(self, dir, prefix, output_dir, output_fps=5, suffix='MP4', verbose=False):
        # Load NMEA data corresponding to the video
        self.load_nmea(dir, prefix)
           
        # Create output_dir if it does not already exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Open CSV file recording metadata
        metadata_filename = os.path.join(output_dir, 'metadata.csv')
        
        if not os.path.exists(metadata_filename):
            csv_file = open(metadata_filename, 'w')
            csv_file.write('filename,prefix,frame_num,lat,lon,altitude,heading\n')
        else:
            csv_file = open(metadata_filename, 'a')
        
        # Split video into frames
        cap = cv2.VideoCapture(os.path.join(self.dir, self.prefix + '.' + suffix))
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_num = 0
        
        #while True:
        for idx in trange(length, desc=prefix):
            ret, frame = cap.read()
            
            if ret == True:
                # Do we want to capture this frame?
                if frame_num % int(self.source_fps / output_fps) == 0:
                    # Determine how many (fractional) seconds we are into the video
                    nmea_index_f = float(frame_num) / float(self.source_fps)
                    
                    # Determine the key frames immediately before and after this frame
                    nmea_index_1 = math.floor(nmea_index_f)
                    nmea_index_2 = math.ceil(nmea_index_f)
                                           
                    # Proportion of the way through surrounding key frames 1 and 2
                    nmea_index_p = nmea_index_f - float(nmea_index_1)
                                        
                    # Protect against high index out of range
                    while nmea_index_2 > 0 and nmea_index_2 not in self.ts_coords.keys():
                        nmea_index_2 -= 1
                        
                    while nmea_index_1 > 0 and nmea_index_1 not in self.ts_coords.keys():
                        nmea_index_1 -= 1
                        
                    # Interpolate details between key frame values
                    coords1 = self.ts_coords[nmea_index_1]
                    coords2 = self.ts_coords[nmea_index_2]
                    
                    coords1_lat = coords1[0]
                    coords1_lon = coords1[1]
                    coords2_lat = coords2[0]
                    coords2_lon = coords2[1]
                    
                    nmea_index_1a = min(nmea_index_1, *(self.ts_altitude.keys()))
                    nmea_index_2a = min(nmea_index_2, *(self.ts_altitude.keys()))
                    
                    altitude1 = self.ts_altitude[nmea_index_1a]
                    altitude2 = self.ts_altitude[nmea_index_2a]
                    
                    # Correct for $GPRMC messages starting at second 1, ts_heading[0] is not defined
                    while nmea_index_1 not in self.ts_heading.keys() and nmea_index_1 < max(*(self.ts_heading.keys())):
                        nmea_index_1 += 1
                        
                    while nmea_index_2 not in self.ts_heading.keys() and nmea_index_2 < max(*(self.ts_heading.keys())):
                        nmea_index_2 += 1
                                            
                    heading1 = self.ts_heading[nmea_index_1]
                    heading2 = self.ts_heading[nmea_index_2]
                                        
                    interpolated_lat = coords1_lat + ((coords2_lat - coords1_lat) * nmea_index_p)
                    interpolated_lon = coords1_lon + ((coords2_lon - coords1_lon) * nmea_index_p)
                    
                    interpolated_altitude = altitude1 + ((altitude2 - altitude1) * nmea_index_p)
                    interpolated_heading  = heading1  + ((heading2  - heading1)  * nmea_index_p)
                    
                    if verbose:
                        print('{0:s} {1:4d} [{2:.7f}, {3:.7f}] {4:f} {5:f}'.format(
                            self.prefix,
                            frame_num,
                            interpolated_lat,
                            interpolated_lon,
                            interpolated_altitude,
                            interpolated_heading
                        ))

                    # Save output image
                    output_filename = os.path.join(output_dir, '{0:s}_{1:04d}.png'.format(self.prefix, frame_num))
                    
                    cv2.imwrite(output_filename, frame)
                    
                    # Log metadata to CSV
                    csv_file.write('{0:s},{1:s},{2:d},{3:.7f},{4:.7f},{5:f},{6:f}\n'.format(
                        output_filename,
                        self.prefix,
                        frame_num,
                        interpolated_lat,
                        interpolated_lon,
                        interpolated_altitude,
                        interpolated_heading
                    ))
                    
                frame_num += 1
            else:
                break
                
        cap.release()
        csv_file.close()
 
 
    def load_nmea(self, dir, prefix):
        self.dir = dir
        self.prefix = prefix

        self.ts_coords.clear()
        self.ts_altitude.clear()
        self.ts_heading.clear()
        
        nmea_filename = os.path.join(dir, prefix + '.NMEA')
        
        file = open(nmea_filename, encoding='utf-8')
        
        first_timestamp = None
        last_good_offset = None
        
        for line in file.readlines():
            if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                # Load generic fields that are common to both record types
                try:
                    msg = NMEAReader.parse(line)
                
                    # How many seconds since the first timestamp for the file is this?
                    if first_timestamp is None:
                        first_timestamp = msg.time
                
                    if msg.time >= first_timestamp:
                        ts_offset = (datetime.combine(date.today(), msg.time) - datetime.combine(date.today(), first_timestamp)).seconds
                    else:
                        ts_offset = (datetime.combine(date.today(), msg.time) - datetime.combine(date.today() - datetime.timedelta(days=1), first_timestamp)).seconds
                
                       
                    self.ts_coords[ts_offset] = [float(msg.lat), float(msg.lon)]

                    # Fetch additional fields that are specific to one record type
                    if line.startswith('$GPGGA'):
                        self.ts_altitude[ts_offset] = msg.alt
                    elif line.startswith('$GPRMC'):
                        self.ts_heading[ts_offset] = msg.cog
                        
                    last_good_offset = ts_offset
                        
                except Exception as e:
                    if last_good_offset is not None:
                        print('WARNING: processing line [{0:s}] [{1:s}], using previous values'.format(prefix, line))
                        self.ts_coords[last_good_offset+1] = self.ts_coords[last_good_offset]
                        self.ts_altitude[last_good_offset+1] = self.ts_altitude[last_good_offset]
                        self.ts_heading[last_good_offset+1] = self.ts_heading[last_good_offset]
                    else:
                        print('ERROR: procesing line [{0:s}] [{1:s}], no previous values'.format(prefix, line))
                        raise Exception("Error parsing NMEA")
                    
                    