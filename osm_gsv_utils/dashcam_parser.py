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

    def __init__(
        self,
        source_fps,
        lane_detector         = None,
        write_lane_images     = False,
        left_lane_mask_top    = 375,
        left_lane_mask_bottom = 640,
        left_lane_mask_margin = 80,
        own_lane_left_prop    = 2/10,
        own_lane_right_prop   = 2/3,
        centre_offset         = -100,
        min_slope             = 0.3,
        max_slope             = 0,
        own_lane_x_min        = 500,
        own_lane_y_min        = 400,
        intercept_bottom      = 640,
        intercept_top         = 486,
        frame_width           = 1920,
        frame_height          = 1080
    ):
        '''
        blah
        '''
        self.source_fps            = source_fps
        self.lane_detector         = lane_detector
        self.write_lane_images     = write_lane_images
        self.left_lane_mask_top    = left_lane_mask_top
        self.left_lane_mask_bottom = left_lane_mask_bottom
        self.left_lane_mask_margin = left_lane_mask_margin
        self.own_lane_left_prop    = own_lane_left_prop
        self.own_lane_right_prop   = own_lane_right_prop
        self.centre_offset         = centre_offset
        self.min_slope             = min_slope
        self.max_slope             = max_slope
        self.own_lane_x_min        = own_lane_x_min
        self.own_lane_y_min        = own_lane_y_min
        self.intercept_bottom      = intercept_bottom
        self.intercept_top         = intercept_top
        self.frame_width           = frame_width
        self.frame_height          = frame_height
        
        self.ts_coords   = {}
        self.ts_altitude = {}
        self.ts_heading  = {}
                
        self.own_lane_vertices = [
            (int(self.own_lane_left_prop * self.frame_width), self.left_lane_mask_bottom),
            (self.frame_width/2 + self.centre_offset, self.left_lane_mask_top),
            (int(self.own_lane_right_prop * self.frame_width), self.left_lane_mask_bottom)
        ]
        
    
    def split_videos(self, dir, output_dir, output_fps=5, suffix='MP4', verbose=False):
        filenames = os.listdir(dir)
        filenames_filtered = [x for x in filenames if x.endswith('.' + suffix)]
        
        # Initialize output CSV and write header
        metadata_filename = os.path.join(output_dir, 'metadata.csv')
        csv_file = open(metadata_filename, 'w')
        csv_file.write('filename,prefix,frame_num,lat,lon,altitude,heading,pixels_bottom,pixels_top,left_slope2,left_int2,left_slope1,left_int1,right_slope1,right_int1\n')
        csv_file.close()

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
        
        if self.write_lane_images:
            lane_dir = os.path.join(output_dir, 'lanes')
            Path(lane_dir).mkdir(parents=True, exist_ok=True)
        
        # Open CSV file recording metadata
        metadata_filename = os.path.join(output_dir, 'metadata.csv')
        
        if not os.path.exists(metadata_filename):
            csv_file = open(metadata_filename, 'w')
            csv_file.write('filename,prefix,frame_num,lat,lon,altitude,heading,pixels_bottom,pixels_top,left_slope2,left_int2,left_slope1,left_int1,right_slope1,right_int1\n')
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
                    
                    # Detect lanes
                    if self.lane_detector is not None:
                        corrected_image = self.lane_detector.correct_image(frame)
                        
                        detected_lanes_image, slopes_and_intercepts = self.lane_detector.detect_lanes(
                            corrected_image,
                            self.own_lane_vertices,
                            left_lane_mask_top    = self.left_lane_mask_top,
                            left_lane_mask_bottom = self.left_lane_mask_bottom,
                            left_lane_mask_margin = self.left_lane_mask_margin,
                            min_slope             = self.min_slope,
                            max_slope             = self.max_slope
                        )
                        
                        left_intersection_x1, own_l_intersection_x1, own_r_intersection_x1 = self.lane_detector.find_intersection_list(
                            detected_lanes_image,
                            slopes_and_intercepts,
                            self.intercept_bottom
                        )
                        
                        left_intersection_x2, own_l_intersection_x2, own_r_intersection_x2 = self.lane_detector.find_intersection_list(
                            detected_lanes_image,
                            slopes_and_intercepts,
                            self.intercept_top
                        )
                        
                        pixel_width_bottom = self.lane_detector.pixel_width(left_intersection_x1, own_l_intersection_x1)
                        pixel_width_top    = self.lane_detector.pixel_width(left_intersection_x2, own_l_intersection_x2)
                        
                        left_slope2, left_int2, left_slope1, left_int1, right_slope1, right_int1 = slopes_and_intercepts
                        
                        if self.write_lane_images:
                            grid_image, intersection_list2 = self.lane_detector.draw_intersection_grid(
                                detected_lanes_image,
                                slopes_and_intercepts,
                                self.intercept_bottom,
                                top=self.intercept_top
                            )
                            
                            cv2.imwrite(os.path.join(lane_dir, '{0:s}_{1:04d}.png'.format(self.prefix, frame_num)), grid_image)
                    else:
                        pixel_width_bottom = 0
                        pixel_width_top    = 0
                        left_slope2        = 0
                        left_int2          = 0
                        left_slope1        = 0
                        left_int1          = 0
                        right_slope1       = 0
                        right_int1         = 0
                            
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
                    csv_file.write('{0:s},{1:s},{2:d},{3:.7f},{4:.7f},{5:f},{6:f},{7:.2f},{8:.2f},{9:s},{10:s},{11:s},{12:s},{13:s},{14:s}\n'.format(
                        output_filename,
                        self.prefix,
                        frame_num,
                        interpolated_lat,
                        interpolated_lon,
                        interpolated_altitude,
                        interpolated_heading,
                        pixel_width_bottom,
                        pixel_width_top,
                        str(left_slope2),
                        str(left_int2),
                        str(left_slope1),
                        str(left_int1),
                        str(right_slope1),
                        str(right_int1)
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
                    
                    