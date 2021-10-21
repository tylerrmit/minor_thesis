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
    
    Videos are split into images in the "split" subfolder, and a "metadata.csv" file is
    created to list each frame in chronological order, along with its metadata such as
    latitude, longitude, heading, altitude, and any other attributes derived by a lane
    detection engine.
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
        Parameters
        ----------
        
        source_fps : int
            The number of frames per second expected in the source video file(s)
            
        lane_detector : lane_detection, optional
            Optional object to perform lane detection on the images as they are loaded, and include
            the results in the output "metadata.csv" file
        
        write_lane_images : boolean, optional
            Specify whether we want to write a copy of the images to the "lanes" directory,
            with a lane detection overlay
            
        left_lane_mask_top : int, optional
            The y-coordinates, in pixels, where the top of any mask should be, whether detecting
            the vehicle's own lane, or a paved shoulder further to the left.
            Should be set based on a typical vanishing point for the road in the image.
            
        left_lane_mask_bottom : int, optional
            The y-coordinate, in pixels, where the bottom of the any mask should be, whether detecting
            the vehicle's own lane, or a paved shoulder further to the left.
            Should be set based on where any visible obstruction from the bonnet of the camera vehicle
            ends.
            
        left_lane_mask_margin : int, optional
            The x-coordinate, in pixels, from which the mask should start from the left hand side of
            the frame, when trying to detect a paved shoulder to the left of the camera vehicle's own
            lane.
        
        own_lane_left_prop : float, optional
            How much space on the left hand side of the frame should be excluded by the mask, when
            looking for the camera vehicle's own lane, specified as a proportion of the total frame width
            
        own_lane_right_prop : float, optional
            How much space on the right hand side of the frame should be excluded by the mask, when
            looking for the camera vehicle's own lane, specified as a proportion of the total frame width
            
        centre_offset : int, optional
            How far many pixels from the centre of the frame is the centre of the camera vehicle.
            This is used to derive where the apex of the detection mask should be, on the horizontal
            axis, when trying to detect the vehicle's own lane
            
        min_slope : float, optional
            Lines that are detected in the image must have a slope coefficient with an absolute value
            of at least this much, to be considered when trying to identify a lane boundary from the average
            of a group of detected lines.  This can eliminate noise from very horizontal lines that are 
            clearly nothing to do with a lane boundary.
            
        max_slope : float, optional
            Lines that are detected in the image must have a slope coefficient with an absolute value
            of now more than this much, to be considered when trying to identify a lane boundary from the average
            of a group of detected lines.  This can eliminate noise from very vertical lines that are
            when considering the lane boundary.  Default value 0 means do not apply a maximum threshold.
            
        own_lane_x_min : int, optional
            Deprecated option, currently does nothing
            
        own_lane_y_min : int, optional
            Deprecated option, currently does nothing
            
        intercept_bottom : int, optional
            The y-coordinate, in pixels, of an arbitrary "bottom" horizontal line that is drawn on the image
            to judge the width of any detected lanes as close as possible to the camera vehicle.  Should be
            set to just above the end of any obstruction from the vehicle bonnet in the image.  The purpose
            of this horizontal line is to provide an easy frame of reference when reviewing a series of images
            in chronological order, and judging the width and stability of any detected paved shoulder,
            and the camera vehicle's own lane.
            
        intercept_top : int, optional
            The y-coordinate, in pixels, of an arbitrary "top" horizontal line that is drawn on the image
            to judge the width of any detected lanes further up the image, towards the vanishing point of the
            road.  The purpose is similar to intercept_top, to provide a frame of reference for assessing the
            width and stability of any detected paved shoulder across multiple images.
            
        frame_width : int, optional
            Expected width of the video footage, in pixels
            
        frame_height : int, optional
            Expected height of the video footage, in pixels
        '''
        # Save any parameters that were passed to the object on instantiation, for later reference
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
        
        # Dictionaries to cache data from the NMEA file for each timestamp/second
        self.ts_coords   = {} # Latitude/longitude per second offset within the video
        self.ts_altitude = {} # Altitude
        self.ts_heading  = {} # Heading
              
        # Calculate the mask to use when detecting the vehicle's own lane, based
        # on other supplied parameters
        self.own_lane_vertices = [
            (int(self.own_lane_left_prop * self.frame_width), self.left_lane_mask_bottom),
            (self.frame_width/2 + self.centre_offset, self.left_lane_mask_top),
            (int(self.own_lane_right_prop * self.frame_width), self.left_lane_mask_bottom)
        ]
        
    
    def split_videos(self, dir, output_dir, output_fps=5, suffix='MP4', verbose=False):
        '''
        Process all videos in a directory
        
        Parameters
        ----------
        
        dir : str
            Path to the directory where all MP4 video files to be processed are held
            
        output_dir : str
            Path to a base output directory where all outputs will be writen
            
        output_fps : int, optional
            The number of frames per second to extract from the video files.
            E.g. if the source videos were recorded at 60 fps, and we decide we only
            need 5 fps, then only every 12th frame will be extracted, the rest will
            be ignored.
            
        suffix : str, optional
            Suffix of the video files to process, usually "MP4"
            
        verbose : boolean, optional
            Specify whether debug messages should be written to STDOUT
        '''
        # Find all the video files to process
        filenames = os.listdir(dir)
        filenames_filtered = [x for x in filenames if x.endswith('.' + suffix)]
        
        # Initialize an output CSV with metadata about each image, and write a header line
        metadata_filename = os.path.join(output_dir, 'metadata.csv')
        csv_file = open(metadata_filename, 'w')
        csv_file.write('filename,prefix,frame_num,lat,lon,altitude,heading,pixels_bottom,pixels_top,left_slope2,left_int2,left_slope1,left_int1,right_slope1,right_int1\n')
        csv_file.close()

        # Process each video file in turn
        # They will be processed in order based on the filename
        # This will be chronological order for the Navman camera that was used in the experiment
        for idx in trange(len(filenames_filtered)):
            filename = filenames_filtered[idx]
            
            # Find the filename prefix without the extension, because we want to process both the video file
            # and the accompanying NMEA file
            prefix = os.path.splitext(filename)[0]
            
            # Process the video file
            if verbose:
                print(prefix)
            self.split_video(dir, prefix, output_dir, output_fps=output_fps, suffix=suffix)
    
    
    def split_video(self, dir, prefix, output_dir, output_fps=5, suffix='MP4', verbose=False):
        '''
        Process an individual video file, splitting it into images at the required output
        frames per second rate, and correlating the image to the accompanying NMEA metadata
        based on the frame number
        
        Parameters
        ----------
        dir : str
            Directory where the video and NMEA file is found
            
        prefix : str
            Filename prefix that is used to find both the video file (with specified suffix)
            and the NMEA file with the same prefix
            
        output_dir : str
            Output directory where any output will be written, including the split image
            files and "metadata.csv"
            
        output_fps : int, optional
            The required output frames per second, if we don't need as many frames per second
            as there are in the original video file
            
        suffix : str, optional
            Suffix to use to find the video file from the dir and prefix
            
        verbose : boolean, optional
            Specify whether debug messages should be written to STDOUT
        '''
        # Load NMEA data corresponding to the video
        self.load_nmea(dir, prefix)
           
        # Create output_dir if it does not already exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # If we are writing output images with a lane detection overlay to a "lanes"
        # subdirectory, make sure that directory exists
        if self.write_lane_images:
            lane_dir = os.path.join(output_dir, 'lanes')
            Path(lane_dir).mkdir(parents=True, exist_ok=True)
        
        # Open CSV file to record metadata for each image
        metadata_filename = os.path.join(output_dir, 'metadata.csv')
        
        # If the medtadata file does not already exist, initialize it with a header record,
        # otherwise open it for appending
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
        
        # Process each frame in turn, and display a progress bar in a Jupyter Notebook
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
                    
                    # Detect lanes, if we initialised this dashcam_parser object with a lane_detector
                    if self.lane_detector is not None:
                        # Correct the image for lens distortion
                        corrected_image = self.lane_detector.correct_image(frame)
                        
                        # Find the vehicle's own lane, and any paved shoulder
                        detected_lanes_image, slopes_and_intercepts = self.lane_detector.detect_lanes(
                            corrected_image,
                            self.own_lane_vertices,
                            left_lane_mask_top    = self.left_lane_mask_top,
                            left_lane_mask_bottom = self.left_lane_mask_bottom,
                            left_lane_mask_margin = self.left_lane_mask_margin,
                            min_slope             = self.min_slope,
                            max_slope             = self.max_slope
                        )
                        
                        # Get the points at which the detected lane boundaries intercept the arbitrary
                        # horizontal lines at intercept_bottom and intercept_top
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
                        
                        # Calculate the width in pixels of any paved shoulder at those top and bottom horizontal lines
                        pixel_width_bottom = self.lane_detector.pixel_width(left_intersection_x1, own_l_intersection_x1)
                        pixel_width_top    = self.lane_detector.pixel_width(left_intersection_x2, own_l_intersection_x2)
                        
                        # Break out the list of slopes and intercepts for the detected lane boundaries into components
                        left_slope2, left_int2, left_slope1, left_int1, right_slope1, right_int1 = slopes_and_intercepts
                        
                        # If we have been asked to output a copy of the image with a lane detection overlay, do that now
                        if self.write_lane_images:
                            grid_image, intersection_list2 = self.lane_detector.draw_intersection_grid(
                                detected_lanes_image,
                                slopes_and_intercepts,
                                self.intercept_bottom,
                                top=self.intercept_top
                            )
                            
                            cv2.imwrite(os.path.join(lane_dir, '{0:s}_{1:04d}.png'.format(self.prefix, frame_num)), grid_image)
                    else:
                        # If we were not asked to apply a lane_detection model, then just fill in placeholder values
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

                    # Save output image (the original version without any overlay, in case we want to run another model on it)
                    output_filename = os.path.join(output_dir, '{0:s}_{1:04d}.png'.format(self.prefix, frame_num))
                    
                    cv2.imwrite(output_filename, frame)
                    
                    # Log image metadata to CSV
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
        '''
        Load the NMEA file and cache the location data for each timestamp/second within the video
        '''
        self.dir    = dir
        self.prefix = prefix

        self.ts_coords.clear()
        self.ts_altitude.clear()
        self.ts_heading.clear()
        
        nmea_filename = os.path.join(dir, prefix + '.NMEA')
        
        file = open(nmea_filename, encoding='utf-8')
        
        first_timestamp = None
        last_good_offset = None
        
        # Read the NMEA file one line at a time
        for line in file.readlines():
            # Only read the interesting NMEA record types
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
                
                    # Store the latitude/longitude based on the time offset
                    self.ts_coords[ts_offset] = [float(msg.lat), float(msg.lon)]

                    # Fetch additional fields that are specific to one record type, and store them, too
                    if line.startswith('$GPGGA'):
                        self.ts_altitude[ts_offset] = msg.alt
                    elif line.startswith('$GPRMC'):
                        self.ts_heading[ts_offset] = msg.cog
                        
                    last_good_offset = ts_offset
                        
                except Exception as e:
                    # If we could not read a line properly, just re-use the last good values if we can, and report a warning
                    if last_good_offset is not None:
                        print('WARNING: processing line [{0:s}] [{1:s}], using previous values'.format(prefix, line))
                        self.ts_coords[last_good_offset+1] = self.ts_coords[last_good_offset]
                        self.ts_altitude[last_good_offset+1] = self.ts_altitude[last_good_offset]
                        self.ts_heading[last_good_offset+1] = self.ts_heading[last_good_offset]
                    # If there were no previous good values to fall back on, report an error
                    else:
                        print('ERROR: procesing line [{0:s}] [{1:s}], no previous values'.format(prefix, line))
                        raise Exception("Error parsing NMEA")
                    
                    