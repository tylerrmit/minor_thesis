'''
Created on 06 Sep 2021

@author: Tyler Saxton
'''

import os
import sys

import cv2
import numpy as np

import pickle

from matplotlib import pyplot as plt

from tqdm.notebook import tqdm, trange

import warnings
warnings.simplefilter('ignore', np.RankWarning)


class lane_detection(object):
    '''
    A class used to "walk" routes in an OSM extract and create a list of sample points and bearings

    '''

    def __init__(self, calibration_config=None, model_dir=None, width_estimation_model=None):
        '''
        Parameters
        ----------
        TODO
        '''

        # Camera calibration matrices
        self.camera_matrix = None
        self.dist_matrix   = None
               
        if calibration_config is not None:
            self.calibrate(calibration_config)
            
        # Width estimation model
        self.model_poly = None
        self.model_pol  = None
        
        if model_dir is not None and width_estimation_model is not None:
            self.load_width_estimation_model(model_dir, width_estimation_model)
            
            
                    
    def calibrate(self, calibration_config):
        f = cv2.FileStorage(calibration_config, cv2.FILE_STORAGE_READ)
        
        self.camera_matrix = f.getNode('K').mat()
        self.dist_matrix   = f.getNode('D').mat()
        
        f.release()
    
    
    def load_width_estimation_model(self, model_dir, width_estimation_model):
        self.model_poly = pickle.load(open(os.path.join(model_dir, width_estimation_model + '_poly.csv'), 'rb'))
        self.model_pol  = pickle.load(open(os.path.join(model_dir, width_estimation_model + '_pol.csv'),  'rb'))
        
        
    def predict_width(self, image_in, left_intersection, right_intersection):
        if self.model_poly is None or self.model_pol is None:
            return None
            
        if left_intersection is None or right_intersection is None:
            return 0
        
        # Calculate offsets from the centre of frame_height
        frame_height = image_in.shape[0]
        frame_width  = image_in.shape[1]
        
        left_offset  = abs((frame_width/2) - left_intersection)
        right_offset = abs((frame_width/2) - right_intersection)
        
        predictions = self.model_pol.predict(self.model_poly.fit_transform([[left_offset], [right_offset]]))
        
        predicted_width = predictions[0] - predictions[1]
        
        return predicted_width
    
    
    def pixel_width(self, left_intersection, right_intersection):
        if left_intersection is None or right_intersection is None:
            return 0
            
        return abs(right_intersection - left_intersection)
        
        
    def load_image(self, path):
        original_image = cv2.imread(path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        return self.correct_image(original_image)

            
    def correct_image(self, image_in):
        if self.camera_matrix is not None and self.dist_matrix is not None:
            return cv2.undistort(image_in, self.camera_matrix, self.dist_matrix, None, None)
        else:
            return image_in
            
            
    def apply_mask(self, image_in, mask_vertices):
        # Initialise a "canvas" to draw the mask on
        mask = np.zeros_like(image_in)
        
        # Determine how many channels we need to mask, e.g. 1 for greyscale, 3 for RGB
        # Create a "color" to fill in the mask with the correct number of channels
        if len(image_in.shape) < 3:
            mask_color = 255
        else:
            channel_count = image_in.shape[2]      
            mask_color = (255,) * channel_count
        
        # Fill in the mask inside the vertices
        cv2.fillPoly(mask, np.array([mask_vertices], np.int32), mask_color)
        
        # Apply the mask to the input image
        masked_image = cv2.bitwise_and(image_in, mask)
        
        # Return the masked image
        return masked_image
    
    def apply_canny(self, image_in, mask_vertices=None, Gaussian_ksize=(5, 5), Gaussian_sigmaX=0, Canny_threshold1=100, Canny_threshold2=200):
        # Convert input image to greyscale
        image_grey = cv2.cvtColor(image_in, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur operation
        image_blur = cv2.GaussianBlur(image_grey, Gaussian_ksize, Gaussian_sigmaX)
        
        # Apply Canny operation
        image_canny = cv2.Canny(image_blur, Canny_threshold1, Canny_threshold2)
        
        # Mask the image, if requested
        if mask_vertices:
            image_canny_masked = self.apply_mask(image_canny, mask_vertices)
            return image_canny_masked
        else:
            return image_canny
            
    def apply_hough(self, image_in_canny, image_in_orig, rho=2, theta=np.pi/60, threshold=100, minLineLength=10, maxLineGap=100, color=(255,255,255), thickness=3):
        # Generate lines
        lines = cv2.HoughLinesP(
            image_in_canny,
            rho           = rho,
            theta         = theta,
            threshold     = threshold,
            lines         = np.array([]),
            minLineLength = minLineLength,
            maxLineGap    = maxLineGap
        )
                
        if lines is not None:
            return lane_detection.draw_lines(image_in_orig, lines.squeeze(), color=color, thickness=thickness)
            
        else:
            return np.zeros((image_in.shape[0], image_in.shape[1], 3), dtype=np.uint8)
            
            
    @staticmethod
    def draw_lines(image_in, lines, image_background=None, color=(255, 0, 0), thickness=3): 
        if image_background is None:
            image_lines = np.zeros((image_in.shape[0], image_in.shape[1], 3), dtype=np.uint8)
        else:
            image_lines = np.copy(image_background)
            
        for line in lines:            
            if line is not None and len(line) >= 4:
                x1 = line[0]
                y1 = line[1]
                x2 = line[2]
                y2 = line[3]
                cv2.line(image_lines, (x1, y1), (x2, y2), (255, 255, 255), thickness=3)        
                
        return image_lines

    def detect_lanes(
        self,
        image_in,
        own_lane_vertices,
        left_lane_mask_top    = 375,
        left_lane_mask_bottom = 640,
        left_lane_mask_margin = 50,
        Canny_threshold1      = 50,
        Canny_threshold2      = 150,
        Hough_rho             = 2,
        Hough_theta           = np.pi/60,
        Hough_threshold       = 100,
        Hough_minLineLength   = 10,
        Hough_maxLineGap      = 100,
        height_limit          = 9/20,
        color_own_lane        = (255, 0, 0),
        color_left_lane       = (0, 255, 0),
        min_slope             = 0.30,
        max_slope             = 0,
    ):
        # Detect first round of lines, focussing on our own lanes first
        canny_image1 = self.apply_canny(
            image_in,
            mask_vertices    = own_lane_vertices,
            Canny_threshold1 = Canny_threshold1,
            Canny_threshold2 = Canny_threshold2
        )
            
        # Generate lines
        lines1 = cv2.HoughLinesP(
            canny_image1,
            rho           = Hough_rho,
            theta         = Hough_theta,
            threshold     = Hough_threshold,
            lines         = np.array([]),
            minLineLength = Hough_minLineLength,
            maxLineGap    = Hough_maxLineGap
        )

        # Obtain average left and right lines, along with their slopes and intercepts
        left_line1, right_line1, left_slope1, left_int1, right_slope1, right_int1 = self.find_average_lines(image_in, lines1, height_limit=height_limit, min_slope=min_slope, max_slope=max_slope)
        
        # Draw our "own lane" lines on a black background
        #own_lane_lines = np.array([averaged_lines1[0], averaged_lines1[1]])
        own_lane_lines = [left_line1, right_line1]
        own_lane_image = self.draw_lines(image_in, own_lane_lines, color=color_own_lane)
        
        # Overlay the detected lines onto the input image
        lanes_image = cv2.addWeighted(image_in, 0.8, own_lane_image,  1, 1)
            
        # Look for a lane further to the left if and only if we found a left margin in the first place
        if left_slope1 is not None:
        
            # Define a new area of interest to focuss on the next lane over to the left,
            # masking off an area just outside our own lane that was just detected
            frame_height = image_in.shape[0]
            frame_width  = image_in.shape[1]
        
            # Derive mask vertices to look for next lane over, derived from the slope and intercept of the
            # left boundary of our own lane, already identified.
        
            # left_lane_mask_top    = 375 = upper bound
            # left_lane_mask_bottom = 690 = lower bound
            # left_lane_mask_margin = 50  = allowance to exclude previously identified left boundary of own lane
        
            left_region_vertices = [
                (0, left_lane_mask_bottom),
                (int(frame_width/2), left_lane_mask_top),
                (int(round((left_lane_mask_bottom - left_int1 + left_lane_mask_margin) / left_slope1)), left_lane_mask_bottom)
            ]
        
            # Use Canny detection again using this new mask
            canny_image2 = self.apply_canny(
                image_in,
                mask_vertices    = left_region_vertices,
                Canny_threshold1 = Canny_threshold1,
                Canny_threshold2 = Canny_threshold2
            )
            
            # Generate lines
            lines2 = cv2.HoughLinesP(
                canny_image2,
                rho           = Hough_rho,
                theta         = Hough_theta,
                threshold     = Hough_threshold,
                lines         = np.array([]),
                minLineLength = Hough_minLineLength,
                maxLineGap    = Hough_maxLineGap
            )
        
            # Obtain average left and right lines, along with their slopes and intercepts
            # This time, the right line will be None, we ignore it
            left_line2, right_line2, left_slope2, left_int2, right_slope2, right_int2 = self.find_average_lines(image_in, lines2, height_limit=height_limit, min_slope=min_slope, max_slope=max_slope)
        
            # Draw the left line for the next lane over, on a black background
            #left_lane_lines = np.array([averaged_lines2[0]])
            left_lane_lines = [left_line2]
            left_lane_image = self.draw_lines(image_in, left_lane_lines, color=color_left_lane)
        
            # Overlay the detected lines onto the input image
            lanes_image = cv2.addWeighted(lanes_image, 0.8, left_lane_image, 1, 1)
        else:
            left_slope2 = None
            left_int2   = None
        
        # Assemble a list of slopes and intercepts for all three lines
        slopes_and_intercepts = [left_slope2, left_int2, left_slope1, left_int1, right_slope1, right_int1]
                
        return lanes_image, slopes_and_intercepts
        
    
    def find_average_lines(self, image_in, lines, height_limit=9/20, min_slope=0.30, max_slope=0, verbose=False):
        left  = []
        right = []

        if verbose:
            print('Processing {0:d} lines'.format(len(lines)))
            
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                
                # Fit a line to the two points, return the slope and y-intercept
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                y_int = parameters[1]
                
                # Sort lines into left vs right, based on the polarity of the slope
                if (slope < 0) and (abs(slope) >= min_slope) and (abs(slope) <= max_slope or max_slope == 0):
                    left.append((slope, y_int))
                    
                    if verbose:
                        print('Left slope {0:f}'.format(slope))
                        
                elif (slope > 0) and (abs(slope) >= min_slope) and (abs(slope) <= max_slope or max_slope == 0):
                    right.append((slope, y_int))
                    
                    if verbose:
                        print('Right slope {0:f}'.format(slope))
                    
                else:
                    if verbose:
                        print('Excluding slope {0:f}'.format(slope))
    
        if verbose:
            print('Left:  ' + str(left))
            print('Right: ' + str(right))
    
        # Take the average slope and intercept for lines of each side
        if len(right) > 0:
            right_avg   = np.average(right, axis=0)
            right_slope = right_avg[0]
            right_int   = right_avg[1]
            right_line  = self.find_line_ends(image_in, right_avg, height_limit)
        else:
            right_avg   = None
            right_slope = None
            right_line  = None
            right_int   = None
        
        if len(left) > 0:
            left_avg   = np.average(left, axis=0)
            left_slope = left_avg[0]
            left_int   = left_avg[1]
            left_line  = self.find_line_ends(image_in, left_avg, height_limit)
        else:
            left_avg   = None
            left_slope = None
            left_line  = None
            left_int   = None
                
        #return np.array([left_line, right_line]), left_slope, left_int, right_slope, right_int
        return left_line, right_line, left_slope, left_int, right_slope, right_int
   
    
    def find_line_ends(self, image_in, average, height_limit=9/20):
        if average is None:
            return None
    
        slope, y_int = average
        y1 = image_in.shape[0]
        
        # How far up the height of the image do we want to extend the lines before they disappear?  E.g. 9/20ths of the image
        y2 = int(y1 * height_limit)
        
        # Determine algebraically
        x1 = int((y1 - y_int) // slope)
        x2 = int((y2 - y_int) // slope)
        
        # Return and array with the two points
        #return np.array([x1, y1, x2, y2])
        return [x1, y1, x2, y2]
    
    
    def draw_intersection_grid(self, image_in, slopes_and_intercepts, horizontal_line_height, color=(255,255,255), thickness=3):
        # Overlay a grid showing where the slopes and intercepts of previously identified lane markings
        # intersect with a horizontal line at a certain height within the image
        # In case this is useful to model lane width based on a fixed height within the image
        
        left_intersection_x, own_l_intersection_x, own_r_intersection_x = self.find_intersection_list(image_in, slopes_and_intercepts, horizontal_line_height)
        
        # Draw the grid
        grid_image = np.copy(image_in)
        
        frame_height = image_in.shape[0]
        frame_width  = image_in.shape[1]
        
        # Draw horizontal line
        cv2.line(grid_image, (0, horizontal_line_height), (frame_width, horizontal_line_height), color, thickness=thickness)
        
        # Draw intersecting vertical lines
        if left_intersection_x is not None:
            cv2.line(grid_image, (int(left_intersection_x),  0), (int(left_intersection_x),  frame_height), color, thickness=thickness)
        if own_l_intersection_x is not None:
            cv2.line(grid_image, (int(own_l_intersection_x), 0), (int(own_l_intersection_x), frame_height), color, thickness=thickness)
        if own_r_intersection_x is not None:
            cv2.line(grid_image, (int(own_r_intersection_x), 0), (int(own_r_intersection_x), frame_height), color, thickness=thickness) 
        
        intersection_list = [left_intersection_x, own_l_intersection_x, own_r_intersection_x]
        
        return grid_image, intersection_list
        
        
    def find_intersection_list(self, image_in, slopes_and_intercepts, horizontal_line_height):
        # Decode slopes_and_intercepts
        left_slope2, left_int2, left_slope1, left_int1, right_slope1, right_int1 = slopes_and_intercepts
                
        # Find x-coordinates where each line intercepts a horizontal line at horizontal_line_height
        if left_slope2 is not None:
            left_intersection_x  = (horizontal_line_height - left_int2)  / left_slope2
        else:
            left_intersection_x  = None
        
        if left_slope1 is not None:
            own_l_intersection_x = (horizontal_line_height - left_int1)  / left_slope1
        else:
            own_l_intersection_x = None
            
        if right_slope1 is not None:
            own_r_intersection_x = (horizontal_line_height - right_int1) / right_slope1
        else:
            own_r_intersection_x = None
                
        return left_intersection_x, own_l_intersection_x, own_r_intersection_x
        
        
    