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

# Suppress a warning that routinely occurs when running some code in this module
# when fitting regression models to estimate the width of a lane
import warnings
warnings.simplefilter('ignore', np.RankWarning)


class lane_detection(object):
    '''
    A class used to apply Canny/Hough techniques to identify the camera vehicle's own lane
    and any paved shoulder, for a single frame.
    
    The outputs are the slopes and intercepts that describe the lane boundary "lines" that
    are detected, and these can be used to create an overlay to visualize the lanes over the
    top of the original image.
    
    To create a map of routes where a paved shoulder exists, look at the outputs for a
    sequence of frames captured along the route to get an overall impression of whether a
    paved shoulder exists or not.  This can help account for "noise" e.g. where an object
    on the side of the road briefly make it appear that there is a paved shoulder, where
    there is none.
    '''

    def __init__(self, calibration_config=None, model_dir=None, width_estimation_model=None):
        '''
        Parameters
        ----------
        calibration_config : str, optional
            Path to a .yml file containing an OpenCV model to correct for the effect of lens
            distortion, custom calibrated to the camera that produced the footage
            
        model_dir : str, optional
            Path to a directory where a regression model to estimate lane width from pixels
            has been pre-saved.  This feature was used to explore if we could get an accurate
            estimation of the width of a paved shoulder, however it was not included in the
            final scope of the research project.  An opportunity for further work, preferably
            after improving lane detection using deep learning to detect the road boundary,
            instea of just Canny/Hough.  Default of None means this feature is disabled.
            
        width_estimation_model : str, optional
            Name of the pre-saved regression model files within model_dir, to use to estimate
            real-world lane widths.  See model_dir.  Not used in the final project.
            Default of None means this feature is disabled.
        '''

        # Load the OpenCV model that will correct for the camera's lens distortion,
        # if required.
        self.camera_matrix = None
        self.dist_matrix   = None
               
        if calibration_config is not None:
            self.calibrate(calibration_config)
            
        # Load a pre-saved polynomial regression model to estimate real-world lane widths,
        # if required.
        self.model_poly = None
        self.model_pol  = None
        
        if model_dir is not None and width_estimation_model is not None:
            self.load_width_estimation_model(model_dir, width_estimation_model)
            
                           
    def calibrate(self, calibration_config):
        '''
        Load the OpenCV model that will correct for the camera's lens distortion
        
        Parameters
        ----------
        calibration_config : str
            Path to the pre-saved .yml file where calibration details were saved, as per the
            OpenCV camera calibration process
        '''
        f = cv2.FileStorage(calibration_config, cv2.FILE_STORAGE_READ)
        
        self.camera_matrix = f.getNode('K').mat()
        self.dist_matrix   = f.getNode('D').mat()
        
        f.release()
    
    
    def load_width_estimation_model(self, model_dir, width_estimation_model):
        '''
        Load a pre-saved polynomial regression model to estimate real-world lane widths
        from the x-coordinates (in pixels) of the left and right intersection points
        between the lane boundaries and a pre-configured horizontal line just above the
        bonnet of the camera vehicle.  If the horizontal line is moved, the regression
        model would need to be re-built.
        
        Parameters
        ----------
        model_dir : str
            Directory where the regression model files are saved
            
        width_estimation_model : str
            Filename prefix for the model to load.  A matching pair of "_poly.csv" and "_pol.csv"
            files are required
        '''
        self.model_poly = pickle.load(open(os.path.join(model_dir, width_estimation_model + '_poly.csv'), 'rb'))
        self.model_pol  = pickle.load(open(os.path.join(model_dir, width_estimation_model + '_pol.csv'),  'rb'))
        
        
    def predict_width(self, image_in, left_intersection, right_intersection):
        '''
        Use the pre-saved, pre-loaded polynomial regression model to estimate real-world lane widths
        based on the x-coordinates of the lane boundary intercepts, see load_width_estimation_model.
        
        Parameters
        ----------
        
        image_in : Numpy image
            The original image, just used to extract the height and width of the frame
            
        left_intersection : int
            The intersection x-coordinate for the left lane boundary
            
        right_intersection : int
            The intersection x-coordinate for the right lane boundary
        '''
        
        # Do nothing if we have not loaded a model
        if self.model_poly is None or self.model_pol is None:
            return None
        
        # Return zero width if we did not find both the left and right lane boundaries
        if left_intersection is None or right_intersection is None:
            return 0
        
        # Convert intersection x-coordinates into offsets from the centre of frame
        frame_height = image_in.shape[0]
        frame_width  = image_in.shape[1]
        
        left_offset  = abs((frame_width/2) - left_intersection)
        right_offset = abs((frame_width/2) - right_intersection)
        
        # Return the predicted number of metres for each of these pixel offsets
        predictions = self.model_pol.predict(self.model_poly.fit_transform([[left_offset], [right_offset]]))
        
        # Return a predicted width (metres)
        predicted_width = predictions[0] - predictions[1]
        
        return predicted_width
    
    
    def pixel_width(self, left_intersection, right_intersection):
        '''
        Calculate the width in pixels
        
        Parameters
        ----------
        left_intersection : int
            x-coordinate for the intersection of the left lane boundary
            
        right_intersection : int
            x-coordinate for the intersection of the right lane boundary
        '''
        if left_intersection is None or right_intersection is None:
            return 0
            
        return abs(right_intersection - left_intersection)
        
        
    def load_image(self, path):
        '''
        Use OpenCV to read an image file into memory, and correct for lens distortion if required
        
        Paramters
        ---------
        path : str
            Path to the image filename
        '''
        original_image = cv2.imread(path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        return self.correct_image(original_image)

            
    def correct_image(self, image_in):
        '''
        Use OpenCV to correct a loaded image for lens distortion, based on a pre-calibrated,
        pre-loaded model for the specific camera
        '''
        if self.camera_matrix is not None and self.dist_matrix is not None:
            return cv2.undistort(image_in, self.camera_matrix, self.dist_matrix, None, None)
        else:
            # Return the original image unaltered if we do not have a correction model
            return image_in
            
            
    def apply_mask(self, image_in, mask_vertices):
        '''
        Apply a mask to an image
        
        Parameters
        ----------
        image_in : image
            An image that has already been loaded into memory
            
        mask_vertices : list
            A list of vertex x/y coordinates to define the shape of the mask
        '''
        # Initialise a blank "canvas" to draw the mask on
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
        '''
        Use OpenCV to apply a Canny edge detector to an image.  If a mask is specified, apply it AFTER performing
        edge detection, so the edges of the mask are not accidentally recognised as edges.
        
        Parameters
        ----------
        image_in : image
            An image that has already been loaded into memory
            
        mask_vertices : list, optional
            A list of vertex x/y coordinates to define the shape of the mask
            
        Gaussian_ksize : tuple, optional
            Option for a Gaussian Blur operation that happens before the Canny operation.  See OpenCV documentation.
            
        Gaussian_sigmaX : int, optional
            Option for a Gaussian Blur operation that happens before the Canny operation. See OpenCV documentation.
            
        Canny_threshold1 : int, optional
            Option for the Canny operation.  See OpenCV documentation.
            
        Canny_threshold2 : int, optional
            Option for the Canny operation.  See OpenCV documentation.
        '''
        # Convert input image to greyscale
        image_grey = cv2.cvtColor(image_in, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur operation
        image_blur = cv2.GaussianBlur(image_grey, Gaussian_ksize, Gaussian_sigmaX)
        
        # Apply Canny operation
        image_canny = cv2.Canny(image_blur, Canny_threshold1, Canny_threshold2)
        
        # Mask the image AFTER edge detection, if requested
        if mask_vertices:
            image_canny_masked = self.apply_mask(image_canny, mask_vertices)
            return image_canny_masked
        else:
            return image_canny
            
            
    def apply_hough(self, image_in_canny, image_in_orig, rho=2, theta=np.pi/60, threshold=100, minLineLength=10, maxLineGap=100, color=(255,255,255), thickness=3):
        '''
        Apply Hough transformation to an image where Canny edge detection has already been performed.
        
        Parameters
        ----------
   
        image_in_canny : image 
            An image that has already been loaded into memory and had Canny edge detection applied
            
        image_in_orig : image
            The original image, used to output an image where the detected lines have been overlaid
            on top of the original image
        
        rho : int, optional
            Option for the Hough transform.  See OpenCV documentation.
            
        theta : float, optional
            Option for the Hough transform.  See OpenCV documnetation.
            
        threshold : int, optional
            Option for the Hough transformation.  See OpenCV documentation.
            
        minLineLength : int, optional
            Option for the Hough transformation.  See OpenCV documentation.
            
        maxLineGap : int, optional
            Option for the Hough transformation.  See OpenCV documentation.
            
        color : tuple, optional
            Colour that should be used when drawing detected line overlays on top of the original image,
            as part of the output
            
        thickness : int, optional
            Thickness in pixels of any detected line overlays that are drawn on top of the original image,
            as part of the output
        '''
        
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
            # Return the original image with detected line overlay
            return lane_detection.draw_lines(image_in_orig, lines.squeeze(), color=color, thickness=thickness)
            
        else:
            # If no lines were detected, just return a blank image in the original format
            return np.zeros((image_in.shape[0], image_in.shape[1], 3), dtype=np.uint8)
            
            
    @staticmethod
    def draw_lines(image_in, lines, image_background=None, color=(255, 255, 255), thickness=3): 
        '''
        Use OpenCV to draw lines over a copy of an original image
    
        It probably does not seem to make much sense to have both "image_in" and "image_background"
        parameters.  However, "image_in" provides the shape for the output if the lines are to be
        drawn on a plain background, and "image_background" provides an optional actual background.
    
        TODO: Refactor this to make it more intuitive.
    
        Parameters
        ----------
        image_in : image
            An image that will be used to determine the shape of the output
        
        lines : list
            A list of lines to draw
            Each line is a list of [x1, y1, x2, y2] coordinates
    
        image_background : image
            The original image that the lines will be drawn on top of, or None to draw on a black background
        
        color : tuple, optional
            Colour that should be used when drawing detected line overlays on top of the original image,
            as part of the output
            
        thickness : int, optional
            Thickness in pixels of any detected line overlays that are drawn on top of the original image,
            as part of the output
        '''
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
                cv2.line(image_lines, (x1, y1), (x2, y2), color=color, thickness=3)        
                
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
        '''
        Major function to detect the lanes in an image
        
        Parameters
        ----------
        
        image_in : image
            Input image that has already been loaded into memory
            
        own_lane_vertices : list
            List of vertices to use as a mask when detecting the camera vehicle's own lane
            
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
        
        Canny_threshold1 : int, optional
            Option for the Canny operation.  See OpenCV documentation.
            
        Canny_threshold2 : int, optional
            Option for the Canny operation.  See OpenCV documentation.
            
        Hough_rho : int, optional
            Option for the Hough transform.  See OpenCV documentation.
            
        Hough_theta : float, optional
            Option for the Hough transform.  See OpenCV documnetation.
            
        Hough_threshold : int, optional
            Option for the Hough transformation.  See OpenCV documentation.
            
        Hough_minLineLength : int, optional
            Option for the Hough transformation.  See OpenCV documentation.
            
        Hough_maxLineGap : int, optional
            Option for the Hough transformation.  See OpenCV documentation.
        
        height_limit : float, optional
            How far up the frame from the bottom, as a proportion of the frame height,
            to draw any detected lane boundary lines before they vanish
            
        color_own_lane : tuple, optional
            Color to draw boundaries of camera vehicle's own lane
            
        color_left_lane : tuple, optional
            Color to draw the left boundary of any paved shoulder
            
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
        '''
        
        # Detect first round of lines, focussing on our camera vehicle's own lanes first
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
        # Based on partitioning lines from the Hough transform into lines that slope upwards and downwards, then
        # taking the averages for each group
        left_line1, right_line1, left_slope1, left_int1, right_slope1, right_int1 = self.find_average_lines(image_in, lines1, height_limit=height_limit, min_slope=min_slope, max_slope=max_slope)
        
        # Draw our "own lane" lines on a black background
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
        
        # Assemble a list of slopes and intercepts for all three lines, some of these values may be None
        slopes_and_intercepts = [left_slope2, left_int2, left_slope1, left_int1, right_slope1, right_int1]
                
        # Return an image with overlay, plus all the slopes and intercepts
        return lanes_image, slopes_and_intercepts
        
    
    def find_average_lines(self, image_in, lines, height_limit=9/20, min_slope=0.30, max_slope=0, verbose=False):
        '''
        Partition the lines found by the Hough transform into lines that slope upwards or downwards,
        because these groups will be assocated with either the left or right boundary lines respectively.
        
        Once the lines are divided into these groups, apply averaging to calculate a consensus lane boundary line
        
        Parameters
        ----------
        
        image_in : image
            Input image, just used for the shape of the images we are working with
            
        lines : list
            List of lines identified by the Hough transform
            
        height_limit : float, optional
            When drawing detected lines, they should be drawn from the bottom of the frame,
            up to this proportion of the frame height up the frame, then vanish
            
        min_slope : float, optional
            Set a minimum absolute value for line slopes, to be considered when determining
            lane boundary lines.  Anything less than this is roughly horizontal, and could not
            be closely related to the lane boundary, probably noise.
            
        max_slope : float, optional
            Set a maximum absolute value for line slopes, to be considered when determining
            lane boundary lines.  Anything more than this is very vertical.  This option is
            not used in practice, you would need a very high number for it to be "too vertical"
            to be related to the lane boundary.
            
        verbose : boolean, optional
            Specify whether debug messages should be written to STDOUT    
        '''
        
        # Initialise lists that will hold "left" lines and "right" lines, based on the overall direction
        # of their slope
        left  = []
        right = []

        if verbose:
            print('Processing {0:d} lines'.format(len(lines)))
            
        if lines is not None:
            for line in lines:
                # Parse each line into x and y coordinates for their end points
                x1, y1, x2, y2 = line.reshape(4)
                
                # Fit a line to the two points, return the slope and y-intercept
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                y_int = parameters[1]
                
                # Sort lines into left vs right, based on the polarity of the slope
                # Exclude lines based on min_slope and max_slope if required
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
                
        # Return the slopes and intercepts, along with lines with end points
        return left_line, right_line, left_slope, left_int, right_slope, right_int
   
    
    def find_line_ends(self, image_in, average, height_limit=9/20):
        '''
        Given the average slope and average intercept for a group of lines,
        define a line that starts at the bottom of the frame, and ends a
        proportion of the way up the frame dictated by height_limit before vanishing
        
        Parameters
        ----------
        
        image_in : image
            An input image that is just used to determine the shape of the frames being analysed
            
        average : numpy array
            A numpy array giving the average slope and average intercept for a group of lines
            
        height_limit : float, optional
            How far up the frame, as a proportion of frame height, to draw the detected line
            before it vanishes
        '''
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
    
    
    def draw_intersection_grid(self, image_in, slopes_and_intercepts, horizontal_line_height, color=(255,255,255), thickness=3, top=None):
        '''
        Overlay detected lane boundaries, horizontal lines, and vertical lines, to visualise the lanes
        that were detected
        
        The two horizontal lines are at arbitrary heights in the frame.  The bottom one is just above
        the camera vehicle's bonnet, and is the closest part of the frame that is unobstructed.  The
        top one is further into the distance, closer to a "vanishing point".  The idea is to be able
        to clearly see the width of the paved shoulder as its boundary lines intersect these fixed
        frames of reference.
        
        Vertical lines are drawn from the top down, and the bottom up, to the intersection points, to
        further help visualise the widths at those intersection points, extending the top and bottom
        widths up and down the frame (with parallel vertical lines) to make it easier to see.  This was
        done as a visual aid to help see changes in the widths from frame to frame, as the operator
        reviews a sequence of frames at relatively high speed.
        
        Parameters
        ----------
        image_in : image
            Original image, the overlay will be drawn over a copy of this
            
        slopes_and_intercepts : list
            A list of slopes and intercepts for the detected lane boundaries
            
        horizontal_line_height : int
            This is the y-coordinate for the bottom horizontal frame of reference line
            
        color : tuple, optional
            Colour that should be used when drawing detected line overlays on top of the original image,
            as part of the output
            
        thickness : int, optional
            Thickness in pixels of any detected line overlays that are drawn on top of the original image,
            as part of the output
        
        top : int, optional
            This is the y-coordinate for the top horizontal frame of reference line, if required
        '''
        
        # Find intersections for the bottom horizontal frame of reference line
        
        left_intersection_x1, own_l_intersection_x1, own_r_intersection_x1 = self.find_intersection_list(image_in, slopes_and_intercepts, horizontal_line_height)
        
        # Copy the original image
        grid_image = np.copy(image_in)
        
        frame_height = image_in.shape[0]
        frame_width  = image_in.shape[1]
        
        # Draw bottom horizontal line
        cv2.line(grid_image, (0, horizontal_line_height), (frame_width, horizontal_line_height), color, thickness=thickness)
        
        # Draw the top horizontal line, if required
        if top is not None:
            cv2.line(grid_image, (0, top), (frame_width, top), color, thickness=thickness)
        
        # Draw intersecting vertical lines from bottom
        if top is None:
            limit = 0
        else:
            limit = horizontal_line_height
            
        if left_intersection_x1 is not None:
            cv2.line(grid_image, (int(left_intersection_x1),  limit), (int(left_intersection_x1),  frame_height), color, thickness=thickness)
        if own_l_intersection_x1 is not None:
            cv2.line(grid_image, (int(own_l_intersection_x1), limit), (int(own_l_intersection_x1), frame_height), color, thickness=thickness)
        if own_r_intersection_x1 is not None:
            cv2.line(grid_image, (int(own_r_intersection_x1), limit), (int(own_r_intersection_x1), frame_height), color, thickness=thickness) 
        
        # Do the top intersections too
        if top is not None:
            left_intersection_x2, own_l_intersection_x2, own_r_intersection_x2 = self.find_intersection_list(image_in, slopes_and_intercepts, top)
        
            if left_intersection_x2 is not None:
                cv2.line(grid_image, (int(left_intersection_x2),  top), (int(left_intersection_x2),  0), color, thickness=thickness)
            if own_l_intersection_x2 is not None:
                cv2.line(grid_image, (int(own_l_intersection_x2), top), (int(own_l_intersection_x2), 0), color, thickness=thickness)
            if own_r_intersection_x2 is not None:
                cv2.line(grid_image, (int(own_r_intersection_x2), top), (int(own_r_intersection_x2), 0), color, thickness=thickness) 
        
        intersection_list = [left_intersection_x1, own_l_intersection_x1, own_r_intersection_x1]
        
        return grid_image, intersection_list
        
        
    def find_intersection_list(self, image_in, slopes_and_intercepts, horizontal_line_height):
        '''
        Find the list of intersection points between the detected lane boundaries (defined by slopes_and_intercepts)
        with a horizontal frame of reference line
        
        Parameters
        ----------
        image_in : image
            Input image, not actually used because even the image dimensions are not required
            
            TODO: Refactor to remove this parameter
            
        slopes_and_intercepts : list
            List of slopes and intercepts for the detected lane boundaries
            
        horizontal_line_height : int
            This is the y-coordinate for the bottom horizontal frame of reference line
        '''
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