'''
Created on 06 Sep 2021

@author: Tyler Saxton
'''

import os
import sys

import tensorflow as tf

import object_detection
from object_detection.builders import model_builder
from object_detection.protos   import pipeline_pb2
from object_detection.utils    import config_util
from object_detection.utils    import label_map_util
from object_detection.utils    import visualization_utils as viz_utils

from google.protobuf import text_format

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tqdm.notebook import tqdm, trange


class tf2_model_wrapper(object):
    '''
    A class used to load a pre-trained detection model (either still in-training or "frozen" to a
    particular release version with a particular number of training steps) and then apply it
    to either a series of images or a video file
    '''

    def __init__(self, trained_model_name, output_directory, download_directory=None, version_suffix=None):
        '''
        Parameters
        ----------
        trained_model_name : str
            The name of the pre-trained model
            
        output_directory: str
            The directory where the detection log and any output images will be written
            
        download_directory : str, optional
            The directory where images to be processed are found.  Required to set up the model
            wrapper for the apply_model() function
        
        version_suffix: str, optional
            The suffix that must be used to identify the correct version of the label map file
            corresponding to this model.  Default = no suffix
        '''

        # Check whether the GPU is recognised, up front
        print(tf.config.list_physical_devices('GPU'))
        
        # Derived/Static Configuration
        self.download_directory    = download_directory
        self.output_directory      = output_directory
        self.output_directory_hits = os.path.join(self.output_directory, 'hits')
        self.output_directory_miss = os.path.join(self.output_directory, 'miss')
        self.trained_model_name    = trained_model_name
        
        if version_suffix is None:
            self.version_suffix = ''
        else:
            self.version_suffix = '_' + version_suffix
        
        print('Output directory for detections: ' + self.output_directory)

        self.label_map_file       = os.path.join('TensorFlow', 'workspace', 'annotations', 'label_map' + self.version_suffix + '.pbtxt')
        
        print('Label Map Path: [' + self.label_map_file + ']')
        
        self.pipeline_config_file = os.path.join('Tensorflow', 'workspace','models', self.trained_model_name, 'pipeline.config')
        
        # Models still in-training and waiting to be exported have their checkpoint files here:
        self.checkpoint_path      = os.path.join('Tensorflow', 'workspace', 'models', self.trained_model_name)

        # Exported models have their checkpoint files in a subdirectory
        checkpoint_subdir = os.path.join(self.checkpoint_path, 'checkpoint') 
        if os.path.exists(checkpoint_subdir) and os.path.isdir(checkpoint_subdir):
            self.checkpoint_path = checkpoint_subdir

        # Create output directory if it does not already exist
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory)
        if not os.path.isdir(self.output_directory_hits):
            os.makedirs(self.output_directory_hits)
        if not os.path.isdir(self.output_directory_miss):
            os.makedirs(self.output_directory_miss)
            
        # Load pipeline config and build a detection model
        self.configs         = config_util.get_configs_from_pipeline_file(self.pipeline_config_file)
        self.detection_model = model_builder.build(model_config=self.configs['model'], is_training=False)

        # Find the latest checkpoint
        self.latest_checkpoint = 'ckpt-1'

        checkpoint_files = os.listdir(self.checkpoint_path)

        for f in checkpoint_files:
            if f.startswith('ckpt-') and f.endswith('index'):
                self.latest_checkpoint = f.split('.')[0]
        print('Latest Checkpoint: ' + self.latest_checkpoint)

        # Restore checkpoint
        self.ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        self.ckpt.restore(os.path.join(self.checkpoint_path, self.latest_checkpoint)).expect_partial()

        self.category_index = label_map_util.create_category_index_from_labelmap(self.label_map_file)
        
        
    @tf.function
    def detect_fn(self, image):
        image, shapes   = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections      = self.detection_model.postprocess(prediction_dict, shapes)
        
        return detections
        

    def apply_model(self, lat, lon, bearing, way_start_id, way_id, node_id, offset_id, filename=None, heading_offsets=[0,90,180,270], min_score=0.5, mask=None, write=True, display=False, log=False, verbose=False):
        '''
        Apply the detection model to a single location in the download_directory that was specified
        when the wrapper was initialised.
        
        If the filename is passed explicitly, then it will be worked out based on the latitude,
        longitude, and the bearing of the road ath the location, and then one image will be loaded
        per heading offset.  A cached Google Street View image filename is constructed from these
        elements based on the filename convention used in the cache.  One image is examined per
        element in heading_offset, to provide 360 degree coverage.
        
        This function is called by an over-arching caller process, one location at a time, as it works
        its way through a batch file.
        
        Parameters
        ----------
        lat: str
            Latitude of the location to be processed

        lon: str
            Longitude of the location to be processed
        
        bearing: int
            Bearing of the road at the location to be processed.  Which was is "forward" for the camera?
            
        way_start_id: int
            The starting way ID from the OpenStreetMap data, for the road that is being sampled, to provide
            traceability of the results back to the map.
            
        way_id: int
            The way ID from the OpenStreetMap data, for the road segment that is being sampled, to provide
            traceability of the results back to the map.
        
        node_id: int
            The node ID from the OpenStreetMap data, for the intersection that is being sampled, to provide
            traceability of the results back to the map.
        
        offset_id: int
            The number of metres offset from the intersection being sampled, positive or negative, relative
            to the bearing of the road at the intersection.  E.g. -20 = 20 metres BEFORE the intersection.
            Provides traceability of the results back to the map.
        
        filename: str, optional
            If a filename is explicitly supplied with this parameter, it will be examined by the model.
            Otherwise a list of images will be examined based on the location details in other fields.
        
        heading_offsets: list, optional
            A list of offsets from the bearing of the road.  Google Street View images are examined
            for each of these offsets, to provide 360 degree coverage.
            Set this to a list with just a single zero value if supplying the filename option.
        
        min_score: float, optional
            The minimum model confidence score at which we accept that the object has been detected
        
        mask: list, optional
            An optional list of (x,y) coordinates in the bitmap image that forms a "mask" to decide which
            parts of the image are examined by the detection model, to mask off one side of the road,
            or exclude areas that are the source of false positives without any prospect of a real
            detection (e.g. reflections from the bonnet of the camera vehicle in dash camera footage)
        
        write: boolean, optional
            Specify whether output images with detection bounding boxes and confidence scores should be written
            to the "hits" and "miss" subdirectories of the output_directory, for examination and troubleshooting
        
        display: boolean, optional
            Specify whether the model should try to display the output image with detection bounding boxes and
            confidence scores to a Jupyter Notebook via pyplot
        
        log: boolean, optional
            Specify whether detection results should be written to "detection_log.csv" in the output_directory
        
        verbose: boolean, optional
            Whether to print debug messages to stdout
        '''
        for heading_offset in heading_offsets:
            # Calculate a heading to examine (if processing Google Street View) based on the
            # original bearing of the road plus a heading offset from a list
            heading = int(round(bearing + heading_offset))
            
            # Make sure the calculated heading is between 0 and 360 after we added the offset
            if heading > 360:
                heading = heading - 360
            
            # Filenames that start with a "-" are annoying
            # Convert negative latitude and longitude to a string with a "s" or "e" prefix
            # Use an "n" or "w" prefix where they are positive
            if lat < 0:
                lat_str = 's{0:.6f}'.format(abs(lat))
            else:
                lat_str = 'n{0:.6f}'.format(abs(lat))
                
            if lon < 0:
                lon_str = 'e{0:.6f}'.format(abs(lon))
            else:
                lon_str = 'w{0:.6f}'.format(abs(lon))
            
            if filename is None:
                # Process a series of Google Street View images for a location, if a filename was not supplied
                image_filename = os.path.join(
                    '{0:.6f}'.format(lat),
                    '{0:.6f}'.format(lon),
                    str(int(heading)),
                    'gsv_0.jpg'
                )
            else:
                # Use an explicit filename if it was supplied
                image_filename = filename
            
            # Assign a filename for the output, based on the location details
            output_filename = '{0:s}_{1:s}_{2:d}.jpg'.format(lat_str, lon_str, heading)
            
            if output_filename.startswith('n0.0'):
                output_filename = os.path.basename(filename)
            
            # Derive the full path for the image that we are about to load and examine
            image_path      = os.path.join(self.download_directory, image_filename)     
        
            if verbose:
                print('Input path: [{0:s}] Output filename: [{1:s}]'.format(image_path, output_filename) )    

            if not os.path.exists(image_path):
                if verbose:
                    # Sometimes GSV just didn't have an image near where we were looking, ignore
                    print('NOTE: [' + image_path + '] does not exist')
                return
        
            # Read the image and convert it into a tensor
            img          = cv2.imread(image_path)
            image_np     = np.array(img)
            
            # Apply a detection mask, if we were asked to
            if mask is not None:
                mask_image_np = np.zeros_like(img)
                channel_count = img.shape[2]
                match_mask_color = (255,) * channel_count
                cv2.fillPoly(mask_image_np, np.int32([mask]), match_mask_color)
                
                masked_image = cv2.bitwise_and(img, mask_image_np)
                image_np = np.array(masked_image)
            else:
                image_np = np.array(img)
            
            # Convert the image to a tensor form, ready to be processed by the model
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
              
            # Detect objects of interest using the model
            self.detections = self.detect_fn(input_tensor)

            num_detections = int(self.detections.pop('num_detections'))
            self.detections = {key: value[0, :num_detections].numpy()
                        for key, value in self.detections.items()}
            self.detections['num_detections'] = num_detections

            # Count detections that met the threshold
            num_detections_threshold = 0
    
            if num_detections > 0:
                for idx, score in enumerate(self.detections['detection_scores']):
                    detected_class = self.detections['detection_classes'][idx] + 1
                    
                    if score >= min_score and detected_class == 1:
                        num_detections_threshold = num_detections_threshold + 1
        
            # Report the results to STDOUT if requested
            if verbose:
                print('num_detections_threshold: ' + str(num_detections_threshold))
        
                if num_detections_threshold > 0:
                    for idx, score in enumerate(self.detections['detection_scores']):
                        detected_class = self.detections['detection_classes'][idx] + 1
                    
                        if score >= min_score:
                            print('Detection class ' + str(detected_class) + ' box:            ' + str(score) + ' ' + str(self.detections['detection_boxes'][idx]))
    
            # Record any detections that met the threshold to the "detection_log.csv" file
            if log and num_detections_threshold > 0:
                try:            
                    detection_log_path = os.path.join(self.output_directory, 'detection_log.csv')
            
                    if not os.path.exists(detection_log_path):
                        detection_log = open(detection_log_path, 'w')
                        detection_log.write('lat,lon,bearing,heading,way_id_start,way_id,node_id,offset_id,score,bbox_0,bbox_1,bbox_2,bbox_3,orig_filename\n')
                    else:
                        detection_log = open(detection_log_path, 'a')
                
                    for idx, score in enumerate(self.detections['detection_scores']):
                        detected_class = self.detections['detection_classes'][idx] + 1
                        
                        if score >= min_score and detected_class == 1:
                            detection_log.write(
                                '{0:.6f},{1:.6f},{2:d},{3:d},{4:d},{5:d},{6:d},{7:f},{8:f},{9:f},{10:f},{11:f},{12:f},{13:s}\n'.format(
                                    lat,
                                    lon,
                                    bearing,
                                    heading,
                                    way_start_id,
                                    way_id,
                                    node_id,
                                    offset_id,
                                    score,
                                    self.detections['detection_boxes'][idx][0],
                                    self.detections['detection_boxes'][idx][1],
                                    self.detections['detection_boxes'][idx][2],
                                    self.detections['detection_boxes'][idx][3],
                                    image_filename
                                )
                            )
            
                    detection_log.close()
            
                except Exception as e:
                    print('Unable to log detections for [' + image_filename + ']: ' + str(e))
        
        
            if write or display:
                # Construct a copy of the image with any detection overlays, ready to write to disk
                # or display in a Jupyter notebook
                
                # Detection_classes should be ints.
                self.detections['detection_classes'] = self.detections['detection_classes'].astype(np.int64)

                # Create a copy of the image with detection boxes overlaid
                label_id_offset = 1
                image_np_with_detections = np.array(img)
                
                # Draw outline of mask
                if mask is not None:
                    for i in range(len(mask)-1):
                        cv2.line(image_np_with_detections, mask[i], mask[i+1], (255, 255, 0), thickness=5)
                 
                # Add detection overlay to the original image
                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    self.detections['detection_boxes'],
                    self.detections['detection_classes']+label_id_offset,
                    self.detections['detection_scores'],
                    self.category_index,
                    use_normalized_coordinates = True,
                    max_boxes_to_draw          = 5,
                    min_score_thresh           = min_score,
                    agnostic_mode              = False
                )

                if write:
                    # Write the output image to disk
                    if num_detections_threshold > 0:
                        output_path = os.path.join(self.output_directory_hits, output_filename)
                    else:
                        output_path = os.path.join(self.output_directory_miss, output_filename)
                    
                    if verbose:
                        print('Writing:     ' + output_path)
                    cv2.imwrite(output_path, image_np_with_detections)

                if display:
                    # Convert color space of the output image
                    image_np_converted = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)

                    plt.imshow(image_np_converted)
                    plt.show()
                
                
    def process_batch_file(self, batch_filename, heading_offsets=[0,90,180,270], min_score=0.5, mask=None, explicit_files=False, progress=False, verbose=False):
        '''
        Read a series of image files to process from a batch file, and call the apply_model() function to
        examine each frame
        
        Parameters
        ----------
        batch_filename : str
            The name of a CSV file containing co-ordinates and bearings of images to examine.
            
        heading_offsets: list, optional
            A list of offsets from the bearing of the road.  Google Street View images are examined
            for each of these offsets, to provide 360 degree coverage.
            Set this to a list with just a single zero value if supplying the filename option.
        
        min_score: float, optional
            The minimum model confidence score at which we accept that the object has been detected
        
        mask: list, optional
            An optional list of (x,y) coordinates in the bitmap image that forms a "mask" to decide which
            parts of the image are examined by the detection model, to mask off one side of the road,
            or exclude areas that are the source of false positives without any prospect of a real
            detection (e.g. reflections from the bonnet of the camera vehicle in dash camera footage)
        
        explicit_files: boolean, optional
            Specify whether the batch file will contain explicit filenames (for dash camera footage images)
            or whether the filename needs to be worked out from the location fields (for Google Street View)
            
        progress: boolean, optional
            Specify whether a progress bar should be displayed in a Jupyter Notebook as it processes,
            each image, via the tqdm package
        
        verbose: boolean, optional
            Whether to print debug messages to stdout
        '''
        
        # Reset existing "detection_log.csv", ready to output the results to the output_directory
        detection_log_path = os.path.join(self.output_directory, 'detection_log.csv')
        
        if os.path.exists(detection_log_path):
            os.remove(detection_log_path)
            
        # Iterate over every requested location in the batch CSV file
        df = pd.read_csv(batch_filename)

        tqdm.pandas()  
            
        if explicit_files:
            # We expect the batch CSV file to give us explicit filenames (dash camera images)
            df.progress_apply(lambda row: self.apply_model(
                row['lat'],
                row['lon'],
                row['bearing'],
                row['way_start_id'],
                row['way_id'],
                row['node_id'],
                row['offset_id'],
                min_score      = min_score,
                mask           = mask,
                filename       = row['image_path'],
                write          = True,
                display        = False,
                log            = True,
                verbose        = verbose),
                axis=1
            )
        else:
            # We expect to have to work out the filenames from the location fields (GSV)
            df.progress_apply(lambda row: self.apply_model(
                row['lat'],
                row['lon'],
                row['bearing'],
                row['way_start_id'],
                row['way_id'],
                row['node_id'],
                row['offset_id'],
                min_score      = min_score,
                mask           = mask,
                write          = True,
                display        = False,
                log            = True,
                verbose        = verbose),
                axis=1
            )
                
        return detection_log_path
        

    def process_split_dir(self, batch_filename, min_score=0.5, mask=None, progress=False, verbose=False):
        '''
        Read a series of image files to process from a batch file, and call the apply_model() function to
        examine each frame
        
        This is basically the same as process_batch_file(), except it has different expectations about
        what information will be supplied in the batch file.  Used when it is not yet known which
        OpenStreetMap way_id_start/way_id/node_id is most closely associated with the image, because
        it came from a dash camera.  If the image came from a dash camera, the frame number within the
        video file is used as the offset_id.
        
        Parameters
        ----------
        batch_filename : str
            The name of a CSV file containing co-ordinates and bearings to download from Google Street View.
            If the file has already been downloaded, it is skipped, to save costs.
        
        min_score: float, optional
            The minimum model confidence score at which we accept that the object has been detected
        
        mask: list, optional
            An optional list of (x,y) coordinates in the bitmap image that forms a "mask" to decide which
            parts of the image are examined by the detection model, to mask off one side of the road,
            or exclude areas that are the source of false positives without any prospect of a real
            detection (e.g. reflections from the bonnet of the camera vehicle in dash camera footage)
            
        progress: boolean, optional
            Specify whether a progress bar should be displayed in a Jupyter Notebook as it processes,
            each image, via the tqdm package
        
        verbose: boolean, optional
            Whether to print debug messages to stdout
        '''
        
        # Reset existing "detection_log.csv", ready to output the results to the output_directory
        detection_log_path = os.path.join(self.output_directory, 'detection_log.csv')
        
        if os.path.exists(detection_log_path):
            os.remove(detection_log_path)
            
        # Iterate over every requested location in the batch
        df = pd.read_csv(batch_filename)

        if progress:
            tqdm.pandas()   

            df.progress_apply(lambda row: self.apply_model(
                row['lat'],
                row['lon'],
                int(row['heading']),
                0,
                0,
                0,
                row['frame_num'],
                filename  = row['filename'],
                heading_offsets = [0],
                min_score = min_score,
                mask      = mask,
                write     = True,
                display   = False,
                log       = True,
                verbose   = verbose),
                axis=1
            )
        else:
            for index, row in df.iterrows():
                self.apply_model(
                    row['lat'],
                    row['lon'],
                    int(row['heading']),
                    0,
                    0,
                    0,
                    row['frame_num'],
                    filename  = row['filename'],
                    heading_offsets = [0],
                    min_score = min_score,
                    mask      = mask,
                    write     = True,
                    display   = False,
                    log       = True,
                    verbose   = verbose
                )
                
        return detection_log_path
        
        
    def process_video(self, directory, video_in, video_out, min_score=0.5, fps=60, mask=None):
        '''
        Process a video file from a dash camera, and output a video file where detection
        bounding boxes and confidence scores have been added as an overlay
        
        Parameters
        ----------
        directory: str
            Directory where the input video file is found, and where the output video
            will be written
        
        video_in: str
            Filename of the video to be processed
            
        video_out: str
            Filename of the output video with detection overlay
        
        min_score: float, optional
            The minimum model confidence score at which we accept that the object has been detected
        
        fps: int, optional
            Number of frames per second expected in the input video, this will be replicated in the output
        
        mask: list, optional
            An optional list of (x,y) coordinates in the bitmap image that forms a "mask" to decide which
            parts of the image are examined by the detection model, to mask off one side of the road,
            or exclude areas that are the source of false positives without any prospect of a real
            detection (e.g. reflections from the bonnet of the camera vehicle in dash camera footage)
        '''
    
        # Derive the full input and output paths
        path_in  = os.path.join(directory, video_in)
        path_out = os.path.join(directory, video_out)
        
        # Open the video and get the frame dimensions and number of frames in the file
        cap = cv2.VideoCapture(path_in)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Open an output video for writing.
        out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))

        # Process each frame in turn, and display a progress bar in Jupyter Notebook
        for frame_num in trange(0, frame_count):
            ret, frame = cap.read()
    
            if ret == True:
                image_np_orig = np.array(frame)
                
                # Apply a mask, if requested
                if mask is not None:
                    mask_image_np = np.zeros_like(image_np_orig)
                    channel_count = image_np_orig.shape[2]
                    match_mask_color = (255,) * channel_count
                    cv2.fillPoly(mask_image_np, np.int32([mask]), match_mask_color)
                
                    masked_image = cv2.bitwise_and(image_np_orig, mask_image_np)
                    image_np = np.array(masked_image)
                else:
                    image_np = image_np_orig.copy()         

                # Convert the image to tensor format, suitable for processing by the model
                input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
                
                # Examine the frame with the model
                detections = self.detect_fn(input_tensor)
    
                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
                detections['num_detections'] = num_detections

                # Force detection class types to be ints
                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                # Add detection overlay to original image
                label_id_offset = 1
                image_np_with_detections = image_np_orig.copy()

                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    self.category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=min_score,
                    agnostic_mode=False
                )

                # Draw outline of mask, if any, so it's clear from the output where the mask was
                if mask is not None:
                    for i in range(len(mask)-1):
                        cv2.line(image_np_with_detections, mask[i], mask[i+1], (255, 255, 0), thickness=5)
                        
                # Write the output image to the output video steam
                out.write(image_np_with_detections)
    
                # Display the image on screen in a new window as it progresses, so the operator
                # can see what is going on
                cv2.imshow('object detection', cv2.resize(image_np_with_detections, (width, height)))
    
                # Allow the operator the opportunity to abort by pressing 'q'
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
        
        # Close input and output files, and destroy any window that was opened to display output
        # as it progressed
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        