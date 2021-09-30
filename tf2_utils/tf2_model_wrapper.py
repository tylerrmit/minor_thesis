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

from tqdm.notebook import tqdm


class tf2_model_wrapper(object):
    '''
    A class used to "walk" routes in an OSM extract and create a list of sample points and bearings

    '''

    def __init__(self, locality, margin, download_directory, output_directory, trained_model_name):
        '''
        Parameters
        ----------
        locality : str
            The name of the locality whose GSV images we are processing
        margin   : int
            The margin that was used around intersections when gathering sample points
        trained_model_name : str
            The name of the pre-trained model
        '''

        # Check whether the GPU is recognised
        print(tf.config.list_physical_devices('GPU'))
        
        # Derived/Static Configuration
        self.locality              = locality
        self.margin                = margin
        self.download_directory    = download_directory
        self.output_directory      = output_directory
        self.output_directory_hits = os.path.join(self.output_directory, 'hits')
        self.output_directory_miss = os.path.join(self.output_directory, 'miss')
        self.trained_model_name    = trained_model_name
        
        print('Output directory for detections: ' + self.output_directory)

        self.label_map_file       = os.path.join('TensorFlow', 'workspace', 'annotations', 'label_map.pbtxt')
        self.pipeline_config_file = os.path.join('Tensorflow', 'workspace','models', self.trained_model_name, 'pipeline.config')
        
        # Models still in-training and waiting to be exported have their checkpoint files here:
        self.checkpoint_path      = os.path.join('Tensorflow', 'workspace', 'models', self.trained_model_name)

        # Exported models have their checkpoint files in a subdirectory
        checkpoint_subdir = os.path.join(self.checkpoint_path, 'checkpoint') 
        if os.path.exists(checkpoint_subdir):
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
            if f.startswith('ckpt-'):
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
        for heading_offset in heading_offsets:
            heading = int(round(bearing + heading_offset))
            if heading > 360:
                heading = heading - 360
            
            if lat < 0:
                lat_str = 's{0:.6f}'.format(abs(lat))
            else:
                lat_str = 'n{0:.6f}'.format(abs(lat))
                
            if lon < 0:
                lon_str = 'e{0:.6f}'.format(abs(lon))
            else:
                lon_str = 'w{0:.6f}'.format(abs(lon))
            
            if filename is None:
                image_filename = os.path.join(
                    '{0:.6f}'.format(lat),
                    '{0:.6f}'.format(lon),
                    str(int(heading)),
                    'gsv_0.jpg'
                )
            else:
                image_filename = filename
                
            output_filename = '{0:s}_{1:s}_{2:d}.jpg'.format(lat_str, lon_str, heading)
            image_path      = os.path.join(self.download_directory, image_filename)     
        
            if verbose:
                print('Input path:  ' + image_path)     

            if not os.path.exists(image_path):
                if verbose:
                    # Sometimes GSV just didn't have an image near where we were looking, ignore
                    print('NOTE: [' + image_path + '] does not exist')
                return
        
            # Read the image and convert it into a tensor
            img          = cv2.imread(image_path)
            image_np     = np.array(img)
            
            if mask is not None:
                mask_image_np = np.zeros_like(img)
                channel_count = img.shape[2]
                match_mask_color = (255,) * channel_count
                cv2.fillPoly(mask_image_np, np.int32([mask]), match_mask_color)
                
                masked_image = cv2.bitwise_and(img, mask_image_np)
                image_np = np.array(masked_image)
            else:
                image_np = np.array(img)
                
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
        
            if verbose:
                print('num_detections_threshold: ' + str(num_detections_threshold))
        
                if num_detections_threshold > 0:
                    for idx, score in enumerate(self.detections['detection_scores']):
                        detected_class = self.detections['detection_classes'][idx] + 1
                    
                        if score >= min_score:
                            print('Detection class ' + str(detected_class) + ' box:            ' + str(score) + ' ' + str(self.detections['detection_boxes'][idx]))
    
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
                # Detection_classes should be ints.
                self.detections['detection_classes'] = self.detections['detection_classes'].astype(np.int64)

                # Create a copy of the image with detection boxes overlaid
                label_id_offset = 1
                image_np_with_detections = np.array(img)
                
                # Draw outline of mask
                if mask is not None:
                    for i in range(len(mask)-1):
                        cv2.line(image_np_with_detections, mask[i], mask[i+1], (255, 255, 0), thickness=5)
                        
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
                
                
    def process_batch_file(self, batch_filename, heading_offsets=[0,90,180,270], min_score=0.5, mask=None, progress=False, verbose=False):
        '''
        Parameters
        ----------
        batch_filename : str
            The name of a CSV file containing co-ordinates and bearings to download from Google Street View.
            If the file has already been downloaded, it is skipped, to save costs.
        '''
        
        # Reset existing detection log
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
                row['bearing'],
                row['way_start_id'],
                row['way_id'],
                row['node_id'],
                row['offset_id'],
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
                    row['bearing'],
                    row['way_start_id'],
                    row['way_id'],
                    row['node_id'],
                    row['offset_id'],
                    min_score = min_score,
                    mask      = mask,
                    write     = True,
                    display   = False,
                    log       = True,
                    verbose   = verbose
                )
                
        return detection_log_path

    def process_split_dir(self, batch_filename, min_score=0.5, mask=None, progress=False, verbose=False):
        '''
        Parameters
        ----------
        batch_filename : str
            The name of a CSV file containing co-ordinates and bearings to download from Google Street View.
            If the file has already been downloaded, it is skipped, to save costs.
        '''
        
        # Reset existing detection log
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