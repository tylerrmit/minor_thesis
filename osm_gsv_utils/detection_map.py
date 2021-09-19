'''
Created on 06 Sep 2021

@author: Tyler Saxton
'''

import json
import os
import sys

import pandas as pd

from tqdm.notebook import tqdm

from ipyleaflet import Map, Marker, GeoJSON


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

        
    @staticmethod  
    def load_layer(filename, color=None):
        try:
            f = open(filename)
            data = json.load(f)
            f.close()
            
            if color is not None:
                layer = GeoJSON(data=data, style={'color': color})
            else:
                layer = GeoJSON(data=data)
            
            return layer, data
            
        except Exception as e:
            print('Unable to load layer [' + str(filename) + ']: ' + str(e))
            return None, None
            
            
    @staticmethod
    def load_points(map, filename):
        df = pd.read_csv(filename)
        
        for index, row in df.iterrows():
            marker = Marker(location=[row['lat'], row['lon']], draggable=False)
            map.add_layer(marker)
            
            
    @staticmethod
    def get_centroid(data):
        lat_min =  999.9
        lat_max = -999.9
        lon_min =  999.9
        lon_max = -999.9
    
        for feature in data['features']:
            geometry = feature['geometry']
            coordinates = geometry['coordinates']
        
            flat_coordinates = detection_map.flatten_coordinates(coordinates)
        
            for point in flat_coordinates:
            
                lat_point = point[1]
                lon_point = point[0]
            
                if lon_point < lon_min:
                    lon_min = lon_point
                if lon_point > lon_max:
                    lon_max = lon_point
                if lat_point < lat_min:
                    lat_min = lat_point
                if lat_point > lat_max:
                    lat_max = lat_point
    
        return [((lat_min + lat_max) / 2), ((lon_min + lon_max) / 2)]

        
    @staticmethod
    def flatten_coordinates(coordinates):
        coordinates_out = []
    
        if type(coordinates[0]) is list:
            if type(coordinates[0][0]) is list:
                for item in coordinates:
                    coordinates_out = coordinates_out + detection_map.flatten_coordinates(item)
                return coordinates_out
            else:
                return coordinates
        else:
            return coordinates