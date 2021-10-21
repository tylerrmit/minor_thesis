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
    Class to hide some of the detail when loading GeoJSON files and displaying them
    in a map in a Jupyter Notebook
    
    ipyleaflet is used to display the maps.  This class can help load a series of
    detection points from a "detection_log.csv", or load a GeoJSON file as a "layer"
    to be included in a map, or work out the centroid latitude/longitude for a GeoJSON
    file so that we can automatically centre the map on the area of interest.
    '''

    def __init__(self, detection_log):
        '''
        Parameters
        ----------
        detection_log : str
            The path to a CSV recording the detections.  If this class is being instantiated,
            the path to the "detection_log.csv" with detection points must be supplied.
            It will then be read into a pandas dataframe.
        '''

        self.detection_log  = detection_log
        
        self.df = pd.read_csv(detection_log)
        
        
    @staticmethod  
    def load_layer(filename, color=None):
        '''
        Load a GeoJSON file, and convert it to a layer object, ready to be added to
        an ipyleaflet map
        
        Parameters
        ----------
        filename : str
            Path to the GeoJSON file to be loaded_points
            
        color : str, optional
            Description of the colour to use when drawing this layer on the map
        '''
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
        '''
        Load a series of points from a "detection_log.csv" file, and display
        them on a map
        
        Parameters
        ----------
        map : ipyleaflet Map
            The map into which the points must be inserted
            
        filename : str
            Path to the "detection_log.csv" containing the list of points
        '''
        df = pd.read_csv(filename)
        
        loaded_points = {}
        
        for index, row in df.iterrows():
            key = '{0:.6f}-{1:.6f}'.format(row['lat'], row['lon'])
            if key not in loaded_points:
                marker = Marker(location=[row['lat'], row['lon']], draggable=False)
                map.add_layer(marker)
                loaded_points[key] = True
                
        return len(loaded_points.keys())
            
            
    @staticmethod
    def get_centroid(data):
        '''
        Calculate the centre latitude/longitude for GeoJSON data that has been
        loaded into memory
        
        Parameters
        ----------
        data : GeoJSON data
            The geographic data that has been loaded into memory from a GeoJSON file
        '''
        
        # Set extreme defaults for min/max latitude and longitudes in the data,
        # as a starting point for our search
        lat_min =  999.9
        lat_max = -999.9
        lon_min =  999.9
        lon_max = -999.9
    
        # Find the bounding box based on the max/min latitude/longitude in the data
        for feature in data['features']:
            geometry = feature['geometry']
            coordinates = geometry['coordinates']
        
            # Use a helper function to flatten tested geometry objects into
            # a single list of coordinates.  E.g. "LineString" is one layer deep
            # but "MultiLineString" is multiple layers deep and we want to tease
            # out all the coordinates into a simple list
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
    
        # The centroid is the average latitude and longitude from the bounding box
        return [((lat_min + lat_max) / 2), ((lon_min + lon_max) / 2)]

        
    @staticmethod
    def flatten_coordinates(coordinates):
        '''
        Flatten any complex geometries (such as MultiLineString) into a simple
        list of co-ordinates
        
        Parameters
        ----------
        coordinates : geometry
            The geometry object to be flattened into a simpler list
        '''
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