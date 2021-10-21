'''
Created on 06 Sep 2021

@author: Tyler Saxton
'''

import os
import sys

import json

import geojson
from geojson import Feature, FeatureCollection, dump

import geopy
import geopy.distance


class osm_filter(object):
    '''
    A class used to filter an OpenStreetMap extract down to a particular locality, based on a GeoJSON
    file that describes the official shape of the locality.  E.g. a shape issued by the government
    on data.gov.au
    '''

    def __init__(self, filename):
        '''
        Parameters
        ----------
        filename : str
            The name of the GeoJSON file to load.
            Either a full path, or assume it's relative to the current working directory
        '''
        with open(filename) as json_file:
            self.gj = geojson.load(json_file)

        json_file.close()


    def save_locality(self, locality, filename_shape, filename_margin, filter_value, filter_key='vic_loca_2', margin=200):
        '''
        Given an OSM extract, and the name of a locality that is "drawn" the in the GeoJSON file that was loaded
        earlier, extract that locality as a smaller GeoJSON file (with just the one locality instead of all of them)
        then print the osmium tool commands that should be used to:
        
        1) Create an OSM extract with JUST that shape, and
        
        2) Create an OSM extract for a bounding box that includes all of the original locality shape, plus an extra
           margin (in metres) to catch any roads just outside the survey area that might be intersections that would
           otherwise be clipped and missed.
           
        Parameters
        ----------
        
        locality : str
            The name of the locality that will be used when saving files
            
        filename_shape : str
            The path to the output file that will be used to store the OpenStreetMap extract of the locality itself
            
        filename_margin : str
            The path to the output file that will be used to store the OpenStreetMap extract of the locality with
            an extra margin
            
        filter_value : str
            The value to look for in the original GeoJSON file to isolate the shape of the locality in question
            
        filter_key : str, optional
            The name of the field to check for filter_value.  Default 'vic_loca_2' is the field where a suburb name
            is found in the GeoJSON file published by the Victorian State Government
            
        margin : int, optional
            The extra number of metres to add to the bounding box for the "margin" extract, on each side.
        '''
        # Find the locality
        gj_index    = 0
        gj_selected = 0

        while True:
            try:
                gj_item = self.gj[gj_index]

                gj_index = gj_index + 1

                if gj_item['properties'][filter_key].upper() == filter_value.upper():
                    gj_selected = gj_item
                    break

            except Error:
                break

        if gj_selected == 0:
            raise Exception("Could not find locality")

        # Find the overall bounding box of the selected locality
        lat_min =  999
        lat_max = -999
        lon_min =  999
        lon_max = -999

        geometry    = gj_selected['geometry']
        coordinates = geometry['coordinates']

        # Flatten all coordinates into a simple list, to handle MultiLineString objects that have multiple layers
        flat_coordinates = osm_filter.flatten_coordinates(coordinates)

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

        print('Bounding box excl margin: {0:.6f}, {1:.6f} => {2:.6f}, {3:.6f}'.format(lat_min, lon_min, lat_max, lon_max));

        # Add the required margin to the bounding box
        origin_min = geopy.Point(lat_min, lon_min)
        margin_min = geopy.distance.distance(meters=margin).destination(point=origin_min, bearing=225) # SW

        origin_max = geopy.Point(lat_max, lon_max)
        margin_max = geopy.distance.distance(meters=margin).destination(point=origin_max, bearing=45)  # NE

        print('Bounding box with margin: {0:.6f}, {1:.6f} => {2:.6f}, {3:.6f}'.format(margin_min.latitude, margin_min.longitude, margin_max.latitude, margin_max.longitude))

        # Save the locality to a new file
        features = []
        features.append(gj_selected)
        feature_collection = FeatureCollection(features)

        with open(filename_shape, 'w') as output_file_shape:
            dump(feature_collection, output_file_shape)

        # Save locality bounding box with margin to a new file
        geometry = {
            'type': 'MultiPolygon',
            'coordinates': [
                [[
                    [lon_min, lat_min],
                    [lon_min, lat_max],
                    [lon_max, lat_max],
                    [lon_max, lat_min],
                    [lon_min, lat_min]
                ]]
            ]
        }
        
        bb_feature = {
            'type': 'Feature',
            'geometry': geometry,
            'properties': {
                'name': 'bounding box'
            }
        }
        
        features = []
        features.append(bb_feature)
        feature_collection = FeatureCollection(features)
        
        with open(filename_margin, 'w') as output_file_margin:
            dump(feature_collection, output_file_margin)
        
        # Output advice on the osmium commands to run to create the extracts
        print('\nRun the following two osmium commands:\n')

        print('First command:  Extract OSM data according to the official "shape" of the Locality\n')

        print('osmium extract --polygon=Locality_' + locality + '.geojson australia-latest.osm.pbf -o Locality_' + locality + '.osm')

        print('\nSecond command:  Extract OSM data for a bounding box ' + str(margin) + ' meters bigger than the "shape"')
        print('This second extract will be used to ensure we do not miss any intersections with streets that are JUST outside the locality\n')

        #print('osmium extract --bbox=' + str(margin_min.longitude) + ',' + str(margin_min.latitude) + ',' + str(margin_max.longitude) + ',' + str(margin_max.latitude) +
        #    ' australia-latest.osm.pbf -o Locality_' + locality + '_margin.osm')
        print('osmium extract --polygon=Locality_' + locality + '_margin.geojson australia-latest.osm.pbf -o Locality_' + locality + '_margin.osm')
                  
    
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
                    coordinates_out = coordinates_out + osm_filter.flatten_coordinates(item)
                return coordinates_out
            else:
                return coordinates
        else:
            return coordinates