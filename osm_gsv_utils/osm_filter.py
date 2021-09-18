'''
Created on 06 Sep 2021

@author: Tyler Saxton
'''

import geojson
from geojson import Feature, FeatureCollection, dump

import geopy
import geopy.distance


class osm_filter(object):
    '''
    A class used to filteran OpenStreetMap extract down to a particular locality
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

        # Find track of the overall bounding box
        lat_min =  999
        lat_max = -999
        lon_min =  999
        lon_max = -999

        geometry    = gj_selected['geometry']
        coordinates = geometry['coordinates']

        flat_coordinates = self.flatten_coordinates(coordinates)

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
        
        print('\nRun the following two osmium commands:\n')

        print('First command:  Extract OSM data according to the official "shape" of the Locality\n')

        print('osmium extract --polygon=Locality_' + locality + '.geojson australia-latest.osm.pbf -o Locality_' + locality + '.osm')

        print('\nSecond command:  Extract OSM data for a bounding box ' + str(margin) + ' meters bigger than the "shape"')
        print('This second extract will be used to ensure we do not miss any intersections with streets that are JUST outside the locality\n')

        #print('osmium extract --bbox=' + str(margin_min.longitude) + ',' + str(margin_min.latitude) + ',' + str(margin_max.longitude) + ',' + str(margin_max.latitude) +
        #    ' australia-latest.osm.pbf -o Locality_' + locality + '_margin.osm')
        print('osmium extract --polygon=Locality_' + locality + '_margin.geojson australia-latest.osm.pbf -o Locality_' + locality + '.osm')
        
        
    def flatten_coordinates(self, coordinates):
        coordinates_out = []

        if type(coordinates[0]) is list:
            if type(coordinates[0][0]) is list:
                for item in coordinates:
                    coordinates_out = coordinates_out + self.flatten_coordinates(item)
                return coordinates_out
            else:
                return coordinates
        else:
            return coordinates
