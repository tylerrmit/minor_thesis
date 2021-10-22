'''
Created on 06 Sep 2021

@author: Tyler Saxton
'''

import json
import math
import os
import sys

import pandas as pd
import numpy as np

import xml.dom.minidom

from geographiclib.geodesic import Geodesic

import geojson
from geojson import Feature, FeatureCollection, dump

from geopy.distance import geodesic

from shapely.geometry import Point, LineString

from tqdm.notebook import tqdm, trange


class osm_walker(object):
    '''
    This class loads an OpenStreetMap XML extract and caches the most relevant parts in memory.
    
    Its original purpose was to "walk" down streets in an OpenStreetMap extract for the survey
    area, and create a list of points to sample from Google Street View.  However it grew to handle
    the details of any operation where we needed to work with the OpenStreetMap (OSM) data, e.g.
    to correlate detection points from dash camera footage to the map data, draw inferred routes,
    etc.
    '''

    def __init__(self, filename_main, filename_margin=None, filter_log=None, verbose=False):
        '''
        Parameters
        ----------
        filename_main : str
            The name of the main OSM XML file to load
            Either a full path, or assume it's relative to the current working directory
            
        filename_margin: str, optional
            The name of a second OSM XML file encompassing the first, with a "margin"
            around it to catch intersecting streets just outside the main area.  Only
            selected parts of the second file are cached, just enough to identify intersection
            nodes
            
        filter_log :  str, optional
            The path to a "filter log" CSV file containing the list of way IDs and node IDs
            to include when loading the OSM data.  If the OSM data is much bigger than the
            route or area we want to look at.  Only used in the early stages of the project,
            and deprecated in favour of using the "osmium" tool to reduce OSM extracts down
            to a selected area
        
        verbose : boolean, optional
            Specify whether debug messages should be written to STDOUT
        '''

        # Dictionaries used to cache OSM information in memory
        self.ways_by_name                = {} # Dictionary giving, for each way name, a list of way ids
        self.ways_by_id                  = {} # Dictionary to look up the way details by the way id
        self.way_names_by_id             = {} # Dictionary to look up the way name by the way id
        self.way_starts                  = {} # Record the start node for each way
        self.way_ends                    = {} # Record the way end for each way
        self.way_is_cycleway             = {} # Record whether a way is tagged as a cycleway or not
        
        self.way_names_per_node          = {} # Dictionary giving, for each node, the list of way NAMEs attached to the node
        self.way_ids_per_node            = {} # Dictionary giving, for each node, the list of way IDs attached to the node
        self.way_ends_per_node           = {} # Dictionary giving, for each node, the list of way IDs that start or end with the node
        self.nodes                       = {} # Dictionary giving node objects by node id
        
        self.node_coords                 = {} # Dictionary giving [lat, lon] by node_id
        
        self.way_node_ids                = {} # Dictionary giving a list of node_ids (in order) for each way, by way_id
        
        self.linked_way_sections         = {} # Dictionary giving ways that have been re-connected by name, list of intersection node_ids in order
        self.linked_way_sections_all     = {} # Dictionary giving ways that have been re-connected by name, list of ALL node_ids in order
        self.linked_way_sections_cwy     = {} # Dictionary to say if any way_id linked to the node in self.linked_way_sections_all is tagged as cycleway
        
        self.linked_linestrings          = {} # Dictionary where linked ways are stored as shapely LineString objects
        self.linked_coord_list           = {} # Dictionary where linked ways are stored as coordinate lists
        
        self.way_id_start_by_way_id      = {} # Dictionary showing the way_id_start for each way_id, where did it go?
        
        self.detection_hits              = {} # Dictionary giving detection hits from a detection log
        
        self.unused_way_ids              = [] # Keep track of any way IDs we have not yet linked to a way segment, to make sure we get them all
        
        self.tagged_features             = [] # Features we are building to draw on a map - cycleway tagged routes
        self.detected_features           = [] # Features we are building to draw on a map - detected routes
        self.both_features               = [] # " - routes that are in both OSM and the detected routes
        self.either_features             = [] # " - routes that are in either OSM or the detected routes
        self.tagged_only_features        = [] # " - routes that are tagged in OSM but were not detected
        self.detected_only_features      = [] # " - routes that were detected but are not tagged in OSM
        
        self.included_nodes              = {} # Nodes that were included in a survey route, value=way_id_start
        self.filtered_nodes              = {} # Nodes that were filtered
        
        
        # If we have been given a CSV file of points to include (because we are filtering to a route)
        # then record them in a dict, ready to compare to the XML data as we load it
        # We want to know this up front so we can disregard OSM data as we parse it,
        # if we need to do that to save memory
        if filter_log is not None:
            self.load_filter_log(filter_log)
            
        # Load the main XML file into memory
        # This assumes that we have reduced the OpenStreetMap data down to a small enough locality
        # that the in-memory approach is feasible
        doc_main = xml.dom.minidom.parse(filename_main)

        # Call a helper function to atually interpret the XML
        self.process_osm_xml(doc_main, intersections_only=False, filter_log=filter_log, verbose=verbose)
        
        if filename_margin:
            # Load a slightly bigger XML file into memory, to catch nodes that are JUST outside the
            # boundary of the locality
            # We only add information to our cache if it relates to ways associated with each node,
            # which allows us to correctly identify intersections (even if the intersecting road is
            # outside the boundary of the main XML file apart from the intersection itself)
            doc_margin = xml.dom.minidom.parse(filename_margin)

            self.process_osm_xml(doc_margin, intersections_only=True, verbose=verbose)
            
        # Report on how many nodes were loaded from the OSM extract(s) and whether any of them were filtered
        if filter_log is not None:
            print('Nodes Loaded: [{0:d}] Filtered = [{1:d}]'.format(len(self.node_coords), len(self.filtered_nodes)))
            
            
    def load_filter_log(self, filter_log):
        '''
        Load a list of way IDs and node IDs from a CSV file to include when loading subsequent
        OSM XML extracts.  If we load a filter log, any other way ID or node ID not on the whitelist
        will be ignored
        '''
        df = pd.read_csv(filter_log)
        
        for index in trange(len(df.index)):
            row = df.iloc[[index]]
                
            way_id_start = str(row['way_id_start'].item())
            node_id      = str(row['node_id'].item())
            
            self.included_nodes[node_id] = way_id_start
            

    def process_osm_xml(self, doc, intersections_only=False, filter_log=None, verbose=True):
        '''
        Parse the OSM XML data
        
        Parameters
        ----------
        doc : xml document
            XML document contents being loaded/interpreted
            
        intersections_only : boolean, optional
            Can be set to True to only load information relating to intersections from this file
        
        filter_log : str, optional
            The path to a "filter log" CSV file containing the list of way IDs and node IDs
            to include when loading the OSM data.  If the OSM data is much bigger than the
            route or area we want to look at.  Only used in the early stages of the project,
            and deprecated in favour of using the "osmium" tool to reduce OSM extracts down
            to a selected area
            
        verbose : boolean, optional
            Specify whether debug messages should be written to STDOUT
        '''
        
        # Get ways and nodes from XML document
        ways_xml  = doc.getElementsByTagName('way')
        nodes_xml = doc.getElementsByTagName('node')

        # First, process the ways in turn, displaying a progress bar in Jupyter Notebook
        for way_idx in trange(0, len(ways_xml)):
            way = ways_xml[way_idx]
            
            # Get the ID for this way
            way_id = way.getAttribute('id')
            
            # Start off assuming no cycleway tag has been found for the way
            is_cycleway = False
                            
            # Examine each "tag" for the Way in turn
            tags = way.getElementsByTagName('tag')
            found_name = False
            found_hwy  = False
            way_name   = 'Unnamed'
            
            for tag in tags:
                k = tag.getAttribute('k')
                v = tag.getAttribute('v').upper()
                
                # We want to find a name for the way
                
                # First preference is to use the actual name
                if not found_name and k == 'name':
                    way_name   = v
                    found_name = True
                
                # Otherwise use generic "ROUNDABOUT" or "SERVICE" for unnamed segments
                elif k == 'junction' or k == 'highway':
                    if not found_name:
                        way_name  = v
                    found_hwy = True
                
                # Skip natural features like cliff, coastline, wood, beach, water, bay
                elif k == 'natural':
                    way_name   = None
                    found_name = True
                # Skip waterways like stream
                elif k == 'waterway':
                    way_name   = None
                    found_name = True
                # Skip railways
                elif k == 'railway':
                    way_name   = None
                    found_name = True
                # Skip reserves
                elif k == 'landuse':
                    way_name   = None
                    found_name = True
                elif k == 'leisure' or k == 'website':
                    way_name   = None
                    found_name = True
                  
                # Exclude unnamed FOOTWAY
                # Otherwise we get a squiggle near the scout hall at Baden Powell Drive,
                # where someone drew a meandering path in the grass.  GSV will give us
                # closest image, which WILL see a bike logo (on the ROAD, from the ROAD)
                # and the whole track looks like a bike path to nowhere
                # way_id 289029036
                # Exclude paths, e.g. Mornington Rail Trail
                # Exclude off-road cycleways e.g. Peninsula Link Trail
                # Exclude pedestrian overpasses
                if k == 'highway' and v in ['FOOTWAY', 'PATH', 'STEPS', 'TRACK', 'CYCLEWAY']:
                    way_name   = None
                    found_name = True
                    found_hwy  = True
                        
                # Identify cycleways
                if k.upper().startswith('CYCLEWAY'):
                    if v not in ['NO']:
                        is_cycleway = True
            
            # If after reading all the tags, we have a name for the way and we
            # didn't set it to None due to it being a cliff, etc. record its information
            # to the cache dictionaries
            if way_name is not None and found_hwy:
                # Remember the name for this way, so we don't have the parse the whole way again later
                self.way_names_by_id[way_id] = way_name
                
                # Add this way to the list of ways by that name
                if not intersections_only:
                    if way_name in self.ways_by_name:
                        self.ways_by_name[way_name].append(way_id)
                    else:
                        self.ways_by_name[way_name] = [way_id]
            
                    # Records the way by its way id
                    # We only add ways that have a name, implicitly excluding "natural" ways such as coastline
                    self.ways_by_id[way_id] = way
    
                    # Record whether this was tagged as a cycleway or not
                    self.way_is_cycleway[way_id] = is_cycleway
                    
                # Record the association betwee this way and each of its nodes
                node_refs = way.getElementsByTagName('nd')
                for idx, node_ref in enumerate(node_refs):
                    ref = node_ref.getAttribute('ref')
                    
                    # Apply node filtering if required
                    if filter_log is None or ref in self.included_nodes:
                        if ref in self.way_names_per_node:
                            if way_name not in self.way_names_per_node[ref]:
                                self.way_names_per_node[ref].append(way_name)
                        else:
                            self.way_names_per_node[ref] = [way_name]
                        
                        if ref in self.way_ids_per_node:
                            if way_id not in self.way_ids_per_node[ref]:
                                self.way_ids_per_node[ref].append(way_id)
                        else:
                            self.way_ids_per_node[ref] = [way_id]
                        
                        if way_id in self.way_node_ids:
                            self.way_node_ids[way_id].append(ref)
                        else:
                            self.way_node_ids[way_id] = [ref]
                        
                        if idx == 0:
                            self.way_starts[way_id] = ref
                        if idx == len(node_refs) - 1:
                            self.way_ends[way_id] = ref
                    else:
                        self.filtered_nodes[ref] = True
                
                # Record the start and end node IDs for the way
                node_ends = [node_refs[0], node_refs[-1]]
                for node_ref in node_ends:
                    ref = node_ref.getAttribute('ref')

                    # Apply node filtering if required
                    if filter_log is None or ref in self.included_nodes:
                        if ref in self.way_ends_per_node:
                            if way_id not in self.way_ends_per_node[ref]:
                                self.way_ends_per_node[ref].append(way_id)
                        else:
                            self.way_ends_per_node[ref] = [way_id]  
                    else:
                        self.filtered_nodes[ref] = True
                
                recorded = True
        
        # Now, process the node details, from a different part of the XML
        # Cache nodes in dict by id/ref
        if not intersections_only:
            for node in nodes_xml:
                id = node.getAttribute('id').upper()
                
                # Apply node filtering if required
                if filter_log is None or id in self.included_nodes:
                    self.nodes[id] = node
                
                    lat = float(node.getAttribute('lat'))
                    lon = float(node.getAttribute('lon'))
                
                    self.node_coords[id] = [lat, lon]
                else:
                    self.filtered_nodes[id] = True
    
            if verbose:
                print('Way count:          %d' % ways_xml.length)
                print('Included ways:      %d' % len(self.ways_by_id.keys()))
                print('Way names:          %d' % len(self.ways_by_name.keys()))
                print('Node count:         %d' % nodes_xml.length)
        
        if verbose:
            # Count intersections
            intersection_count = 0
    
            for node in self.way_names_per_node.keys():
                if len(self.way_names_per_node[node]) > 1:
                    intersection_count = intersection_count + 1
    
            print('Intersection count: %d' % intersection_count)


    # Get the bearing from one node to the next
    @staticmethod
    def bearing_from_nodes(prev_node, next_node):
        '''
        Get a bearing from one node to the next, by looking up the XML object latitude
        and longitude attributes, and using the geodesic library to calculate it
        
        Parmeters
        ----
        prev_node : xml
            An XML entity representing the first node, with "lat" and "lon" attributes
            
        next_node : xml
            An XML entity representing the second node, with "lat" and "lon" attributes
        
        '''
        lat1 = float(prev_node.getAttribute('lat'))
        lon1 = float(prev_node.getAttribute('lon'))
        lat2 = float(next_node.getAttribute('lat'))
        lon2 = float(next_node.getAttribute('lon'))
    
        bearing = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)['azi1']
        
        # Convert any negative bearings into a number between 0 and 360
        if bearing < 0:
            bearing = bearing + 360
        
        return int(round(bearing))

        
    # Get a series of points from one point to the next, at regular intervals up to a desired offset
    @staticmethod
    def expand_offsets(lat1, lon1, lat2, lon2, max_offset, interval, way_id_start, way_id, node_id, way_name='-'):
        '''
        Given two points, generate a list of points every "interval" metres between them, up to "max_offset" offset
        
        Useful if trying to generate a list of sample points every X metres away from a no
        de, in the direction
        of the next node
        road in a survey area
        
        Parameters
        ----------
        
        lat1 : float
            Latitude of point 1, which will correspond to node_id
            
        lon1 : float
            Longitude of point 1, which will correspond to node_id
            
        lat2 : float
            Latitude of point 2, which is the direction of the next node, showing us which direction to move
            
        lon2 : float
            Longitude of point 2, which is the direction of the next node, showing which direction to move
            
        max_offset : ?
        
        interval : int
            Number of metres between each point in the output series
            
        way_id_start : int
            The way_id_start that this pair of points was sampled from
            This is a special way_id that is the first way in a string of adjoining
            ways with the same name that were re-connected
            OSM breaks up roads into multiple ways if e.g. the speed limit or some
            other characteristic changes
        
        way_id : int
            The way_id that this pair of points is sampled from
            
        node_id : int
            The node_id that this point is sampled from, probably an intersection node
            
        way_name : str, optional
            The name of the way that this pair of points is sampled from
        '''
        
        # Initialise the list of sample point sthat will be returned
        # Each piont is a list with elements in a set order, instead of a custom class as it probably should be
        sample_points = []
    
        # Calculate a bearing from the node to the next point
        bearing = Geodesic.WGS84.Inverse(float(lat1), float(lon1), float(lat2), float(lon2))['azi1']
    
        # Generate a line between the points, we need this to make sure we don't go
        # beyond the next point, that would be a waste because we might go too far off
        # track, and we might end up with far more points than we need
        line = Geodesic.WGS84.InverseLine(float(lat1), float(lon1), float(lat2), float(lon2))
    
        # Calculate how many steps are required to get to max_offset by interval
        num_steps     = int(math.ceil(abs(max_offset) / interval))
        
        # Calculate a maximum number of steps based on the distance between the two nodes and the interval
        # line and num_steps_max were useful when building a list of sample points every interval,
        # along every road.  When we switched to just sampling the immediate area around intersections,
        # "num_steps_max" became no longer used.  Leaving them here in the code in case we later want to go
        # back and support the original option
        num_steps_max = int(math.ceil(line.s13 / interval))
    
        # Iterate through the required number of steps
        if max_offset < 0:
            polarity = -1
        else:
            polarity = 1
        
        for step_i in range(num_steps + 1):
            if step_i > 0:
                # Make sure the step size does not go beyond the next point, potentially off the path
                s = min(interval * step_i, line.s13)
                
                # Generate a new point s distance along the line
                g = line.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            
                # Build the list that represents the point and the metadata to show where it was
                # sampled from
                sample_point = [
                    g['lat2'],
                    g['lon2'],
                    int(round(bearing)),
                    step_i * interval * polarity,
                    way_id_start,
                    way_id,
                    node_id,
                    way_name
                ]
            
                # Append this point to the output
                sample_points.append(sample_point)
    
        return sample_points


    # Get every sample point for a way
    def walk_way_intersections_by_id(self, way_id_start, way_id, min_offset=0, max_offset=0, interval=10, debug=False):
        '''
        Generate a list of sample points around intersections on a linked series of ways
        
        A sample point will be generated very interval metres to either side of every intersection,
        within a range of offsets given by min_offset and max_offset.  If min_offset and max_offset are both
        zero, then only the intersection point itself will be sampled, right in the middle of the intersection.
        
        Parameters
        ----------
        way_id_start : int
            The way_id at the start of a linked sequence of ways with the same name, that we are traversing
            
        way_id : int
            The way_id within the linked sequence of ways
            
        min_offset : int, optional
            minimum offset metres from intersection to sample
            
        max_offset : int, optional
            maximum offset metres from intersection to sample
            
        interval : int, optional
            The number of metres between each sample point
            
        debug : boolean
            Whether debug messages should be printed to STDOUT
        '''
        
        # Initialise list of points that will be returned
        sample_points = []
    
        # Retrieve the way
        way = self.ways_by_id[way_id]
    
        # Iterate through nodes in the way
        node_refs = way.getElementsByTagName('nd')
    
        idx_first = 0
        idx_last  = len(node_refs) - 1
    
        for idx, node_ref in enumerate(node_refs):
            ref = node_ref.getAttribute('ref')
        
            # Only examine intersection nodes, and the start and end nodes
            if (self.is_intersection_node(ref)) or (idx == 0) or (idx == len(node_refs) - 1):
                if debug:
                    print('Debug Node Intersection: {0:s} {1:d} {2:.6f}, {3:.6f}'.format(ref, len(self.way_names_for_node[ref]),
                        float(self.nodes[ref].getAttribute('lat')), float(self.nodes[ref].getAttribute('lon'))))
            
                # Found an intersection!  We will output for this one
            
                # Find any negative offset samples required
                if idx > idx_first and min_offset < 0 and interval > 0:
                    # Create a list of sample points for the intersection based on the heading
                    # from the previous node on the way
                    prev_points = osm_walker.expand_offsets(
                        self.nodes[node_refs[idx  ].getAttribute('ref')].getAttribute('lat'),
                        self.nodes[node_refs[idx  ].getAttribute('ref')].getAttribute('lon'),
                        self.nodes[node_refs[idx-1].getAttribute('ref')].getAttribute('lat'),
                        self.nodes[node_refs[idx-1].getAttribute('ref')].getAttribute('lon'),
                        min_offset,
                        interval,
                        way_id_start,
                        way_id,
                        ref,
                        self.way_names_by_id[way_id]
                    )
                
                    # Add to the output points for the way
                    sample_points = sample_points + prev_points[::-1] # Reversed with slicing
            
                # Find the bearing at the node itself, and output the node itself
                if idx == idx_first:
                    bearing = osm_walker.bearing_from_nodes(
                        self.nodes[node_refs[idx  ].getAttribute('ref')],
                        self.nodes[node_refs[idx+1].getAttribute('ref')]
                    )
                elif idx == idx_last:
                    bearing = osm_walker.bearing_from_nodes(
                        self.nodes[node_refs[idx-1].getAttribute('ref')],
                        self.nodes[node_refs[idx  ].getAttribute('ref')]
                    )
                else:
                    bearing = osm_walker.bearing_from_nodes(
                        self.nodes[node_refs[idx-1].getAttribute('ref')],
                        self.nodes[node_refs[idx+1].getAttribute('ref')]
                    )

                sample_point = [
                    float(self.nodes[ref].getAttribute('lat')), float(self.nodes[ref].getAttribute('lon')), bearing, 0, way_id_start, way_id, ref, self.way_names_by_id[way_id]
                ]
                        
                sample_points.append(sample_point)
                        
                # Find any postive offset samples required
                if idx < idx_last and max_offset > 0 and interval > 0:
                    sample_points = sample_points + osm_walker.expand_offsets(
                        self.nodes[node_refs[idx  ].getAttribute('ref')].getAttribute('lat'),
                        self.nodes[node_refs[idx  ].getAttribute('ref')].getAttribute('lon'),
                        self.nodes[node_refs[idx+1].getAttribute('ref')].getAttribute('lat'),
                        self.nodes[node_refs[idx+1].getAttribute('ref')].getAttribute('lon'),
                        max_offset,
                        interval,
                        way_id_start,
                        way_id,
                        ref,
                        self.way_names_by_id[way_id]
                    )
            else:
                if debug:
                    print('Debug Node NON-Intersection: {0:s} {1:d} {2:.6f}, {3:.6f}'.format(ref, len(self.way_names_for_node),
                        float(self.nodes[ref].getAttribute('lat')), float(self.nodes[ref].getAttribute('lon'))))
        
        return sample_points


    # Given a way by its ID, are there any other ways with the same name that intersect
    # with its first node?
    def is_way_start(self, way_id, verbose=False):
        '''
        Given a way by its IDS, are there any other ways WITH THE SAME NAME that intersect with
        its first node?  If not, then it is a way_id_start, the first way in potentially a sequence
        of linked ways.  Otherwise, it is just one of the later ways in a sequence.
        
        Parameters
        ----------
        way_id : int
            The way id to check
            
            
        verbose : boolean, optional
            Whether to output debug messages to STDOUT
        '''
        
        # Retrieve the cached XML details of the way, and find its first node ID
        way           = self.ways_by_id[way_id]
        node_refs     = way.getElementsByTagName('nd')
        first_node_id = node_refs[0].getAttribute('ref')
        this_way_name = self.way_names_by_id[way_id]
        
        if verbose:
            print('First node: ' + first_node_id)
            
        # Are there any other ways that join, end-to-end, with this node, with any way name?
        if first_node_id not in self.way_ends_per_node:
            intersecting_ways = []
        else:
            intersecting_ways = self.way_ends_per_node[first_node_id]

        # If there is only one way linked to the first node, then it must be this
        # way, and therefore this is the start of a way
        if len(intersecting_ways) <= 1:
            if verbose:
                print('No intersecting way ends for ' + first_node_id)
            return True
    
        # How many ways with the same name are linked to the first node?
        intersecting_ways_with_this_name = 0
              
        for intersecting_way_id in intersecting_ways:
            intersecting_way_name = self.way_names_by_id[intersecting_way_id]
            
            if verbose:
                print('Intersecting way end: ' + intersecting_way_name + ' (' + intersecting_way_id + ')')
                        
            if intersecting_way_name == this_way_name and intersecting_way_id in self.ways_by_id:
                intersecting_ways_with_this_name = intersecting_ways_with_this_name + 1
        
        if verbose:
            print('Intersecting way ends with same name: ' + str(intersecting_ways_with_this_name))
            
        # If there is only one way with this name linked to the first node, then it is this one!
        if intersecting_ways_with_this_name <= 1:
            return True
    
        return False
        
        
    # At the end of a way, does it join to another way by the same name?
    def find_next_way(self, way_id, match_name=True, include_used=False, verbose=False):
        '''
        At the end of a way, does it join to another way by the same name?
        
        Parameters
        ----------
        way_id : int
            The id of the way being checked
            
        match_name : boolean, optional
            Whether we are requiring that the adjoining way have the same name,
            or whether it could have a different name e.g. "ROUNDABOUT"
            
        include_used : boolean, optional
            Whether to include or exclude ways that we have already traversed as we work
            our way through the survey area
            
        verbose : boolean, optional
            Whether debug messages should be written to STDOUT
        '''
        
        # Retrieve the details of the way, and find its last node ID
        way           = self.ways_by_id[way_id]
        node_refs     = way.getElementsByTagName('nd')
        last_node_id  = node_refs[-1].getAttribute('ref')
        this_way_name = self.way_names_by_id[way_id]
    
        # Are there any other ways that intersect with this node, with any way name?
        if last_node_id not in self.way_ends_per_node:
            intersecting_ways = []
        else:
            intersecting_ways = self.way_ids_per_node[last_node_id]
        
        if verbose:
            print('Last node ID: {0:s}'.format(last_node_id))
            print('Intersecting way count: {0:d}'.format(len(intersecting_ways)))

        # See if any of them have the same name
        for intersecting_way_id in intersecting_ways:
            # Exclude this way itself
            if intersecting_way_id != way_id:
                if verbose:
                    print('Checking way {0:s}'.format(intersecting_way_id))
                    
                if intersecting_way_id in self.unused_way_ids or include_used:
                    if match_name:
                        intersecting_way_name = self.way_names_by_id[intersecting_way_id]
            
                        if intersecting_way_name == this_way_name and intersecting_way_id in self.ways_by_id:
                            return intersecting_way_id
                    else:
                        return intersecting_way_id
    
        return None

  
    # Find intersection points (and offsets from intersections) for all ways
    # Walking down ajoining ways of the same name in a sensible order, where possible
    def sample_all_way_intersections(self, min_offset, max_offset, interval=10, ordered=False, verbose=False):
        '''
        Find sample points at every intersection along every way in the survey area
        
        Parameters
        ----------
        min_offset : int
            minimum offset metres from intersection to sample
            
        max_offset : int
            maximum offset metres from intersection to sample
            
        interval : int, optional
            The number of metres between each sample point
            
        ordered : boolean, optional
            If this is set to false, we will just traverse every way, in any random order.
            If it is set to true, we will attempt to link up ways into coherent chains
            with a way_id_start, which is a little more involved but might make the output
            sample images easier for a human to wrap their head around due to more coherent
            sample order.
            
        verbose : boolean, optional
            Whether debug messages should be written to STDOUT
        
        '''
        all_points = []

        # If we don't want to try to walk in a natural order, do it the simple and fast way
        if not ordered:
            for way_id in self.ways_by_id.keys():
                points = self.walk_way_intersections_by_id(way_id, way_id, min_offset=min_offset, max_offset=max_offset, interval=interval)
    
            all_points = all_points + points

            return all_points
    
        # Process ways in name order, so we can walk down streets that are divided into multiple ways in an
        # order that appears sensible to a human
        
        # Link together way sections that belong to the same name and join end-to-end
        self.link_way_sections(verbose=verbose)
        
        # Iterate through each way section that we've linked back up into a chain (or included as-is
        # if it does not belong to a larger chain)
        all_points = []
        
        for way_id_start in self.linked_way_sections.keys():
            section = self.linked_way_sections[way_id_start]
            
            if verbose:
                print('{0:s} {1:s} {2:4d}'.format(way_id_start, self.way_names_by_id[way_id_start], len(section)))
        
            section_points = []
            
            for way_id in section:
                if verbose:
                    print('==> {0:s}'.format(way_id))
                    
                section_points = section_points + self.walk_way_intersections_by_id(way_id_start, way_id, min_offset=min_offset, max_offset=max_offset, interval=interval)
            
            all_points = all_points + section_points
                
        return all_points
    
 
    def link_way_sections(self, filter_way_name=None, verbose=False):
        '''
        Where continuous roads have been fragmented into ways due to a change of characteritic
        (e.g. speed limit change) we link them back up and store information about
        the resulting chains.
        
        Parameters
        ----------
        filter_way_name : str, optional
            Optionally only look to link up way segments for ways of one particular name
            
        verbose : boolean, optional
            Specify whether debug messages should be written to STDOUT
        '''
        # Process ways in name order, so we can walk down streets that are divided into multiple ways in an
        # order that appears sensible to a human
    
        # Reset linkages between ways of same name
        self.linked_way_sections      = {}
        self.linked_way_sections_all  = {}
        self.linked_way_sections_cway = {}
        
        self.linked_linestrings       = {}
        self.linked_coord_list        = {}
        
        # Iterate through each distinct way name (including generic ways like "junction")
        for way_name in self.ways_by_name.keys():
            if filter_way_name is not None and way_name != filter_way_name:
                continue
                
            if verbose:
                print('WAY: ' + way_name)
            
            # Find a list of way_ids with this name where the first node does not intersect
            # with another way of the same name
            way_starts = []
        
            # Find all the ways by this name
            way_ids = self.ways_by_name[way_name]
        
            for way_id in way_ids:
                # Check if this way appears to be a "starting way"
                if self.is_way_start(way_id):
                    way_starts.append(way_id)
                                    
            # Initialise a list to keep track of ways that we haven't walked yet
            self.unused_way_ids = way_ids.copy()
                    
            # Process the next way start
            while len(way_starts) > 0:
                # Process the next start way id
                way_id_start = way_starts.pop()
                way_id       = way_id_start
                                                
                section     = [way_id]
                
                self.way_id_start_by_way_id[way_id] = way_id_start
                                
                # Retrieve the details of the way, and find each node ID
                section_all = []
                section_cwy = []
                coord_list  = []
                                
                way         = self.ways_by_id[way_id]
                node_refs   = way.getElementsByTagName('nd')
                
                for node_ref in node_refs:
                    ref = node_ref.getAttribute('ref')
                    if len(self.included_nodes) < 1 or ref in self.included_nodes:
                        if ref not in section_all:
                            section_all.append(ref)
                            section_cwy.append(self.way_is_cycleway[way_id])
                            these_coords = self.node_coords[ref]
                            coord_list.append((these_coords[0], these_coords[1]))
                        
                if verbose:
                    print('{3:10d} {0:5s} {1:10s} {2:s} {4:d}'.format('START', way_id, self.way_names_by_id[way_id], len(self.unused_way_ids), len(coord_list)))
                 
                # Remove this way start from the list of ways we have not processed yet
                self.unused_way_ids.remove(way_id)
            
                # Recursively walk down adjoining ways in order
                next_way_id = self.find_next_way(way_id, match_name=True)
            
                while next_way_id:
                    way_id = next_way_id
                                        
                    section = section + [way_id]
                    
                    self.way_id_start_by_way_id[way_id] = way_id_start
                                    
                    way         = self.ways_by_id[way_id]
                    node_refs   = way.getElementsByTagName('nd')
                
                    for node_ref in node_refs:
                        ref = node_ref.getAttribute('ref')
                        if len(self.included_nodes) < 1 or ref in self.included_nodes:
                            if ref not in section_all:
                                section_all.append(ref)
                                section_cwy.append(self.way_is_cycleway[way_id])
                                these_coords = self.node_coords[ref]
                                coord_list.append((these_coords[0], these_coords[1]))
                                           
                    if verbose:
                        print('{3:10d} {0:5s} {1:10s} {2:s} {4:d}'.format('CONT', way_id, self.way_names_by_id[way_id], len(self.unused_way_ids), len(coord_list)))
                                  
                    # Remove this way start from the list of ways we have not processed yet
                    self.unused_way_ids.remove(way_id)
                    if way_id in way_starts:
                        way_starts.remove(way_id)
                
                    next_way_id = self.find_next_way(way_id, match_name=True)
                
                self.linked_way_sections_all[way_id_start] = section_all
                self.linked_way_sections_cwy[way_id_start] = section_cwy
                self.linked_way_sections[way_id_start]     = section
                if len(coord_list) > 1:
                    self.linked_linestrings[way_id_start] = LineString(coord_list)
                    self.linked_coord_list[way_id_start]  = coord_list
                
                if verbose:
                    print('Saving {0:s} {1:s}'.format(way_name, way_id_start))
                    print('section:     {0:3d} {1:s}'.format(len(section), str(section)))
                    print('section_all: {0:3d} {1:s}'.format(len(section_all), str(section_all)))
                    
            # Handle any unused ways that weren't a "way start" and weren't linked to one
            while len(self.unused_way_ids) > 0:
                # Process the next way id
                way_id_start = self.unused_way_ids.pop()
                way_id       = way_id_start
                                
                section = [way_id]
                
                self.way_id_start_by_way_id[way_id] = way_id_start
            
                # Retrieve the details of the way, and find each node ID
                section_all = []
                section_cwy = []
                coord_list  = []
                
                way         = self.ways_by_id[way_id]
                node_refs   = way.getElementsByTagName('nd')
                
                for node_ref in node_refs:
                    ref = node_ref.getAttribute('ref')
                    if len(self.included_nodes) < 1 or ref in self.included_nodes:
                        if ref not in section_all:
                            section_all.append(ref)
                            section_cwy.append(self.way_is_cycleway[way_id])
                            these_coords = self.node_coords[ref]
                            coord_list.append((these_coords[0], these_coords[1]))
                                        
                if verbose:
                    print('{3:10d} {0:5s} {1:10s} {2:s} {4:d}'.format('NEXT', way_id, self.way_names_by_id[way_id], len(self.unused_way_ids), len(coord_list)))
            
                # Recursively walk down any ajoining ways in order
                next_way_id = self.find_next_way(way_id, match_name=False)
            
                while next_way_id:
                    way_id = next_way_id
                                        
                    section = section + [way_id]
                    
                    self.way_id_start_by_way_id[way_id] = way_id_start
                
                    way         = self.ways_by_id[way_id]
                    node_refs   = way.getElementsByTagName('nd')
                
                    for node_ref in node_refs:
                        ref = node_ref.getAttribute('ref')
                        if len(self.included_nodes) < 1 or ref in self.included_nodes:
                            if ref not in section_all:
                                section_all.append(ref)
                                section_cwy.append(self.way_is_cycleway[way_id])
                                these_coords = self.node_coords[ref]
                                coord_list.append((these_coords[0], these_coords[1]))
                                            
                    if verbose:
                        print('{3:10d} {0:5s} {1:10s} {2:s} {4:d}'.format('CONT', way_id, self.way_names_by_id[way_id], len(self.unused_way_ids), len(coord_list)))
            
                    # Remove this way start from the list of ways we have not processed yet
                    self.unused_way_ids.remove(way_id)
                
                    next_way_id = self.find_next_way(way_id, match_name=False)
                
                self.linked_way_sections_all[way_id_start] = section_all
                self.linked_way_sections_cwy[way_id_start] = section_cwy
                self.linked_way_sections[way_id_start]     = section
                if len(coord_list) > 1:
                    self.linked_linestrings[way_id_start] = LineString(coord_list)
                    self.linked_coord_list[way_id_start]  = coord_list
                
                if verbose:
                    print('Saving {0:s} {1:s}'.format(way_name, way_id))
                    print('section:     {0:3d} {1:s}'.format(len(section), str(section)))
                    print('section_all: {0:3d} {1:s}'.format(len(section_all), str(section_all)))                 
    
    
    def load_detection_log(self, filename):
        '''
        Load a detection log into memory, indexed by latitude-longitude
        
        Parameters
        ----------
        filename : str
            Path to the detection_log.csv file to be loaded
        '''
        df = pd.read_csv(filename)
        
        for index, row in df.iterrows():
            # row['lat'], lon, bearing, way_id_start, way_id, node_id, offset_id, score, bbox_0, bbox_1, bbox_2, bbox_3
            key = '{0:.0f}-{1:.0f}'.format(row['way_id_start'], row['node_id'])

            if key in self.detection_hits:
                self.detection_hits[key].append([row['offset_id'], row['score']])
            else:
                self.detection_hits[key] = [[row['offset_id'], row['score']]]
                
                
    def snap_detection_log(self, filename_in, filename_out):
        '''
        Load a detection log, and for each point, find the closest corresponding way/node
        in the OpenStreetMap data, so that the detection is "aligned" to the road network,
        and we can easily compare the detected routes to what is tagged in OpenStreetMap.
        
        Parameters
        ----------
        filename_in : str
            Path to the input detection_log.csv
            
        filename_out : str
            Path to the output detection log where the points have been aligned
            to the closest way_id and node_id
        '''
        
        # Read the original file
        df = pd.read_csv(filename_in)
        
        # Initialise the output file, with header
        output_file = open(filename_out, 'w')
        output_file.write('lat,lon,bearing,heading,way_id_start,way_id,node_id,offset_id,score,bbox_0,bbox_1,bbox_2,bbox_3,way_name\n')
            
        # Process the input detection log one entry at a time
        tqdm.pandas() 
             
        df.progress_apply(lambda row: self.snap_row(
            output_file,
            row['lat'],
            row['lon'],
            row['bearing'],
            row['heading'],
            row['score'],
            row['bbox_0'],
            row['bbox_1'],
            row['bbox_2'],
            row['bbox_3']),
            axis=1
        )
               

    def snap_row(self, output, lat, lon, bearing, heading, score, bbox_0, bbox_1, bbox_2, bbox_3):
        '''
        Align a single row/point in a detection log to the closest corresponding way_id and node_id
        and that node_id's precise location
        
        Parameters
        ----------
        lat : float
            The latitude of the original point (e.g. from the dash camera GPS sensor
            
        lon : float
            The longitude of the original point (e.g. from the dash camera GPS sensor
        
        bearing : int
            The original bearing at the detection point
            
        heading : int
            The original bearing at the detection point, will be a copy of heading
            
        score : float
            The original score in the detection log
            
        bbox_0 : float
            The original bounding box recorded in the detection log
            
        bbox_1 : float
            The original bounding box recorded in the detection log
            
        bbox_2 : float
            The original bounding box recorded in the detection log
            
        bbox_3 : float
            The original bounding box recorded in the detection log
        '''
        
        # Create a Point object from the original location
        p = Point(lat, lon)
        
        # Do this TWICE, once to find the closest intersection node, and once to find the closest node
        # of any type`
        for option in [True, False]:
            # Find the nearest node in the OpenStreetMap data for the point
            # Returns the way_id_start and node_id, plus the distance and whether it is an intersection
            way_id_start, node_id, distance, is_intersection = self.find_nearest_node(p, want_intersection=option)
        
            # If the closest node is an intersection, don't output it twice
            if (option and is_intersection) or (not option and not is_intersection):
                way_name = self.way_names_by_id[way_id_start]
            
                output.write('{0:.6f},{1:.6f},{2:d},{3:d},{4:d},{5:d},{6:d},{7:f},{8:f},{9:f},{10:f},{11:f},{12:f},{13:s}\n'.format(
                    lat,
                    lon,
                    int(bearing),
                    int(heading),
                    int(way_id_start),
                    int(way_id_start),
                    int(node_id),
                    distance,
                    score,
                    bbox_0,
                    bbox_1,
                    bbox_2,
                    bbox_3,
                    way_name + ' ' + str(option)
                ))

   
    def is_intersection_node(self, node_id):
        '''
        Check if a node is an intersection
        
        A node is an intersection if there are multiple ways that share the node, with more than one
        distinct way name, where the way is a road and not a property boundary or natural feature
        (Property boundaries and natural features etc. are already filtered out as we load the OpenStreetMap
        XML data into memory)
        
        Parameters
        ----------
        node_id : int
            The node to check
        '''
        if len(self.way_names_per_node[node_id]) > 1:
            return True
        else:
            return False
        
        
    def write_geojsons(self, name, geojson_directory, intersection_skip_limit=1, infer_ends=True, verbose=False):
        '''
        Once we have detected our routes, write a series of geojson files to show our detected routes,
        vs. what is tagged in OpenStreetMap.  Additional geojson files are created to show where they agree
        and disagree
        
        Parameters
        ----------
        name : str
            A locality name or label to include as the prefix to all output geojson files
        
        geojson_directory : str
            The directory where all geojson files will be written
            
        intersection_skip_limit : int, optional 
            The number of intersections between "hits" that can be "misses" before the assume that a bicycle
            lane route was interrupted
            
        infer_ends : boolean, optional
            When intersection_skip_limit > 0, should that be applied to fill in a small gap at the
            end of a road?  Or just to small gaps sandwiched between "hits" on the road
            
        verbose : boolean, optional
            Specify whether debug messages should be written to STDOUT
        '''
        
        # Initialise lists of geojson "features" representing lines on each map
        self.detected_features      = []
        self.tagged_features        = []
        self.both_featurtes         = []
        self.detected_only_features = []
        self.tagged_only_features   = []
        
        # For each linked chain of ways, find the parts of the way that appear to have bicycle lane routes,
        # and save the output to the above features lists
        for way_id_start in self.linked_way_sections.keys():
            self.draw_way_segment(way_id_start, intersection_skip_limit=intersection_skip_limit, infer_ends=infer_ends, verbose=verbose)
        
        # Take the features lists and write them to the corresponding geojson files
        self.write_geojson(name, geojson_directory, 'hit',      self.detected_features)
        self.write_geojson(name, geojson_directory, 'tag',      self.tagged_features)
        self.write_geojson(name, geojson_directory, 'both',     self.both_features)
        self.write_geojson(name, geojson_directory, 'either',   self.either_features)
        self.write_geojson(name, geojson_directory, 'hit_only', self.detected_only_features)
        self.write_geojson(name, geojson_directory, 'tag_only', self.tagged_only_features)
        
        # Measure the total distance of routes in each geojson files, to provide a comparison measure
        for part in ['hit', 'tag', 'both', 'either', 'hit_only', 'tag_only']:
            geojson_filename = os.path.join(geojson_directory, part + '.geojson')
            
            distance = osm_walker.geojson_distance(geojson_filename)
            
            print('{0:8s}: Total distance {1:10.2f}m'.format(part, distance))
        
        
    def write_geojson(self, name, geojson_directory, mode, features):
        '''
        Write a geojson file based on a list of features we have detected as routes for the map
        
        Parameters
        ----------
        name : str
            A locality name or label to include as the prefix to all output geojson files
        
        geojson_directory : str
            The directory where all geojson files will be written
            
        mode : str
            A string to include in the output filename, after the "name" to describe the content,
            e.g. "hit", "both", etc.
            
        features : list
            The list of features to write to the geojson file
        '''
        
        print('Writing ' + mode + ', feature count: ' + str(len(features)))
        
        # Determine the output filename
        label = '{0:s}_{1:s}'.format(name, mode)
        
        geojson_filename = os.path.join(geojson_directory, mode + '.geojson')
        
        # Construct a FeatureCollection from the Features list
        featurecollection = {
            'type':     'FeatureCollection',
            'name':     label,
            'features': features
        }
        
        # Write the file
        print('Writing to: ' + geojson_filename)
        with open(geojson_filename, 'w') as outfile:
            json.dump(featurecollection, outfile, indent=4)
            outfile.close()
    
    
    def draw_way_segment(self, way_id_start, intersection_skip_limit=1, infer_ends=True, verbose=False):
        '''
        For a given way, infer bicycle lane routes based on the detections we have loaded from
        a detection log, and possibly aligned to the OpenStreetMap nodes if the data came from
        a dash camera where the points were not already aligned.
        
        Parameters
        ----------
        way_id_start : int
            The first way_id in a linked chain of ways to assess
            
         intersection_skip_limit : int, optional 
            The number of intersections between "hits" that can be "misses" before the assume that a bicycle
            lane route was interrupted
            
        infer_ends : boolean, optional
            When intersection_skip_limit > 0, should that be applied to fill in a small gap at the
            end of a road?  Or just to small gaps sandwiched between "hits" on the road
            
        verbose : boolean, optional
            Specify whether debug messages should be written to STDOUT           
        '''
        
        # Find each way_id representing a section that was linked to this way_id_start
        self.section_node_list = self.linked_way_sections_all[way_id_start]
        
        # Retrieve info on whether we found any cycleway tags for any way_id associated with way_id_strt
        self.section_node_cwys = self.linked_way_sections_cwy[way_id_start]
        
        self.node_is_intersection = {}
        self.node_hit_detected    = {}
        self.node_hit_assumed     = {}
        self.node_tagged          = {}
        
        # Determine whether each node is an intersection, whether a hit was detected
        for idx, node_id in enumerate(self.section_node_list):
            
            #if (self.is_intersection_node(node_id)) or (idx == 0) or (idx == len(self.section_node_list) - 1):
            if self.is_intersection_node(node_id):
                self.node_is_intersection[node_id] = 1
            else:
                self.node_is_intersection[node_id] = 0
                
            key = str(way_id_start) + '-' + str(node_id)
            
            if key in self.detection_hits:
                self.node_hit_detected[node_id] = 1
            else:
                self.node_hit_detected[node_id] = 0
                
            if self.section_node_cwys[idx]:
                self.node_tagged[node_id] = 1
            elif node_id not in self.node_tagged:
                self.node_tagged[node_id] = 0
                
            self.node_hit_assumed[node_id] = 0       
            
        # Infer between hits if there aren't too many missed intersections
        if intersection_skip_limit > 0:
            prev_hit = self.draw_find_next_hit(-1)
            next_hit = self.draw_find_next_hit(prev_hit)
            
            while (prev_hit is not None) and (next_hit is not None):
                missed_intersections = self.draw_count_intersection_miss_between(prev_hit, next_hit)
            
                if (missed_intersections is not None) and (missed_intersections <= intersection_skip_limit):
                    for idx in range(prev_hit+1, next_hit):
                        self.node_hit_assumed[self.section_node_list[idx]] = 1 + missed_intersections
                    
                prev_hit = next_hit
                next_hit = self.draw_find_next_hit(next_hit)
            
        
        # Infer ends if there are no prior intersections
        if infer_ends:
            # Assume hits before first hit if there are no prior intersections (edge of map)
            # and there were continuous hits assumed to that point
            first_hit          = self.draw_find_next_hit(-1)
            second_hit         = self.draw_find_next_hit(first_hit)
            first_intersection = self.draw_find_next_intersection(-1)
        
            if (first_hit is not None) and (second_hit is not None) and (first_intersection is not None) and (first_intersection >= first_hit):
                missed_intersections = self.draw_count_intersection_miss_between(first_hit, second_hit)
                if missed_intersections <= intersection_skip_limit:
                    for idx in range(0, first_hit):
                        self.node_hit_assumed[self.section_node_list[idx]] = 1
                    
            # Assume hits after last hit if there are no further intersections (edge of map)
            # and there were continuous hits assumed to that point
            last_hit          = self.draw_find_prev_hit(len(self.section_node_list))
            second_last_hit   = self.draw_find_prev_hit(last_hit)
            last_intersection = self.draw_find_prev_intersection(len(self.section_node_list))
        
            if (last_hit is not None) and (second_last_hit is not None) and (last_intersection is not None) and (last_intersection <= last_hit):
                missed_intersections = self.draw_count_intersection_miss_between(second_last_hit, last_hit)
                if missed_intersections <= intersection_skip_limit:
                    for idx in range(last_hit+1, len(self.section_node_list)):
                        self.node_hit_assumed[self.section_node_list[idx]] = 1 + missed_intersections
                
        # Output the conclusion
        if verbose:
            for idx, node_id in enumerate(self.section_node_list):       
                print('{0:12s} {1:3d} => {2:d} {3:d} {4:d} {5:d} {6:.6f}, {7:.6f}   {8:s}'.format(
                    node_id,
                    idx,
                    self.node_is_intersection[node_id],
                    self.node_hit_detected[node_id],
                    self.node_hit_assumed[node_id],
                    self.node_tagged[node_id],
                    self.node_coords[node_id][0],
                    self.node_coords[node_id][1],
                    self.way_names_by_id[way_id_start]
                ))
            
        # Generate geographic feature lists for each map
        self.detected_features      = self.detected_features      + self.draw_way_segment_features(way_id_start, 'hit')
        self.tagged_features        = self.tagged_features        + self.draw_way_segment_features(way_id_start, 'tag')
        self.both_features          = self.both_features          + self.draw_way_segment_features(way_id_start, 'both')
        self.either_features        = self.either_features        + self.draw_way_segment_features(way_id_start, 'either')
        self.detected_only_features = self.detected_only_features + self.draw_way_segment_features(way_id_start, 'hit_only')
        self.tagged_only_features   = self.tagged_only_features   + self.draw_way_segment_features(way_id_start, 'tag_only')   
                

    def draw_way_segment_features(self, way_id_start, mode):
        '''
        Return a list of features for a way_id_start that should be drawn on a map
        
        Parameters
        ----------
        
        way_id_start : int
            The first way_id in a linked chain of ways to assess
            
        mode : str
            The label associated with the features, e.g. "hit", "both", etc.        
        '''
        
        # Retrieve the ordered list of nodes in the linked ways associated with way_id_start
        section_node_list = self.linked_way_sections_all[way_id_start]
        
        # Start with an empty feature list, and an empty list of coordinate lists
        features = []
        coordinates_list_list = []
        
        coordinates_open = False
        coordinates      = []
               
        # Look at each node in turn...
        for idx in range(0, len(section_node_list)):
            node_id = section_node_list[idx]
            
            # Did we detect a bicycle route here?
            # Yes, if either we explicitly detected it, or if we assumed it based on filling in a gap with intersection_skip_limit
            hit = self.node_hit_detected[node_id] + self.node_hit_assumed[node_id]
            
            # Was a bicycle route tagged in the OpenStreetMap data here?`
            tag = self.node_tagged[node_id]
                        
            # Did they both agree there is a cycleway here?    
            if hit and tag:
                both = 1
            else:
                both = 0
                
            # Was a route detected here but not tagged in OpenStreetMap?
            if hit and not tag:
                hit_only = 1
            else:
                hit_only = 0
                
            # Was a route tagged in OpenStreetMap but not detected?                
            if tag and not hit:
                tag_only = 1
            else:
                tag_only = 0
                
            # Did either of them believe there is a bicycle lane here?
            if hit or tag:
                either = 1
            else:
                either = 0
               
            # Based on the "mode" -- which map are we drawing -- decide whether our "pen"
            # is down here.  If the pen is down here and the next node, that gets added
            # to the Feature list and becomes a line on the map
            if mode == 'hit' and hit:
                pen_down = 1
            elif mode == 'tag' and tag:
                pen_down = 1
            elif mode == 'both' and both:
                pen_down = 1
            elif mode == 'either' and either:
                pen_down = 1
            elif mode == 'hit_only' and hit_only:
                pen_down = 1
            elif mode == 'tag_only' and tag_only:
                pen_down = 1
            else:
                pen_down = 0
            
            # If our pen is down, add this to the list of coordinates
            if pen_down:
                coordinates_open = True
                coordinates      = coordinates + [[self.node_coords[node_id][1], self.node_coords[node_id][0]]]
            # If our pen is no longer down, but it was, finish off the line
            elif coordinates_open:
                if len(coordinates) > 1:
                    coordinates_list_list = coordinates_list_list + [coordinates]
                coordinates_open = False
                coordinates      = []
                      
        if coordinates_open:
            if len(coordinates) > 1:
                coordinates_list_list = coordinates_list_list + [coordinates]
            coordinates_open = False
            coordinates      = []
                
        # Take the coordinate list lists for each line, and build the actual "Feature" data structure, then append it
        # to the results
        if len(coordinates_list_list) > 0:               
            for i in range(0, len(coordinates_list_list)):
                feature = {
                    'type': 'Feature',
                    'properties': {
                        'id': (mode + ' ' + str(way_id_start) + '-' + str(i)),
                        'name': (mode + ' ' + str(way_id_start) + '-' + str(i) + '-' + str(len(coordinates_list_list[i]))),
                        'version': '1'
                    },
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': coordinates_list_list[i]
                    }
                }
                
                features.append(feature)
        
        return features
       
    
    def draw_find_next_intersection(self, idx):  
        '''
        Find the next intersection in the list of nodes in the road segment currently being examined
        
        Parameters
        ----------
        idx : int
            The index of the node we are currently looking at within the road segment
        '''
        if idx is None:
            return None
            
        idx += 1
        while idx < len(self.section_node_list):
            if self.node_is_intersection[self.section_node_list[idx]]:
                return idx
            idx += 1
        return None
       
       
    def draw_find_prev_intersection(self, idx):  
        '''
        Find the previous intersection in the list of nodes in the road segment currently being examined
        
        Parameters
        ----------
        idx : int
            The index of the node we are currently looking at within the road segment
        '''    
        if idx is None:
            return None
            
        idx -= 1
        while idx >= 0:
            if self.node_is_intersection[self.section_node_list[idx]]:
                return idx
            idx -= 1
        return None


    def draw_find_next_hit(self, idx): 
        '''
        Find the next node with a "hit" in the list of nodes in the road segment currently being examined
        
        Parameters
        ----------
        idx : int
            The index of the node we are currently looking at within the road segment
        '''    
        if idx is None:
            return None
            
        idx += 1
        while idx < len(self.section_node_list):
            if self.node_hit_detected[self.section_node_list[idx]]:
                return idx
            idx += 1
        return None      
        
        
    def draw_find_prev_hit(self, idx):   
        '''
        Find the previous node with a "hit" in the list of nodes in the road segment currently being examined
        
        Parameters
        ----------
        idx : int
            The index of the node we are currently looking at within the road segment
        '''     
        if idx is None:
            return None
            
        idx -= 1
        while idx >= 0:
            if self.node_hit_detected[self.section_node_list[idx]]:
                return idx   
            idx -= 1
        return None
      
      
    def draw_count_intersection_miss_between(self, prev_hit, next_hit):  
        '''
        Count how many "missed" intersections occur between two "hits" in the list of nodes
        in the road segment currently being examined
        
        Parameters
        ----------
        prev_hit : int
            The index of the first node with a hit, within the road segment
            
        next_hit : int
            The index of the next node with a hit, within the road segment
        ''' 
    
        # Sanity check of inputs
        if prev_hit is None or next_hit is None:
            return None
            
        intersection_miss_between = 0
        
        for idx in range(prev_hit+1, next_hit):
            if (self.node_is_intersection[self.section_node_list[idx]]) and (self.node_hit_detected[self.section_node_list[idx]] == 0):
                intersection_miss_between += 1
                
        return intersection_miss_between
       
       
    def dump_way(self, way_id):
        '''
        Debugging function to show all node ids in a way 
        
        Parameters
        ----------
        way_id : int
            The way id to dump to STDOUT
        '''
        way_name = self.way_names_by_id[way_id]
        way_start = self.way_starts[way_id]
        way_end   = self.way_ends[way_id]
        way_start_coords = self.node_coords[way_start]
        way_end_coords   = self.node_coords[way_end]
        
        print('{0:s} [{1:s}] {2:s} {3:s} -> {4:s} {5:s}'.format(way_id, way_name, way_start, str(way_start_coords), way_end, str(way_end_coords)))
        

    @staticmethod
    def geojson_distance(filename, verbose=False):
        '''
        Calculate the total distance of all lines in a geojson file
        
        Useful for comparing the total length of routes, and measure the amount of agreement/disagreement
        depending on the geojson file being examined
        
        Parameters
        ----------
        filename : str
            The geojson file to measure
            
        verbose : boolean, optional
            Specify whether debug message will be written to STDOUT
        '''
        with open(filename) as json_file:
            gj = geojson.load(json_file)

        total_distance = 0

        for feature in gj['features']:
            geom = feature['geometry']
            coordinates = geom['coordinates']
    
            # Loop through all points
            for i in range(1, len(coordinates)):
                # Compare point i to the previous point i-1
                coord_a = coordinates[i-1]
                coord_b = coordinates[i]
        
                # Measure the distance between them, and add to the total
                distance = geodesic((coord_a[1], coord_a[0]), (coord_b[1], coord_b[0])).m
                if verbose:
                    print('{0:f}, {1:f} -> {2:f}, {3:f} = {3:f}'.format(coord_a[1], coord_a[0], coord_b[1], coord_b[0], distance))
        
                total_distance += distance

        return total_distance
            
    
    def find_nearest_way_segment(self, point, qualitative_mode=True, verbose=False):
        '''
        For a given point (lat/lon) find the closest way in the OpenStreetMap data
        
        This is the first part of the process to align a detection location from the dash camera
        to the actual layout of the road according to OpenStreetMap, which is useful to compare routes.
        
        Once we have quickly found the closest way (with the shapely library, which seems
        to be faster than brute force) we can narrow down to the closest nodes on the way.
        
        Parameters
        ----------
        point : Point (lat, lon)
            The point we are searching for
            
        qualitative_mode : boolean, optional
            If this is set to True, we use a shapely library that gives us the closest distance
            in "degrees" rather than metres.  This is a little tricky to convert into metres, because
            it depends on where on Earth it is, the Earth is not a perfect sphere.  But it is
            good enough to qualitatively distinguish the closest way.  If it is false, we will use
            a more brute-force method that will actually give us the distance to the way in metres.

        verbose : boolean, optional
            Specify whether debug message will be written to STDOUT           
        '''
        
        # Find the closest way (so far) and its distance from teh point
        closest_way      = None
        closest_distance = None

        if not self.linked_coord_list:
            self.link_way_sections()
            
        for way_id in self.linked_coord_list.keys():
            # Limit search to named ways
            way_name = self.way_names_by_id[way_id]
            
            if way_name not in ['Unnamed', 'FOOTWAY', 'PATH', 'PEDESTRIAN', 'RESIDENTIAL', 'ROUNDABOUT', 'SERVICE', 'TERTIARY_LINK', 'TRACK', 'TRUNK', 'TRUNK_LINK']:
                if qualitative_mode:
                    distance = point.distance(self.linked_linestrings[way_id]) # degrees
                else:
                    distance = self.find_distance_to_way(point, way_id, verbose=False) # metres
            
                if verbose:
                    if distance is not None:
                        print('Distance {0:s} {1:s} {2:f}'.format(way_id, way_name, distance))
                    else:
                        print('Distance {0:s} {1:s} NONE'.format(way_id, way_name))
                    
                if closest_distance is None or distance < closest_distance:
                    if verbose and closest_distance is not None:
                        print('New closest distance was {0:f} now {1:f}'.format(closest_distance, distance))
                    closest_way      = way_id
                    closest_distance = distance                  
            
        return closest_way


    def find_distance_to_way(self, point, way_id, verbose=False):
        '''
        Brute force search to find the closest distance between the point and the way
        
        Faster to use the other method
        
        Parameters
        ----------
        point : Point (lat, lon)
            The point we are searching for
            
        way_id : int
            The way we want to check the distance to
            
        verbose : boolean, optional
            Specify whether debug message will be written to STDOUT 
        '''
        closest_distance_for_way = None
        
        coord_list = self.linked_coord_list[way_id]
        way_name   = self.way_names_by_id[way_id]
        
        # Check every node in the way
        for i in range(0, len(coord_list)):
            # Find the coordinates of the node
            coords = coord_list[i]
            
            # Measure the distance from the point to those coordinates
            distance = geodesic((coords[0], coords[1]), (point.coords[0][0], point.coords[0][1])).m
            if verbose:
                print('Coords {0:f}, {1:f}'.format(coords[0], coords[1]))
                
            if closest_distance_for_way is None or distance < closest_distance_for_way:
                if verbose:
                    print('New closest distance in way {0:s} {1:s} = {2:f}'.format(way_id, way_name, distance))
                closest_distance_for_way = distance
        
        return closest_distance_for_way
    
    
    def find_distance_to_node(self, node_id, point):
        '''
        Find the distance from a point to a node
        
        Parameters
        ----------
        node_id : int
            node id to check
            
        point : Point (lat, lon)
            point to check
        '''
        node_coords = self.node_coords[node_id]
        point_coords = point.coords[0]
        
        distance = geodesic((node_coords[0], node_coords[1]), (point_coords[0], point_coords[1])).m
        
        return distance # metres
    
    
    def find_nearest_node(self, point, want_intersection=True, verbose=False):
        '''
        Find the nearest node to a point, from any way in the OpenStreetMap extract
        
        Parameters
        ----------
        point : Point (lat, lon)
            The point to check
            
        want_intersectdion : boolean, optional
            Specify whether the nearest node search must be limited to intersections
            
        verbose : boolean, optional
            Specify whether debug message will be written to STDOUT 
        '''
        # First, find the nearest way
        way_id_start = self.find_nearest_way_segment(point, verbose=False)
        
        if way_id_start is None:
            return None
            
        # Next, find the nearest node from the list of intersection nodes on the way
        closest_node_id  = None
        closest_distance = None
        
        if verbose:
            print('Checking way [{0:s} {1:s}] having nodes [{2:d}]'.format(
                way_id_start, 
                self.way_names_by_id[way_id_start], 
                len(self.linked_way_sections_all[way_id_start])
            ))
            
        for node_id in self.linked_way_sections_all[way_id_start]:
            # Limit search to intersections, unless we specifically asked for any node
            if not want_intersection or self.is_intersection_node(node_id):    
                # Calculate distance (in metres)
                distance = self.find_distance_to_node(node_id, point)
                
                if verbose:
                    print('Checking {0:s} distance {1:f}'.format(node_id, distance))
                    
                if closest_distance is None or distance < closest_distance:
                    closest_node_id  = node_id
                    closest_distance = distance
                    if verbose:
                        print('This is the closest so far')
            else:
                if verbose:
                    print('Skipping non-intersection {0:s}'.format(node_id))
        
        if closest_node_id is not None:
            return way_id_start, closest_node_id, closest_distance, self.is_intersection_node(closest_node_id)
            
        # Resort to finding a node that is not an intersection, even if we really wanted one
        # Better than nothing
        for node_id in self.linked_way_sections_all[way_id_start]:
            if not self.is_intersection_node(node_id):
                if verbose:
                    print('Checking non-intersection {0:s}'.format(node_id))
                    
                # Calculate distance (in degrees)
                distance = self.find_distance_to_node(node_id, point)
                
                if closest_distance is None or distance < closest_distance:
                    closest_node_id  = node_id
                    closest_distance = distance
            
        return way_id_start, closest_node_id, closest_distance, False


    def find_nearest_node_pair(self, point, want_intersection=True, verbose=False):
        '''
        Find the nearest pair of adjacent nodes on a way for a point
        
        Useful for aligning dashcam images to a road segment between two intersections,
        e.g. when creating a map of paved shoulders for research question 4
        
        Parameters
        ----------
        point : Point (lat, lon)
            The point to check
            
        want_intersectdion : boolean, optional
            Specify whether the nearest node search must be limited to intersections
            
        verbose : boolean, optional
            Specify whether debug message will be written to STDOUT       
        '''
        # First, find the nearest way
        way_id_start = self.find_nearest_way_segment(point, verbose=False)
        
        if way_id_start is None:
            return None
            
        # Next, find the THREE nearest nodes from the list of intersection nodes on the way
        closest_node_id1  = None
        closest_distance1 = None
        
        closest_node_id2  = None
        closest_distance2 = None
        
        closest_node_id3  = None
        closest_distance3 = None
        
        if verbose:
            print('Checking way [{0:s} {1:s}] having nodes [{2:d}]'.format(
                way_id_start, 
                self.way_names_by_id[way_id_start], 
                len(self.linked_way_sections_all[way_id_start])
            ))
            
        for node_id in self.linked_way_sections_all[way_id_start]:
            # Limit search to intersections, unless we specifically asked for any node
            if not want_intersection or self.is_intersection_node(node_id):    
                # Calculate distance (in metres)
                distance = self.find_distance_to_node(node_id, point)
                
                if verbose:
                    print('Checking {0:s} distance {1:f}'.format(node_id, distance))
                    
                if closest_distance1 is None or distance < closest_distance1:
                    # Shuffle 1st and 2nd down the list to 2nd and 3rd
                    # There is bound to be a better python data structure for this like deque
                    closest_node_id3  = closest_node_id2
                    closest_distance3 = closest_distance2
                    
                    closest_node_id2  = closest_node_id1
                    closest_distance2 = closest_distance1
                    
                    closest_node_id1  = node_id
                    closest_distance1 = distance
                    if verbose:
                        print('This is the closest so far')
                        
                elif closest_distance2 is None or distance < closest_distance2:
                    # Shuffle 2nd down the list to 3rd
                    closest_node_id3  = closest_node_id2
                    closest_distance3 = closest_distance2
                    
                    closest_node_id2  = node_id
                    closest_distance2 = distance
                    if verbose:
                        print('This is the second closest so far')
                        
                elif closest_distance3 is None or distance < closest_distance3:
                    # No shuffling required                    
                    closest_node_id3       = node_id
                    closest_distance3      = distance
                    if verbose:
                        print('This is the third closest so far')
                        
            else:
                if verbose:
                    print('Skipping non-intersection {0:s}'.format(node_id))
        
        if closest_node_id2 is None:
            return way_id_start, closest_node_id1, closest_node_id2, closest_distance1, closest_distance2
            
        # If the distance between the point and 2nd is greater than the distance between 1st and 2nd
        # then assume the point is actually between the 1st and the 3rd
        # E.g. |--------------*-|-|
        closest_node_id1_coords = self.node_coords[closest_node_id1]
        closest_node_id2_coords = self.node_coords[closest_node_id2]
        point_coords            = point.coords[0]
               
        dist_p_2 = geodesic((closest_node_id2_coords[0], closest_node_id2_coords[1]), (point_coords[0], point_coords[1])).m
        dist_1_2 = geodesic((closest_node_id2_coords[0], closest_node_id2_coords[1]), (closest_node_id1_coords[0], closest_node_id1_coords[1])).m     
        
        if dist_p_2 >= dist_1_2:
            closest_node_idB = closest_node_id3
            closest_distanceB = closest_distance3            
        else:
            closest_node_idB  = closest_node_id2
            closest_distanceB = closest_distance2
        
        # Always report the node with the lower ID first, we don't split the segment into two parts,
        # where one part is closer to one end, and the other part is closer to the other end
        if closest_node_idB is not None and closest_node_id1 is not None and closest_node_idB < closest_node_id1:
            closest_node_idA  = closest_node_idB
            closest_distanceA = closest_distanceB
            closest_node_idB  = closest_node_id1
            closest_distanceB = closest_distance1
        else:
            closest_node_idA  = closest_node_id1
            closest_distanceA = closest_distance1
            
        return way_id_start, closest_node_idA, closest_node_idB, closest_distanceA, closest_distanceB


    def find_nearest_intersections_for_csv(self, filename_in, filename_out, intercept_bottom=640, intercept_top=486):
        '''
        For every point in a CSV file, find the point at which the paved shoulder lane boundaries intersect
        
        Parameters
        ----------
        filename_in : str
            Path to the input CSV
            
        filename_out : str
            Path to the output CSV
        
        intercept_bottom : int, optional
            Arbirary height of the bottom horizontal line where we will check where the paved shoulder lane boundaries
            intersect it
            
        intercept_top : int, optional
            Arbitrary height of the top horizontal line wehre we will heck where the paved shoulder lane boundaries
            intersect it
        '''
        
        # Read the input file
        df = pd.read_csv(filename_in)
        
        # Initialize the output file        
        output_file = open(filename_out, 'w')
        output_file.write('filename,prefix,frame_num,lat,lon,altitude,heading,pixels_bottom,pixels_top,left_slope2,left_int2,left_slope1,left_int1,right_slope1,right_int1,way_id_start,node_id1,node_id2,distance1,distance2,lat1,lon1,lat2,lon2,intersection_x,intersection_y,x2_bottom,x1_bottom,x2_top,x1_top,slope_diff,way_name\n')
         
        # Process every record, one at a time, displaying a progress bar in Jupyter Notebook
        tqdm.pandas()
         
        df.progress_apply(lambda row: self.find_nearest_intersections_row(
            output_file,
            intercept_bottom,
            intercept_top,
            row['filename'],
            row['prefix'],
            row['frame_num'],
            row['lat'],
            row['lon'],
            row['altitude'],
            row['heading'],
            row['pixels_bottom'],
            row['pixels_top'],
            row['left_slope2'],
            row['left_int2'],
            row['left_slope1'],
            row['left_int1'],
            row['right_slope1'],
            row['right_int1']),
            axis=1
        )
        

    def find_nearest_intersections_row(self, output, intercept_bottom, intercept_top, filename, prefix, frame_num, lat, lon, altitude, heading, pixels_bottom, pixels_top, left_slope2, left_int2, left_slope1, left_int1, right_slope1, right_int1):
        '''
        For one point in a CSV file, find the point at which the paved shoulder lane boundaries intersect
        
        Parameters
        ----------
        output : str
            Path to the output file where the results will be written as a single record
            
        intercept_bottom : int, optional
            Arbirary height of the bottom horizontal line where we will check where the paved shoulder lane boundaries
            intersect it
            
        intercept_top : int, optional
            Arbitrary height of the top horizontal line wehre we will heck where the paved shoulder lane boundaries
            intersect it
            
        filename : str
            Field passed from the original file
            
        prefix : str
            Field passed from the original file
            
        frame_num : int
            Field passed from the original file
            
        lat : float
            Field passed from the original file
            
        lon : float
            Field passed from the original file
        
        altitude : float
            Field passed from the original file
            
        heading : int
            Field passed from the original file
            
        pixels_bottom : int
            Field passed from the original file
            
        pixels_top : int
            Field passed from the original file
            
        left_slope2 : float
            Field passed from the original file
            
        left_int2 : float
            Field passed from the original file
            
        left_slope1 : float
            Field passed from the original file
            
        left_int1 : float
            Field passed from the original file
            
        right_slope1 : float
            Field passed from the original file
            
        right_int1 : float
            Field passed from the original file
        '''
        p = Point(lat, lon)
        
        way_id_start, closest_node_id1, closest_node_id2, closest_distance1, closest_distance2 = self.find_nearest_node_pair(p)
        
        # Only output records where the segment was found_hwy
        # Exclude records where we were too close to the intersection, to remove uncertainty about which road we were on
        # And it will hopefully remove noise/uncertainty around intersections as shoulders disappear
        # And uncertainty around roundabouts
        if closest_node_id1 is not None and closest_node_id2 is not None and closest_distance1 >= 30 and closest_distance2 >= 30:
            coords1 = self.node_coords[closest_node_id1]
            coords2 = self.node_coords[closest_node_id2]
            
            # Work out where the lines intersect
            if left_slope2 is not None and left_slope1 is not None and left_slope2 != 'None' and left_slope1 != 'None' and left_slope2 != 0 and left_slope1 != 0:
                y1 = intercept_bottom
                x1 = int((intercept_bottom - float(left_int2)) / float(left_slope2))
            
                y2 = intercept_top
                x2 = int((intercept_top    - float(left_int2)) / float(left_slope2))
            
                y3 = intercept_bottom
                x3 = int((intercept_bottom - float(left_int1)) / float(left_slope1))
            
                y4 = intercept_top
                x4 = int((intercept_top    - float(left_int1)) / float(left_slope1))
            
                intersection = self.findIntersection(x1, y1, x2, y2, x3, y3, x4, y4)
                
                intersection_x = int(intersection[0])
                intersection_y = int(intersection[1])
                
                x2_top    = x2
                x2_bottom = x1
                
                x1_top    = x4
                x1_bottom = x3
            else:
                intersection_x = 9999
                intersection_y = 9999
                
                x2_top    = 0
                x2_bottom = 0
                x1_top    = 0
                x1_bottom = 0
            
            if left_slope2 is None or left_slope1 is None or left_slope2 == 'None' or left_slope1 == 'None':
                slope_diff = None
            else:
                slope_diff = float(left_slope1) - float(left_slope2)
                
            output.write('{0:s},{1:s},{2:d},{3:.6f},{4:.6f},{5:d},{6:d},{7:d},{8:d},{9:s},{10:s},{11:s},{12:s},{13:s},{14:s},{15:s},{16:s},{17:s},{18:d},{19:d},{20:.6f},{21:.6f},{22:.6f},{23:.6f},{24:d},{25:d},{26:d},{27:d},{28:d},{29:d},{30:s},{31:s}\n'.format(
                filename,
                prefix,
                frame_num,
                lat,
                lon,
                int(altitude),
                int(heading),
                int(pixels_bottom),
                int(pixels_top),
                left_slope2,
                left_int2,
                left_slope1,
                left_int1,
                right_slope1,
                right_int1,
                way_id_start,
                closest_node_id1,
                closest_node_id2,
                int(closest_distance1),
                int(closest_distance2),
                coords1[0],
                coords1[1],
                coords2[0],
                coords2[1],
                intersection_x,
                intersection_y,
                int(x2_bottom),
                int(x1_bottom),
                int(x2_top),
                int(x1_top),
                str(slope_diff),
                self.way_names_by_id[way_id_start]
            ))
    
    # https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
    # Modified to protect against divide by zero and provide extreme defaults to the application in that case
    # 9999 is well outside the bounds of the image size 1920x1080 so will be seen as an outlier
    @staticmethod
    def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
        '''
        Given two lines (defined by two pairs of start/end points) find the point at which they intersect
        
        Parameters
        ----------
        x1 : int
            X Coordinate of the first point
            
        y1 : int
            Y Coordinate of the first point
            
        x2 : int
            X Coordinate of the second point
            
        y2 : int
            Y Coordinate of the second point
            
        x3 : int
            X Coordinate of the third point
            
        y3 : int
            Y Coordinate of the third point
            
        x4 : int
            X Coordinate of the fourth point
            
        y4 : int
            Y Coordinate of the fourth point
        '''
        denominator = ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
        
        if denominator != 0:
            px = ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / denominator
            py = ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / denominator
        else:
            px = 9999
            py = 9999
            
        return [px, py]
        
        
    def summarise_lane_detections_csv(self, filename_in, filename_out):
        '''
        Given a CSV file where each point has been matched to the closest way and pair of intersections,
        create additional columns aggregating attributes of the paved shoulder stats across all records
        associated with that same way and pair of intersections.
        
        Parameters
        ----------
        
        filename_in : str
            Input CSV file
            
        filename_out : str
            Output CSV flle with extra summary columns for each group        
        '''
        # Read the original CSV
        df = pd.read_csv(filename_in)
        
        # Flag rows where either of the left lane lines are missing
        df1 = df[['way_id_start', 'node_id1', 'node_id2', 'intersection_x', 'intersection_y', 'left_slope2', 'left_slope1', 'x2_top', 'x1_top']]

        pd.options.mode.chained_assignment = None

        df1.loc[(df['left_slope2'] != 'None') & (df['left_slope1'] != 'None'), "missing"] = 0
        df1.loc[(df['left_slope2'] == 'None') | (df['left_slope1'] == 'None'), "missing"] = 1
        
        # Exclude missing values from average and standard deviation
        df1.loc[(abs(df['intersection_x']) >= 9999), 'intersection_x'] = None
        df1.loc[(abs(df['intersection_y']) >= 9999), 'intersection_y'] = None

        df1['width_top'] = df1['x1_top'] - df1['x2_top']
        df1.loc[(df1['width_top'] == 0), 'width_top'] = None

        # Create summary statistics per road segment
        df2 = df1.groupby(['way_id_start', 'node_id1', 'node_id2']).agg({
            'intersection_x': ['std'],
            'intersection_y': ['mean', 'std'],
            'width_top':      ['mean', 'std'],
            'missing':        ['mean']
        })
        df2.columns = ['intersection_x_std', 'intersection_y_mean', 'intersection_y_std', 'width_top_mean', 'width_top_std', 'prop_missing']
        df2 = df2.reset_index()
        
        # Merge summary statistics back onto the main dataframe
        merge_columns = ['way_id_start', 'node_id1', 'node_id2']
        combined_df = pd.merge(df, df2, how='left', left_on=merge_columns, right_on=merge_columns)

        # Write the merged data to a new CSV
        combined_df.to_csv(filename_out, index=False)
        

    def draw_lane_detections(self, csv_in, geojson_filename, prop_missing=0.2, intersection_y_std=50, intersection_x_std=50, width_top_mean=75):
        '''
        For every group of frames in the CSV (by closest way_id and two closest intersection node ids)
        assess the group statistics and decide whether there is a paved shoudler on that route segment
        at all.  If there is, draw it in a geojson file.
        
        Parameters
        ----------
        csv_in : str
            Input CSV name
            
        geojson_filename : str
            Output filename for geojson files
            
        prop_missing : float, optional
            The maximum proportion of frames in the group that are allowed to have no paved shoulder
            lane boundaries detected, before we assume there is no paved shoulder on the road segment.
            
        intersection_y_std : float, option
            The maximum standard deviation of the y-coordinate of the intersection between paved shoulder
            lane boundaries, before we assumne the lines are moving too much from frame to frame to be real.

        intersection_x_std : float, option
            The maximum standard deviation of the x-coordinate of the intersection between paved shoulder
            lane boundaries, before we assumne the lines are moving too much from frame to frame to be real.
           
        width_top_mean : float, option
            The minimum mean width of the image, in pixels, at the top horizontal reference line
        '''
        # Read the CSV with summary information joined to individual detection information
        df = pd.read_csv(csv_in)
        
        # Apply detection thresholds for drawing a path
        
        # Suppress warnings about chained assignments
        pd.options.mode.chained_assignment = None
        
        df = df[df['prop_missing']       <= prop_missing]
        df = df[df['intersection_y_std'] <= intersection_y_std]
        df = df[df['intersection_x_std'] <= intersection_x_std]
        df = df[df['width_top_mean']     >= width_top_mean]
        
        # Reduce columns to the interesting ones
        df = df[['way_id_start', 'node_id1', 'node_id2', 'way_name', 'prop_missing', 'intersection_y_std', 'intersection_x_std', 'width_top_mean']]
        
        # Group down to one record per combination of way_id_start x node_id1 x node_id2
        grouped = df.groupby(['way_id_start', 'node_id1', 'node_id2'])
        first_values = grouped.first()
        first_values = first_values.reset_index()
        
        # Process each row
        self.lane_features = []
        
        for i in trange(0, len(first_values)):
            way_id_start = str(first_values.loc[i, 'way_id_start'])
            node_id1     = str(first_values.loc[i, 'node_id1'])
            node_id2     = str(first_values.loc[i, 'node_id2'])
            
            self.draw_lane(way_id_start, node_id1, node_id2)
        
        if len(self.lane_features) > 0:
            featurecollection = {
                'type':     'FeatureCollection',
                'name':     os.path.basename(geojson_filename),
                'features': self.lane_features
            }
        
            print('Writing {0:d} features to: {1:s}'.format(len(self.lane_features), geojson_filename))
            with open(geojson_filename, 'w') as outfile:
                json.dump(featurecollection, outfile, indent=4)
                outfile.close()
        else:
            print('No features to write to: ' + geojson_filename)
            
            
    def draw_lane(self, way_id_start, node_id1, node_id2):  
        '''
        If we determined that a route segment (on a way, between a pair of intersections) has a
        paved shoulder, then add it to the Features that will be written to a geojson file
        
        Parameters
        ----------
        way_id_start : int
            The way being drawn
        
        node_id1 : int
            The first node ID where the line will start
            
        node_id2 : int
            The second node ID where the line will end (unless it continues in the next road segment, too)
        '''
        # Get a list of all nodes in way_start_id, in order
        # Note: we might encounter either node_id1 or node_id2 first
        
        section_node_list = self.linked_way_sections_all[way_id_start]
        
        pen_down         = 0
        coordinates_list = []
        
        for node_id in section_node_list:
            if node_id in [node_id1, node_id2]:
                # Include the start and end nodes
                coordinates_list.append([self.node_coords[node_id][1], self.node_coords[node_id][0]])
                
                # Keep track of how many end nodes we have encountered
                pen_down += 1
            elif pen_down % 2 == 1:
                # If we have only encountered one end node so far, we are between them, so include this one
                coordinates_list.append([self.node_coords[node_id][1], self.node_coords[node_id][0]])

        if len(coordinates_list) > 0:               
            feature = {
                'type': 'Feature',
                'properties': {
                    'id':   '{0:s}_{1:s}_{2:s}'.format(way_id_start, node_id1, node_id2),
                    'name': '{0:s}_{1:s}_{2:s}_{3:s}'.format(self.way_names_by_id[way_id_start], way_id_start, node_id1, node_id2),
                    'version': '1'
                },
                'geometry': {
                    'type': 'LineString',
                    'coordinates': coordinates_list
                }
            }
            
            self.lane_features.append(feature)