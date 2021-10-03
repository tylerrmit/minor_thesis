'''
Created on 06 Sep 2021

@author: Tyler Saxton
'''

import json
import math
import os
import sys

import pandas as pd

import xml.dom.minidom

from geographiclib.geodesic import Geodesic

import geojson
from geojson import Feature, FeatureCollection, dump

from geopy.distance import geodesic

from shapely.geometry import Point, LineString

from tqdm.notebook import tqdm, trange


class osm_walker(object):
    '''
    A class used to "walk" routes in an OSM extract and create a list of sample points and bearings

    '''

    def __init__(self, filename_main, filename_margin=None, verbose=False):
        '''
        Parameters
        ----------
        filename_main : str
            The name of the main OSM XML file to load
            Either a full path, or assume it's relative to the current working directory
        filename_margin: str
            The name of a second OSM XML file encompassing the first, with a "margin"
            around it to catch intersecting streets just outside the main area
        '''

        # Cache ways in dict by name
        self.ways_by_name                = {} # Dictionary giving, for each way name, a list of ways
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
        self.linked_way_sections_cwy     = {} # Dictionary to if any way_id linked to the node in self.linked_way_sections_all is tagged as cycleway
        
        self.linked_linestrings          = {} # Dictionary where linked ways are stored as shapely LineString objects
        self.linked_coord_list           = {}
        
        self.way_id_start_by_way_id      = {} # Dictionary showing the way_id_start for each way_id, where did it go?
        
        self.detection_hits              = {} # Dictionary giving detection hits from a detection log
        
        self.unused_way_ids              = [] # Keep track of any way IDs we have not yet linked to a way segment, to make sure we get them all
        
        self.tagged_features             = [] # Features we are building to draw on a map - cycleway tagged routes
        self.detected_features           = [] # Features we are building to draw on a map - detected routes
        self.both_features               = []
        self.either_features             = []
        self.tagged_only_features        = []
        self.detected_only_features      = []
                
        
        # Load the main XML file into memory
        # This assumes that we have reduced the OpenStreetMap data down to a small enough locality
        # that the in-memory approach is feasible
        doc_main = xml.dom.minidom.parse(filename_main)

        self.process_osm_xml(doc_main, intersections_only=False, verbose=verbose)
        
        if filename_margin:
            # Load a slightly bigger XML file into memory, to catch nodes that are JUST outside the
            # boundary of the locality
            doc_margin = xml.dom.minidom.parse(filename_margin)

            self.process_osm_xml(doc_margin, intersections_only=True, verbose=verbose)
            self.process_osm_xml(doc_margin, intersections_only=True, verbose=verbose)


    def process_osm_xml(self, doc, intersections_only=False, verbose=True):
        # Get ways and nodes from XML document
        ways_xml  = doc.getElementsByTagName('way')
        nodes_xml = doc.getElementsByTagName('node')

        for way in ways_xml:
            # Get the ID for this way
            way_id = way.getAttribute('id')
            
            is_cycleway = False
                            
            # Find the name for the way based on a 'tag' element where k='name'
            tags = way.getElementsByTagName('tag')
            found_name = False
            found_hwy  = False
            way_name   = 'Unnamed'
            
            for tag in tags:
                k = tag.getAttribute('k')
                v = tag.getAttribute('v').upper()
                
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
                    
                # Record the association with this way against the node
                # We can tell that an intersection is a node associated with multiple ways
                node_refs = way.getElementsByTagName('nd')
                for idx, node_ref in enumerate(node_refs):
                    ref = node_ref.getAttribute('ref')
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
                
                node_ends = [node_refs[0], node_refs[-1]]
                for node_ref in node_ends:
                    ref = node_ref.getAttribute('ref')                        
                    if ref in self.way_ends_per_node:
                        if way_id not in self.way_ends_per_node[ref]:
                            self.way_ends_per_node[ref].append(way_id)
                    else:
                        self.way_ends_per_node[ref] = [way_id]                
                
                recorded = True
                 
        # Cache nodes in dict by id/ref
        if not intersections_only:
            for node in nodes_xml:
                id = node.getAttribute('id').upper()
                self.nodes[id] = node
                
                lat = float(node.getAttribute('lat'))
                lon = float(node.getAttribute('lon'))
                
                self.node_coords[id] = [lat, lon]
    
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
        lat1 = float(prev_node.getAttribute('lat'))
        lon1 = float(prev_node.getAttribute('lon'))
        lat2 = float(next_node.getAttribute('lat'))
        lon2 = float(next_node.getAttribute('lon'))
    
        bearing = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)['azi1']
        if bearing < 0:
            bearing = bearing + 360
        
        return int(round(bearing))

        
    # Get a series of points from one point to the next, at regular intervals up to a desired offset
    @staticmethod
    def expand_offsets(lat1, lon1, lat2, lon2, max_offset, interval, way_id_start, way_id, node_id):
        sample_points = []
    
        bearing = Geodesic.WGS84.Inverse(float(lat1), float(lon1), float(lat2), float(lon2))['azi1']
    
        line = Geodesic.WGS84.InverseLine(float(lat1), float(lon1), float(lat2), float(lon2))
    
        num_steps     = int(math.ceil(abs(max_offset) / interval))
        num_steps_max = int(math.ceil(line.s13 / interval))
    
        if max_offset < 0:
            polarity = -1
        else:
            polarity = 1
        
        for step_i in range(num_steps + 1):
            if step_i > 0:
                s = min(interval * step_i, line.s13)
                g = line.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            
                sample_point = [
                    g['lat2'],
                    g['lon2'],
                    int(round(bearing)),
                    step_i * interval * polarity,
                    way_id_start,
                    way_id,
                    node_id
                ]
            
                sample_points.append(sample_point)
    
        return sample_points


    # Get every sample point for a way
    def walk_way_intersections_by_id(self, way_id_start, way_id, min_offset=0, max_offset=0, interval=10, debug=False):
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
        
            if (self.is_intersection_node(ref)) or (idx == 0) or (idx == len(node_refs) - 1):
                if debug:
                    print('Debug Node Intersection: {0:s} {1:d} {2:.6f}, {3:.6f}'.format(ref, len(self.way_names_for_node[ref]),
                        float(self.nodes[ref].getAttribute('lat')), float(self.nodes[ref].getAttribute('lon'))))
            
                # Found an intersection!  We will output for this one
            
                # Find any negative offset samples required
                if idx > idx_first and min_offset < 0 and interval > 0:
                    prev_points = osm_walker.expand_offsets(
                        self.nodes[node_refs[idx  ].getAttribute('ref')].getAttribute('lat'),
                        self.nodes[node_refs[idx  ].getAttribute('ref')].getAttribute('lon'),
                        self.nodes[node_refs[idx-1].getAttribute('ref')].getAttribute('lat'),
                        self.nodes[node_refs[idx-1].getAttribute('ref')].getAttribute('lon'),
                        min_offset,
                        interval,
                        way_id_start,
                        way_id,
                        ref
                    )
                
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
                    float(self.nodes[ref].getAttribute('lat')), float(self.nodes[ref].getAttribute('lon')), bearing, 0, way_id_start, way_id, ref
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
                        ref
                    )
            else:
                if debug:
                    print('Debug Node NON-Intersection: {0:s} {1:d} {2:.6f}, {3:.6f}'.format(ref, len(self.way_names_for_node),
                        float(self.nodes[ref].getAttribute('lat')), float(self.nodes[ref].getAttribute('lon'))))
        
        return sample_points

    # Given a way by its ID, are there any other ways with the same name that intersect
    # with its first node?
    def is_way_start(self, way_id, verbose=False):
        # Retrieve the details of the way, and find the first node ID
        way           = self.ways_by_id[way_id]
        node_refs     = way.getElementsByTagName('nd')
        first_node_id = node_refs[0].getAttribute('ref')
        this_way_name = self.way_names_by_id[way_id]
        
        if verbose:
            print('First node: ' + first_node_id)
            
        # Are there any other ways that join, end-to-end, with this node, with any way name?
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
        # Retrieve the details of the way, and find the last node ID
        way           = self.ways_by_id[way_id]
        node_refs     = way.getElementsByTagName('nd')
        last_node_id  = node_refs[-1].getAttribute('ref')
        this_way_name = self.way_names_by_id[way_id]
    
        # Are there any other ways that intersect with this node, with any way name?
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
        
        # Iterate through each way section
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
                self.linked_linestrings[way_id_start]      = LineString(coord_list)
                self.linked_coord_list[way_id_start]       = coord_list
                
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
                self.linked_linestrings[way_id_start]      = LineString(coord_list)
                self.linked_coord_list[way_id_start]       = coord_list
                
                if verbose:
                    print('Saving {0:s} {1:s}'.format(way_name, way_id))
                    print('section:     {0:3d} {1:s}'.format(len(section), str(section)))
                    print('section_all: {0:3d} {1:s}'.format(len(section_all), str(section_all)))
                    
    
    
    def load_detection_log(self, filename):        
        df = pd.read_csv(filename)
        
        for index, row in df.iterrows():
            # row['lat'], lon, bearing, way_id_start, way_id, node_id, offset_id, score, bbox_0, bbox_1, bbox_2, bbox_3
            key = '{0:.0f}-{1:.0f}'.format(row['way_id_start'], row['node_id'])

            if key in self.detection_hits:
                self.detection_hits[key].append([row['offset_id'], row['score']])
            else:
                self.detection_hits[key] = [[row['offset_id'], row['score']]]
                

    def snap_detection_log(self, filename_in, filename_out):
        df = pd.read_csv(filename_in)
        
        tqdm.pandas()
        
        output_file = open(filename_out, 'w')
        output_file.write('lat,lon,bearing,heading,way_id_start,way_id,node_id,offset_id,score,bbox_0,bbox_1,bbox_2,bbox_3,way_name\n')
                        
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
        p = Point(lat, lon)
        
        
        for option in [True, False]:
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
        if len(self.way_names_per_node[node_id]) > 1:
            return True
        else:
            return False
        
        
    def write_geojsons(self, name, geojson_directory, intersection_skip_limit=1, infer_ends=True, verbose=False):
        self.detected_features      = []
        self.tagged_features        = []
        self.both_featurtes         = []
        self.detected_only_features = []
        self.tagged_only_features   = []
        
        for way_id_start in self.linked_way_sections.keys():
        #for way_id_start in ['841124847']:
            self.draw_way_segment(way_id_start, intersection_skip_limit=intersection_skip_limit, infer_ends=infer_ends, verbose=verbose)
        
        self.write_geojson(name, geojson_directory, 'hit',      self.detected_features)
        self.write_geojson(name, geojson_directory, 'tag',      self.tagged_features)
        self.write_geojson(name, geojson_directory, 'both',     self.both_features)
        self.write_geojson(name, geojson_directory, 'either',   self.either_features)
        self.write_geojson(name, geojson_directory, 'hit_only', self.detected_only_features)
        self.write_geojson(name, geojson_directory, 'tag_only', self.tagged_only_features)
        
        # Compare distances
        for part in ['hit', 'tag', 'both', 'either', 'hit_only', 'tag_only']:
            geojson_filename = os.path.join(geojson_directory, part + '.geojson')
            
            distance = osm_walker.geojson_distance(geojson_filename)
            
            print('{0:8s}: Total distance {1:10.2f}m'.format(part, distance))
        
        
    def write_geojson(self, name, geojson_directory, mode, features):
        print('Writing ' + mode + ', feature count: ' + str(len(features)))
        
        label = '{0:s}_{1:s}'.format(name, mode)
        
        geojson_filename = os.path.join(geojson_directory, mode + '.geojson')
        
        featurecollection = {
            'type':     'FeatureCollection',
            'name':     label,
            'features': features
        }
        
        print('Writing to: ' + geojson_filename)
        with open(geojson_filename, 'w') as outfile:
            json.dump(featurecollection, outfile, indent=4)
            outfile.close()
    
    
    def draw_way_segment(self, way_id_start, intersection_skip_limit=1, infer_ends=True, verbose=False):
        self.section_node_list = self.linked_way_sections_all[way_id_start]
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
        section_node_list = self.linked_way_sections_all[way_id_start]
        
        features = []
        
        coordinates_list_list = []
        
        coordinates_open = False
        coordinates      = []
                
        for idx in range(0, len(section_node_list)):
            node_id = section_node_list[idx]
            
            hit = self.node_hit_detected[node_id] + self.node_hit_assumed[node_id]
            tag = self.node_tagged[node_id]
            
            # Is this node tagged as a cycleway in any way that is part of this segment?
            # Check each way in the segment...
            
            #tag = 0
            #for way_id_part in self.linked_way_sections[way_id_start]:                
            #    # where the node is linked to that way...
            #    if way_id_part in self.way_ids_per_node[node_id]:
            #        if self.way_is_cycleway[way_id_part]:
            #            tag = 1
                
            if hit and tag:
                both = 1
            else:
                both = 0
                
            if hit and not tag:
                hit_only = 1
            else:
                hit_only = 0
                
            if tag and not hit:
                tag_only = 1
            else:
                tag_only = 0
                
            if hit or tag:
                either = 1
            else:
                either = 0
                
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
                
            if pen_down:
                coordinates_open = True
                coordinates      = coordinates + [[self.node_coords[node_id][1], self.node_coords[node_id][0]]]
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
        if idx is None:
            return None
            
        idx += 1
        while idx < len(self.section_node_list):
            if self.node_is_intersection[self.section_node_list[idx]]:
                return idx
            idx += 1
        return None
       
       
    def draw_find_prev_intersection(self, idx):        
        if idx is None:
            return None
            
        idx -= 1
        while idx >= 0:
            if self.node_is_intersection[self.section_node_list[idx]]:
                return idx
            idx -= 1
        return None


    def draw_find_next_hit(self, idx):        
        if idx is None:
            return None
            
        idx += 1
        while idx < len(self.section_node_list):
            if self.node_hit_detected[self.section_node_list[idx]]:
                return idx
            idx += 1
        return None      
        
        
    def draw_find_prev_hit(self, idx):        
        if idx is None:
            return None
            
        idx -= 1
        while idx >= 0:
            if self.node_hit_detected[self.section_node_list[idx]]:
                return idx   
            idx -= 1
        return None
      
      
    def draw_count_intersection_miss_between(self, prev_hit, next_hit):       
        if prev_hit is None or next_hit is None:
            return None
            
        intersection_miss_between = 0
        
        for idx in range(prev_hit+1, next_hit):
            if (self.node_is_intersection[self.section_node_list[idx]]) and (self.node_hit_detected[self.section_node_list[idx]] == 0):
                intersection_miss_between += 1
                
        return intersection_miss_between
       
       
    def dump_way(self, way_id):
        way_name = self.way_names_by_id[way_id]
        way_start = self.way_starts[way_id]
        way_end   = self.way_ends[way_id]
        way_start_coords = self.node_coords[way_start]
        way_end_coords   = self.node_coords[way_end]
        
        print('{0:s} [{1:s}] {2:s} {3:s} -> {4:s} {5:s}'.format(way_id, way_name, way_start, str(way_start_coords), way_end, str(way_end_coords)))
        

    @staticmethod
    def geojson_distance(filename, verbose=False):
        with open(filename) as json_file:
            gj = geojson.load(json_file)

        total_distance = 0

        for feature in gj['features']:
            geom = feature['geometry']
            coordinates = geom['coordinates']
    
            for i in range(1, len(coordinates)):
                coord_a = coordinates[i-1]
                coord_b = coordinates[i]
        
                distance = geodesic((coord_a[1], coord_a[0]), (coord_b[1], coord_b[0])).m
                if verbose:
                    print('{0:f}, {1:f} -> {2:f}, {3:f} = {3:f}'.format(coord_a[1], coord_a[0], coord_b[1], coord_b[0], distance))
        
                total_distance += distance

        return total_distance
            
    
    def find_nearest_way_segment(self, point, qualitative_mode=True, verbose=False):
        closest_way      = None
        closest_distance = None

        if not self.linked_coord_list:
            self.link_way_sections()
            
        for way_id in self.linked_coord_list.keys():
            # Limit search to named ways
            way_name = self.way_names_by_id[way_id]
            
            if way_name not in ['Unnamed', 'FOOTWAY', 'PATH', 'PEDESTRIAN', 'RESIDENTIAL', 'ROUNDABOUT', 'SERVICE', 'TERTIARY_LINK', 'TRACK', 'TRUNK', 'TRUNK_LINK']:
                if qualitative_mode:
                    distance = point.distance(self.linked_linestrings[way_id])
                else:
                    distance = self.find_distance_to_way(point, way_id, verbose=False)
            
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
        closest_distance_for_way = None
        
        coord_list = self.linked_coord_list[way_id]
        way_name   = self.way_names_by_id[way_id]
        
        for i in range(0, len(coord_list)):
            coords = coord_list[i]
            distance = geodesic((coords[0], coords[1]), (point.coords[0][0], point.coords[0][1])).m
            if verbose:
                print('Coords {0:f}, {1:f}'.format(coords[0], coords[1]))
                
            if closest_distance_for_way is None or distance < closest_distance_for_way:
                if verbose:
                    print('New closest distance in way {0:s} {1:s} = {2:f}'.format(way_id, way_name, distance))
                closest_distance_for_way = distance
        
        return closest_distance_for_way
    
    
    def find_distance_to_node(self, node_id, point):
        node_coords = self.node_coords[node_id]
        point_coords = point.coords[0]
        
        distance = geodesic((node_coords[0], node_coords[1]), (point_coords[0], point_coords[1])).m
        
        return distance # metres
    
    
    def find_nearest_node(self, point, want_intersection=True, verbose=False):
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
            
        # Resort to finding a node that is not an intersection
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
        # First, find the nearest way
        way_id_start = self.find_nearest_way_segment(point, verbose=False)
        
        if way_id_start is None:
            return None
            
        # Next, find the three nearest nodes from the list of intersection nodes on the way
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
            return way_id_start, closest_node_id1, closest_node_id3, closest_distance1, closest_distance3
        else:
            return way_id_start, closest_node_id1, closest_node_id2, closest_distance1, closest_distance2

    def find_nearest_intersections_for_csv(self, filename_in, filename_out):
        df = pd.read_csv(filename_in)
        
        tqdm.pandas()
        
        output_file = open(filename_out, 'w')
        output_file.write('filename,prefix,frame_num,lat,lon,altitude,heading,pixels_bottom,pixels_top,left_slope2,left_int2,left_slope1,left_int1,right_slope1,right_int1,way_id_start,node_id1,node_id2,distance1,distance2,lat1,lon1,lat2,lon2,way_name\n')
                        
        df.progress_apply(lambda row: self.find_nearest_intersections_row(
            output_file,
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
        

    def find_nearest_intersections_row(self, output, filename, prefix, frame_num, lat, lon, altitude, heading, pixels_bottom, pixels_top, left_slope2, left_int2, left_slope1, left_int1, right_slope1, right_int1):
        p = Point(lat, lon)
        
        way_id_start, closest_node_id1, closest_node_id2, closest_distance1, closest_distance2 = self.find_nearest_node_pair(p)
        
        # Only output records where the segment was found_hwy
        # Exclude records where we were too close to the intersection, to remove uncertainty about which road we were on
        # And it will hopefully remove noise/uncertainty around intersections as shoulders disappear
        # And uncertainty around roundabouts
        if closest_node_id1 is not None and closest_node_id2 is not None and closest_distance1 >= 20 :
            coords1 = self.node_coords[closest_node_id1]
            coords2 = self.node_coords[closest_node_id2]
            
            #output.write('{0:s},{1:s},{2:d},{3:.6f},{4:.6f},{5:d},{6:d},{7:d},{8:d},{9:f},{10:f},{11:f},{12:f},{13:f},{14:f},{15:s},{16:s},{17:s},{18:d},{19:d},{20:.6f},{21:.6f},{22:.6f},{23:.6f},{24:s}\n'.format(
            output.write('{0:s},{1:s},{2:d},{3:.6f},{4:.6f},{5:d},{6:d},{7:d},{8:d},{9:s},{10:s},{11:s},{12:s},{13:s},{14:s},{15:s},{16:s},{17:s},{18:d},{19:d},{20:.6f},{21:.6f},{22:.6f},{23:.6f},{24:s}\n'.format(
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
                self.way_names_by_id[way_id_start]
            ))