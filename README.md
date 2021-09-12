# minor_thesis
Code used for Minor Thesis research

## Jupyter Notebooks

GPU_test.ipynb = train a simple TensorFlow 2 model to recognise characters (MNIST)
as a demo/test to ensure GPU is being utilized in the environment

Google_Streetview_API.ipynb = Demonstrate how to download a GSV image for a
co-ordinate (lat/lon) and how to calculate a bearing from one co-ordinate to the next

PBN_Geocoding.ipynb = Use local Nominatim server to perform reverse-geocoding
of streets in PBN geojson file.  Superceded by Find_Intersections.ipynb
and eventually PBN_Distinct_Intersections.ipynb.

Find_Intersections.ipynb = Original notebook that uses an OSM extract of Victoria +
a local Nominatim server + the CSV data extracted from the PBN .geojson file
to reverse-geocode all "existing" routes in PBN, then find the intersections

Points_and_Bearings.ipynb = Demo of how to calculate the bearing from one co-ordinate
to the next, AND to expand the co-ordinate pair into a a list of co-ordinates
at 10m intervals

### PBN_Distinct_Intersections.ipynb
Load PBN geojson file + OSM extract of Victoria,
use local Nominatim serve to perform reverse-geocoding, use OSM to find the
intersections within the bounding box of the original street, find the bearings,
save as CSV for use in GSV_Demo.ipynb.  Appears to supercede Find_Intersections.ipynb.

Google_Streetview_API.ipynb = Demonstrate how to download a GSV image for a
co-ordinate (lat/lon) and how to calculate a bearing from one co-ordinate to the next

GSV_Demo.ipynb = given a list of intersection co-ordinates along "EXISTING" PBN routes 
and the bearing at each co-ordinate, sample one co-ordinate at a time, allow manual
specification of metres forward/backward from the intersection along the bearing and
manual override of the bearing, then download and display 4x GSV images at 0/90/180/270
from the bearing.  Allow "hits" (where a bicycle lane logo etc. is found in an image)
to be recorded in a CSV "hits.csv" for later labelling.

### Filter_PBN_Existing.ipynb
Take the original PBN .geojson file and filter it
into both "EXISTING" vs not "EXISTING" (planned) so that we can draw them
as differenct colours in Map_GEOJSON.ipynb

### OSM_Filter.ipynb
Extract the "shape" of a "Locality" from a govt .geojson file,
and then save it to a new .geojson file.  Includes instructions to then use "osmium"
to filter an OSM extract down to just that locality, and save as an OXM file (XML)

### Parse_OSM.ipynb
Load an OSM file into memory/dict objects, find the ways that
have a "cycleway" tag, then save out as a .geojson file for drawing on a map

### Map_GEOJSON.ipynb
Given .geojson files that show bicycle networks as lines from
one co-ordinate (lat/lon) to the next, draw them on a map for comparison

### Parse_GeoJSON.ipynb
Read a version of OSM cycleway or road data that has been converted into a .geojson
via Parse_OSM.ipynb.  Turn it into a list of co-ordinate pairs along each way,
and then expand those co-ordinate pairs into a list of co-ordinates at 10m intervals
=> In "Mount Eliza" there are 1804 "ways" (road segments) and 75740 points if we
sample at 10m intervals.  At $7 USD per 1000 images and 4 images per point, this
would be $303 USD for the one town.


## Python Classes

osm_filter.py = Called by OSM_Filter.ipynb to load a "locality" geojson file and then
filter it down to only one feature that matches the "vic_loca_2" property, then
save it as a smaller geosjon file representing the "shape" of the selected locality

