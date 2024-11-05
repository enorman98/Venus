import numpy as np
import pandas as pd 
from sklearn.neighbors import NearestNeighbors
from math import radians, sin, cos, sqrt, atan2 

#points=pd.read_csv('Venus_Craters.csv',usecols=['lat', 'lon'])
# Number of points
size = 947

rng = np.random.default_rng()
u = rng.uniform(size=size)
v = rng.uniform(size=size)

theta = 2 * np.pi * u
phi = np.arccos(2 * v - 1)

lon = np.rad2deg(theta - np.pi)  # longitude in the range [-180, 180]
lat = np.rad2deg(phi - np.pi / 2.0)

points = pd.DataFrame({'lat': lat, 'lon': lon})



rVenus = 6051.8

def latlon_to_cartesian(lat, lon, R=rVenus):
    """
    Converts arrays of latitude and longitude coordinates to 3D Cartesian coordinates.

    Args:
        lat: Array of latitudes in degrees.
        lon: Array of longitudes in degrees.
        R: Radius of Venus in kilometers (default: 6051.8 km).

    Returns:
        Tuple of arrays (x, y, z) representing Cartesian coordinates.
    """

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)

    return x, y, z



longitude=points['lon'].values
latitude=points['lat'].values



# Pass the entire array directly
x, y, z = latlon_to_cartesian(latitude, longitude)

coordinates = np.vstack((x, y, z)).T  


neigh = NearestNeighbors(n_neighbors=2, algorithm='kd_tree')


neigh.fit(coordinates)

distances, indices = neigh.kneighbors(coordinates)

"""
Now I will match the longitude and latitudes back to the respective indices 
"""

neighbor_latitudes = []
neighbor_longitudes = []

# Loop over each set of neighbors
for neighbor_indices in indices:
    # Retrieve the latitude and longitude for each neighbor
    lons = longitude[neighbor_indices]
    lats = latitude[neighbor_indices]
    neighbor_latitudes.append(lats)
    neighbor_longitudes.append(lons)


"""
To get the Distances
"""

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the moon given in decimal degrees using the Haversine formula.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Radius of the moon in kilometers
    return c * rVenus



distances = []

# Loop through each pair of latitudes and longitudes
for lat_pair, lon_pair in zip(neighbor_latitudes, neighbor_longitudes):
    # Unpack each pair (lat1, lat2) and (lon1, lon2)
    lat1, lat2 = lat_pair
    lon1, lon2 = lon_pair
    distance = haversine(lon1, lat1, lon2, lat2)
    distances.append(distance)
