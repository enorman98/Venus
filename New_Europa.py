import numpy as np
import pandas as pd 
from sklearn.neighbors import NearestNeighbors


"""
First I am going to put random points on a circle. This will be used for determining the standard deviation. 
"""
def randomp_points(size):
    # Number of points

    rng = np.random.default_rng()
    u = rng.uniform(size=size)
    v = rng.uniform(size=size)

    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)

    lon = theta - np.pi  # longitude in the range [-pi, pi]
    lat = phi - np.pi / 2.0

    points = pd.DataFrame({'lat': lat, 'lon': lon})
    return points


"""
These are the knowns that will be used in the code
"""
europa_radius = 1561
radius_km = 403.81
center_lat_deg, center_lon_deg = 10, 20
center_lat_rad, center_lon_rad = np.radians(center_lat_deg), np.radians(center_lon_deg)

"""
For this part of the code, I created another number generater so that it will produce one random point at a time. 
"""
def random_points_generator():
    rng = np.random.default_rng()
    while True:
        # Generate one random point
        u = rng.uniform()
        v = rng.uniform()

        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)

        lon = theta - np.pi  
        lat = phi - np.pi / 2.0

        yield {'lat': lat, 'lon': lon}


point_gen = random_points_generator()

#Create empty lists so that I can call on them later for z-score
points = []

#These are the points outside the masked area
points_outside = []

#There are all the points generated
all_points = []

for i, point in enumerate(point_gen):
    # Add the new point to the list of all points
    points.append([point['lat'], point['lon']])
    all_points.append({'lat': point['lat'], 'lon': point['lon']})  # Save all points

    #This is the array will it will only save the new points. 
    points_array = np.array([point])  

    # Compute distance for the new point
    center_coords = np.array([[center_lat_rad, center_lon_rad]])
    neigh = NearestNeighbors(radius=radius_km / europa_radius, metric='haversine')
    neigh.fit(np.array(points))  
    distances = neigh.radius_neighbors(center_coords, return_distance=False)

    # Check if the new point is outside the circle. If the index isnt in distance[0], then its considered outside and saved.
    if i not in distances[0]:  
        points_outside.append({'lat': point['lat'], 'lon': point['lon']})

    # Check if the number of outside points has reached 56 (the number of craters I counted)
    if len(points_outside) >= 56:
        print(f"Reached 56 points outside the circle. Breaking loop at iteration {i + 1}.")
        break

"""
Original code
"""
def compute_mean_distance(coordinates,R=1.0):
    neigh = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', metric='haversine').fit(coordinates)
    distances, indices = neigh.kneighbors(coordinates)
    # The first column of distances is the point itself, so we ignore it
    distances = np.array([d[1] for d in distances])*R

    mean_dist = np.mean(distances)
    return mean_dist

def zscore(coordinates,R=1.0):
    n=len(coordinates)
    mean_dist = compute_mean_distance(coordinates,R)
    A = 4*np.pi*R**2
    dexpected = 1.0 / (2 * np.sqrt(n / A))
    sigma = 0.26136 / np.sqrt(n**2 / A)
    Z = (mean_dist - dexpected) / sigma
    return Z


ncraters=len(all_points)
rnd_z=[]

all_points= np.array([[point['lat'], point['lon']] for point in all_points])
points_outside= np.array([[point['lat'], point['lon']] for point in points_outside])

for _ in range(1000):
    rnd_z.append(zscore(randomp_points(len(all_points)), R=europa_radius))

rnd_z_stdev = np.std(rnd_z)
rnd_z_mean = np.mean(rnd_z)


print(f"Z-score (Europa No Mask) : {zscore(all_points)}")
print(f"Z-score (Europa Masked) : {zscore(points_outside)}")
print(f"Z-score (Random): {rnd_z_mean:.2f} ± {rnd_z_stdev:.1f}")