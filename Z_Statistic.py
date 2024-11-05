import numpy as np
import pandas as pd 
from sklearn.neighbors import NearestNeighbors

def randomp_points(size):
    # Number of points
    size = 947

    rng = np.random.default_rng()
    u = rng.uniform(size=size)
    v = rng.uniform(size=size)

    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)

    lon = theta - np.pi  # longitude in the range [-pi, pi]
    lat = phi - np.pi / 2.0

    points = pd.DataFrame({'lat': lat, 'lon': lon})
    return points

def compute_mean_distance(coordinates,R=1.0):
    neigh = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', metric='haversine').fit(coordinates)
    distances, indices = neigh.kneighbors(coordinates)
    # The first column of distances is the point itself, so we ignore it
    distances = np.array([d[1] for d in distances])*R

    mean_dist = np.mean(distances)
    return mean_dist

rVenus = 6051.8
vpoints=pd.read_csv('Venus_Craters.csv',usecols=['lat', 'lon'])

longitude=np.deg2rad(vpoints['lon'].values)
latitude=np.deg2rad(vpoints['lat'].values)
coordinates = np.vstack((latitude,longitude)).T 

ven_mean_dist = compute_mean_distance(coordinates,R=rVenus)

rnd_mean_dist = []
for _ in range(100):
    rnd_mean_dist.append(compute_mean_distance(randomp_points(len(coordinates)),R=rVenus))
   
rnd_mean_stdev = np.std(rnd_mean_dist)
rnd_mean_dist = np.mean(rnd_mean_dist) 

print(f"Mean distance (Venus) : {ven_mean_dist:.1f} km")
print(f"Mean distance (Random): {rnd_mean_dist:.1f} ± {rnd_mean_stdev:.1f} km")
