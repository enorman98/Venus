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

def zscore(coordinates,R=1.0):
    n=len(coordinates)
    mean_dist = compute_mean_distance(coordinates,R)
    A = 4*np.pi*R**2
    dexpected = 1.0 / (2 * np.sqrt(n / A))
    sigma = 0.26136 / np.sqrt(n**2 / A)
    Z = (mean_dist - dexpected) / sigma
    return Z

rVenus = 6051.8
vpoints=pd.read_csv('Venus_Craters.csv',usecols=['lat', 'lon'])

longitude=np.deg2rad(vpoints['lon'].values)
latitude=np.deg2rad(vpoints['lat'].values)
ven_coord = np.vstack((latitude,longitude)).T 
ncraters=len(ven_coord)
rnd_z= []

for _ in range(1000):
    rnd_z.append(zscore(randomp_points(ncraters),R=rVenus))
   
rnd_z_stdev = np.std(rnd_z)
rnd_z_mean = np.mean(rnd_z)


print(f"Z-score (Venus) : {zscore(ven_coord)}" )
print(f"Z-score (Random): {rnd_z_mean:.1f} ± {rnd_z_stdev:.1f}")
