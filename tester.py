import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np
import geopy.distance
import imageio
from timeit import default_timer as timer
import pandas as pd
import seaborn as sns
import scipy
from scipy.stats import norm
import requests
import json
import os
from os.path import join, dirname, abspath
from glob import glob
import io
import pathlib
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
from shapely import geometry
import random
import shapely.geometry as ge

import itertools
import networkx as nx

import shapely

from shapely.geometry import LineString, Point
from timeit import default_timer as timer
from datetime import timedelta


import time
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import geopandas as gpd

import networkx as nx
import osmnx as ox
from descartes import PolygonPatch
from shapely.geometry import Point, LineString, Polygon

from shapely.affinity import rotate
from shapely.geometry import LineString, Point
from shapely.ops import split


import json
import random

import socket



from sshtunnel import SSHTunnelForwarder
import pymongo
import os.path
from pymongo import MongoClient




try:
    from data_from_mongo import read_from_mongo
except:
    import read_from_mongo



REMOTE_ADDRESS = ('docdb-2019-06-13-11-43-18.cluster-cybs9fpwjg54.eu-west-1.docdb.amazonaws.com',27017)
MONGO_HOST = 'automotive.vizible.zone'

MONGO_DB = "VizibleZone"
MONGO_USER = "ubuntu"

pem_server_file = 'viziblezone-prod.pem'
pem_ca_file = 'rds-combined-ca-bundle.pem'
pem_path = '../pems/'

if not os.path.exists(pem_path + pem_server_file):
    pem_path=pem_path[1:]

server = SSHTunnelForwarder(
    MONGO_HOST,
    ssh_pkey = pem_path + pem_server_file,
    ssh_username=MONGO_USER,
    remote_bind_address=REMOTE_ADDRESS
)
server.stop()
server.start()

client = MongoClient('127.0.0.1',
                     server.local_bind_port,
                     username='viziblezone',
                     password='vz123456',
                     ssl=True,
                     ssl_match_hostname=False,
                     ssl_ca_certs=(pem_path + pem_ca_file),
                     authMechanism='SCRAM-SHA-1') # server.local_bind_port is assigned local port
db = client[MONGO_DB]
print('you are connected to Production server')
print(db.collection_names())


ids_sessions=[['5db00a4345d6764d7480fd4b',
  '5db00a4445d6764d7480fd6d',
  '5db00a4545d6764d748104a9',
  '5db00a4645d6764d74810be8',
  '5db00a4745d6764d74811326',
  '5db00a4c45d6764d748121a7',
  '5db00a4745d6764d74811a65',
  '5db00a4d45d6764d748128e9',
  '5db00a4e45d6764d7481302c',
  '5db00a4e45d6764d7481376b',
  '5db00a4f45d6764d74813eaa',
  '5db00a4f45d6764d748145ab',
  '5db00a5045d6764d74814ce5',
  '5db00a5245d6764d74815426',
  '5db00a5245d6764d74815b64',
  '5db00a5345d6764d748162a8',
  '5db00a5345d6764d748169e8',
  '5db00a5445d6764d7481712a',
  '5db00a5445d6764d7481786c',
  '5db00a5545d6764d74817fad',
  '5db00a5545d6764d748186ef',
  '5db00a5745d6764d74818e28',
  '5db00a5845d6764d7481955e',
  '5db00a5845d6764d74819ca2',
  '5db00a5945d6764d7481a3e6',
  '5db00a5945d6764d7481ab24',
  '5db00a5a45d6764d7481b264',
  '5db00a5c45d6764d7481b9a3',
  '5db00a5c45d6764d7481c0e3',
  '5db00a5d45d6764d7481c826',
  '5db00a5d45d6764d7481cf67',
  '5db00a5e45d6764d7481d6a5',
  '5db00a5f45d6764d7481dde0',
  '5db00a5f45d6764d7481e520',
  '5db00a6145d6764d7481ec62',
  '5db00a6145d6764d7481f397',
  '5db00a6245d6764d7481fad7',
  '5db00a6345d6764d74820219',
  '5db00a6345d6764d74820955',
  '5db00a6445d6764d74821094',
  '5db00a6445d6764d748217d4',
  '5db00a6645d6764d74821f12',
  '5db00a6645d6764d7482264a',
  '5db00a6745d6764d74822d8e',
  '5db00a6745d6764d748234cf',
  '5db00a6845d6764d74823c15',
  '5db00a6945d6764d74824354',
  '5db00a6b45d6764d74824a98',
  '5db00a6d45d6764d748251d9',
  '5db00a6d45d6764d7482591b',
  '5db00a6e45d6764d7482605a',
  '5db00a6e45d6764d74826790',
  '5db00a6f45d6764d74826ec8',
  '5db00a6f45d6764d748275f8',
  '5db00a7045d6764d74827d35',
  '5db00a7145d6764d74828475',
  '5db00a7145d6764d74828bb6',
  '5db00a7245d6764d748292f4',
  '5db00a7245d6764d74829a37',
  '5db00a7345d6764d7482a179',
  '5db00a7345d6764d7482a8b9',
  '5db00a7545d6764d7482afef',
  '5db00a7545d6764d7482b05f',
  '5db00a7645d6764d7482b7a8',
  '5db00a7645d6764d7482bef0',
  '5db00a7745d6764d7482c639',
  '5db00a7d45d6764d7482cd85',
  '5db00b7045d6764d7482cdee',
  '5db00bac45d6764d7482ce13'],
 ['5db00e4a45d6764d748334ca'],
 ['5db01f7345d6764d7484aca7'],
 ['5db02d8c45d6764d7484b4d6', '5db02e7e45d6764d7484b50f'],
 ['5db0411845d6764d748595c7'],
 ['5db0428945d6764d7485b043', '5db042c445d6764d7485b0bf'],
 ['5db0565a45d6764d748a9dce'],
 ['5db0580545d6764d748a9df6'],
 ['5db0647445d6764d748b300c'],
 ['5db0671645d6764d748c4509',
  '5db0678e45d6764d748c452b',
  '5db0680645d6764d748c454c',
  '5db0684445d6764d748c456e',
  '5db068bc45d6764d748c458f'],
 ['5db0801645d6764d748e4005',
  '5db0801645d6764d748e4026',
  '5db0801745d6764d748e46f4',
  '5db0801845d6764d748e4e3c',
  '5db0801945d6764d748e557d',
  '5db0801a45d6764d748e5cbf',
  '5db0801b45d6764d748e6402',
  '5db0801c45d6764d748e6b4b',
  '5db0801d45d6764d748e728c',
  '5db0801e45d6764d748e79a6',
  '5db0801f45d6764d748e80db',
  '5db0802045d6764d748e881b',
  '5db0802145d6764d748e8f5d',
  '5db0802245d6764d748e969f',
  '5db0802345d6764d748e9de1',
  '5db0802445d6764d748ea525',
  '5db0802545d6764d748eac65',
  '5db0802645d6764d748eb3a6',
  '5db0802745d6764d748ebae7',
  '5db0802845d6764d748ec220',
  '5db0802945d6764d748ec958',
  '5db0802a45d6764d748ed09a',
  '5db0802b45d6764d748ed7d8',
  '5db0802d45d6764d748edf19',
  '5db0802d45d6764d748ee658',
  '5db0802e45d6764d748eed96',
  '5db0802f45d6764d748ef4d7',
  '5db0803045d6764d748efc19',
  '5db0803145d6764d748f0a8f',
  '5db0803245d6764d748f11ce',
  '5db0803345d6764d748f203d',
  '5db0803445d6764d748f277c',
  '5db0803545d6764d748f35ea',
  '5db0803645d6764d748f4457',
  '5db0803745d6764d748f4b9a',
  '5db0803845d6764d748f5a0c',
  '5db0803945d6764d748f6140',
  '5db0803b45d6764d748f6883',
  '5db0803c45d6764d748f6fca',
  '5db0803d45d6764d748f76f5',
  '5db0803e45d6764d748f7e26',
  '5db0803f45d6764d748f8566',
  '5db0804045d6764d748f8ca9',
  '5db0804145d6764d748f93eb',
  '5db0804245d6764d748f9b2e',
  '5db0804345d6764d748fa271',
  '5db0804445d6764d748fa9b8',
  '5db0804345d6764d748fa9b4',
  '5db0804545d6764d748fb0fa',
  '5db0804645d6764d748fb83b',
  '5db0804745d6764d748fbf75',
  '5db0804845d6764d748fc6a9',
  '5db0804945d6764d748fcde7',
  '5db0804a45d6764d748fd527',
  '5db0804c45d6764d748fdc68',
  '5db0804d45d6764d748fe3a7',
  '5db0804e45d6764d748feae9',
  '5db0804f45d6764d748ff228',
  '5db0805045d6764d748ff969',
  '5db0805145d6764d749000a7',
  '5db0805345d6764d749007e2',
  '5db0805445d6764d74900f17',
  '5db0805545d6764d7490165a',
  '5db0805645d6764d74901d98',
  '5db0805745d6764d749024d6',
  '5db0805845d6764d74902c15',
  '5db0805945d6764d74903358',
  '5db0805a45d6764d74903a97',
  '5db0805b45d6764d749041db',
  '5db0805c45d6764d7490491d',
  '5db0805d45d6764d74905058',
  '5db0805e45d6764d7490578a',
  '5db0805f45d6764d74905eca',
  '5db0806045d6764d7490660b',
  '5db0806145d6764d74906d47',
  '5db0806245d6764d74907488']]


print('retreiving data from server')

#ids_sessions = get_id_list_from_user(user_id, start_date, end_date)
ids=ids_sessions[0]


print('get_df_for_ids')
mc = read_from_mongo.MongoConnection()
if not mc.connect('prod'):
    print ("cannot connect to server")

df_AS=read_from_mongo.get_df_for_ids(mc,ids)

print('dataframe is ready')

df_AS=df_AS.sort_values('timestamps_value').reset_index(drop=True)

df_AS=df_AS.iloc[22400:24000]

print('map is ready')

unique_locs_df= df_AS[['gps_longitude','gps_latitude']].drop_duplicates(['gps_longitude','gps_latitude']).reset_index()

walk_pois=[ge.Point(row[1]['gps_longitude'],row[1]['gps_latitude']) for row in unique_locs_df.iterrows()]


walk_pois_df =df_AS[['gps_longitude','gps_latitude','gps_speed','timestamps_value']].drop_duplicates(['gps_longitude','gps_latitude'])

walk_pois_df['timestamps_value']=pd.to_datetime(walk_pois_df['timestamps_value'])

walk_pois_df['next_node']=None

middle_loc=((df_AS['gps_latitude'].max()+df_AS['gps_latitude'].min())/2, (df_AS['gps_longitude'].max()+df_AS['gps_longitude'].min())/2)

radius=max(geopy.distance.distance((df_AS['gps_latitude'].max(),df_AS['gps_longitude'].mean()),(df_AS['gps_latitude'].min(),df_AS['gps_longitude'].mean())).m,
           geopy.distance.distance((df_AS['gps_latitude'].mean(),df_AS['gps_longitude'].max()),(df_AS['gps_latitude'].mean(),df_AS['gps_longitude'].min())).m)/1

G = ox.graph_from_point(middle_loc, distance= radius+100, network_type='drive',simplify=False)
G_projected = ox.project_graph(G)
ox.plot_graph(G_projected)

walk_pois_df.timestamps_value=walk_pois_df.timestamps_value.astype(str)



gps_azimuth=random.randint(0, 360)

gps_longitude=random.uniform(df_AS['gps_longitude'].min(), df_AS['gps_longitude'].max())
gps_latitude =random.uniform(df_AS['gps_latitude'].min(), df_AS['gps_latitude'].max())


'''
gps_longitude=35.021983217645904
gps_latitude =31.808269377665198

'''

speed = 0   #meanwhile we always count 5 meters
pedestrian1_dict = {"id":1, "gps_azimuth": gps_azimuth, "gps_longitude": gps_longitude, "gps_latitude": gps_latitude, "speed":speed }


gps_azimuth=random.randint(0, 360)

gps_longitude=random.uniform(df_AS['gps_longitude'].min(), df_AS['gps_longitude'].max())
gps_latitude =random.uniform(df_AS['gps_latitude'].min(), df_AS['gps_latitude'].max())

speed = 0   #meanwhile we always count 5 meters
pedestrian2_dict = {"id":2, "gps_azimuth": gps_azimuth, "gps_longitude": gps_longitude, "gps_latitude": gps_latitude, "speed":speed }


gps_azimuth=random.randint(0, 360)

gps_longitude=random.uniform(df_AS['gps_longitude'].min(), df_AS['gps_longitude'].max())
gps_latitude =random.uniform(df_AS['gps_latitude'].min(), df_AS['gps_latitude'].max())

speed = 0   #meanwhile we always count 5 meters
pedestrian3_dict = {"id":3, "gps_azimuth": gps_azimuth, "gps_longitude": gps_longitude, "gps_latitude": gps_latitude, "speed":speed }

gps_azimuth=random.randint(0, 360)

gps_longitude=random.uniform(df_AS['gps_longitude'].min(), df_AS['gps_longitude'].max())
gps_latitude =random.uniform(df_AS['gps_latitude'].min(), df_AS['gps_latitude'].max())

speed = 0   #meanwhile we always count 5 meters
pedestrian4_dict = {"id":4, "gps_azimuth": gps_azimuth, "gps_longitude": gps_longitude, "gps_latitude": gps_latitude, "speed":speed }


pedestrians_list=[pedestrian1_dict, pedestrian2_dict, pedestrian3_dict,pedestrian4_dict]

import socket

# local host IP '127.0.0.1'
host = '127.0.0.1'

# Define the port on which you want to connect
port = 12345

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# connect to server on local computer
s.connect((host, port))

# message you send to server
for i in range(len(walk_pois_df) - 1):
    driver_dict = dict(walk_pois_df.iloc[i])
    ped_driver_dict = {"id": i, "driver": driver_dict, "pedestrians": pedestrians_list}

    message = json.dumps(ped_driver_dict)  # message sent to server
    print(message)
    s.send(message.encode('ascii'))
    print('waiting for data')

    # messaga received from server
    data = s.recv(1024)

    # print the received message
    # here it would be a reverse of sent message
    print('Received from the server :', str(data.decode('ascii')))

    # ask the client whether he wants to continue
    time.sleep(1)
# close the connection
print('end', i)
s.close()