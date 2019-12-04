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


middle_loc=(31.816938, 35.0153425)

G = ox.graph_from_point(middle_loc, distance=3000, network_type='drive', simplify=False)

fig, ax = ox.plot_graph(G, fig_height=10,
                        show=False, close=False,
                        edge_color='#777777')

#plt.show(block=False)
plt.pause(5)
plt.close()
