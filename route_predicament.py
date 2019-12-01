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
from descartes.patch import PolygonPatch
import random
import itertools


import socket


try:
    from OSM_graph_manager import graph_from_dataframe
    from data_from_mongo import read_from_mongo

except:
    import graph_from_dataframe
    import read_from_mongo


def node_to_point(n):
    dic=G.nodes[n]
    return (dic['x'],dic['y'])


def is_reachable(n1,n2):
    try:
        nx.shortest_path_length(G, n1 ,n2)
        return True
    except:
        return False

def same_edge(e1,e2):
    if {e1[1],e1[2]}=={e2[1],e2[2]}:
        return True
    return False

def is_one_way(e):
    if not is_reachable(e[1] , e[2]) or nx.shortest_path_length(G, e[1] , e[2]) != 1:
        if is_reachable(e[2] , e[1]) and nx.shortest_path_length(G, e[2] , e[1]) == 1:
            return True, (e[2], e[1])
        else:
            'Error, parameter is not an edge'
    elif not is_reachable(e[2] , e[1]) or nx.shortest_path_length(G, e[2] , e[1]) != 1:
            return True,(e[1], e[2])
    else:
        return False,(None,None)

def edge_length_by_points(G, n1, n2):
    return geopy.distance.distance(node_to_point(n1) , node_to_point(n2)).m

def drop_dups(l):
    return [key for key, grp in itertools.groupby(l)]


def find_direction():
    print(path_df.shape)
    start_row = path_df.iloc[-1]
    start_point = (start_row['gps_longitude'], start_row['gps_latitude'])
    nearest_edge_start = ox.utils.get_nearest_edge(G, (start_point[1], start_point[0]))
    print(start_point, nearest_edge_start)

    prev_row = path_df.iloc[-2]
    prev_point = (prev_row['gps_longitude'], prev_row['gps_latitude'])
    nearest_edge_prev = ox.utils.get_nearest_edge(G, (prev_point[1], prev_point[0]))
    print(prev_point, nearest_edge_prev)

    print('start_point ', start_point)

    if same_edge(nearest_edge_start, nearest_edge_prev):
        #        print('same edge')
        is_ow, (former_node, next_node) = is_one_way(nearest_edge_start)
        print('one_way ', is_ow, former_node, next_node)
        if not is_ow:
            dist_fn_start = geopy.distance.distance(
                (nearest_edge_start[0].coords[0][0], nearest_edge_start[0].coords[0][1]), start_point).m

            dist_fn_prev = geopy.distance.distance(
                (nearest_edge_start[0].coords[0][0], nearest_edge_start[0].coords[0][1]), prev_point).m

            next_node = None
            if dist_fn_prev < dist_fn_start:
                next_node = nearest_edge_start[2]
                former_node = nearest_edge_start[1]
            else:
                next_node = nearest_edge_start[1]
                former_node = nearest_edge_start[2]

    else:
        #        print('different edge')

        if nearest_edge_start[1] == nearest_edge_prev[1] or nearest_edge_start[1] == nearest_edge_prev[2]:
            former_node = nearest_edge_start[1]
            next_node = nearest_edge_start[2]
        elif nearest_edge_start[2] == nearest_edge_prev[1] or nearest_edge_start[2] == nearest_edge_prev[2]:
            former_node = nearest_edge_start[2]
            next_node = nearest_edge_start[1]
        else:
            print("WARNING:  cannot find direction")
            return (None, None, None, None)
    return former_node, next_node, start_point, prev_point


def find_next_tree(nns, former_node, tree_edges, dist):
    global leaves
    global graph_edges

    while len(nns) > 0:
        nn = nns[0]
        nns = nns[1:]

        nearest_edges = [e for e in graph_edges if
                         (e[2] != former_node and e[1] == nn[0] and is_reachable(e[1],
                                                                                 e[2]) and nx.shortest_path_length(G,
                                                                                                                   e[1],
                                                                                                                   e[
                                                                                                                       2]) == 1)
                         or (e[1] != former_node and e[2] == nn[0] and is_reachable(e[2],
                                                                                    e[1]) and nx.shortest_path_length(G,
                                                                                                                      e[
                                                                                                                          2],
                                                                                                                      e[
                                                                                                                          1]) == 1)]
        tree_edges += nearest_edges
        nnids = list(set([n for n in sum([[e[1], e[2]] for e in nearest_edges], []) if n != nn[0]]))
        nnids = [n for n in nnids if is_reachable(nn[0], n)]
        leaves += [[n, edge_length_by_points(G, nn[0], n) + nn[1]] for n in nnids if
                   is_reachable(nn[0], n) and edge_length_by_points(G, nn[0], n) + nn[1] > dist]
        nns_ = [[n, edge_length_by_points(G, nn[0], n) + nn[1]] for n in nnids if
                n not in [l[0] for l in leaves]]  # problematic in case of rounds

        if len(nns_) > 0:
            find_next_tree(nns_, nn[0], tree_edges, dist)


def shortest_path_length_extended(G, p1, p2):  # not only for nodes,  p1 and p2 assumed to be on edges
    e1 = ox.utils.get_nearest_edge(G, p1)
    e2 = ox.utils.get_nearest_edge(G, p2)
    if same_edge(e1, e2):
        return geopy.distance.distance(p1, p2).m
    op1 = geopy.distance.distance(p1, e1[1]).m + nx.shortest_path_length(G, e1[1], e2[1]) + geopy.distance.distance(
        e2[1], p2).m
    op2 = geopy.distance.distance(p1, e1[2]).m + nx.shortest_path_length(G, e1[2], e2[1]) + geopy.distance.distance(
        e2[1], p2).m
    op2 = geopy.distance.distance(p1, e1[1]).m + nx.shortest_path_length(G, e1[1], e2[2]) + geopy.distance.distance(
        e2[2], p2).m
    op3 = geopy.distance.distance(p1, e1[2]).m + nx.shortest_path_length(G, e1[2], e2[2]) + geopy.distance.distance(
        e2[2], p2).m
    return min([op1, op2, op3, op4])


def discontinuity_detected(path_df):
    print('ddd')
    path_df = path_df[path_df['next_node'].apply(lambda x: pd.notnull(x))]

    if len(set(list(path_df['next_node']))) < 2:
        print(drop_dups(list(path_df['next_node'])))
        print('False len 2')
        return False

    time_from_prev = (path_df.iloc[-1]['timestamps_value'] - path_df.iloc[-2]['timestamps_value']).seconds
    print('time_from_prev ', time_from_prev)

    if not is_reachable(path_df.iloc[0]['next_node'], path_df.iloc[-1]['next_node']):
        print('True not reachable ', path_df.iloc[0]['next_node'], ' ', path_df.iloc[-1]['next_node'])
        return True

    #    print(path_df)

    nodes = drop_dups(list(path_df['next_node']))
    print(nodes)
    # the new node shouldnt be too far from the former one
    if not is_reachable(nodes[-2], nodes[-1]) or (
            nx.shortest_path_length(G, nodes[-2], nodes[-1]) > time_from_prev * 4):
        print('True too many nodes')
        return True

    speed = path_df['gps_speed'].mean()

    time = (path_df.iloc[-1]['timestamps_value'] - path_df.iloc[0]['timestamps_value']).seconds
    print(path_df.iloc[0]['next_node'], path_df.iloc[-1]['next_node'])

    points = list(zip(path_df['gps_longitude'], path_df['gps_latitude']))

    distance = shortest_path_length_extended(G, points[0], points[-1])
    print('distance ', distance, 'speed*time', speed * time)
    return not (speed * time > distance - distance / 3 and speed * time < distance + distance / 3)


def snap_locations_to_road():
    print('a')
    path_df['point_location'] = tuple(zip(path_df['gps_latitude'], path_df['gps_longitude']))
    print('ab1')

    strlocs = str(list(path_df['point_location']))[2:-2].replace('), (', '|').replace(', ', ',')
    print('ab2')

    file = open('api_key.txt', 'r')
    api_key = file.read()
    print('ab3')

    response = requests.get(
        'https://roads.googleapis.com/v1/snapToRoads?path=' + strlocs + '&interpolate=false&key=' + api_key)
    print('ab3')

    response = response.json()
    if 'error' in response.keys():
        print (response['error']['code'])
    print('ab4')

    snapped = [(d['location']['latitude'], d['location']['longitude']) for d in response['snappedPoints']]
    print('b')
    path_df['point_location'] = snapped
    print('c')
    path_df['gps_latitude'] = path_df['point_location'].apply(lambda x: x[0])
    path_df['gps_longitude'] = path_df['point_location'].apply(lambda x: x[1])


def getExtrapoledLine(p1, p2, EXTRAPOL_RATIO=2):
    'Creates a line extrapoled in p1->p2 direction'

    a = p1
    b = (p1[0] + EXTRAPOL_RATIO * (p2[0] - p1[0]), p1[1] + EXTRAPOL_RATIO * (p2[1] - p1[1]))
    return LineString([a, b])


def find_target_point(point, angle, dist_in_m): #point: y is lat, x is long
    dx_in_m=dist_in_m*np.sin(angle*np.pi/180)
    dy_in_m=dist_in_m*np.cos(angle*np.pi/180)
    dx_in_long=dx_in_m/40075000*360/np.cos(point.y/180*np.pi)
    dy_in_lat=dy_in_m/111700
    return ge.Point(point.x+dx_in_long,point.y+dy_in_lat)


import socket

from _thread import *
import threading

path_df = pd.DataFrame(columns=['gps_longitude', 'gps_latitude', 'gps_speed', 'timestamps_value', 'next_node'])
# print(path_df)

host = ""

# reverse a port on your computer
# in our case it is 12345 but it
# can be anything
port = 12345
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))
print("socket binded to port", port)

# put the socket into listening mode
s.listen(5)
print("socket is listening")

# establish connection with client
c, addr = s.accept()

# lock acquired by client
#    print_lock.acquire()
print('Connected to :', addr[0], ':', addr[1])

G = None
graph_edges = None
leaves = None

# a forever loop until client wants to exit
while True:

    # Start a new thread and return its identifier
    print(len(path_df))
    data = c.recv(1024)
    if not data:
        print('Bye')

        # lock released on exit
        #            print_lock.release()
        break

    # reverse the given string from client
    #        data = data[::-1]
    print(data)
    # send back reversed string to client
    c.send('Got it'.encode('ascii'))

    jsns = ['{' + j + '}' for j in data.decode("ascii").split('}{')]

    print(jsns)

    jsns[0] = jsns[0][1:]
    jsns[-1] = jsns[-1][:-1]

    packets = [json.loads(j) for j in jsns]

    pedestrians = [p['pedestrians'] for p in packets if p['id'] == max([p['id'] for p in packets])][0]

    sorted(packets, key=lambda i: i['id'])
    drivers = [p['driver'] for p in packets]

    for d in drivers:
        newrow = d

        newrow['timestamps_value'] = pd.to_datetime(newrow['timestamps_value'])

        path_df = path_df.append(newrow, ignore_index=True)

    nn = None

    if len(path_df) < 2:
        continue
    newloc = (path_df.iloc[-1]['gps_latitude'], path_df.iloc[-1]['gps_longitude'])
    if G is None or ge.Point(tuple(reversed(newloc))).distance(
            ox.utils.get_nearest_edge(G, tuple(reversed(newloc)))[0]) * 10000 > 30:
        G = ox.graph_from_point(newloc, distance=3000, network_type='drive', simplify=False)

        gdf = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
        graph_edges = gdf[["geometry", "u", "v"]].values.tolist()

    dist = 25 * path_df.iloc[-1]['gps_speed']

    former_node, next_node, start_point, prev_point = find_direction()
    if next_node is None:
        continue

    nn = next_node

    path_df.loc[path_df.index[-1], 'next_node'] = next_node

    if discontinuity_detected(path_df):
        path_df.loc[path_df.index[-1], 'next_node'] = None

        print('Warning: discontinuity detected')
        snap_locations_to_road()
        #            print(path_df[['gps_latitude','gps_longitude','point_location']])

        former_node, next_node, start_point, prev_point = find_direction()
        nn = next_node

        path_df.loc[path_df.index[-1], 'next_node'] = next_node

        if next_node is None or discontinuity_detected(path_df):
            print('Warning: discontinuity detected in snapped locations')
            continue

    print(former_node, nn, '************************')
    print('start_point ', start_point)

    start_edge_len = geopy.distance.distance(start_point, node_to_point(nn)).m
    full_edges = []
    partial_edges = []
    repaint = False
    if start_edge_len > dist:
        only_edge = LineString([start_point, node_to_point(nn)])
        only_edge = LineString([start_point, only_edge.interpolate(
            only_edge.length * dist / geopy.distance.distance(start_point, node_to_point(nn)).m)])
        print('start_edge_len', start_edge_len)
        first_mid_edge = only_edge
        repaint = True
    else:
        nns = [[nn, start_edge_len]]
        tree_edges = []
        leaves = []
        #    starttimer = timer()
        #    print('leaves', leaves)
        print(former_node, nns, '*********************')
        find_next_tree(nns, former_node, tree_edges, dist)
        print('leaves', leaves)
        print('len_tree_edges', len(tree_edges))
        #    print('leaves', leaves)
        #    endtimer = timer()
        #    print(timedelta(seconds=endtimer-starttimer))

        tree_edges_nodes = [[e[1], e[2]] for e in tree_edges]
        tree_edges_nodes = sum(tree_edges_nodes, [])

        leaves_edges = [[LineString([Point(G.nodes[path[-2]]['x'], G.nodes[path[-2]]['y']),
                                     Point(G.nodes[path[-1]]['x'], G.nodes[path[-1]]['y'])]), path[-2], path[-1]] for
                        path in
                        [nx.shortest_path(G, nn, l[0], weight='length') for l in
                         [l for l in leaves if is_reachable(nn, l[0])]
                         if all(elem in tree_edges_nodes for elem in nx.shortest_path(G, nn, l[0], weight='length'))
                         ] if len(path) > 1]
        [print(le) for le in leaves_edges]
        #       leaves_edges=[le for le in leaves_edges if nx.shortest_path_length(G, nn, le[2] , weight='length')+start_edge_len-dist > 0]

        #        print('--', (nx.shortest_path_length(G, nn, leaves_edges[0][2] , weight='length')+start_edge_len-dist)/nx.shortest_path_length(G, leaves_edges[0][1], leaves_edges[0][2] , weight='length'))

        # if nx.shortest_path(G, nn, l[0] , weight='length') in tree_edges_nodes -- cases when the shortest path is different will be cut!!!!
        # the edge cases that are not covered are when the driver destination is before the leaf node (bc shortest path are only between nodes)
        # or if the driver decided not to take the shortest path

        if len(leaves_edges) > 0:
            print('----------------------')
            print(leaves_edges)
            [print('le', edge_length_by_points(G, nn, le[2]) + start_edge_len - dist) for le in leaves_edges]
            print('--',
                  (edge_length_by_points(G, nn, leaves_edges[0][2]) + start_edge_len - dist) / edge_length_by_points(G,
                                                                                                                     leaves_edges[
                                                                                                                         0][
                                                                                                                         1],
                                                                                                                     leaves_edges[
                                                                                                                         0][
                                                                                                                         2]))

            mid_edges = [[le[0].interpolate(le[0].length * (
                    1 - (edge_length_by_points(G, nn, le[2]) + start_edge_len - dist) / edge_length_by_points(G, le[1],
                                                                                                              le[2]))),
                          le]
                         for le in leaves_edges]

            print('mid_edges ', mid_edges[0][0].x)

            full_edges = [t for t in tree_edges if
                          t[2] not in [l[0] for l in leaves] and t[1] not in [l[0] for l in leaves]]
            full_edges = [f[0] for f in full_edges]

            #       print('full_edges',  len(full_edges))
            print('fe', len(full_edges))
            partial_edges = [LineString([node_to_point(m[1][1]), m[0]]) for m in mid_edges]
            #        print('partial_edges', len(partial_edges))

            print('pe', len(partial_edges))

            first_mid_edge = LineString([start_point, node_to_point(nn)])
            repaint = True
        else:
            print('no leaves edges')

    if repaint:

        try:
            b = 0.0006

            fig, ax = ox.plot_graph(G, fig_height=10,
                                    show=False, close=False,
                                    edge_color='#777777')

            patch = PolygonPatch(Point(prev_point).buffer(b), fc='#9900ff', ec='k', linewidth=0, alpha=0.5, zorder=-1)
            ax.add_patch(patch)

            patch = PolygonPatch(Point(start_point).buffer(b), fc='#ff00aa', ec='k', linewidth=0, alpha=0.5, zorder=-1)
            ax.add_patch(patch)

            patch = PolygonPatch(Point(G.nodes[next_node]['x'], G.nodes[next_node]['y']).buffer(b), fc='#0000ff',
                                 ec='k', linewidth=0, alpha=0.5, zorder=-1)
            ax.add_patch(patch)

            for point in [Point(G.nodes[l[0]]['x'], G.nodes[l[0]]['y']) for l in leaves]:
                patch = PolygonPatch(point.buffer(b), fc='#00ffff', ec='k', linewidth=0, alpha=0.5, zorder=-1)
                ax.add_patch(patch)

            for line in full_edges:
                patch = PolygonPatch(line.buffer(b), fc='#0f800f', ec='k', linewidth=0, alpha=0.5, zorder=-1)
                ax.add_patch(patch)

            for line in partial_edges:
                patch = PolygonPatch(line.buffer(b), fc='#0f899f', ec='k', linewidth=0, alpha=0.5, zorder=-1)
                ax.add_patch(patch)

            patch = PolygonPatch(first_mid_edge.buffer(b), fc='#ffff00', ec='k', linewidth=0, alpha=0.5, zorder=-1)
            ax.add_patch(patch)

            for p in pedestrians:

                poi = Point(p['gps_longitude'], p['gps_latitude'])
                direction_point = find_target_point(poi, (180 + p['gps_azimuth']) % 360, 70000)

                circle = Point(poi).buffer(5 * (
                    100) / 111700)  # 5 meters, remove the multiplication by 1000,  its only to make it visible on the graph

                walk_ls = LineString([direction_point, poi])

                left_border = rotate(walk_ls, -135, origin=poi)
                right_border = rotate(walk_ls, 135, origin=poi)

                splitter = LineString([*left_border.coords, *right_border.coords[::-1]])
                if len(split(circle, splitter)) < 2:
                    continue
                minarea = sorted([s.area for s in split(circle, splitter)])[
                    -2]  # sometimes it spits to three, third one looks empty
                sector = [s for s in split(circle, splitter) if s.area == minarea][0]

                patch = PolygonPatch(Point(poi).buffer(0.0003), fc='#0f800f', ec='k', linewidth=0, alpha=0.5, zorder=-1)
                ax.add_patch(patch)

                patch = PolygonPatch(Point(direction_point).buffer(0.0003), fc='#0f800f', ec='k', linewidth=0,
                                     alpha=0.5, zorder=-1)
                ax.add_patch(patch)

                patch = PolygonPatch(walk_ls.buffer(0.0003), fc='#ff0000', ec='k', linewidth=0, alpha=0.5, zorder=-1)
                ax.add_patch(patch)

                patch = PolygonPatch(sector, fc='#ffff00', ec='k', linewidth=0, alpha=0.5, zorder=-1)
                ax.add_patch(patch)

                for line in [sector.intersection(e) for e in full_edges + partial_edges + [first_mid_edge] if
                             type(sector.intersection(e)) is LineString]:
                    patch = PolygonPatch(line.buffer(0.0008), fc='#000000', ec='k', linewidth=0, alpha=0.5, zorder=-1)
                    ax.add_patch(patch)

                    # this code will prevent from the graphs to be displayed.  enable it when you need the intersection results as linestring rather then graphs
                '''                 
                intersections = [sector.intersection(e) for e in full_edges+partial_edges+[first_mid_edge] if type(sector.intersection(e)) is LineString]
                print('sector.intersection ', intersections)

                if(len(intersections) > 0):
                    road_ls = intersections[0]
                    road_ls = getExtrapoledLine(road_ls.coords[0] , road_ls.coords[1], EXTRAPOL_RATIO = 5)
                    walk_ls = getExtrapoledLine(walk_ls.coords[0] , walk_ls.coords[1], EXTRAPOL_RATIO = 5)


                angle = getAngle(ls1.coords[0], tuple(ls1.intersection(ls2).coords)[0] ,ls2.coords[1])
                print('crossing angle: ', angle)

                '''

            plt.show(block=False)





        except:
            s.close()

