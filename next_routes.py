import sys
import connect_to_db as mongoConnection
import pandas as pd

import hostnameManager
import datetime
import getopt
import traceback
import timerange_for_user
import read_from_mongo
import Graph_from_dataframe
import osmnx as ox

from descartes.patch import PolygonPatch
import shapely.geometry as ge
import geopy.distance
import networkx as nx
from shapely.geometry import LineString, Point
import random

from shapely.affinity import rotate
from shapely.geometry import LineString, Point
from shapely.ops import split
import itertools

import requests


def snap_locations_to_road(path_df):
    path_df['point_location'] = tuple(zip(path_df['gps_latitude'], path_df['gps_longitude']))
    strlocs = str(list(path_df['point_location']))[2:-2].replace('), (', '|').replace(', ', ',')

    api_key = 'AIzaSyBBYVg8i8oh8syurQ1K3jxUgkTMd3sL8FI'

    response = requests.get(
        'https://roads.googleapis.com/v1/snapToRoads?path=' + strlocs + '&interpolate=false&key=' + api_key)
    response = response.json()
    if 'error' in response.keys():
        print (response['error']['code'])

    snapped = [(d['location']['latitude'], d['location']['longitude']) for d in response['snappedPoints']]

    path_df['point_location'] = snapped

    path_df['gps_latitude'] = path_df['point_location'].apply(lambda x: x[0])
    path_df['gps_longitude'] = path_df['point_location'].apply(lambda x: x[1])


def drop_dups(l):
    return [key for key, grp in itertools.groupby(l)]

def node_to_point(n):
    dic=G.nodes[n]
    return (dic['x'],dic['y'])

def same_edge(e1,e2):
    if {e1[1],e1[2]}=={e2[1],e2[2]}:
        return True
    return False

def is_reachable(n1,n2):
    try:
        nx.shortest_path_length(G, n1 ,n2)
        return True
    except:
        return False

    return False, (None, None)

def shortest_path_length_by_points(G, n1, n2):
    return geopy.distance.distance(node_to_point(n1) , node_to_point(n2)).m


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

def shortest_path_length_extended(G, p1, p2):    #not only for nodes,  p1 and p2 assumed to be on edges
    e1 = ox.utils.get_nearest_edge(G, p1)
    e2 = ox.utils.get_nearest_edge(G, p2)
    if same_edge(e1,e2):
        return geopy.distance.distance(p1, p2).m
    op1 = geopy.distance.distance(p1, e1[1]).m + nx.shortest_path_length(G, e1[1], e2[1]) + geopy.distance.distance(e2[1], p2).m
    op2 = geopy.distance.distance(p1, e1[2]).m + nx.shortest_path_length(G, e1[2], e2[1]) + geopy.distance.distance(e2[1], p2).m
    op2 = geopy.distance.distance(p1, e1[1]).m + nx.shortest_path_length(G, e1[1], e2[2]) + geopy.distance.distance(e2[2], p2).m
    op3 = geopy.distance.distance(p1, e1[2]).m + nx.shortest_path_length(G, e1[2], e2[2]) + geopy.distance.distance(e2[2], p2).m
    return min([op1, op2, op3, op4])



def find_direction(start, path_df):
    prev = start - 1
    print(path_df.shape)
    start_row = path_df.iloc[start]
    start_point = (start_row['gps_longitude'], start_row['gps_latitude'])
    nearest_edge_start = ox.utils.get_nearest_edge(G, (start_point[1], start_point[0]))
    print(start_point, nearest_edge_start)

    prev_row = path_df.iloc[prev]
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

    while len(nns) > 0:
        #        print(nns)
        nn = nns[0]
        nns = nns[1:]
        #        print(nns)
        print('former_node', former_node)
        print('nn0', nn)
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
        #        print(nearest_edges)
        print('nearest_edges', len(nearest_edges))
        #        print(nearest_edges)
        tree_edges += nearest_edges
        #        paint_edges(tree_edges)
        nnids = list(set([n for n in sum([[e[1], e[2]] for e in nearest_edges], []) if n != nn[0]]))
        nnids = [n for n in nnids if is_reachable(nn[0], n)]
        print('nnids', nnids)
        #        print('dist',dist,[ [n,shortest_path_length_by_points(G, nn[0], n )+nn[1]]  for n in nnids if is_reachable(nn[0],n)]  )
        leaves += [[n, shortest_path_length_by_points(G, nn[0], n) + nn[1]] for n in nnids if
                   is_reachable(nn[0], n) and shortest_path_length_by_points(G, nn[0], n) + nn[1] > dist]
        print('leaves ', leaves)
        nns_ = [[n, shortest_path_length_by_points(G, nn[0], n) + nn[1]] for n in nnids if
                n not in [l[0] for l in leaves]]  # problematic in case of rounds
        print('nns_ ', nns_)

        #        print(nns_)

        if len(nns_) > 0:
            find_next_tree(nns_, nn[0], tree_edges, dist)


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


def main():
    global leaves
    global graph_edges
    global G

    try:
        (opts, args) = getopt.getopt(sys.argv[1:], 'hs:u:d:', ['help', 'server_type=', 'userid=', 'date='])
    # raised whenever an option (which requires a value) is passed without a value.
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    user_id = hostname = date = None
    connectiontype = ''

    # check if any options were passed
    if len(opts) != 0:
        print(opts)
        for (o, a) in opts:
            if o in ('-h', '--help'):
                usage()
                sys.exit()
            elif o in ('-s', '--server_type'):
                connectiontype = a
                hostname = hostnameManager.getHostName(a)
            elif o in ('-u', '--userid'):
                user_id = a
            elif o in ('-d', '--date'):
                date = a

    if not date:
        date = datetime.datetime.today().strftime("%Y-%m-%d")

    print('\nHostname is: ' + hostname)
    if mongoConnection.connectToDB(connectiontype):
        print(hostname, date, user_id)
        start_date = (datetime.datetime.fromisoformat(date) - datetime.timedelta(days=1)).strftime(
            "%Y-%m-%d") + " 00:00:00+0000"
        end_date = date + " 00:00:00+0000"
        print(user_id, start_date, end_date)
        ids_sessions = timerange_for_user.get_id_list_from_user(user_id, start_date, end_date)
        if len(ids_sessions) == 0:
            print('no data for this input')
            return
        print(ids_sessions)

        print('retreiving data from server')

#        ids = ids_sessions[6]
        ids=['5db00a4345d6764d7480fd4b',
 '5db00a4445d6764d7480fd6d',
 '5db00a4545d6764d748104a9',
 '5db00a4645d6764d74810be8',
 '5db00a4745d6764d74811326',
 '5db00a4745d6764d74811a65',
 '5db00a4c45d6764d748121a7',
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
 '5db00bac45d6764d7482ce13']
        print('get_df_for_ids')
        print('get_df_for_ids')

        df_AS = read_from_mongo.get_df_for_ids(ids)
        df_AS = df_AS.iloc[22400:24000]
        df_AS = df_AS.sort_values('timestamps_value').reset_index(drop=True)
        df_AS.gps_speed.plot()

        print('dataframe is ready')

        G = Graph_from_dataframe.get_distance_graph(df_AS, 300)

        G_projected = ox.project_graph(G)
        #    ox.plot_graph(G_projected)
        print('map is ready')

        unique_locs_df = df_AS[['gps_longitude', 'gps_latitude']].drop_duplicates(
            ['gps_longitude', 'gps_latitude']).reset_index()

        walk_pois = [ge.Point(row[1]['gps_longitude'], row[1]['gps_latitude']) for row in unique_locs_df.iterrows()]

        walk_pois_df = df_AS[['gps_longitude', 'gps_latitude', 'gps_speed', 'timestamps_value']].drop_duplicates(
            ['gps_longitude', 'gps_latitude'])

        walk_pois_df['timestamps_value'] = pd.to_datetime(walk_pois_df['timestamps_value'])

        # Create the plot fig and ax objects but prevent Matplotlib

        # from plotting and closing out the plot operation

        fig, ax = ox.plot_graph(G, fig_height=10,
                                show=False, close=False,
                                edge_color='#777777')

        for point in walk_pois:
            patch = PolygonPatch(point.buffer(0.00020), fc='#ff0000', ec='k', linewidth=0, alpha=0.5, zorder=-1)
            ax.add_patch(patch)
        ox.plot_graph(G)

        gdf = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)

        graph_edges = gdf[["geometry", "u", "v"]].values.tolist()

        start = 1
        prev = 0

        path_df = walk_pois_df.iloc[0:start + 1]
        path_df['next_node'] = None

        speed_ms = 33
        dist = 15 * speed_ms

        no_sector=True
        while no_sector:
            poi1 = (random.uniform(df_AS['gps_longitude'].min(), df_AS['gps_longitude'].max()),
                random.uniform(df_AS['gps_latitude'].min(), df_AS['gps_latitude'].max()))

            poi2 = (random.uniform(df_AS['gps_longitude'].min(), df_AS['gps_longitude'].max()),
                random.uniform(df_AS['gps_latitude'].min(), df_AS['gps_latitude'].max()))

            circle = Point(poi2).buffer(0.005)

            ls = LineString([poi1, poi2])

            left_border = rotate(ls, -135, origin=poi2)
            right_border = rotate(ls, 135, origin=poi2)

            splitter = LineString([*left_border.coords, *right_border.coords[::-1]])
            if len(split(circle, splitter)) > 1:
                no_sector=False

        minarea = sorted([s.area for s in split(circle, splitter)])[-2]  # sometimes it spits to three, third one looks empty
        sector = [s for s in split(circle, splitter) if s.area == minarea][0]

        while start < len(walk_pois_df) - 1:
            nn = None
            while nn is None and start < len(walk_pois_df) - 1:
                start += 1
                prev += 1
                new_row = walk_pois_df.iloc[start]
                path_df = path_df.append({'gps_longitude': new_row['gps_longitude'],
                                          'gps_latitude': new_row['gps_latitude'],
                                          'gps_speed': new_row['gps_speed'],
                                          'timestamps_value': new_row['timestamps_value'],
                                          'next_node': None}, ignore_index=True)

                print('find direction ', start)
                former_node, next_node, start_point, prev_point = find_direction(start, path_df)
                nn = next_node

                path_df['next_node'].iloc[start] = next_node

                if discontinuity_detected(path_df):
                    path_df['next_node'].iloc[start] = None

                    print('Warning: discontinuity detected')
                    snap_locations_to_road(path_df)
                    #            print(path_df[['gps_latitude','gps_longitude','point_location']])

                    former_node, next_node, start_point, prev_point = find_direction(start,path_df)
                    nn = next_node

                    path_df['next_node'].iloc[start] = next_node

                    if discontinuity_detected(path_df):
                        print('Warning: discontinuity detected in snapped locatons')
                        nn = None

            #            nn=None
            if start == len(walk_pois_df) - 1:
                break
            print(former_node, nn, '************************')
            print(start)
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
                                             Point(G.nodes[path[-1]]['x'], G.nodes[path[-1]]['y'])]), path[-2],
                                 path[-1]] for path in
                                [nx.shortest_path(G, nn, l[0], weight='length') for l in
                                 [l for l in leaves if is_reachable(nn, l[0])]
                                 if all(
                                    elem in tree_edges_nodes for elem in nx.shortest_path(G, nn, l[0], weight='length'))
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
                    [print('le', shortest_path_length_by_points(G, nn, le[2]) + start_edge_len - dist) for le in
                     leaves_edges]
                    print('--', (shortest_path_length_by_points(G, nn, leaves_edges[0][
                        2]) + start_edge_len - dist) / shortest_path_length_by_points(G, leaves_edges[0][1],
                                                                                      leaves_edges[0][2]))

                    mid_edges = [[le[0].interpolate(le[0].length * (
                            1 - (shortest_path_length_by_points(G, nn, le[
                        2]) + start_edge_len - dist) / shortest_path_length_by_points(G, le[1], le[2]))), le]
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
                print('repaint', nn)
                fig, ax = ox.plot_graph(G, node_color='w', node_edgecolor='k', node_size=1,
                                        node_zorder=3, edge_color='r', edge_linewidth=1,
                                        fig_height=10, show=False, close=False)

                b = 0.0006

                patch = PolygonPatch(Point(poi1).buffer(0.0004), fc='#0f800f', ec='k', linewidth=0, alpha=0.5,
                                     zorder=-1)
                ax.add_patch(patch)

                patch = PolygonPatch(Point(poi2).buffer(0.0005), fc='#0f800f', ec='k', linewidth=0, alpha=0.5,
                                     zorder=-1)
                ax.add_patch(patch)

                patch = PolygonPatch(ls.buffer(0.0002), fc='#ff0000', ec='k', linewidth=0, alpha=0.5, zorder=-1)
                ax.add_patch(patch)

                patch = PolygonPatch(Point(prev_point).buffer(b), fc='#9900ff', ec='k', linewidth=0, alpha=0.5,
                                     zorder=-1)
                ax.add_patch(patch)

                patch = PolygonPatch(Point(start_point).buffer(b), fc='#ff00aa', ec='k', linewidth=0, alpha=0.5,
                                     zorder=-1)
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

                patch = PolygonPatch(sector, fc='#ffff00', ec='k', linewidth=0, alpha=0.5, zorder=-1)
                ax.add_patch(patch)

                for line in [sector.intersection(e) for e in full_edges + partial_edges + [first_mid_edge] if
                             type(sector.intersection(e)) is LineString]:
                    patch = PolygonPatch(line.buffer(0.0008), fc='#000000', ec='k', linewidth=0, alpha=0.5, zorder=-1)
                    ax.add_patch(patch)

                print('sector.intersection ',
                      [sector.intersection(e) for e in full_edges + partial_edges + [first_mid_edge] if
                       type(sector.intersection(e)) is LineString])

        print('End of script')


main()
