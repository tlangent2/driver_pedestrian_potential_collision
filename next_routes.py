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


def node_to_point(n):
    dic = G.node[n]
    return (dic['x'], dic['y'])


def find_next_tree(nns, former_node, tree_edges, dist):
    global leaves

    while len(nns) > 0:
        #        print(nns)
        nn = nns[0]
        nns = nns[1:]
        #        print(nns)
        #        print('former_node', former_node)
        nearest_edges = [e for e in graph_edges if
                         e[2] != former_node and e[1] == nn[0] or e[1] != former_node and e[2] == nn[0]]
        #        print(nearest_edges)
        tree_edges += nearest_edges
        #        paint_edges(tree_edges)
        nnids = list(set([n for n in sum([[e[1], e[2]] for e in nearest_edges], []) if n != nn[0]]))
        leaves += [[n, nx.shortest_path_length(G, n, nn[0], weight='length') + nn[1]] for n in nnids if
                   nx.shortest_path_length(G, n, nn[0], weight='length') + nn[1] > dist]
        nns_ = [[n, nx.shortest_path_length(G, n, nn[0], weight='length') + nn[1]] for n in nnids if
                n not in [l[0] for l in leaves]]  # problematic in case of rounds

        #        print(nns_)

        if len(nns_) > 0:
            find_next_tree(nns_, nn[0], tree_edges, dist)

def get_edge_points(ls):
    return [ls.interpolate(ls.length*x) for x in [i/int(ls.length*20000) for i in range(int(ls.length*20000))]  ]

def same_edge(e1,e2):
    if {e1[1],e1[2]}=={e2[1],e2[2]}:
        return True
    return False


def find_direction(start):
    prev = start - 1
    start_row = walk_pois_df.iloc[start]
    start_point = (start_row['gps_longitude'], start_row['gps_latitude'])
    nearest_edge_start = ox.utils.get_nearest_edge(G, (start_point[1], start_point[0]))

    prev_row = walk_pois_df.iloc[prev]
    prev_point = (prev_row['gps_longitude'], prev_row['gps_latitude'])
    nearest_edge_prev = ox.utils.get_nearest_edge(G, (prev_point[1], prev_point[0]))

    if same_edge(nearest_edge_start, nearest_edge_prev):
        #        print('same edge')
        dist_fn_start = geopy.distance.distance(
            (nearest_edge_start[0].coords[0][0], nearest_edge_start[0].coords[0][1]), start_point).m

        dist_fn_prev = geopy.distance.distance((nearest_edge_start[0].coords[0][0], nearest_edge_start[0].coords[0][1]),
                                               prev_point).m

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


def main():
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

        ids_sessions = timerange_for_user.get_id_list_from_user(user_id, start_date, end_date)

        print(ids_sessions)

        print('retreiving data from server')

        #   ids = ids_sessions[3]
        ids = ['5d60d4b1d997ab0b6038c85a', '5d60d13fd997ab0b6038c7c8',
               '5d60d11cd997ab0b6038c6f0']
        print('get_df_for_ids')

        df_AS = read_from_mongo.get_df_for_ids(ids)
        df_AS = df_AS.sort_values('timestamps_value').reset_index(drop=True)
        df_AS.gps_speed.plot()
        print('dataframe is ready')

        G = Graph_from_dataframe.get_distance_graph(df_AS, 100)

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

        dist_df = df_AS[['gps_latitude', 'gps_longitude']].drop_duplicates()

        dist_df['gps_latitude_next'] = list(dist_df['gps_latitude'])[1:] + [0]

        dist_df['gps_longitude_next'] = list(dist_df['gps_longitude'])[1:] + [0]

        dist_df['dist'] = dist_df.apply(lambda row: geopy.distance.distance((row['gps_latitude'], row['gps_longitude']),
                                                                            (row['gps_latitude_next'],
                                                                             row['gps_longitude_next'])).m, axis=1)

        start = 1
        prev = 0

        speed_ms = 60
        dist = 6 * speed_ms

        while start < len(walk_pois_df) - 1:
            nn = None
            while nn is None:
                start += 1
                prev += 1
                former_node, next_node, start_point, prev_point = find_direction(start)
                nn = next_node
            print(start)
            print('start_point ', start_point)
            start_edge_len = geopy.distance.distance(start_point, node_to_point(nn)).m
            if start_edge_len > dist:
                only_edge = LineString([start_point, node_to_point(nn)])
                only_edge = LineString([start_point, only_edge.interpolate(
                    only_edge.length * dist / geopy.distance.distance(start_point, node_to_point(nn)).m)])
                print('start_edge_len', start_edge_len)
                edge_points_start = get_edge_points(only_edge)
                edge_points_full = []
                edge_points_partial = []
            else:
                nns = [[nn, start_edge_len]]
                tree_edges = []
                leaves = []
                #    starttimer = timer()
                #    print('leaves', leaves)

                find_next_tree(nns, former_node, tree_edges, dist)

                #    print('leaves', leaves)
                #    endtimer = timer()
                #    print(timedelta(seconds=endtimer-starttimer))

                tree_edges_nodes = [[e[1], e[2]] for e in tree_edges]
                tree_edges_nodes = sum(tree_edges_nodes, [])

                leaves_edges = [[LineString([Point(G.node[path[-2]]['x'], G.node[path[-2]]['y']),
                                             Point(G.node[path[-1]]['x'], G.node[path[-1]]['y'])]), path[-2], path[-1]]
                                for path in
                                [nx.shortest_path(G, nn, l[0], weight='length') for l in leaves
                                 if all(
                                    elem in tree_edges_nodes for elem in nx.shortest_path(G, nn, l[0], weight='length'))
                                 ] if len(path) > 1]

                # if nx.shortest_path(G, nn, l[0] , weight='length') in tree_edges_nodes -- cases when the shortest path is different will be cut!!!!
                # the edge cases that are not covered are when the driver destination is before the leaf node (bc shortest path are only between nodes)
                # or if the driver decided not to take the shortest path
                print('----------------------')
                print(leaves_edges[0][0].length)
                print((nx.shortest_path_length(G, nn, leaves_edges[0][2],
                                               weight='length') + start_edge_len - dist) / nx.shortest_path_length(G,
                                                                                                                   leaves_edges[
                                                                                                                       0][
                                                                                                                       1],
                                                                                                                   leaves_edges[
                                                                                                                       0][
                                                                                                                       2],
                                                                                                                   weight='length'))

                mid_edges = [[le[0].interpolate(le[0].length * (
                        1 - (nx.shortest_path_length(G, nn, le[2],
                                                     weight='length') + start_edge_len - dist) / nx.shortest_path_length(
                    G, le[1], le[2], weight='length'))), le]
                             for le in leaves_edges]

                print('mid_edges ', mid_edges[0][0].x)

                full_edges = [t for t in tree_edges if
                              t[2] not in [l[0] for l in leaves] and t[1] not in [l[0] for l in leaves]]
                #       print('full_edges',  len(full_edges))

                partial_edges = [LineString([node_to_point(m[1][1]), m[0]]) for m in mid_edges]
                #        print('partial_edges', len(partial_edges))

                edge_points_full = sum([get_edge_points(f) for f in [f[0] for f in full_edges]], [])
                edge_points_partial = sum([get_edge_points(f) for f in partial_edges], [])
                edge_points_start = get_edge_points(LineString([start_point, node_to_point(nn)]))
                print(len(edge_points_full), len(edge_points_partial), len(edge_points_start))

            #    print('edge_points', len(edge_points))
            fig, ax = ox.plot_graph(G, node_color='w', node_edgecolor='k', node_size=1,
                                    node_zorder=3, edge_color='r', edge_linewidth=1,
                                    fig_height=10, show=False, close=False)

            for point in [Point(prev_point)]:
                patch = PolygonPatch(point.buffer(0.0003), fc='#9900ff', ec='k', linewidth=0, alpha=0.5, zorder=-1)
                ax.add_patch(patch)

            for point in [Point(start_point)]:
                patch = PolygonPatch(point.buffer(0.0003), fc='#ff00aa', ec='k', linewidth=0, alpha=0.5, zorder=-1)
                ax.add_patch(patch)

            for point in [Point(G.node[next_node]['x'], G.node[next_node]['y'])]:
                patch = PolygonPatch(point.buffer(0.0001), fc='#0000ff', ec='k', linewidth=0, alpha=0.5, zorder=-1)
                ax.add_patch(patch)

            for point in [Point(G.node[l[0]]['x'], G.node[l[0]]['y']) for l in leaves]:
                patch = PolygonPatch(point.buffer(0.0004), fc='#00ffff', ec='k', linewidth=0, alpha=0.5, zorder=-1)
                ax.add_patch(patch)

            for point in edge_points_full:
                patch = PolygonPatch(point.buffer(0.0002), fc='#0f800f', ec='k', linewidth=0, alpha=0.5, zorder=-1)
                ax.add_patch(patch)

            for point in edge_points_partial:
                patch = PolygonPatch(point.buffer(0.0002), fc='#0f899f', ec='k', linewidth=0, alpha=0.5, zorder=-1)
                ax.add_patch(patch)

            for point in edge_points_start:
                patch = PolygonPatch(point.buffer(0.0002), fc='#aaaaaa', ec='k', linewidth=0, alpha=0.5, zorder=-1)
                ax.add_patch(patch)

        mongoConnection.dispose()

        print('End of script')


main()
