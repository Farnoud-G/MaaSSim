################################################################################
# Module: utils.py
# Reusable functions and methods used throughout the simulator
# Rafal Kucharski @ TU Delft
################################################################################

import pandas as pd
from dotmap import DotMap
import math
import random
import numpy as np
import os

from osmnx.distance import get_nearest_node
import osmnx as ox
import networkx as nx
import json
import h3

from .traveller import travellerEvent
from .driver import driverEvent


def rand_node(df):
    # returns a random node of a graph
    return df.loc[random.choice(df.index)].name


def generic_generator(generator, n):
    # to create multiple passengers/vehicles/etc
    return pd.concat([generator(i) for i in range(1, n + 1)], axis=1, keys=range(1, n + 1)).T


def empty_series(df, name=None):
    # returns empty Series from a given DataFrame, to be used for consistency of adding new rows to DataFrames
    if name is None:
        name = len(df.index) + 1
    return pd.Series(index=df.columns, name=name)


def initialize_df(df):
    # deletes rows in DataFrame and leaves the columns and index
    # returns empty DataFrame
    if type(df) == pd.core.frame.DataFrame:
        cols = df.columns
    else:
        cols = list(df.keys())
    df = pd.DataFrame(columns=cols)
    df.index.name = 'id'
    return df


def get_config(path, root_path=None, set_t0=False):
    """
    reads a .json file with MaaSSim configuration
    use as: params = get_config(config.json)
    :param path:
    :param root_path: adjust the paths with the main path while reading (used mainly for Travis tests on linux server)
    :param set_t0: adjust the t0 string to pandas datetime
    :return: params DotMap
    """
    with open(path) as json_file:
        data = json.load(json_file)
        params = DotMap(data)
    if root_path is not None:
        params.paths.G = os.path.join(root_path, params.paths.G)  # graphml of a current .city
        params.paths.skim = os.path.join(root_path, params.paths.skim)  # csv with a skim between the nodes of the .city
    if set_t0:
        params.t0 = pd.to_datetime(params.t0)
    return params


def save_config(_params, path=None):
    if path is None:
        path = os.path.join(_params.paths.params, _params.NAME + ".json")
    with open(path, "w") as write_file:
        json.dump(_params, write_file)


def set_t0(_params, now=True):
    if now:
        _params.t0 = pd.Timestamp.now().floor('1s')
    else:
        _params.t0 = pd.to_datetime(_params.t0)
    return _params


def networkstats(inData):
    """
    for a given network calculates it center of gravity (avg of node coordinates),
    gets nearest node and network radius (75th percentile of lengths from the center)
    returns a dictionary with center and radius
    """
    center_x = pd.DataFrame((inData.G.nodes(data='x')))[1].mean()
    center_y = pd.DataFrame((inData.G.nodes(data='y')))[1].mean()

    nearest = get_nearest_node(inData.G, (center_y, center_x))
    ret = DotMap({'center': nearest, 'radius': inData.skim[nearest].quantile(0.75)})
    return ret


def load_G(_inData, _params=None, stats=True, set_t=True):
    # loads graph and skim from a params paths
    if set_t:
        _params = set_t0(_params)
    _inData.G = ox.load_graphml(_params.paths.G)
    _inData.nodes = pd.DataFrame.from_dict(dict(_inData.G.nodes(data=True)), orient='index')
    skim = pd.read_csv(_params.paths.skim, index_col='Unnamed: 0')
    skim.columns = [int(c) for c in skim.columns]
    _inData.skim = skim
    if stats:
        _inData.stats = networkstats(_inData)  # calculate center of network, radius and central node
    return _inData

def dynamic_paricing(_inData, _params, level):
    lat = []
    lng = []
    for i in _inData.G.nodes:
        lat.append(_inData.G.nodes[i]['y'])
        lng.append(_inData.G.nodes[i]['x'])

    sdf = pd.DataFrame() # surge dataframe
    sdf['lat'] = lat
    sdf['lng'] = lng

    # hex_col = 'hex'+str(level)
    sdf['hex_address'] = sdf.apply(lambda x: h3.geo_to_h3(x.lat,x.lng,level),axis=1)
    sdf = sdf.groupby(['hex_address']).size().to_frame('cnt').reset_index()
    ds_dict = {100:'max_ds_100new.csv', 200:'max_ds_200new.csv', 300:'max_ds_300new.csv',
               400:'max_ds_400new.csv', 500:'max_ds_400new.csv', 600:'max_ds_400new.csv',
               700:'max_ds_400new.csv', 800:'max_ds_400new.csv', 900:'max_ds_400new.csv', 1000:'max_ds_500.csv'}
    max_ds_df = pd.read_csv(ds_dict[_params.nV])
    max_ds_df.set_index('hex_address',inplace=True)
    _inData.max_ds_df = max_ds_df
    _inData.sdf = sdf
    #------------------------------------------------------------------
    RD = [-0.43, -0.31, -0.47, -0.47, -0.33, -0.60, -0.80, -0.99, -0.99, -0.99, -0.99, -0.99, -0.34
      , -0.34, -0.34, -0.34, -0.34, -0.34, -0.34, -0.61, -0.61, -0.61, -0.61, -0.61, -0.61, -0.61,
      -0.61, -0.61, -0.61, -0.61, -0.61, -0.61, -0.61, -0.61, -0.61, -0.61, -0.61, -0.61, -0.61]
    SMP = list(np.arange(1.2, 5.1, 0.1))
    SMP = [round(item, 1) for item in SMP]
    pr = [1]
    smp = [1]

    for i in range(0,len(RD)):
        dp = (SMP[i]-smp[-1])/smp[-1]
        dd = dp*(RD[i])
        pr.append((1+dd)*pr[-1])
        smp.append(SMP[i])

    surge_dict = {}
    for i in range(0, len(smp)):
        surge_dict[smp[i]] = pr[i]
    surge_dict[1.1] = (surge_dict[1] + surge_dict[1.2])/2
    _inData.surge_dict = surge_dict
    
    return _inData


def download_G(inData, _params, make_skims=True):
    # uses osmnx to download the graph
    print('Downloading network for {} witn osmnx'.format(_params.city))
    inData.G = ox.graph_from_place(_params.city, network_type='drive')
    inData.nodes = pd.DataFrame.from_dict(dict(inData.G.nodes(data=True)), orient='index')
    if make_skims:
        inData.skim_generator = nx.all_pairs_dijkstra_path_length(inData.G,
                                                                  weight='length')
        inData.skim_dict = dict(inData.skim_generator)  # filled dict is more usable
        inData.skim = pd.DataFrame(inData.skim_dict).fillna(_params.dist_threshold).T.astype(
            int)  # and dataframe is more intuitive

    return inData


def save_G(inData, _params, path=None):
    # saves graph and skims to files
    ox.save_graphml(inData.G, filepath=_params.paths.G)
    inData.skim.to_csv(_params.paths.skim, chunksize=20000000)

    
def generate_platforms(_inData, _params, nPM): #f#
    plats = list()
    for i in range(nPM + 1):
        plats.append(empty_series(_inData.platforms, name=i))
    plats = pd.concat(plats, axis=1, keys=range(1, nPM + 1)).T
    plats.fare = _params.platforms.get('fare', 1)
    plats['base_fare'] = _params.platforms.get('base_fare',1)
    plats['min_fare'] = _params.platforms.get('min_fare',2)
    return plats    


def generate_vehicles(_inData, _params, nV):
    """
    generates single vehicle (database row with structure defined in DataStructures)
    index is consecutive number if dataframe
    position is random graph node
    status is IDLE
    """
    vehs = list()
    for i in range(nV + 1):
        vehs.append(empty_series(_inData.vehicles, name=i))

    vehs = pd.concat(vehs, axis=1, keys=range(1, nV + 1)).T
    vehs.event = driverEvent.STARTS_DAY
    vehs.platform = 1 #f#
    vehs.shift_start = 0
    vehs.shift_end = 60 * 60 * 24
    
    distances = _inData.skim[_inData.stats['center']].to_frame().dropna()  #f#
    distances.columns = ['distance'] #f#
    distances = distances[distances['distance'] < _params.dist_threshold] #f#
    distances['weight'] =distances['distance'].apply(lambda x:math.exp(_params.demand_structure.origins_dispertion * x)) #f#
    vehs.pos = list(distances.sample(_params.nV, random_state= _params.seed, weights='weight', replace=True).index) #f#
    # vehs.pos = vehs.pos.apply(lambda x: int(rand_node(_inData.nodes)))

    return vehs


def generate_demand(_inData, _params=None, avg_speed=False):
    # generates nP requests with a given temporal and spatial distribution of origins and destinations
    # returns _inData with dataframes requests and passengers populated.

    try:
        _params.t0 = pd.to_datetime(_params.t0)
    except:
        pass

    min_dist = _params.get('dist_threshold_min',0)
    
    df = pd.DataFrame(index=np.arange(0, _params.nP), columns=_inData.passengers.columns)
    df.status = travellerEvent.STARTS_DAY
    df.pos = _inData.nodes.sample(_params.nP, replace=True).index  # df.pos = df.apply(lambda x: rand_node(_inData.nodes), axis=1)
    _inData.passengers = df
    requests = pd.DataFrame(index=df.index, columns=_inData.requests.columns)
    distances = _inData.skim[_inData.stats['center']].to_frame().dropna()  # compute distances from center
    distances.columns = ['distance']
    distances = distances[distances['distance'] < _params.dist_threshold]
    # apply negative exponential distributions
    distances['p_origin'] = distances['distance'].apply(lambda x:
                                                        math.exp(
                                                            _params.demand_structure.origins_dispertion * x))

    distances['p_destination'] = distances['distance'].apply(
        lambda x: math.exp(_params.demand_structure.destinations_dispertion * x))
    if _params.demand_structure.temporal_distribution == 'uniform':
        treq = np.random.uniform(-_params.simTime * 60 * 60 / 2, _params.simTime * 60 * 60 / 2,
                                 _params.nP)  # apply uniform distribution on request times
    elif _params.demand_structure.temporal_distribution == 'normal':
        treq = np.random.normal(_params.simTime * 60 * 60 / 2,
                                _params.demand_structure.temporal_dispertion * _params.simTime * 60 * 60 / 2,
                                _params.nP)  # apply normal distribution on request times
    else:
        treq = None
    requests.treq = [_params.t0 + pd.Timedelta(int(_), 's') for _ in treq]
    requests.origin = list(
        distances.sample(_params.nP, weights='p_origin', replace=True).index)  # sample origin nodes from a distribution
    requests.destination = list(distances.sample(_params.nP, weights='p_destination',
                                                 replace=True).index)  # sample destination nodes from a distribution

    requests['dist'] = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)
    while len(requests[requests.dist >= _params.dist_threshold]) + len(requests[requests.dist < min_dist]) > 0:
        requests.origin = requests.apply(lambda request: (distances.sample(1, weights='p_origin').index[0]
                                                          if request.dist >= _params.dist_threshold or request.dist < min_dist else
                                                          request.origin),
                                         axis=1)
        requests.destination = requests.apply(lambda request: (distances.sample(1, weights='p_destination').index[0]
                                                               if request.dist >= _params.dist_threshold or request.dist < min_dist else
                                                               request.destination),
                                              axis=1)
        requests.dist = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)

    requests['ttrav'] = requests.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
    # requests.ttrav = pd.to_timedelta(requests.ttrav)
    if avg_speed:
        requests.ttrav = (pd.to_timedelta(requests.ttrav) / _params.speeds.ride).dt.floor('1s')
    requests.tarr = [request.treq + request.ttrav for _, request in requests.iterrows()]
    requests = requests.sort_values('treq')
    requests.index = df.index
    requests.pax_id = df.index
    requests.shareable = False

    _inData.requests = requests
    _inData.passengers.pos = _inData.requests.origin

    _inData.passengers.platforms = _inData.passengers.platforms.apply(lambda x: [1]) #f#

    return _inData

# def read_requests_csv(_inData,_params, path): #f#
#     from .data_structures import structures
#     # try:
#     #     _params.t0 = pd.to_datetime(_params.t0)
#     # except:
#     #     pass
    
#     _params.t0 = pd.to_datetime(_params.t0)
#     df = pd.read_csv(path)
#     df = df[df.dist>_params.dist_threshold_min]
#     df.treq = df.apply(lambda row: pd.Timestamp(row.treq), axis=1)
#     df = df.loc[pd.Timestamp(_params.start_time)<df.treq]
#     df = df.loc[(_params.start_time < df.treq) & (df.treq < _params.end_time)]
    
#     df = df.sample(_params.nP, random_state=_params.seed, replace= True)
#     _inData.passengers = pd.DataFrame(index=np.arange(0, _params.nP), columns=_inData.passengers.columns)
#     requests = pd.DataFrame(index=_inData.passengers.index, columns=_inData.requests.columns)
#     requests.pax_id = _inData.passengers.index
#     requests.ride_id = _inData.passengers.index
#     requests.origin = list(df.origin) #[1419265970, 46396602] #df.origin
#     requests.destination = list(df.destination) #[46423397, 1686005089]#df.destination
    
#     requests.treq = list(df.treq)
#     # requests.treq = requests.apply(lambda row: pd.Timestamp(row.treq), axis=1)


# #     if _params.demand_structure.temporal_distribution == 'uniform':
# #         treq = np.random.uniform(-_params.simTime * 60 * 60 / 2, _params.simTime * 60 * 60 / 2,
# #                                  _params.nP)  # apply uniform distribution on request times
        
# #     requests.treq = [_params.t0 + pd.Timedelta(int(_), 's') for _ in treq]
#     requests['dist'] = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)
#     requests['ttrav'] = requests.apply(lambda request: pd.Timedelta(request.dist/_params.speeds.ride, 's').floor('s'), axis=1)    
#     requests.tarr = [request.treq + request.ttrav for _, request in requests.iterrows()]
#     requests = requests.sort_values('treq')
#     requests.shareable = False
#     _inData.requests = requests
    
#     _inData.passengers['pax_id'] = _inData.passengers.index.copy()
#     _inData.passengers.pos = _inData.requests.origin.copy()
#     _inData.passengers.platforms = _inData.passengers.platforms.apply(lambda x: [1])
    
#     return _inData

def read_requests_csv(_inData,_params, path): #f# with synthetic treq distribution
    from .data_structures import structures
    # try:
    #     _params.t0 = pd.to_datetime(_params.t0)
    # except:
    #     pass
    
    _params.t0 = _params.start_time
    df = pd.read_csv(path)
    df = df[df.dist>_params.dist_threshold_min]
    df = df.sample(_params.nP, random_state=_params.seed, replace= True)
    _inData.passengers = pd.DataFrame(index=np.arange(0, _params.nP), columns=_inData.passengers.columns)
    requests = pd.DataFrame(index=_inData.passengers.index, columns=_inData.requests.columns)
    requests.pax_id = _inData.passengers.index
    requests.ride_id = _inData.passengers.index
    requests.origin = list(df.origin) #[1419265970, 46396602] #df.origin
    requests.destination = list(df.destination) #[46423397, 1686005089]#df.destination

    if _params.demand_structure.temporal_distribution == 'uniform':
        treq = np.random.uniform(0, _params.simTime * 60 * 60, _params.nP)  # apply uniform distribution on request times
        
    requests.treq = [_params.t0 + pd.Timedelta(int(_), 's') for _ in treq]
    requests['dist'] = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)
    requests['ttrav'] = requests.apply(lambda request: pd.Timedelta(request.dist/_params.speeds.ride, 's').floor('s'), axis=1)    
    requests.tarr = [request.treq + request.ttrav for _, request in requests.iterrows()]
    requests = requests.sort_values('treq')
    requests.shareable = False
    _inData.requests = requests
    
    _inData.passengers['pax_id'] = _inData.passengers.index.copy()
    _inData.passengers.pos = _inData.requests.origin.copy()
    _inData.passengers.platforms = _inData.passengers.platforms.apply(lambda x: [1])
    
    return _inData

# def read_requests_csv(inData, path):
#     from MaaSSim.data_structures import structures
#     inData.requests = pd.read_csv(path, index_col=1)
#     inData.requests.treq = pd.to_datetime(inData.requests.treq)
#     inData.requests['pax_id'] = inData.requests.index.copy()
#     inData.requests.ttrav = pd.to_timedelta(inData.requests.ttrav)
#     inData.passengers = pd.DataFrame(index=inData.requests.index, columns=structures.passengers.columns)
#     inData.passengers['pax_id'] = inData.passengers.index.copy()
#     inData.passengers.pos = inData.requests.origin.copy()
#     inData.passengers.platforms = inData.passengers.platforms.apply(lambda x: [0])
#     return inData

def read_vehicle_positions(inData, path):
    inData.vehicles = pd.read_csv(path, index_col=0)
    return inData


def make_config_paths(params, main=None, rel = False):
    # call it whenever you change a city name, or main path
    if main is None:
        if rel:
            main = '../..'
        else:
            main = os.path.join(os.getcwd(), "../..")
    if rel:
        params.paths.main = main
    else:
        params.paths.main = os.path.abspath(main)  # main repo folder


    params.paths.data = os.path.join(params.paths.main, 'data')  # data folder (not synced with repo)
    params.paths.params = os.path.join(params.paths.data, 'configs')
    params.paths.postcodes = os.path.join(params.paths.data, 'postcodes',
                                          "PC4_Nederland_2015.shp")  # PCA4 codes shapefile
    params.paths.albatross = os.path.join(params.paths.data, 'albatross')  # albatross data
    params.paths.sblt = os.path.join(params.paths.data, 'sblt')  # sblt results
    params.paths.G = os.path.join(params.paths.data, 'graphs',
                                  params.city.split(",")[0] + ".graphml")  # graphml of a current .city
    params.paths.skim = os.path.join(params.paths.main, 'data', 'graphs', params.city.split(",")[
        0] + ".csv")  # csv with a skim between the nodes of the .city
    params.paths.NYC = os.path.join(params.paths.main, 'data',
                                    'fhv_tripdata_2018-01.csv')  # csv with a skim between the nodes of the .city
    return params


def prep_supply_and_demand(_inData, params):
    _inData = generate_demand(_inData, params, avg_speed=True)
    _inData.vehicles = generate_vehicles(_inData, params.nV)
    _inData.vehicles.platform = _inData.vehicles.apply(lambda x: 0, axis=1)
    _inData.passengers.platforms = _inData.passengers.apply(lambda x: [0], axis=1)
    _inData.requests['platform'] = _inData.requests.apply(lambda row: _inData.passengers.loc[row.name].platforms[0],
                                                          axis=1)

    _inData.platforms = initialize_df(_inData.platforms)
    _inData.platforms.loc[0] = [1, 'Platform', 1]
    return _inData


#################
# PARALLEL RUNS #
#################


def test_space():
    # to see if code works
    test_space = DotMap()
    test_space.nP = [30, 40]  # number of requests per sim time
    test_space.nV = [10, 20]  # number of requests per sim time
    return test_space


def slice_space(s, replications=1, _print=False):
    # util to feed the np.optimize.brute with a search space
    def sliceme(l):
        return slice(0, len(l), 1)

    ret = list()
    sizes = list()
    size = 1
    for key in s.keys():
        ret += [sliceme(s[key])]
        sizes += [len(s[key])]
        size *= sizes[-1]
    if replications > 1:
        sizes += [replications]
        size *= sizes[-1]
        ret += [slice(0, replications, 1)]
    if _print:
        print('Search space to explore of dimensions {} and total size of {}'.format(sizes, size))
    return tuple(ret)


def collect_results(path):
    from pathlib import Path
    import zipfile
    collections = DotMap()
    first = True
    for archive in Path(path).rglob('*.zip'):
        zf = zipfile.ZipFile(archive)
        if first:
            for file in zf.namelist():
                collections[file[:-4]] = list()
            first = False
        for file in zf.namelist():
            df = pd.read_csv(zf.open(file))
            for key in archive.stem.split('-')[1:]:
                field, value = key.split('_')
                df[field] = value

            collections[file[:-4]].append(df)
    for key in collections.keys():
        collections[key] = pd.concat(collections[key])
    return collections
