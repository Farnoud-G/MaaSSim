################################################################################
# Module: runners.py
# Description: Wrappers to prepare and run simulations
# Rafal Kucharski @ TU Delft
################################################################################


from MaaSSim.maassim import Simulator
from MaaSSim.shared import prep_shared_rides
from MaaSSim.utils import get_config, load_G, generate_demand, generate_vehicles, initialize_df, empty_series, \
    slice_space, read_requests_csv, read_vehicle_positions, generate_platforms
import pandas as pd
from scipy.optimize import brute
import logging
import re


def single_pararun(one_slice, *args):
    # function to be used with optimize brute
    inData, params, search_space = args  # read static input
    _inData = inData.copy()
    _params = params.copy()
    stamp = dict()
    # parameterize
    for i, key in enumerate(search_space.keys()):
        val = search_space[key][int(one_slice[int(i)])]
        stamp[key] = val
        _params[key] = val

    stamp['dt'] = str(pd.Timestamp.now()).replace('-','').replace('.','').replace(' ','')

    filename = ''
    for key, value in stamp.items():
        filename += '-{}_{}'.format(key, value)
    filename = re.sub('[^-a-zA-Z0-9_.() ]+', '', filename)
    _inData.passengers = initialize_df(_inData.passengers)
    _inData.requests = initialize_df(_inData.requests)
    _inData.vehicles = initialize_df(_inData.vehicles)

    sim = simulate(inData=_inData, params=_params, logger_level=logging.WARNING)
    sim.dump(dump_id=filename, path = _params.paths.get('dumps', None))  # store results

    print(filename, pd.Timestamp.now(), 'end')
    return 0


def simulate_parallel(config="../data/config/parallel.json", inData=None, params=None, search_space=None, **kwargs):
    if inData is None:  # othwerwise we use what is passed
        from MaaSSim.data_structures import structures
        inData = structures.copy()  # fresh data
    if params is None:
        params = get_config(config, root_path = kwargs.get('root_path'))  # load from .json file

    if len(inData.G) == 0:  # only if no graph in input
        inData = load_G(inData, params, stats=True)  # download graph for the 'params.city' and calc the skim matrices
    if len(inData.passengers) == 0:  # only if no passengers in input
        inData = generate_demand(inData, params, avg_speed=True)
    if len(inData.vehicles) == 0:  # only if no vehicles in input
        inData.vehicles = generate_vehicles(inData, params.nV)
    if len(inData.platforms) == 0:  # only if no platforms in input
        inData.platforms = initialize_df(inData.platforms)
        inData.platforms.loc[0] = empty_series(inData.platforms)
        inData.platforms.fare = [1]
        inData.vehicles.platform = 0
        inData.passengers.platforms = inData.passengers.apply(lambda x: [0], axis=1)


    inData = prep_shared_rides(inData, params.shareability)  # obligatory to prepare schedules


    brute(func=single_pararun,
          ranges=slice_space(search_space, replications=params.parallel.get("nReplications",1)),
          args=(inData, params, search_space),
          full_output=True,
          finish=None,
          workers=params.parallel.get('nThread',1))


def simulate(config="data/config.json", inData=None, params=None, **kwargs):
    """
    main runner and wrapper
    loads or uses json config to prepare the data for simulation, run it and process the results
    :param config: .json file path
    :param inData: optional input data
    :param params: loaded json file
    :param kwargs: optional arguments
    :return: simulation object with results
    """

    if inData is None:  # otherwise we use what is passed
        from MaaSSim.data_structures import structures
        inData = structures.copy()  # fresh data
    if params is None:
        params = get_config(config, root_path = kwargs.get('root_path'))  # load from .json file
    if kwargs.get('make_main_path',False):
        from MaaSSim.utils import make_config_paths
        params = make_config_paths(params, main = kwargs.get('make_main_path',False), rel = True)

    if params.paths.get('vehicles', False):
        inData = read_vehicle_positions(inData, path=params.paths.vehicles)

    if len(inData.G) == 0:  # only if no graph in input
        inData = load_G(inData, params, stats=True)  # download graph for the 'params.city' and calc the skim matrices
        
    if params.paths.get('requests', False):
        inData = read_requests_csv(inData, params, path=params.paths.requests)
        
    if len(inData.passengers) == 0:  # only if no passengers in input
        inData = generate_demand(inData, params, avg_speed=True)
    if len(inData.vehicles) == 0:  # only if no vehicles in input
        inData.vehicles = generate_vehicles(inData, params, params.nV)
    if len(inData.platforms) == 0:  # only if no platforms in input
        # inData.platforms = initialize_df(inData.platforms)
        # inData.platforms.loc[0] = empty_series(inData.platforms)
        # inData.platforms.fare = [1]
        inData.platforms = generate_platforms(inData, params, params.get('nPM', 1))

    inData = prep_shared_rides(inData, params.shareability)  # prepare schedules

    sim = Simulator(inData, params=params, **kwargs)  # initialize
    
    for day in range(params.get('nD', 1)):  # run iterations
        nP = sim.res[day-1].pax_exp.OUT.value_counts().get(False, 0) if day>0 else 0
        nV = sim.res[day-1].veh_exp.OUT.value_counts().get(False, 0) if day>0 else 0
        print('nP = ', nP, '   nV = ', nV)
        print('Day = ', day)
        
        # platfrom_profit = sim.res[day-1].pax_kpi.plat_revenue['sum'] if len(sim.res)>0 else 0 # - marketing cost 
        #Strategy============================================================        
        
        # 1- Trip fare adjustment -------------------------------------------
        # sim.platforms.fare = params.platforms.fare
        
        # 2- Commission rate adjustment -------------------------------------
        # if 300==day:
        #     # sim.platforms.fare[1] = 2 #euro/km
        #     sim.platforms.comm_rate[1] = 0.50
        #     print('Tragedy STARTS!')
            
        # if 150<=day<250:
        #     sim.platforms.comm_rate[1] = 0.20
        # elif 250<=day<350:
        #     sim.platforms.comm_rate[1] = 0.50
        # elif 350<=day:
        #     sim.platforms.comm_rate[1] = 0.20
        # else:
        #     sim.platforms.comm_rate[1] = 0.0

        # 3- Discount adjustment -------------------------------------------
        # params.platforms.discount = 0.20 if 300<=day<350 else 0
        # if 0<=day<200:
        #     params.platforms.discount = 0.40
        # else:
        #     params.platforms.discount = 0
            
        # if day==100:
        #     sim.platforms.comm_rate[1] = 0.15
        #     sim.platforms.fare[1] = 2 #euro/km     
        
        # 4- Marketing adjustment ------------------------------------------
        sim.platforms.daily_marketing[1] = True if len(sim.res) in range(0, 50) else False
        
        #====================================================================

        sim.make_and_run(run_id=day)  # prepare and SIM
        sim.output()  # calc results

        if sim.functions.f_stop_crit(sim=sim):
            break
    return sim


if __name__ == "__main__":
    simulate(make_main_path='..')  # single run

    from MaaSSim.utils import test_space

    simulate_parallel(search_space = test_space())

