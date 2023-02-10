################################################################################
# Module: runners.py
# Description: Wrappers to prepare and run simulations
# Rafal Kucharski @ TU Delft
################################################################################
import numpy as np

from MaaSSim.maassim import Simulator
from MaaSSim.shared import prep_shared_rides
from MaaSSim.utils import get_config, load_G, generate_demand, generate_vehicles, initialize_df, empty_series, \
    slice_space, read_requests_csv, read_vehicle_positions, generate_platforms
import pandas as pd
from scipy.optimize import brute
import logging
import re
import datetime

from dqn import DQNAgent


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

    stamp['dt'] = str(pd.Timestamp.now()).replace('-', '').replace('.', '').replace(' ', '')

    filename = ''
    for key, value in stamp.items():
        filename += '-{}_{}'.format(key, value)
    filename = re.sub('[^-a-zA-Z0-9_.() ]+', '', filename)
    _inData.passengers = initialize_df(_inData.passengers)
    _inData.requests = initialize_df(_inData.requests)
    _inData.vehicles = initialize_df(_inData.vehicles)

    sim = simulate(inData=_inData, params=_params, logger_level=logging.WARNING)
    sim.dump(dump_id=filename, path=_params.paths.get('dumps', None))  # store results

    print(filename, pd.Timestamp.now(), 'end')
    return 0


def simulate_parallel(config="../data/config/parallel.json", inData=None, params=None, search_space=None, **kwargs):
    if inData is None:  # othwerwise we use what is passed
        from MaaSSim.data_structures import structures
        inData = structures.copy()  # fresh data
    if params is None:
        params = get_config(config, root_path=kwargs.get('root_path'))  # load from .json file

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
          ranges=slice_space(search_space, replications=params.parallel.get("nReplications", 1)),
          args=(inData, params, search_space),
          full_output=True,
          finish=None,
          workers=params.parallel.get('nThread', 1))


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
        params = get_config(config, root_path=kwargs.get('root_path'))  # load from .json file
    if kwargs.get('make_main_path', False):
        from MaaSSim.utils import make_config_paths
        params = make_config_paths(params, main=kwargs.get('make_main_path', False), rel=True)

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
        print('Day = ', day)

        platfrom_profit = sim.res[day - 1].pax_kpi.plat_revenue['sum'] if len(sim.res) > 0 else 0  # - marketing cost
        # Strategy============================================================

        # 1- Trip fare adjustment -------------------------------------------
        # sim.platforms.fare = params.platforms.fare

        # 2- Commission rate adjustment -------------------------------------
        if 300 == day:
            # sim.platforms.fare[1] = 2 #euro/km
            sim.platforms.comm_rate[1] = 0.50
            print('Tragedy STARTS!')

        # 3- Discount adjustment -------------------------------------------
        # params.platforms.discount = 0.20 if 300<=day<350 else 0
        if 100 <= day < 200:
            params.platforms.discount = 0.40
        else:
            params.platforms.discount = 0

        # 4- Marketing adjustment ------------------------------------------
        sim.platforms.daily_marketing[1] = True if len(sim.res) in range(0, 100) else False

        # ====================================================================

        sim.make_and_run(run_id=day)  # prepare and SIM
        sim.output()  # calc results

        if sim.functions.f_stop_crit(sim=sim):
            break
    return sim


def simulate_rldqn_3act(config="data/config.json", inData=None, params=None, **kwargs):
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
        params = get_config(config, root_path=kwargs.get('root_path'))  # load from .json file
    if kwargs.get('make_main_path', False):
        from MaaSSim.utils import make_config_paths
        params = make_config_paths(params, main=kwargs.get('make_main_path', False), rel=True)

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

    state_size = 2
    action_size = 3

    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32
    state = np.asarray([params.get('nV', 1), params.get('nP', 1)])
    state = np.reshape(state, [1, state_size])

    sim.platforms.comm_rate[1] = 0.20 if 'initial_comm_rate' not in kwargs else float(kwargs['initial_comm_rate'])
    params.platforms.discount = 0.10
    sim.platforms.fare[1] = 2  # euro/km

    stp = 0.01 if 'stp' not in kwargs else float(kwargs['stp'])

    f = open(kwargs['file_res'], 'a')
    f.write('New simulation started at:    ' + str(datetime.datetime.now()) + '\n')
    f.write('act_size:    ' + str(action_size) + '\n')
    f.close()

    print('stp is: ', stp)
    print('type(stp): ', type(stp))
    print('initial comm rate: ', sim.platforms.comm_rate[1])
    print('type(comm_rate): ', type(sim.platforms.comm_rate[1]))

    revs = []

    for day in range(params.get('nD', 1)):  # run iterations

        print('---------------->', 'day:', day, 'of', params.get('nD', 1), '<--------------------')

        # number of active passengers out of All passengers
        nP = 0 if day == 0 else sim.res[day - 1].pax_exp.OUT.value_counts().get(False, 0)
        # number of active drivers out of All drivers
        nV = 0 if day == 0 else sim.res[day - 1].veh_exp.OUT.value_counts().get(False, 0)
        state = np.asarray([nP, nV])
        state = np.reshape(state, [1, state_size])

        # Strategy============================================================

        # 1- Trip fare adjustment -------------------------------------------
        # sim.platforms.fare = params.platforms.fare

        # 2- Commission rate adjustment -------------------------------------
        # Here model selects action base on current state
        action = agent.act(state)
        if action == 0:
            sim.platforms.comm_rate[1] = sim.platforms.comm_rate[1] + stp if sim.platforms.comm_rate[1] + stp < 1 else 1
        elif action == 1:
            sim.platforms.comm_rate[1] = sim.platforms.comm_rate[1] - stp if sim.platforms.comm_rate[1] - stp > 0 else 0
        elif action == 2:
            sim.platforms.comm_rate[1] = sim.platforms.comm_rate[1]
        sim.platforms.comm_rate[1] = round(sim.platforms.comm_rate[1], 2)

        # 3- Discount adjustment -------------------------------------------
        # params.platforms.discount = 0.20 if 300<=day<350 else 0
        if 100 <= day < 200:
            params.platforms.discount = 0.40
        else:
            params.platforms.discount = 0

        # 4- Marketing adjustment ------------------------------------------
        sim.platforms.daily_marketing[1] = True if len(sim.res) in range(0, 100) else False

        # ====================================================================

        sim.make_and_run(run_id=day)  # prepare and SIM
        sim.output()  # calc results

        # Calculating new state
        nP = 0 if day == 0 else sim.res[day].pax_exp.OUT.value_counts().get(False, 0)
        nV = 0 if day == 0 else sim.res[day].veh_exp.OUT.value_counts().get(False, 0)
        next_state = np.asarray([nP, nV])
        next_state = np.reshape(next_state, [1, state_size])

        reward = sim.res[day].pax_kpi.plat_revenue['sum'] if len(sim.res) > 0 else 0  # - marketing cost
        revs.append(reward)

        agent.memorize(state, action, reward, next_state, done)

        print('nP: ',state[0][0],'  nV: ', state[0][1],' Action: ', action,' Commrate: ', sim.platforms.comm_rate[1],' fare: ', sim.platforms.fare.iloc[0],
           ' discount: ', params.platforms.discount,' daily_marketing: ', sim.platforms.daily_marketing[1], ' reward: ',reward,' new nV: ', next_state[0][0], ' new nP: ',next_state[0][1])

        f = open(kwargs['file_res'], 'a')
        f.write(str(state[0][0]) + ',' + str(state[0][1]) + ',' + str(action) + ',' + str(sim.platforms.comm_rate[
            1]) + ',' + str(sim.platforms.fare.iloc[0]) + ',' + str(params.platforms.discount) + ',' + str(sim.platforms.daily_marketing[
                    1]) + ',' + str(reward) + ',' + str(next_state[0][0]) + ',' + str(next_state[0][1]) + '\n')
        f.close()

        if sim.functions.f_stop_crit(sim=sim):
            break
    return sim


def simulate_tune4(config="data/config.json", inData=None, params=None, **kwargs):
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
        params = get_config(config, root_path=kwargs.get('root_path'))  # load from .json file
    if kwargs.get('make_main_path', False):
        from MaaSSim.utils import make_config_paths
        params = make_config_paths(params, main=kwargs.get('make_main_path', False), rel=True)

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
    state_size = 2
    action_size = 3

    f = open(kwargs['file_res'], 'a')
    f.write('act_size:    ' + str(action_size) + '\n')
    f.close()

    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    state = np.asarray([params.get('nV', 1), params.get('nP', 1)])
    state = np.reshape(state, [1, state_size])

    sim.platforms.comm_rate[1] = 0.20 if 'initial_comm_rate' not in kwargs else float(kwargs['initial_comm_rate'])
    params.platforms.discount = 0.10
    sim.platforms.fare[1] = 2  # euro/km

    stp = 0.01 if 'stp' not in kwargs else float(kwargs['stp'])

    print('stp is: ', stp)
    print('type(stp): ', type(stp))
    print('initial comm rate: ', sim.platforms.comm_rate[1])
    print('type(comm_rate): ', type(sim.platforms.comm_rate[1]))

    revs = []
    for day in range(params.get('nD', 1)):  # run iterations

        # number of active passengers out of 2000 passengers
        nP = 0 if day == 0 else sim.res[day - 1].pax_exp.OUT.value_counts().get(False, 0)

        # number of active drivers out of 200 drivers
        nV = 0 if day == 0 else sim.res[day - 1].veh_exp.OUT.value_counts().get(False, 0)

        state = np.asarray([nP, nV])
        state = np.reshape(state, [1, state_size])

        print('-------->', 'iter:', day, 'of', params.get('nD', 1), '<--------')

        action = agent.act(state)

        print('action: ', str(action), 'comm_rate: ', str(sim.platforms.comm_rate[1]))

        if (action == 0):
            sim.platforms.comm_rate[1] = sim.platforms.comm_rate[1] + stp if sim.platforms.comm_rate[1] + stp < 1 else 1
        elif (action == 1):
            sim.platforms.comm_rate[1] = sim.platforms.comm_rate[1] - stp if sim.platforms.comm_rate[1] - stp > 0 else 0
        elif (action == 2):
            sim.platforms.comm_rate[1] = sim.platforms.comm_rate[1]

        sim.platforms.comm_rate[1] = round(sim.platforms.comm_rate[1], 2)

        print('new comm_rate: ', sim.platforms.comm_rate[1])
        print('current state: ', str(state))

        # Strategy============================================================
        if 300 <= day:
            sim.platforms.fare[1] = 2  # euro/km

        params.platforms.discount = 0.20 if 300 <= day < 350 else 0
        if 25 <= day < 100:
            params.platforms.discount = 0.40
        else:
            params.platforms.discount = 0

        if day == 100:
            sim.platforms.fare[1] = 2  # euro/km
        # ====================================================================

        sim.make_and_run(run_id=day)  # prepare and SIM
        sim.output()  # calc results

        next_state = state

        reward = sim.res[day].pax_kpi.plat_revenue['sum'] if len(sim.res) > 0 else 0
        revs.append(reward)

        agent.memorize(state, action, reward, next_state, done)
        state = next_state

        print(str(action) + ',' + str(sim.platforms.comm_rate[1]) + ',' + str(reward) + ' state:' + str(state))

        f = open(kwargs['file_res'], 'a')
        f.write(str(action) + ',' + str(sim.platforms.comm_rate[1]) + ',' + str(reward) + ' state:' + str(state) + '\n')
        f.close()

        if sim.functions.f_stop_crit(sim=sim):
            break

    return sim


if __name__ == "__main__":
    simulate(make_main_path='..')  # single run

    from MaaSSim.utils import test_space

    simulate_parallel(search_space=test_space())
