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
import random
import numpy as np
import copy


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
        if day>0:
            df = sim.res[day-1].pax_exp
            fd = sim.res[day-1].veh_exp
            np1 = len(df[df.platform_id==1])
            np2 = len(df[df.platform_id==2])
            vp1 = len(fd[fd.platform_id==1])
            vp2 = len(fd[fd.platform_id==2])
            print('np1 = ', np1, '  np2 = ', np2)
            print('vp1 = ', vp1, '  vp2 = ', vp2)
        print('Day = ', day)

        #Strategy============================================================        
        # print(sim.platforms)
        # 1- Trip fare adjustment -------------------------------------------
        # sim.platforms.fare = params.platforms.fare
        
        # 2- Commission rate adjustment -------------------------------------
        # if day==300:
        #     # sim.platforms.fare[1] = 2 #euro/km
        #     sim.platforms.comm_rate[1] = 0.50
        #     sim.platforms.comm_rate[2] = 0.50
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
        if 0<=day<100:
            sim.platforms.discount[1] = 0.40
            # sim.platforms.discount[2] = 0.40
            # sim.platforms.comm_rate[1] = 0.40
        else:
            sim.platforms.discount[1] = 0
            # sim.platforms.discount[2] = 0
            # sim.platforms.comm_rate[1] = 0
            
        if 25<=day<125:
            # sim.platforms.discount[1] = 0.40
            sim.platforms.discount[2] = 0.40
            # sim.platforms.comm_rate[2] = -0.40
        else:
            # sim.platforms.discount[1] = 0
            sim.platforms.discount[2] = 0
            # sim.platforms.comm_rate[2] = 0
            
    
        # if day>=200:
        #     if 200<=day<400:
        #         sim.platforms.discount[1] = 0.30
        #         sim.platforms.discount[2] = 0
        #         sim.platforms.comm_rate[1] = 0.30
        #         sim.platforms.comm_rate[2] = 0
        #     else:
        #         sim.platforms.discount[1] = 0
        #         sim.platforms.discount[2] = 0
        #         sim.platforms.comm_rate[1] = 0
        #         sim.platforms.comm_rate[2] = 0
        
        # 4- Marketing adjustment ------------------------------------------
        if 0<=day<100:
            sim.platforms.daily_marketing[1] = True
            # sim.platforms.daily_marketing[2] = True
        else:
            sim.platforms.daily_marketing[1] = False
            # sim.platforms.daily_marketing[2] = False
        
        if 25<=day<125:
            sim.platforms.daily_marketing[2] = True
        else:
            sim.platforms.daily_marketing[2] = False
        
        # price-cutting -----------------------------------------------------
#         if day>149:
            
#             df = sim.res[day-1].pax_exp
#             nP_p1 = len(df[df.platform_id==1])
#             nP_p2 = len(df[df.platform_id==2])

#             if nP_p2<nP_p1:
#                 sim.platforms.discount[2] = 0.40
#                 sim.platforms.discount[1] = 0
#             elif nP_p1<nP_p2:
#                 sim.platforms.discount[1] = 0.40
#                 sim.platforms.discount[2] = 0
#             else:
#                 sim.platforms.discount[1] = 0
#                 sim.platforms.discount[2] = 0
            # print('nP_p1 = ',nP_p1, '  nP_p2 = ',nP_p2 )
        #====================================================================
        
        sim.make_and_run(run_id=day)  # prepare and SIM
        sim.output()  # calc results

        if sim.functions.f_stop_crit(sim=sim):
            break
    return sim


def Markov(config="data/config.json", inData=None, params=None, **kwargs):

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
        inData.platforms = generate_platforms(inData, params, params.get('nPM', 1))

    inData = prep_shared_rides(inData, params.shareability)  # prepare schedules
    sim = Simulator(inData, params=params, **kwargs)  # initialize
    
    # Initialization ------------------------------------------------------
    interval = params.interval
    step = params.step
    threshold_u = params.threshold_u
    max_revenue = params.max_revenue # maximum revenue with the initial fare
    alpha = params.alpha
    min_fare = params.min_fare
    max_fare = params.max_fare
    
    sim.platforms.fare[1] = params.initial_fares[0]
    sim.platforms.fare[2] = params.initial_fares[1]
    # Initialization ------------------------------------------------------

    for day in range(params.get('nD', 1)):  # run iterations
        
        # Other levers -----------------------------------------------------
        sim.platforms.comm_rate[1] = 0.20
        sim.platforms.comm_rate[2] = 0.20

        sim.platforms.discount[1] = 0.0
        sim.platforms.discount[2] = 0.0
        
        # No marketing due to the created noise into competition
        sim.platforms.daily_marketing[1] = False
        sim.platforms.daily_marketing[2] = False

        #--------------------------------------------------------------------
        
        sim.make_and_run(run_id=day)  # prepare and SIM
        sim.output()  # calc results
                
        print('Day = ', day, ' -------------------------------------------')
        df = sim.res[day].pax_exp; fd = sim.res[day].veh_exp
        np1 = len(df[df.platform_id==1]); vp1 = len(fd[fd.platform_id==1])
        np2 = len(df[df.platform_id==2]); vp2 = len(fd[fd.platform_id==2])
        print('np1 = ', np1, '  np2 = ', np2); print('vp1 = ', vp1, '  vp2 = ', vp2)
        
        # Far adjustment for inter-platform competition -----------------------
        if (day+1)%interval==0 and day!=0:
                        
            for p in range(1, params.nPM+1):
                # utility calculation for the last move
                capital = sum(sim.res[i].plat_exp.remaining_capital[p] for i in range(day+1-interval,day+1)) / interval
                revenue = sum(sim.res[i].plat_exp.revenue[p] for i in range(day+1-interval,day+1)) / interval
                P_market_share = sum(sim.res[i].plat_exp.P_market_share[p] for i in range(day+1-interval,day+1)) / interval
                V_market_share = sum(sim.res[i].plat_exp.V_market_share[p] for i in range(day+1-interval,day+1)) / interval
                market_share = sum(sim.res[i].plat_exp.market_share[p] for i in range(day+1-interval,day+1)) / interval
                utility = alpha*(revenue/max_revenue) + (1-alpha)*market_share # revenue and market share are normalized to 0-1
                sim.trajectory['P{}'.format(p)].append((sim.platforms.fare[p], utility, revenue, market_share))
                
                # Reactive decision making: utility evaluation of last adjustment to determine the next step
                if len(sim.trajectory['P{}'.format(p)])<2:
                    sim.platforms.fare[p] += random.choice([-step, 0, step])
                    # sim.platforms.fare[p] += step
                else:
                    delta_f = sim.trajectory['P{}'.format(p)][-1][0] - sim.trajectory['P{}'.format(p)][-2][0]
                    delta_u = sim.trajectory['P{}'.format(p)][-1][1] - sim.trajectory['P{}'.format(p)][-2][1]
                    # delta_u of apponent might be consdired as well.
                    
                    if abs(delta_u) > threshold_u:
                        if delta_f != 0:
                            uf = delta_u/delta_f
                            if uf>0: # direct relationship between utility and fare
                                sim.platforms.fare[p] += step
                            else:    # inverse relationship
                                sim.platforms.fare[p] -= step
                        else:        # no relation due to delta_f = 0
                            if delta_u > 0:
                                pass
                            else:
                                sim.platforms.fare[p] += random.choice([-step, 0, step])
                    else:
                        if P_market_share > 0.1:
                            sim.platforms.fare[p] += random.choice([0, step])
                        else:
                            sim.platforms.fare[p] += random.choice([0, -step])
                
                # pass # do not change the fare
                
                sim.platforms.fare[p] = min(max(sim.platforms.fare[p], min_fare), max_fare)

                print('-------------------------------------------------------')
                print('P{} trajectory: '.format(p), sim.trajectory['P{}'.format(p)][-1])  
                print('-------------------------------------------------------')
                print('New P{} fare: '.format(p), sim.platforms.fare[p])
                print('-------------------------------------------------------')
            # Far adjustment for inter-platform competition -----------------------
            
        if sim.functions.f_stop_crit(sim=sim):
            break
    return sim
#------------------------------------------------------------------------------

def Try_and_Select(config="data/config.json", inData=None, params=None, **kwargs):

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
        inData.platforms = generate_platforms(inData, params, params.get('nPM', 1))

    inData = prep_shared_rides(inData, params.shareability)  # prepare schedules
    sim = Simulator(inData, params=params, **kwargs)  # initialize
    
    # Initialization ------------------------------------------------------
    threshold_u = params.threshold_u
    # max_revenue = params.max_revenue # maximum revenue with the initial fare
    # alpha = params.alpha
    turnover_interval = params.turnover_interval
    step_size = params.step_size
    initial_fares = params.initial_fares
    min_fare = params.min_fare
    max_fare = params.max_fare
    fares = np.arange(min_fare, max_fare+step_size, step_size)
    fares = np.round(fares, 1)
    max_i = len(fares)-1
    P1_moves = [(-float('inf'),initial_fares[0])] #(max utility, max utility fare)
    P2_moves = [(-float('inf'),initial_fares[1])]
    # fare_grid is consistent with the visuals where on each row P1 changes and P2 is fixed and vice versa in each column.
    fare_grid = np.array([[(x, y) for y in fares] for x in fares])
    fare_grid = np.round(fare_grid, 1)
    
    if params.random_ini_position == True:
        p1_i = np.random.randint(0, max_i+1)
        p2_i = np.random.randint(0, max_i+1)
    else: 
        p1_i = np.where(fares == initial_fares[0])[0][0]
        p2_i = np.where(fares == initial_fares[1])[0][0]
            
    turn_count = 0
    
    sim.platforms.fare[1] = fare_grid[p2_i, p1_i][1]
    sim.platforms.fare[2] = fare_grid[p2_i, p1_i][0]
    sim.fare_trajectory.append((sim.platforms.fare[2], sim.platforms.fare[1]))

    sim.platforms.comm_rate[1] = 0.20
    sim.platforms.comm_rate[2] = 0.20

    sim.platforms.discount[1] = 0.0
    sim.platforms.discount[2] = 0.0

    sim.platforms.daily_marketing[1] = True
    sim.platforms.daily_marketing[2] = True
    # Initialization ------------------------------------------------------

    for day in range(params.get('nD', 1)):
                
        if day%turnover_interval==0:
            
            p1_trun, p2_trun = (turn_count % 2 == 0, turn_count % 2 != 0)
            ava_moves = []
            res_list = []
            
            #============================ P1 ============================
            if p1_trun==True:
                print('============ P1 TURN ============')
                
            #---------------------------- P1 Left -----------------------
                if p1_i!=0:
                    for d in range(day, day+turnover_interval):
                        sim.run_id = d
                        sim.platforms.fare[1] = fare_grid[p2_i, p1_i-1][1] 
                        sim.make_and_run(run_id=d)
                        sim.output(run_id=d)
                        cell_print(sim, d, cell='Left', end=day+turnover_interval, fare=sim.platforms.fare[1])

                    res_copyL = copy.deepcopy(sim.res) 
                    L_point = u_calculator(sim, res=res_copyL, platform_id=1, day=day)
                    ava_moves.append((L_point,sim.platforms.fare[1]))
                    res_list.append(res_copyL)
                else:
                    ava_moves.append((u_calculator(sim, res=None, platform_id=1, day=day),-1))
                    res_list.append(None)
            #---------------------------- P1 Left -----------------------
            
            #---------------------------- P1 Middle ---------------------
                for d in range(day, day+turnover_interval):
                    sim.run_id = d
                    sim.platforms.fare[1] = fare_grid[p2_i, p1_i][1]                        
                    sim.make_and_run(run_id=d)
                    sim.output(run_id=d)
                    cell_print(sim, d, cell='Middle', end=day+turnover_interval, fare=sim.platforms.fare[1])

                res_copyM = copy.deepcopy(sim.res) 
                M_point = u_calculator(sim, res=res_copyM, platform_id=1, day=day)
                ava_moves.append((M_point,sim.platforms.fare[1]))
                res_list.append(res_copyM)
            #---------------------------- P1 Middle ---------------------
                
            #---------------------------- P1 Right ----------------------
                if p1_i!=max_i:
                    for d in range(day, day+turnover_interval):
                        sim.run_id = d
                        sim.platforms.fare[1] = fare_grid[p2_i, p1_i+1][1]                        
                        sim.make_and_run(run_id=d)
                        sim.output(run_id=d)
                        cell_print(sim, d, cell='Right', end=day+turnover_interval, fare=sim.platforms.fare[1])

                    res_copyR = copy.deepcopy(sim.res)   
                    R_point = u_calculator(sim, res=res_copyR, platform_id=1, day=day)
                    ava_moves.append((R_point, sim.platforms.fare[1]))
                    res_list.append(res_copyR)
                else:
                    ava_moves.append((u_calculator(sim, res=None, platform_id=1, day=day),-1))
                    res_list.append(None)
            #---------------------------- P1 Right ---------------------
                
                move_with_max_u = max(ava_moves, key=lambda x: x[0])
                max_utility_for_max_fare = max((utility for utility, fare in P1_moves if fare == move_with_max_u[1]), default=-float('inf'))
                last_move = P1_moves[-1]
                
                if move_with_max_u[0] - last_move[0] > threshold_u:
                    if move_with_max_u[0] - ava_moves[1][0] > threshold_u and move_with_max_u[0] > max_utility_for_max_fare: 
                        next_move = ava_moves.index(move_with_max_u)
                    else:
                        next_move = 1
                
                elif -threshold_u < ava_moves[1][0] - last_move[0] < threshold_u:
                    next_move = 1
                        
                else:
                    if len(P1_moves) > 1:
                        last_valid_tuple = next((item for item in reversed(P1_moves) if item[1] != P1_moves[-1][1]), None) 
                        matching_move = next((item for item in ava_moves if item[1] == last_valid_tuple[1]), None)
                        next_move = ava_moves.index(matching_move) 
                    else:
                        next_move = 1
                
                if d<200: next_move=1 ################################################
                if next_move ==0: next_move=1#########################################
                p1_i = max(0, min(max_i, p1_i + [-1, 0, 1][next_move]))
                sim.platforms.fare[1] = fare_grid[p2_i, p1_i][1]
                sim.res = copy.deepcopy(res_list[next_move])
                P1_moves.append((ava_moves[next_move][0],sim.platforms.fare[1]))
                
                print('available moves = ', ava_moves)
                print('P1 fare = {},   P2 fare = {}'.format(fare_grid[p2_i, p1_i][1], fare_grid[p2_i, p1_i][0]))
            #============================ P1 ============================
            
            #============================ P2 ============================
            elif p2_trun==True:
                print('============ P2 TURN ============')
                
            #---------------------------- P2 Lower ----------------------    
                if p2_i!=0:
                    for d in range(day, day+turnover_interval):
                        sim.run_id = d
                        sim.platforms.fare[2] = fare_grid[p2_i-1, p1_i][0]                        
                        sim.make_and_run(run_id=d)
                        sim.output(run_id=d)
                        cell_print(sim, d, cell='Lower', end=day+turnover_interval, fare=sim.platforms.fare[2])

                    res_copyL = copy.deepcopy(sim.res) 
                    L_point = u_calculator(sim, res=res_copyL, platform_id=2, day=day)
                    ava_moves.append((L_point,sim.platforms.fare[2]))
                    res_list.append(res_copyL)
                else:
                    ava_moves.append((u_calculator(sim, res=None, platform_id=2, day=day), -1))
                    res_list.append(None)
            #---------------------------- P2 Lower ----------------------
            
            #---------------------------- P2 Middle ---------------------
                for d in range(day, day+turnover_interval):
                    sim.run_id = d
                    sim.platforms.fare[2] = fare_grid[p2_i, p1_i][0]                        
                    sim.make_and_run(run_id=d)
                    sim.output(run_id=d)
                    cell_print(sim, d, cell='Middle', end=day+turnover_interval, fare=sim.platforms.fare[2])

                res_copyM = copy.deepcopy(sim.res) 
                M_point = u_calculator(sim, res=res_copyM, platform_id=2, day=day)
                ava_moves.append((M_point,sim.platforms.fare[2]))
                res_list.append(res_copyM)
            #---------------------------- P2 Middle ---------------------
            
            #---------------------------- P2 Upper ----------------------
                if p2_i!=max_i:
                    for d in range(day, day+turnover_interval):
                        sim.run_id = d
                        sim.platforms.fare[2] = fare_grid[p2_i+1, p1_i][0]                        
                        sim.make_and_run(run_id=d)
                        sim.output(run_id=d)
                        cell_print(sim, d, cell='Upper', end=day+turnover_interval, fare=sim.platforms.fare[2])

                    res_copyU = copy.deepcopy(sim.res)
                    # R_point = res_copyR[d].plat_exp.remaining_capital.loc[2]
                    U_point = u_calculator(sim, res=res_copyU, platform_id=2, day=day)
                    ava_moves.append((U_point, sim.platforms.fare[2]))
                    res_list.append(res_copyU)
                else:
                    ava_moves.append((u_calculator(sim, res=None, platform_id=2, day=day), -1))
                    res_list.append(None)
            #---------------------------- P2 Upper ----------------------
                
                move_with_max_u = max(ava_moves, key=lambda x: x[0])
                max_utility_for_max_fare = max((utility for utility, fare in P2_moves if fare == move_with_max_u[1]), default=-float('inf'))
                last_move = P2_moves[-1]
                
                if move_with_max_u[0] - last_move[0] > threshold_u:
                    if move_with_max_u[0] - ava_moves[1][0] > threshold_u and move_with_max_u[0] > max_utility_for_max_fare: 
                        next_move = ava_moves.index(move_with_max_u)
                    else:
                        next_move = 1
                
                elif -threshold_u < ava_moves[1][0] - last_move[0] < threshold_u:
                    next_move = 1
                        
                else:
                    if len(P2_moves) > 1:
                        last_valid_tuple = next((item for item in reversed(P2_moves) if item[1] != P2_moves[-1][1]), None) 
                        matching_move = next((item for item in ava_moves if item[1] == last_valid_tuple[1]), None)
                        next_move = ava_moves.index(matching_move)
                    else:
                        next_move = 1
                
                if d<200: next_move=1 ################################################
                if next_move ==0: next_move=1#########################################
                p2_i = max(0, min(max_i, p2_i + [-1, 0, 1][next_move]))
                sim.platforms.fare[2] = fare_grid[p2_i, p1_i][0] 
                sim.res = copy.deepcopy(res_list[next_move]) 
                P2_moves.append((ava_moves[next_move][0],sim.platforms.fare[2]))
                
                print('available moves = ', ava_moves)
                print('P1 fare = {},   P2 fare = {}'.format(fare_grid[p2_i, p1_i][1], fare_grid[p2_i, p1_i][0]))
            #============================ P2 ============================

            turn_count += 1
            sim.fare_trajectory.append((sim.platforms.fare[2], sim.platforms.fare[1]))
            day = day+turnover_interval
            
        if sim.functions.f_stop_crit(sim=sim):
            break
    return sim


def u_calculator(sim, res, platform_id, day):
    if res==None:
        sim.competition_trajectory['P{}'.format(platform_id)].append((None))
        return -float('inf')
    else:
        params = sim.params
        ti = params.turnover_interval
        mean_days = 10
        s = day+ti-mean_days
        # remaining_capital as the utility evaluation tool is not consistent with Try and Select model.
        # remaining_capital = sum(res[i].plat_exp.remaining_capital[platform_id] for i in range(s, day+ti)) / mean_days
        # capital_share = remaining_capital/params.initial_capital
        profit = sum(res[i].plat_exp.profit[platform_id] for i in range(s, day+ti)) / mean_days
        profit_share = profit/params.expense_per_day
        revenue = sum(res[i].plat_exp.revenue[platform_id] for i in range(s, day+ti)) / mean_days
        revenue_share = revenue/params.max_revenue
        P_market_share = sum(res[i].plat_exp.P_market_share[platform_id] for i in range(s, day+ti)) / mean_days
        V_market_share = sum(res[i].plat_exp.V_market_share[platform_id] for i in range(s, day+ti)) / mean_days
        market_share = sum(res[i].plat_exp.market_share[platform_id] for i in range(s, day+ti)) / mean_days
        # utility = params.alpha*revenue_share + (1-params.alpha)*market_share # revenue and market share are normalized to 0-
        utility = profit_share
        sim.competition_trajectory['P{}'.format(platform_id)].append((sim.platforms.fare[platform_id], utility, revenue_share, market_share))
        print('P{}: '.format(platform_id), sim.competition_trajectory['P{}'.format(platform_id)][-1])

        return utility


def cell_print(sim, d, cell, end, fare):
    print('Day = ', d, '--- on the {} cell with fare {}'.format(cell, fare))
    df = sim.res[d].pax_exp;fd = sim.res[d].veh_exp
    np1 = len(df[df.platform_id==1]);np2 = len(df[df.platform_id==2])
    vp1 = len(fd[fd.platform_id==1]);vp2 = len(fd[fd.platform_id==2])
    print('(np1,vp1) = ', (np1,vp1), '    (np2,vp2) = ', (np2,vp2))
    print('--------------------------------------------')
    if d == end-1:
        # print('Left cell: Fare = {} & P1 Remaining capital = {} & P2 Remaining capital = {}'.format(sim.platforms.fare[1], sim.res[d].platforms.P_remaining_capital.loc[1], sim.res[d].platforms.P_remaining_capital.loc[2]))
        print('--------------------------------------------')

        


def Try_and_Select_X(config="data/config.json", inData=None, params=None, **kwargs):

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
        inData.platforms = generate_platforms(inData, params, params.get('nPM', 1))

    inData = prep_shared_rides(inData, params.shareability)  # prepare schedules
    sim = Simulator(inData, params=params, **kwargs)  # initialize
    
    # Initialization ------------------------------------------------------
    threshold_u = params.threshold_u
    # max_revenue = params.max_revenue # maximum revenue with the initial fare
    # alpha = params.alpha
    turnover_interval = params.turnover_interval
    step_size = params.step_size
    initial_fares = params.initial_fares
    min_fare = params.min_fare
    max_fare = params.max_fare
    fares = np.arange(min_fare, max_fare+step_size, step_size)
    fares = np.round(fares, 1)
    max_i = len(fares)-1
    P1_moves = [(-float('inf'),initial_fares[0])] #(max utility, max utility fare)
    P2_moves = [(-float('inf'),initial_fares[1])]
    # fare_grid is consistent with the visuals where on each row P1 changes and P2 is fixed and vice versa in each column.
    fare_grid = np.array([[(x, y) for y in fares] for x in fares])
    fare_grid = np.round(fare_grid, 1)
    
    if params.random_ini_position == True:
        p1_i = np.random.randint(0, max_i+1)
        p2_i = np.random.randint(0, max_i+1)
    else: 
        p1_i = np.where(fares == initial_fares[0])[0][0]
        p2_i = np.where(fares == initial_fares[1])[0][0]
            
    turn_count = 0
    
    sim.platforms.fare[1] = fare_grid[p2_i, p1_i][1]
    sim.platforms.fare[2] = fare_grid[p2_i, p1_i][0]
    sim.fare_trajectory.append((sim.platforms.fare[2], sim.platforms.fare[1]))

    sim.platforms.comm_rate[1] = 0.20
    sim.platforms.comm_rate[2] = 0.20

    sim.platforms.discount[1] = 0.0
    sim.platforms.discount[2] = 0.0

    sim.platforms.daily_marketing[1] = True
    sim.platforms.daily_marketing[2] = True
    # Initialization ------------------------------------------------------

    for day in range(params.get('nD', 1)):
                
        if day%turnover_interval==0:
            
            p1_trun, p2_trun = (turn_count % 2 == 0, turn_count % 2 != 0)
            ava_moves = []
            res_list = []
            
            #============================ P1 ============================
            if p1_trun==True:
                print('============ P1 TURN ============')
                
            #---------------------------- P1 Left -----------------------
                if p1_i!=0:
                    for d in range(day, day+turnover_interval):
                        sim.run_id = d
                        sim.platforms.fare[1] = fare_grid[p2_i, p1_i-1][1] 
                        sim.make_and_run(run_id=d)
                        sim.output(run_id=d)
                        cell_print(sim, d, cell='Left', end=day+turnover_interval, fare=sim.platforms.fare[1])

                    res_copyL = copy.deepcopy(sim.res) 
                    L_point = u_calculator(sim, res=res_copyL, platform_id=1, day=day)
                    ava_moves.append((L_point,sim.platforms.fare[1]))
                    res_list.append(res_copyL)
                else:
                    ava_moves.append((u_calculator(sim, res=None, platform_id=1, day=day),-1))
                    res_list.append(None)
            #---------------------------- P1 Left -----------------------
            
            #---------------------------- P1 Middle ---------------------
                for d in range(day, day+turnover_interval):
                    sim.run_id = d
                    sim.platforms.fare[1] = fare_grid[p2_i, p1_i][1]                        
                    sim.make_and_run(run_id=d)
                    sim.output(run_id=d)
                    cell_print(sim, d, cell='Middle', end=day+turnover_interval, fare=sim.platforms.fare[1])

                res_copyM = copy.deepcopy(sim.res) 
                M_point = u_calculator(sim, res=res_copyM, platform_id=1, day=day)
                ava_moves.append((M_point,sim.platforms.fare[1]))
                res_list.append(res_copyM)
            #---------------------------- P1 Middle ---------------------
                
            #---------------------------- P1 Right ----------------------
                if p1_i!=max_i:
                    for d in range(day, day+turnover_interval):
                        sim.run_id = d
                        sim.platforms.fare[1] = fare_grid[p2_i, p1_i+1][1]                        
                        sim.make_and_run(run_id=d)
                        sim.output(run_id=d)
                        cell_print(sim, d, cell='Right', end=day+turnover_interval, fare=sim.platforms.fare[1])

                    res_copyR = copy.deepcopy(sim.res)   
                    R_point = u_calculator(sim, res=res_copyR, platform_id=1, day=day)
                    ava_moves.append((R_point, sim.platforms.fare[1]))
                    res_list.append(res_copyR)
                else:
                    ava_moves.append((u_calculator(sim, res=None, platform_id=1, day=day),-1))
                    res_list.append(None)
            #---------------------------- P1 Right ---------------------
                
                move_with_max_u = max(ava_moves, key=lambda x: x[0])
                max_utility_for_max_fare = max((utility for utility, fare in P1_moves if fare == move_with_max_u[1]), default=-float('inf'))
                last_move = P1_moves[-1]
                
                if move_with_max_u[0] - last_move[0] > threshold_u:
                    if move_with_max_u[0] - ava_moves[1][0] > threshold_u and move_with_max_u[0] > max_utility_for_max_fare: 
                        next_move = ava_moves.index(move_with_max_u)
                    else:
                        next_move = 1
                
                elif -threshold_u < ava_moves[1][0] - last_move[0] < threshold_u:
                    next_move = 1
                        
                else:
                    if len(P1_moves) > 1:
                        last_valid_tuple = next((item for item in reversed(P1_moves) if item[1] != P1_moves[-1][1]), None) 
                        matching_move = next((item for item in ava_moves if item[1] == last_valid_tuple[1]), None)
                        next_move = ava_moves.index(matching_move) 
                    else:
                        next_move = 1
                
                
                p1_i = max(0, min(max_i, p1_i + [-1, 0, 1][next_move]))
                sim.platforms.fare[1] = fare_grid[p2_i, p1_i][1]
                sim.res = copy.deepcopy(res_list[next_move])
                P1_moves.append((ava_moves[next_move][0],sim.platforms.fare[1]))
                
                print('available moves = ', ava_moves)
                print('P1 fare = {},   P2 fare = {}'.format(fare_grid[p2_i, p1_i][1], fare_grid[p2_i, p1_i][0]))
            #============================ P1 ============================
            
            #============================ P2 ============================
            elif p2_trun==True:
                print('============ P2 TURN ============')
                
            #---------------------------- P2 Lower ----------------------    
                if p2_i!=0:
                    for d in range(day, day+turnover_interval):
                        sim.run_id = d
                        sim.platforms.fare[2] = fare_grid[p2_i-1, p1_i][0]                        
                        sim.make_and_run(run_id=d)
                        sim.output(run_id=d)
                        cell_print(sim, d, cell='Lower', end=day+turnover_interval, fare=sim.platforms.fare[2])

                    res_copyL = copy.deepcopy(sim.res) 
                    L_point = u_calculator(sim, res=res_copyL, platform_id=2, day=day)
                    ava_moves.append((L_point,sim.platforms.fare[2]))
                    res_list.append(res_copyL)
                else:
                    ava_moves.append((u_calculator(sim, res=None, platform_id=2, day=day), -1))
                    res_list.append(None)
            #---------------------------- P2 Lower ----------------------
            
            #---------------------------- P2 Middle ---------------------
                for d in range(day, day+turnover_interval):
                    sim.run_id = d
                    sim.platforms.fare[2] = fare_grid[p2_i, p1_i][0]                        
                    sim.make_and_run(run_id=d)
                    sim.output(run_id=d)
                    cell_print(sim, d, cell='Middle', end=day+turnover_interval, fare=sim.platforms.fare[2])

                res_copyM = copy.deepcopy(sim.res) 
                M_point = u_calculator(sim, res=res_copyM, platform_id=2, day=day)
                ava_moves.append((M_point,sim.platforms.fare[2]))
                res_list.append(res_copyM)
            #---------------------------- P2 Middle ---------------------
            
            #---------------------------- P2 Upper ----------------------
                if p2_i!=max_i:
                    for d in range(day, day+turnover_interval):
                        sim.run_id = d
                        sim.platforms.fare[2] = fare_grid[p2_i+1, p1_i][0]                        
                        sim.make_and_run(run_id=d)
                        sim.output(run_id=d)
                        cell_print(sim, d, cell='Upper', end=day+turnover_interval, fare=sim.platforms.fare[2])

                    res_copyU = copy.deepcopy(sim.res)
                    # R_point = res_copyR[d].plat_exp.remaining_capital.loc[2]
                    U_point = u_calculator(sim, res=res_copyU, platform_id=2, day=day)
                    ava_moves.append((U_point, sim.platforms.fare[2]))
                    res_list.append(res_copyU)
                else:
                    ava_moves.append((u_calculator(sim, res=None, platform_id=2, day=day), -1))
                    res_list.append(None)
            #---------------------------- P2 Upper ----------------------
                
                move_with_max_u = max(ava_moves, key=lambda x: x[0])
                max_utility_for_max_fare = max((utility for utility, fare in P2_moves if fare == move_with_max_u[1]), default=-float('inf'))
                last_move = P2_moves[-1]
                
                if move_with_max_u[0] - last_move[0] > threshold_u:
                    if move_with_max_u[0] - ava_moves[1][0] > threshold_u and move_with_max_u[0] > max_utility_for_max_fare: 
                        next_move = ava_moves.index(move_with_max_u)
                    else:
                        next_move = 1
                
                elif -threshold_u < ava_moves[1][0] - last_move[0] < threshold_u:
                    next_move = 1
                        
                else:
                    if len(P2_moves) > 1:
                        last_valid_tuple = next((item for item in reversed(P2_moves) if item[1] != P2_moves[-1][1]), None) 
                        matching_move = next((item for item in ava_moves if item[1] == last_valid_tuple[1]), None)
                        next_move = ava_moves.index(matching_move)
                    else:
                        next_move = 1
                                
                p2_i = max(0, min(max_i, p2_i + [-1, 0, 1][next_move]))
                sim.platforms.fare[2] = fare_grid[p2_i, p1_i][0] 
                sim.res = copy.deepcopy(res_list[next_move]) 
                P2_moves.append((ava_moves[next_move][0],sim.platforms.fare[2]))
                
                print('available moves = ', ava_moves)
                print('P1 fare = {},   P2 fare = {}'.format(fare_grid[p2_i, p1_i][1], fare_grid[p2_i, p1_i][0]))
            #============================ P2 ============================

            turn_count += 1
            sim.fare_trajectory.append((sim.platforms.fare[2], sim.platforms.fare[1]))
            day = day+turnover_interval
            
        if sim.functions.f_stop_crit(sim=sim):
            break
    return sim       
        
        
        
        
        
        
        
                

if __name__ == "__main__":
    simulate(make_main_path='..')  # single run

    from MaaSSim.utils import test_space

    simulate_parallel(search_space = test_space())

