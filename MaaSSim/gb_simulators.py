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
        inData.platforms = generate_platforms(inData, params, params.get('nPM', 1))

    inData = prep_shared_rides(inData, params.shareability)  # prepare schedules

    sim = Simulator(inData, params=params, **kwargs)  # initialize
    
    # Competition ===========================================================
    point_grid = np.load('point_grid.npy', allow_pickle=True)
    
    fares = [1.0, 1.2, 1.4, 1.6]
    fare_grid = np.array([[(x, y) for y in fares] for x in fares])
    
    if params.random_ini_position == True:
        p1_i = np.random.randint(0, len(fares))
        p2_i = np.random.randint(0, len(fares))
        
    else: 
        p1_i = fares.index(1.6)
        p2_i = fares.index(1.6)
    
    turn_count = 0
    sim.competition_trajectory = [(fare_grid[p2_i, p1_i][1], fare_grid[p2_i, p1_i][0])]
    
    for day in range(params.get('nD', 1)):  # run iterations

        # Trip fare adjustment --------------------------------------------
        sim.platforms.fare[1] = fare_grid[p2_i, p1_i][1]
        sim.platforms.fare[2] = fare_grid[p2_i, p1_i][0]
        
        # Other levers -----------------------------------------------------
        sim.platforms.comm_rate[1] = 0.20
        sim.platforms.comm_rate[2] = 0.20

        sim.platforms.discount[1] = 0.0
        sim.platforms.discount[2] = 0.0
            
        sim.platforms.daily_marketing[1] = True
        sim.platforms.daily_marketing[2] = True
        
        #-------------------------------------------------------------------
        
        sim.make_and_run(run_id=day)  # prepare and SIM
        sim.output()  # calc results
        
        #-------------------------------------------------------------------
        if day%2==0 and day!=0:
            if turn_count%2==0:
                p1_trun = True
                p2_trun = False
            else: 
                p1_trun = False
                p2_trun = True
             
            current_points = point_grid[p2_i, p1_i]
            ava_points = []
            
            if p1_trun==True:
                if p1_i!=0:
                    adj1_point = point_grid[p2_i, p1_i-1][1]
                    ava_points.append(adj1_point)
                else: 
                    ava_points.append(-float('inf'))
                ava_points.append(current_points[1])
                if p1_i!=3:
                    adj2_point = point_grid[p2_i, p1_i+1][1]
                    ava_points.append(adj2_point)
                else: 
                    ava_points.append(-float('inf'))
                    
                next_move = ava_points.index(max(ava_points))
                if next_move==0:
                    p1_i = p1_i-1
                elif next_move==1:
                    p1_i = p1_i
                elif next_move==2:
                    p1_i = p1_i+1
                    
            elif p2_trun==True:
                if p2_i!=0:
                    adj1_point = point_grid[p2_i-1, p1_i][0]
                    ava_points.append(adj1_point)
                else: 
                    ava_points.append(-float('inf'))
                ava_points.append(current_points[0])
                if p2_i!=3:
                    adj2_point = point_grid[p2_i+1, p1_i][0]
                    ava_points.append(adj2_point)
                else: 
                    ava_points.append(-float('inf'))
                    
                # ava_points = [-x for x in ava_points]
                next_move = ava_points.index(max(ava_points))
                if next_move==0:
                    p2_i = p2_i-1
                elif next_move==1:
                    p2_i = p2_i
                elif next_move==2:
                    p2_i = p2_i+1                
            
            turn_count += 1
            sim.competition_trajectory.append((fare_grid[p2_i, p1_i][0], fare_grid[p2_i, p1_i][1]))
        # Print -------------------------------------------------------------------
        print('Day = ', day)
        if day%2==0 and day!=0:
            print('p1_trun = ', p1_trun, 'p2_trun = ', p2_trun)
            print('p2_i',p2_i,'  ','p1_i',p1_i)
        # df = sim.res[day].pax_exp
        # fd = sim.res[day].veh_exp
        # np1 = len(df[df.platform_id==1]);np2 = len(df[df.platform_id==2])
        # vp1 = len(fd[fd.platform_id==1]);vp2 = len(fd[fd.platform_id==2])
        # print('np1 = ', np1, '  np2 = ', np2);print('vp1 = ', vp1, '  vp2 = ', vp2)
        print('--------------------------------------')
        #-------------------------------------------------------------------
        if sim.functions.f_stop_crit(sim=sim):
            break
    return sim


def simulate_Try_and_Select(config="data/config.json", inData=None, params=None, **kwargs):

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
    
    # Competition ===========================================================    
    fares = params.fares
    max_i = len(fares)-1
    fare_grid = np.array([[(x, y) for y in fares] for x in fares])
    
    if params.random_ini_position == True:
        p1_i = np.random.randint(0, max_i+1)
        p2_i = np.random.randint(0, max_i+1)
        
    else: 
        p1_i = fares.index(1.4)
        p2_i = fares.index(1.6)
    
    turn_count = 0
    turnover_interval = params.turnover_interval
    sim.competition_trajectory = [(fare_grid[p2_i, p1_i][1], fare_grid[p2_i, p1_i][0])]
    # levers -----------------------------------------------------
    sim.platforms.fare[1] = fare_grid[p2_i, p1_i][1]
    sim.platforms.fare[2] = fare_grid[p2_i, p1_i][0]
    
    sim.platforms.comm_rate[1] = 0.20
    sim.platforms.comm_rate[2] = 0.20

    sim.platforms.discount[1] = 0.0
    sim.platforms.discount[2] = 0.0

    sim.platforms.daily_marketing[1] = True
    sim.platforms.daily_marketing[2] = True
    
    # sim.competition_trajectory = [(sim.platforms.fare[2], sim.platforms.fare[1])]
    
    for day in range(params.get('nD', 1)):
        
        if day%turnover_interval==0:
            
            p1_trun, p2_trun = (turn_count % 2 == 0, turn_count % 2 != 0)
            ava_points = []
            res_list = []
            
            if p1_trun==True:
                print('------------------ P1 TURN ------------------')
                
                if p1_i!=0:
                    for d in range(day, day+turnover_interval):
                        print('Day = ', d, '  Left cell' )
                        sim.run_id = d
                        sim.platforms.fare[1] = fare_grid[p2_i, p1_i-1][1]                        
                        sim.make_and_run(run_id=d)
                        sim.output(run_id=d)
                        #-----------------------------------
                        df = sim.res[d].pax_exp;fd = sim.res[d].veh_exp
                        np1 = len(df[df.platform_id==1]);np2 = len(df[df.platform_id==2])
                        vp1 = len(fd[fd.platform_id==1]);vp2 = len(fd[fd.platform_id==2])
                        print('np1 = ', np1, '  np2 = ', np2);print('vp1 = ', vp1, '  vp2 = ', vp2)
                        #-----------------------------------
                    print('Left cell: Fare = {} & P1 Remaining capital = {} & P2 Remaining capital = {}'.format(sim.platforms.fare[1], sim.res[d].platforms.P_remaining_capital.loc[1], sim.res[d].platforms.P_remaining_capital.loc[2]))
                    print('--------------------------------------------')
                    res_copyL = copy.deepcopy(sim.res) 
                    L_point = res_copyL[d].platforms.P_remaining_capital.loc[1]                        
                    ava_points.append(L_point)
                    res_list.append(res_copyL)
                else:
                    ava_points.append(-float('inf'))
                    res_list.append(None)
                
                for d in range(day, day+turnover_interval):
                    print('Day = ', d, '  Middle cell' )
                    sim.run_id = d
                    sim.platforms.fare[1] = fare_grid[p2_i, p1_i][1]                        
                    sim.make_and_run(run_id=d)
                    sim.output(run_id=d)
                    #-----------------------------------
                    df = sim.res[d].pax_exp;fd = sim.res[d].veh_exp
                    np1 = len(df[df.platform_id==1]);np2 = len(df[df.platform_id==2])
                    vp1 = len(fd[fd.platform_id==1]);vp2 = len(fd[fd.platform_id==2])
                    print('np1 = ', np1, '  np2 = ', np2);print('vp1 = ', vp1, '  vp2 = ', vp2)
                     #-----------------------------------
                print('Middle cell: Fare = {} & P1 Remaining capital = {} & P2 Remaining capital = {}'.format(sim.platforms.fare[1], sim.res[d].platforms.P_remaining_capital.loc[1], sim.res[d].platforms.P_remaining_capital.loc[2]))
                print('--------------------------------------------')
                res_copyS = copy.deepcopy(sim.res) 
                # sim.resS = res_copyS #####
                S_point = res_copyS[d].platforms.P_remaining_capital.loc[1]                        
                ava_points.append(S_point)
                res_list.append(res_copyS)
                
                if p1_i!=max_i:
                    for d in range(day, day+turnover_interval):
                        print('Day = ', d, '  Right cell' )
                        sim.run_id = d
                        sim.platforms.fare[1] = fare_grid[p2_i, p1_i+1][1]                        
                        sim.make_and_run(run_id=d)
                        sim.output(run_id=d)
                        #-----------------------------------
                        df = sim.res[d].pax_exp;fd = sim.res[d].veh_exp
                        np1 = len(df[df.platform_id==1]);np2 = len(df[df.platform_id==2])
                        vp1 = len(fd[fd.platform_id==1]);vp2 = len(fd[fd.platform_id==2])
                        print('np1 = ', np1, '  np2 = ', np2);print('vp1 = ', vp1, '  vp2 = ', vp2)
                        #-----------------------------------
                    print('Right cell: Fare = {} & P1 Remaining capital = {} & P2 Remaining capital = {}'.format(sim.platforms.fare[1], sim.res[d].platforms.P_remaining_capital.loc[1], sim.res[d].platforms.P_remaining_capital.loc[2]))
                    print('--------------------------------------------')
                    res_copyR = copy.deepcopy(sim.res)   
                    # sim.resR = res_copyR #####
                    R_point = res_copyR[d].platforms.P_remaining_capital.loc[1]                        
                    ava_points.append(R_point)
                    res_list.append(res_copyR)
                else:
                    ava_points.append(-float('inf'))
                    res_list.append(None)
                
                # ava_points = [3,2,1]
                next_move = ava_points.index(max(ava_points))
                p1_i = max(0, min(max_i, p1_i + [-1, 0, 1][next_move]))
                sim.platforms.fare[1] = fare_grid[p2_i, p1_i][1] 
                sim.res = copy.deepcopy(res_list[next_move])
                print('fares = ', fare_grid[p2_i, p1_i])
            
            elif p2_trun==True:
                print('------------ P2 TURN ------------')
                
                if p2_i!=0:
                    for d in range(day, day+turnover_interval):
                        print('Day = ', d, '  Lower cell' )
                        sim.run_id = d
                        sim.platforms.fare[2] = fare_grid[p2_i-1, p1_i][0]                        
                        sim.make_and_run(run_id=d)
                        sim.output(run_id=d)
                        #-----------------------------------
                        df = sim.res[d].pax_exp;fd = sim.res[d].veh_exp
                        np1 = len(df[df.platform_id==1]);np2 = len(df[df.platform_id==2])
                        vp1 = len(fd[fd.platform_id==1]);vp2 = len(fd[fd.platform_id==2])
                        print('np1 = ', np1, '  np2 = ', np2);print('vp1 = ', vp1, '  vp2 = ', vp2)
                        #-----------------------------------
                    print('Lower cell: Fare = {} & P1 Remaining capital = {} & P2 Remaining capital = {}'.format(sim.platforms.fare[2], sim.res[d].platforms.P_remaining_capital.loc[1], sim.res[d].platforms.P_remaining_capital.loc[2]))
                    print('--------------------------------------------')
                    res_copyL = copy.deepcopy(sim.res) 
                    L_point = res_copyL[d].platforms.P_remaining_capital.loc[2]                        
                    ava_points.append(L_point)
                    res_list.append(res_copyL)
                else:
                    ava_points.append(-float('inf'))
                    res_list.append(None)
                
                for d in range(day, day+turnover_interval):
                    print('Day = ', d, '  Middle cell' )
                    sim.run_id = d
                    sim.platforms.fare[2] = fare_grid[p2_i, p1_i][0]                        
                    sim.make_and_run(run_id=d)
                    sim.output(run_id=d)
                    #-----------------------------------
                    df = sim.res[d].pax_exp;fd = sim.res[d].veh_exp
                    np1 = len(df[df.platform_id==1]);np2 = len(df[df.platform_id==2])
                    vp1 = len(fd[fd.platform_id==1]);vp2 = len(fd[fd.platform_id==2])
                    print('np1 = ', np1, '  np2 = ', np2);print('vp1 = ', vp1, '  vp2 = ', vp2)
                    #-----------------------------------
                print('Middle cell: Fare = {} & P1 Remaining capital = {} & P2 Remaining capital = {}'.format(sim.platforms.fare[2], sim.res[d].platforms.P_remaining_capital.loc[1], sim.res[d].platforms.P_remaining_capital.loc[2]))
                print('--------------------------------------------')
                res_copyS = copy.deepcopy(sim.res) 
                S_point = res_copyS[d].platforms.P_remaining_capital.loc[2]                        
                ava_points.append(S_point)
                res_list.append(res_copyS)
                
                if p2_i!=max_i:
                    for d in range(day, day+turnover_interval):
                        print('Day = ', d, '  Upper cell' )
                        sim.run_id = d
                        sim.platforms.fare[2] = fare_grid[p2_i+1, p1_i][0]                        
                        sim.make_and_run(run_id=d)
                        sim.output(run_id=d)
                        #-----------------------------------
                        df = sim.res[d].pax_exp;fd = sim.res[d].veh_exp
                        np1 = len(df[df.platform_id==1]);np2 = len(df[df.platform_id==2])
                        vp1 = len(fd[fd.platform_id==1]);vp2 = len(fd[fd.platform_id==2])
                        print('np1 = ', np1, '  np2 = ', np2);print('vp1 = ', vp1, '  vp2 = ', vp2)
                        #-----------------------------------
                    print('Upper cell: Fare = {} & P1 Remaining capital = {} & P2 Remaining capital = {}'.format(sim.platforms.fare[2], sim.res[d].platforms.P_remaining_capital.loc[1], sim.res[d].platforms.P_remaining_capital.loc[2]))
                    print('--------------------------------------------')
                    res_copyR = copy.deepcopy(sim.res)   
                    R_point = res_copyR[d].platforms.P_remaining_capital.loc[2]                        
                    ava_points.append(R_point)
                    res_list.append(res_copyR)
                else:
                    ava_points.append(-float('inf'))
                    res_list.append(None)
                    
                # ava_points = [3,2,1]
                next_move = ava_points.index(max(ava_points))
                p2_i = max(0, min(max_i, p2_i + [-1, 0, 1][next_move]))
                sim.platforms.fare[2] = fare_grid[p2_i, p1_i][0] 
                sim.res = copy.deepcopy(res_list[next_move]) 
                print('fares = ', fare_grid[p2_i, p1_i])
        
            turn_count += 1
            sim.competition_trajectory.append((sim.platforms.fare[2], sim.platforms.fare[1]))
            day = day+turnover_interval
            if  day==499:
                if input('Say stop if you want to break: ')=='stop':
                    break
            
#-------------------------------------------------------------------------------------------
        if sim.functions.f_stop_crit(sim=sim):
            break
    return sim








#===============================================================================================

def simulate_S(config="data/config.json", inData=None, params=None, **kwargs):

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
    
    # Competition ===========================================================
    # point_grid = np.load('point_grid.npy', allow_pickle=True)
    
    fares = [1.0, 1.2, 1.4, 1.6]
    max_i = len(fares)-1
    fare_grid = np.array([[(x, y) for y in fares] for x in fares])
    
    if params.random_ini_position == True:
        p1_i = np.random.randint(0, max_i+1)
        p2_i = np.random.randint(0, max_i+1)
        
    else: 
        p1_i = fares.index(1.6)
        p2_i = fares.index(1.6)
    
    turn_count = 0
    turnover_interval = 2
    sim.competition_trajectory = [(fare_grid[p2_i, p1_i][1], fare_grid[p2_i, p1_i][0])]
    # Other levers -----------------------------------------------------
    sim.platforms.comm_rate[1] = 0.20
    sim.platforms.comm_rate[2] = 0.20

    sim.platforms.discount[1] = 0.0
    sim.platforms.discount[2] = 0.0

    sim.platforms.daily_marketing[1] = True
    sim.platforms.daily_marketing[2] = True
    
    for day in range(params.get('nD', 1)):

        # Trip fare adjustment --------------------------------------------
        sim.platforms.fare[1] = fare_grid[p2_i, p1_i][1]
        sim.platforms.fare[2] = fare_grid[p2_i, p1_i][0]
        
        #-------------------------------------------------------------------
        sim.make_and_run(run_id=day)  
        sim.output()
        
        #-------------------------------------------------------------------
        if day%turnover_interval==0 and day!=0:
            L_sim = copy.deepcopy(sim)
            S_sim = copy.deepcopy(sim)
            R_sim = copy.deepcopy(sim)
            if turn_count%2==0:
                p1_trun = True
                p2_trun = False
            else: 
                p1_trun = False
                p2_trun = True
            ava_points = []
            
            if p1_trun==True:
                
                if p1_i!=0:
                    for d in range(day+1, day+turnover_interval):
                        print('L_sim    ', 'Day = ', day, '  d in Day = ', d)
                        L_sim.platforms.fare[1] = fare_grid[p2_i, p1_i-1][1]
                        L_sim.make_and_run(run_id=d)
                        L_sim.output()
                    L_point = L_sim.res[d].platforms.P_remaining_capital.loc[1]                        
                    ava_points.append(L_point)
                else:
                    ava_points.append(-float('inf'))
                    
                for d in range(day+1, day+turnover_interval):
                    print('S_sim    ', 'Day = ', day, '  d in Day = ', d)
                    S_sim.platforms.fare[1] = fare_grid[p2_i, p1_i][1]
                    S_sim.make_and_run(run_id=d)
                    S_sim.output()
                S_point = S_sim.res[d].platforms.P_remaining_capital.loc[1]                        
                ava_points.append(S_point) 
                
                if p1_i!=max_i:
                    for d in range(day+1, day+turnover_interval):
                        print('R_sim    ', 'Day = ', day, '  d in Day = ', d)
                        R_sim.platforms.fare[1] = fare_grid[p2_i, p1_i+1][1]
                        R_sim.make_and_run(run_id=d)
                        R_sim.output()
                    R_point = R_sim.res[d].platforms.P_remaining_capital.loc[1]                        
                    ava_points.append(R_point)
                else:
                    ava_points.append(-float('inf'))
                
            
#             elif p2_trun==True:
        
        
        
#         turn_count += 1
#         sim.competition_trajectory.append((fare_grid[p2_i, p1_i][0], fare_grid[p2_i, p1_i][1]))
# #-------------------------------------------------------------------------------------------
             
#             current_points = point_grid[p2_i, p1_i]
#             ava_points = []
            
#             if p1_trun==True:
#                 if p1_i!=0:
#                     adj1_point = point_grid[p2_i, p1_i-1][1]
#                     ava_points.append(adj1_point)
#                 else: 
#                     ava_points.append(-float('inf'))
#                 ava_points.append(current_points[1])
#                 if p1_i!=3:
#                     adj2_point = point_grid[p2_i, p1_i+1][1]
#                     ava_points.append(adj2_point)
#                 else: 
#                     ava_points.append(-float('inf'))
                    
#                 next_move = ava_points.index(max(ava_points))
#                 if next_move==0:
#                     p1_i = p1_i-1
#                 elif next_move==1:
#                     p1_i = p1_i
#                 elif next_move==2:
#                     p1_i = p1_i+1
                    
#             elif p2_trun==True:
#                 if p2_i!=0:
#                     adj1_point = point_grid[p2_i-1, p1_i][0]
#                     ava_points.append(adj1_point)
#                 else: 
#                     ava_points.append(-float('inf'))
#                 ava_points.append(current_points[0])
#                 if p2_i!=3:
#                     adj2_point = point_grid[p2_i+1, p1_i][0]
#                     ava_points.append(adj2_point)
#                 else: 
#                     ava_points.append(-float('inf'))
                    
#                 # ava_points = [-x for x in ava_points]
#                 next_move = ava_points.index(max(ava_points))
#                 if next_move==0:
#                     p2_i = p2_i-1
#                 elif next_move==1:
#                     p2_i = p2_i
#                 elif next_move==2:
#                     p2_i = p2_i+1                
            
#             turn_count += 1
#             sim.competition_trajectory.append((fare_grid[p2_i, p1_i][0], fare_grid[p2_i, p1_i][1]))
        # Print -------------------------------------------------------------------
        print('Day = ', day)
        if day%turnover_interval==0 and day!=0:
            print('p1_trun = ', p1_trun, 'p2_trun = ', p2_trun)
            print('p2_i',p2_i,'  ','p1_i',p1_i)
        # df = sim.res[day].pax_exp
        # fd = sim.res[day].veh_exp
        # np1 = len(df[df.platform_id==1]);np2 = len(df[df.platform_id==2])
        # vp1 = len(fd[fd.platform_id==1]);vp2 = len(fd[fd.platform_id==2])
        # print('np1 = ', np1, '  np2 = ', np2);print('vp1 = ', vp1, '  vp2 = ', vp2)
        print('--------------------------------------')
        #-------------------------------------------------------------------
        if sim.functions.f_stop_crit(sim=sim):
            break
    return sim








if __name__ == "__main__":
    simulate(make_main_path='..')  # single run

    from MaaSSim.utils import test_space

    simulate_parallel(search_space = test_space())

