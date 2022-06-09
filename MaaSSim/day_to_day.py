from .traveller import travellerEvent
from .driver import driverEvent
import numpy as np
import pandas as pd
import math
from numpy import log as ln
import random
from statsmodels.tsa.stattools import adfuller


def driver_opt_out(veh, **kwargs): # user defined function to represent agent participation choice
    """
    This function depends on stochasticity and heterogeneity of model
    """
    sim = veh.sim
    params = sim.params
    expected_income = params.d2d.ini_exp_income if len(sim.res) == 0 else sim.res[len(sim.res)-1].veh_exp.EXPECTED_INC.loc[veh.id]
    
    working_U = params.d2d.get('B_income',1)*(expected_income + veh.veh.get('exp_income_eps', 0))
    not_working_U = params.d2d.get('B_income',1)*(params.d2d.res_wage + veh.veh.get('res_wage_eps', 0))
    
    if params.d2d.probabilistic:
        working_P = (math.exp(working_U))/(math.exp(working_U) + math.exp(not_working_U))
        sim.driver_p.append(working_P)
        return bool(working_P < random.uniform(0,1))
    else:
        return bool(working_U < not_working_U)
    
    
def S_driver_opt_out(veh, **kwargs): # user defined function to represent agent participation choice
    """
    This function depends on stochasticity and heterogeneity of model
    """
    sim = veh.sim
    params = sim.params    
    informed = False if len(sim.res) == 0 else sim.res[len(sim.res)-1].veh_exp.INFORMED.loc[veh.id]
    if informed==False:
        return True
    
    EXPERIENCE_U = 0.5 if len(sim.res) == 0 else sim.res[len(sim.res)-1].veh_exp.EXPERIENCE_U.loc[veh.id]
    MARKETING_U = 0.5 if len(sim.res) == 0 else sim.res[len(sim.res)-1].veh_exp.MARKETING_U.loc[veh.id]
    WOM_U = 0.5 if len(sim.res) == 0 else sim.res[len(sim.res)-1].veh_exp.WOM_U.loc[veh.id]
    
    b1 = 0.5
    b2 = 0.25
    b3 = 0.25
    working_U = b1*EXPERIENCE_U + b2*MARKETING_U + b3*WOM_U
    not_working_U = b1*0.5 + b2*0.5 #+ b3*0.5
    # print('veh id ', veh.id,'U= ',  working_U)
    veh.veh.working_U = working_U
    
    if params.d2d.probabilistic:
        s = 20
        working_P = (math.exp(s*working_U))/(math.exp(s*working_U) + math.exp(s*not_working_U))
        # print('working_p= ',working_P)
        # sim.driver_p.append(working_P)
        return bool(working_P < random.uniform(0,1))
    else:
        return bool(working_U < not_working_U)
    
def S_traveller_opt_out(pax, **kwargs):
    
    sim = pax.sim
    params = sim.params
    exp_wait_t = params.d2d.ini_exp_wt if len(sim.res) == 0 else sim.res[len(sim.res)-1].pax_exp.EXPECTED_WT.loc[pax.id]
    
    req = pax.request
    plat = sim.platforms.loc[1]
    rh_fare = max(plat.get('base_fare',0) + plat.fare*req.dist/1000, plat.get('min_fare',0))
    ASC = 0 if len(sim.res) == 0 else sim.res[len(sim.res)-1].veh_exp.COMMISSION.sum()/10000
    # if pax.id == 1:
    #     print('ASC = ', ASC)
    
    rh_U = -params.d2d.get('B_fare',1)*rh_fare -params.d2d.get('B_inveh_time',1)*req.ttrav.total_seconds()/3600-params.d2d.get('B_exp_time',1)*exp_wait_t/60 + ASC + pax.pax.get('exp_utility_eps', 0)
    alt_U = -params.d2d.get('B_fare',1)*params.PT_fare*req.dist/1000 -params.d2d.get('B_inveh_time',1)*(req.dist/params.PT_speed)/3600
    
    S = 8
    rh_U = rh_U*S
    alt_U = alt_U*S

    if params.d2d.probabilistic:
        rh_P = (math.exp(rh_U))/(math.exp(rh_U)+math.exp(alt_U))
        sim.traveller_p.append(rh_P)
        return bool(rh_P < random.uniform(0,1))
    else:
        return bool(rh_U < alt_U)


    
def traveller_opt_out(pax, **kwargs):
    
    sim = pax.sim
    params = sim.params
    exp_wait_t = params.d2d.ini_exp_wt if len(sim.res) == 0 else sim.res[len(sim.res)-1].pax_exp.EXPECTED_WT.loc[pax.id]
    
    req = pax.request
    plat = sim.platforms.loc[1]
    rh_fare = max(plat.get('base_fare',0) + plat.fare*req.dist/1000, plat.get('min_fare',0))
    ASC = 0 if len(sim.res) == 0 else sim.res[len(sim.res)-1].veh_exp.COMMISSION.sum()/10000
    # if pax.id == 1:
    #     print('ASC = ', ASC)
    
    rh_U = -params.d2d.get('B_fare',1)*rh_fare -params.d2d.get('B_inveh_time',1)*req.ttrav.total_seconds()/3600-params.d2d.get('B_exp_time',1)*exp_wait_t/60 + ASC + pax.pax.get('exp_utility_eps', 0)
    
    alt_U = -params.d2d.get('B_fare',1)*params.PT_fare*req.dist/1000 -params.d2d.get('B_inveh_time',1)*(req.dist/params.PT_speed)/3600
    
    S = 8
    rh_U = rh_U*S
    alt_U = alt_U*S
    
    #=================================================================
#     print('rh_U= ',rh_U,' fare= ', -params.d2d.get('B_fare',1)*rh_fare,' time= ',-params.d2d.get('B_inveh_time',1)*req.ttrav.total_seconds()/3600-params.d2d.get('B_exp_time',1)*exp_wait_t/60)
    
#     print('alt_U= ',alt_U,' fare= ',-params.d2d.get('B_fare',1)*params.PT_fare*req.dist/1000, 'time= ', -params.d2d.get('B_inveh_time',1)*(req.dist/params.PT_speed)/3600)
    #=================================================================
    
    if params.d2d.probabilistic:
        rh_P = (math.exp(rh_U))/(math.exp(rh_U)+math.exp(alt_U))
        sim.traveller_p.append(rh_P)
        return bool(rh_P < random.uniform(0,1))
    else:
        return bool(rh_U < alt_U)

    
def d2d_kpi_veh(*args,**kwargs):

    """
    calculate vehicle KPIs (global and individual)
    apdates driver expected income
    """
    sim =  kwargs.get('sim', None)
    params = sim.params
    run_id = kwargs.get('run_id', None)
    simrun = sim.runs[run_id]
    vehindex = sim.inData.vehicles.index
    df = simrun['rides'].copy()  # results of previous simulation
    DECIDES_NOT_TO_DRIVE = df[df.event == driverEvent.DECIDES_NOT_TO_DRIVE.name].veh  # track drivers out
    dfs = df.shift(-1)  # to map time periods between events
    dfs.columns = [_ + "_s" for _ in df.columns]  # columns with _s are shifted
    df = pd.concat([df, dfs], axis=1)  # now we have time periods
    df = df[df.veh == df.veh_s]  # filter for the same vehicles only
    df = df[~(df.t == df.t_s)]  # filter for positive time periods only
    df['dt'] = df.t_s - df.t  # make time intervals
    ret = df.groupby(['veh', 'event_s'])['dt'].sum().unstack()  # aggreagted by vehicle and event
    ret.columns.name = None
    ret = ret.reindex(vehindex)  # update for vehicles with no record
    ret['nRIDES'] = df[df.event == driverEvent.ARRIVES_AT_DROPOFF.name].groupby(['veh']).size().reindex(ret.index)
    ret['nREJECTED'] = df[df.event==driverEvent.IS_REJECTED_BY_TRAVELLER.name].groupby(['veh']).size().reindex(ret.index)
    for status in driverEvent:
        if status.name not in ret.columns:
            ret[status.name] = 0
    DECIDES_NOT_TO_DRIVE.index = DECIDES_NOT_TO_DRIVE.values
    ret['OUT'] = DECIDES_NOT_TO_DRIVE
    ret['OUT'] = ~ret['OUT'].isnull()
    ret['DRIVING_TIME'] = ret.ARRIVES_AT_PICKUP + ret.ARRIVES_AT_DROPOFF
    ret['DRIVING_DIST'] = ret['DRIVING_TIME']*(params.speeds.ride/1000)  #here we assume the speed is constant on the network
    # ret['REVENUE'] = (ret.ARRIVES_AT_DROPOFF * (params.speeds.ride/1000) * params.platforms.fare).add(params.platforms.base_fare * ret.nRIDES) * (1-params.platforms.comm_rate)
    
    d = df[df['event_s']=='ARRIVES_AT_DROPOFF']
    if len(d) != 0:
        d['TRIP_FARE'] = d.apply(lambda row: max(row['dt'] * (params.speeds.ride/1000) * params.platforms.fare + params.platforms.base_fare, params.platforms.min_fare), axis=1)
        ret['TRIP_FARE'] = d.groupby(['veh']).sum().TRIP_FARE
    else:
        ret['TRIP_FARE'] = 0
    ret['REVENUE'] = ret['TRIP_FARE']*(1-params.platforms.comm_rate)
    ret['COMMISSION'] = ret['TRIP_FARE']*params.platforms.comm_rate
    ret['COST'] = ret['DRIVING_DIST'] * (params.d2d.fuel_cost) # Operating Cost (OC)
    ret['PROFIT'] = ret['REVENUE'] - ret['COST']
    ret['mu'] = ret.apply(lambda row: 1 if row['OUT'] == False else 0, axis=1)
    ret['nDAYS_WORKED'] = ret['mu'] if run_id == 0 else sim.res[run_id-1].veh_exp.nDAYS_WORKED + ret['mu']
    ret.fillna(0, inplace=True)
    
    # Driver adaptation (learning) --------------------------------------------------------------------------------- #
    ret['ACTUAL_INC'] = ret.PROFIT    
    
    #update_learning_status(sim, ret)
    #---------------------------------------------------------
    # Djavadian & Chow (2017)
    pre_exp_inc = params.d2d.ini_exp_income if run_id == 0 else sim.res[run_id-1].veh_exp.EXPECTED_INC
    ave_income = 0 if ret.mu.sum() == 0 else ret.ACTUAL_INC.sum()/ret.mu.sum()
    ret['EXPECTED_INC'] = (1-params.d2d.veh_omega)*pre_exp_inc + params.d2d.veh_omega*ret.mu*ret.ACTUAL_INC+ \
                           params.d2d.veh_omega*(1-ret.mu)*ave_income
    #---------------------------------------------------------
    # Arjan (2021)
    # ret['pre_exp_inc'] = params.d2d.ini_exp_income if run_id == 0 else sim.res[run_id-1].veh_exp.EXPECTED_INC
    # ret['EXPECTED_INC'] = ret.apply(lambda row: row['pre_exp_inc'] if row['mu']==0 or sim.vehs[row.name].veh.get('learning','on')=='off' else (1-(row['nDAYS_WORKED']+1)**(-(params.d2d.kappa)))*row['pre_exp_inc'] + ((row['nDAYS_WORKED']+1)**(-(params.d2d.kappa)))*row['ACTUAL_INC'], axis=1)
    #---------------------------------------------------------
    # Nejc model
    #====================================================================
    # Rafal & Farnoud (2022)
    
    ret['INFORMED'] = False if run_id == 0 else sim.res[run_id-1].veh_exp.INFORMED
    #-------------------------------------------------------
    """ Utility gained through experience"""
    
    ret['pre_EXPERIENCE_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].veh_exp.EXPERIENCE_U
    
    if run_id == 0:
        ret['pre_ACTUAL_INC'] = params.d2d.res_wage
    else:
        veh_exp = sim.res[run_id-1].veh_exp
        ret['pre_ACTUAL_INC'] = ret.apply(lambda row: veh_exp.pre_ACTUAL_INC[row.name] if veh_exp.mu[row.name] == 0 else veh_exp.ACTUAL_INC[row.name], axis=1)
    
    ret['inc_dif'] = ret.apply(lambda row: 0 if row.mu==0 else (row['pre_ACTUAL_INC']-row['ACTUAL_INC'])/params.d2d.res_wage, axis=1)
    ret['EXPERIENCE_U'] = ret.apply(lambda row: 1/(1+math.exp(params.d2d.learning_d*(ln((1/row.pre_EXPERIENCE_U)-1)+row.inc_dif))), axis=1)
    #--------------------------------------------------------
    """ Utility gained through marketing"""

    ret['pre_MARKETING_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].veh_exp.MARKETING_U
    ret['MARKETING_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].veh_exp.MARKETING_U
    percentage = 10/100 # this percentage should be a function of platfrom profit
    retx = ret.sample(int(percentage*params.nV))
    retx['MARKETING_U'] = retx.apply(lambda row: 1/(1+math.exp(params.d2d.learning_d*(ln((1/row.pre_MARKETING_U)-1)+row.pre_MARKETING_U-1))), axis=1)
    retx['INFORMED'] = True
    ret.update(retx)
    #--------------------------------------------------------
    """ Utility gained through Word of Mouth (WOM)"""
    
    # print(ret.INFORMED)
    ret['pre_WOM_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].veh_exp.WOM_U
    ret['WOM_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].veh_exp.WOM_U
    
    v_list = [v for v in range(1, params.nV+1)]
    tuples = []
    for v in range(1, params.nV+1):
        if v in v_list:
            v_list.remove(v)
            interlocutor = random.choice(v_list)
            v_list.remove(interlocutor)
            tuples.append((v,interlocutor))
            
    for tup in tuples:
        v1 = tup[0]
        v2 = tup[1]
        ret['WOM_U'].loc[v1] = 1/(1+math.exp(params.d2d.learning_d*(ln((1/ret['pre_WOM_U'].loc[v1])-1)+ret['pre_WOM_U'].loc[v1]-sim.vehs[v2].veh.working_U)))
        ret['WOM_U'].loc[v2] = 1/(1+math.exp(params.d2d.learning_d*(ln((1/ret['pre_WOM_U'].loc[v2])-1)+ret['pre_WOM_U'].loc[v2]-sim.vehs[v1].veh.working_U)))
        if (ret['INFORMED'].loc[v1] == False and ret['INFORMED'].loc[v2] == True) | (ret['INFORMED'].loc[v2] == False and ret['INFORMED'].loc[v1] == True):
            ret['INFORMED'].loc[v1] = True
            ret['INFORMED'].loc[v2] = True
    
    #===================================================================
    
    ret = ret[['nRIDES','nREJECTED', 'nDAYS_WORKED', 'DRIVING_TIME', 'DRIVING_DIST', 'REVENUE',
               'COST','COMMISSION','TRIP_FARE','ACTUAL_INC', 'EXPECTED_INC','OUT','mu','INFORMED',
               'EXPERIENCE_U','MARKETING_U','WOM_U','pre_ACTUAL_INC'] + [_.name for _ in driverEvent]]
    ret.index.name = 'veh'
    
    # KPIs
    kpi = ret.agg(['sum', 'mean', 'std'])
    kpi['nV'] = ret.shape[0]
    return {'veh_exp': ret, 'veh_kpi': kpi}
    
    

def update_learning_status(sim, ret):
    
    if len(sim.runs) > 3: # stationarity test needs at least 4 values.
        f = pd.DataFrame()
        for run_id in range(0,len(sim.runs)-1):
            f['{}'.format(run_id)] = sim.res[run_id].veh_exp['ACTUAL_INC']
        #we can't add the last day's ACTUAL_INC from res, since it is not calculated yet.
        f['{}'.format(len(sim.runs)-1)] = ret['ACTUAL_INC']
        for veh in f.index:
            if sim.vehs[veh].veh['learning'] == 'on':
                a = f.loc[veh]
                a = [_ for _ in a if _ != 0]
                if len(a) > 3:
                    adf = adfuller(a)
                    # if adf[0] < 0.05:
                    if adf[0] < adf[4]["5%"]:
                        sim.vehicles.at[veh,'learning'] = 'off'
                        print('vehid ',veh)
                        print('day----------------------------',len(sim.runs))
    return sim
    
    
################################################################################################################

def d2d_kpi_pax(*args,**kwargs):
    # calculate passenger indicators (global and individual)

    sim = kwargs.get('sim', None)
    params = sim.params
    run_id = kwargs.get('run_id', None)
    simrun = sim.runs[run_id]
    paxindex = sim.inData.passengers.index
    df = simrun['trips'].copy()  # results of previous simulation
    PREFERS_OTHER_SERVICE = df[df.event == travellerEvent.PREFERS_OTHER_SERVICE.name].pax  # track drivers out
    dfs = df.shift(-1)  # to map time periods between events
    dfs.columns = [_ + "_s" for _ in df.columns]  # columns with _s are shifted
    df = pd.concat([df, dfs], axis=1)  # now we have time periods
    df = df[df.pax == df.pax_s]  # filter for the same vehicles only
    df = df[~(df.t == df.t_s)]  # filter for positive time periods only
    df['dt'] = df.t_s - df.t  # make time intervals
    ret = df.groupby(['pax', 'event_s'])['dt'].sum().unstack()  # aggreagted by vehicle and event

    ret.columns.name = None
    ret = ret.reindex(paxindex)  # update for vehicles with no record

    ret.index.name = 'pax'
    ret = ret.fillna(0)

    for status in travellerEvent:
        if status.name not in ret.columns:
            ret[status.name] = 0  # cover all statuses
    PREFERS_OTHER_SERVICE.index = PREFERS_OTHER_SERVICE.values
    ret['OUT'] = PREFERS_OTHER_SERVICE
    ret['OUT'] = ~ret['OUT'].isnull()   
    ret['mu'] = ret.apply(lambda row: 1 if row['OUT'] == False else 0, axis=1)
    ret['nDAYS_HAILED'] = ret['mu'] if run_id == 0 else sim.res[run_id-1].pax_exp.nDAYS_HAILED + ret['mu']
    ret['TRAVEL'] = ret['ARRIVES_AT_DROPOFF']  # time with traveller (paid time)
    ret['ACTUAL_WT'] = (ret['RECEIVES_OFFER'] + ret['MEETS_DRIVER_AT_PICKUP'] + ret.get('LOSES_PATIENCE', 0))/60  #in minute
    ret['OPERATIONS'] = ret['ACCEPTS_OFFER'] + ret['DEPARTS_FROM_PICKUP'] + ret['SETS_OFF_FOR_DEST']
    ret.fillna(0, inplace=True)
    
    # Traveller adaptation (learning) --------------------------------------------------------------------------------- #
    # Djavadian-------------------------------------------------------------------------------------------------------- #
    pre_exp_wt = params.d2d.ini_exp_wt if run_id == 0 else sim.res[run_id-1].pax_exp.EXPECTED_WT
    ave_wt = 0 if ret.mu.sum() == 0 else ret.ACTUAL_WT.sum()/ret.mu.sum()
    ret['EXPECTED_WT'] = (1-params.d2d.pax_omega)*pre_exp_wt + params.d2d.pax_omega*ret.mu*ret.ACTUAL_WT+ \
                           params.d2d.pax_omega*(1-ret.mu)*ave_wt
    # Arjan------------------------------------------------------------------------------------------------------------ #
    # ret['pre_exp_wt'] = params.d2d.ini_exp_wt if run_id == 0 else sim.res[run_id-1].pax_exp.EXPECTED_WT
    # ret['EXPECTED_WT'] = ret.apply(lambda row: row['pre_exp_wt'] if row['mu']==0 or sim.pax[row.name].pax.get('learning','on')=='off' else (1-(row['nDAYS_HAILED']+1)**(-(params.d2d.kappa)))*row['pre_exp_wt'] + ((row['nDAYS_HAILED']+1)**(-(params.d2d.kappa)))*row['ACTUAL_WT'], axis=1)                                    
    # ----------------------------------------------------------------------------------------------------------------- #

    ret = ret[['ACTUAL_WT', 'EXPECTED_WT', 'OUT','mu','nDAYS_HAILED'] + [_.name for _ in travellerEvent]]
    ret.index.name = 'pax'

    kpi = ret.agg(['sum', 'mean', 'std'])
    kpi['nP'] = ret.shape[0]
    return {'pax_exp': ret, 'pax_kpi': kpi}





























    
def my_function(veh, **kwargs): # user defined function to represent agent decisions
    sim = veh.sim
    if  veh.veh.expected_income < sim.params.d2d.res_wage:
        return True
    else:
        return False
    
    
#sim.logger.info("Heyyoooooooooooooooooooooooooooooooooooooooo")
# df = df[df['event'].isin(['IS_ACCEPTED_BY_TRAVELLER', 'ARRIVES_AT_PICKUP', 'DEPARTS_FROM_PICKUP'])] 

def exp_income(sim):
    params = sim.params 
    run_id = sim.run_ids[-1]
    act_income = sim.res[run_id].veh_exp.PROFIT
    mu = sim.res[run_id].veh_exp.mu
    if mu.sum() == 0:
        ave_income = 0
    else:
        ave_income = act_income.sum()/mu.sum()
    # update the expected_income
    sim.inData.vehicles.expected_income = (1-params.d2d.omega)*sim.inData.vehicles.expected_income +\
                                              params.d2d.omega*mu*act_income + \
                                              params.d2d.omega*(1-mu)*ave_income
    sim.income.expected['run {}'.format(run_id+1)] = sim.inData.vehicles.expected_income.copy()
    sim.income.actual['run {}'.format(run_id)] = act_income.copy()
    
    
    
    
# def driver_opt_out(veh, **kwargs): # user defined function to represent agent decisions
        
#     sim = veh.sim
#     params = sim.params
#     expected_income = params.d2d.ini_exp_income if len(sim.res) == 0 else sim.res[len(sim.res)-1].veh_exp.EXPECTED_INC.loc[veh.id]
#     working_U = expected_income
#     not_working_U = veh.veh.res_wage
#     working_P = (math.exp(working_U))/(math.exp(working_U)+math.exp(not_working_U))
#     if  working_P < random.uniform(0, 1): # probabilistic
#     #if  working_U < not_working_U: # deterministic
#         return True
#     else:
#         return False