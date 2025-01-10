from .traveller import travellerEvent
from .driver import driverEvent
import numpy as np
import pandas as pd
import math
from numpy import log as ln
import random
from statsmodels.tsa.stattools import adfuller
    

def S_driver_opt_out_TS(veh, **kwargs): # user defined function to represent agent participation choice
    """
    This function depends on stochasticity and heterogeneity of model
    """
    sim = veh.sim
    params = sim.params
    run_id = sim.run_id
    
    if params.lock_out and run_id>0:
        veh.veh.P1_U = sim.res[run_id-1].veh_exp.P1_U.loc[veh.id]
        veh.veh.P2_U = sim.res[run_id-1].veh_exp.P2_U.loc[veh.id]
        veh.platform_id = sim.res[run_id-1].veh_exp.TOM_platform_id.loc[veh.id]
        veh.veh.platform = sim.res[run_id-1].veh_exp.TOM_platform_id.loc[veh.id]
        veh.platform = veh.sim.plats[veh.veh.platform]
        
        return sim.res[run_id-1].veh_exp.TOM_OUT.loc[veh.id]
    
    else:
        RW_U = params.d2d.B_Experience*0.5 + params.d2d.B_Marketing*0.5 + params.d2d.B_WOM*0.5
        alts_u = {'RW': RW_U, 'plats':{'P1': 'Nan', 'P2': 'Nan'}}
        alts_x = {'RW': 'Nan', 'P1': 'Nan', 'P2': 'Nan', 'plats':{'P1': 'Nan', 'P2': 'Nan'}}
        alts_p = {'RW': 'Nan', 'plats':{'P1': 'Nan', 'P2': 'Nan'}}
        nPM = 0

        p1_informed = False if run_id == 0 else sim.res[run_id-1].veh_exp.P1_INFORMED.loc[veh.id]
        p2_informed = False if run_id == 0 else sim.res[run_id-1].veh_exp.P2_INFORMED.loc[veh.id]
        # p1_informed = True; p2_informed = True

        P1_hate = 0 if run_id == 0 else sim.res[run_id-1].veh_exp.P1_hate.loc[veh.id]
        P2_hate = 0 if run_id == 0 else sim.res[run_id-1].veh_exp.P2_hate.loc[veh.id]

        p1_informed = P1_hate <= 0 and p1_informed
        p2_informed = P2_hate <= 0 and p2_informed

        # P1 utilization-------------------------------------------------------------------
        if p1_informed==True:

            P1_EXPERIENCE_U = 0 if run_id == 0 else sim.res[run_id-1].veh_exp.P1_EXPERIENCE_U.loc[veh.id]
            P1_MARKETING_U = 0 if run_id == 0 else sim.res[run_id-1].veh_exp.P1_MARKETING_U.loc[veh.id]
            P1_WOM_U = 0 if run_id == 0 else sim.res[run_id-1].veh_exp.P1_WOM_U.loc[veh.id]

            P1_U = params.d2d.B_Experience*P1_EXPERIENCE_U + params.d2d.B_Marketing*P1_MARKETING_U + params.d2d.B_WOM*P1_WOM_U
            alts_u['plats']['P1'] = P1_U
            nPM += 1
            veh.veh.P1_U = P1_U

        #P2 utilization---------------------------------------------------------------------
        if p2_informed==True:

            P2_EXPERIENCE_U = 0 if run_id == 0 else sim.res[run_id-1].veh_exp.P2_EXPERIENCE_U.loc[veh.id]
            P2_MARKETING_U = 0 if run_id == 0 else sim.res[run_id-1].veh_exp.P2_MARKETING_U.loc[veh.id]
            P2_WOM_U = 0 if run_id == 0 else sim.res[run_id-1].veh_exp.P2_WOM_U.loc[veh.id]

            P2_U = params.d2d.B_Experience*P2_EXPERIENCE_U + params.d2d.B_Marketing*P2_MARKETING_U + params.d2d.B_WOM*P2_WOM_U
            alts_u['plats']['P2'] = P2_U
            nPM += 1
            veh.veh.P2_U = P2_U
        #------------------------------------------------------------------------------------

        if nPM == 0: # there is only one choice RW
            return True  

        else: # Nested Logit (NL)
            for p in alts_u['plats']: # calculate X in platform nest
                alts_x['plats'][p] = math.exp(params.d2d.mn*alts_u['plats'][p]) if alts_u['plats'][p]!= 'Nan' else 0

            w = (1/params.d2d.mn)*ln(sum(alts_x['plats'].values()))    # RH satisfaction - Logsum
            for p in alts_u['plats']: # calculate probability in platform nest
                alts_p['plats'][p] = alts_x['plats'][p]/sum(alts_x['plats'].values())
                alts_u[p] = w if alts_u['plats'][p]!= 'Nan' else 'Nan'

            alts_x.pop('plats', None)
            for alt in alts_x:
                alts_x[alt] = math.exp(params.d2d.m*alts_u[alt]) if alts_u[alt]!= 'Nan' else 0

            x_w = math.exp(params.d2d.m*w)
            for alt in alts_x:
                alts_p[alt] = (alts_x[alt]/(alts_x['RW'] + x_w))*alts_p['plats'][alt] if alt!='RW' else alts_x[alt]/(alts_x['RW'] + x_w)

        #-----------------------------------------------------------------------------------------
        rand_v = random.uniform(0,1)

        if rand_v <= alts_p['RW']:
            return True  # opts for RW
        elif rand_v <= alts_p['RW'] + alts_p['P1']:
            veh.platform_id = 1
            veh.veh.platform = 1  # opts for platform number 1
            veh.platform = veh.sim.plats[veh.veh.platform]
            return False
        else:
            veh.platform_id = 2
            veh.veh.platform = 2  # opts for platform number 2
            veh.platform = veh.sim.plats[veh.veh.platform]
            return False

def S_traveller_opt_out_TS(pax, **kwargs):
    
    sim = pax.sim
    params = sim.params
    run_id = sim.run_id
    PT_U = params.d2d.B_Experience*0.5 + params.d2d.B_Marketing*0.5 + params.d2d.B_WOM*0.5
    
    alts_u = {'PT': PT_U, 'plats':{'P1': 'Nan', 'P2': 'Nan'}}
    alts_x = {'PT': 'Nan', 'P1': 'Nan', 'P2': 'Nan', 'plats':{'P1': 'Nan', 'P2': 'Nan'}}
    alts_p = {'PT': 'Nan', 'plats':{'P1': 'Nan', 'P2': 'Nan'}}
    nPM = 0
    
    p1_informed = False if run_id == 0 else sim.res[run_id-1].pax_exp.P1_INFORMED.loc[pax.id]
    p2_informed = False if run_id == 0 else sim.res[run_id-1].pax_exp.P2_INFORMED.loc[pax.id]
    # p1_informed = True; p2_informed = True
    
    P1_hate = 0 if run_id == 0 else sim.res[run_id-1].pax_exp.P1_hate.loc[pax.id]
    P2_hate = 0 if run_id == 0 else sim.res[run_id-1].pax_exp.P2_hate.loc[pax.id]
    
    p1_informed = P1_hate <= 0 and p1_informed
    p2_informed = P2_hate <= 0 and p2_informed
    
    #P1 utilization---------------------------------------------------------------------
    if p1_informed==True:
        
        P1_EXPERIENCE_U = 0 if run_id == 0 else sim.res[run_id-1].pax_exp.P1_EXPERIENCE_U.loc[pax.id]    
        P1_MARKETING_U = 0 if run_id == 0 else sim.res[run_id-1].pax_exp.P1_MARKETING_U.loc[pax.id]
        P1_WOM_U = 0 if run_id == 0 else sim.res[run_id-1].pax_exp.P1_WOM_U.loc[pax.id]
    
        P1_U = params.d2d.B_Experience*P1_EXPERIENCE_U + params.d2d.B_Marketing*P1_MARKETING_U + params.d2d.B_WOM*P1_WOM_U
        alts_u['plats']['P1'] = P1_U
        nPM += 1
        pax.pax.P1_U = P1_U
        
    #P2 utilization---------------------------------------------------------------------
    if p2_informed==True:
        
        P2_EXPERIENCE_U = 0 if run_id == 0 else sim.res[run_id-1].pax_exp.P2_EXPERIENCE_U.loc[pax.id]    
        P2_MARKETING_U = 0 if run_id == 0 else sim.res[run_id-1].pax_exp.P2_MARKETING_U.loc[pax.id]
        P2_WOM_U = 0 if run_id == 0 else sim.res[run_id-1].pax_exp.P2_WOM_U.loc[pax.id]
    
        P2_U = params.d2d.B_Experience*P2_EXPERIENCE_U + params.d2d.B_Marketing*P2_MARKETING_U + params.d2d.B_WOM*P2_WOM_U
        alts_u['plats']['P2'] = P2_U
        nPM += 1
        pax.pax.P2_U = P2_U
        
    # -----------------------------------------------------------------------------------
    if nPM == 0: # there is only one choice RW
        return True   
            
    else: # Nested Logit (NL)
        for p in alts_u['plats']: # calculate X in platform nest
            alts_x['plats'][p] = math.exp(params.d2d.mn*alts_u['plats'][p]) if alts_u['plats'][p]!= 'Nan' else 0

        w = (1/params.d2d.mn)*ln(sum(alts_x['plats'].values()))    # RH satisfaction - Logsum
        for p in alts_u['plats']: # calculate probability in platform nest
            alts_p['plats'][p] = alts_x['plats'][p]/sum(alts_x['plats'].values())
            alts_u[p] = w if alts_u['plats'][p]!= 'Nan' else 'Nan'

        alts_x.pop('plats', None)
        for alt in alts_x:
            alts_x[alt] = math.exp(params.d2d.m*alts_u[alt]) if alts_u[alt]!= 'Nan' else 0

        x_w = math.exp(params.d2d.m*w)
        for alt in alts_x:
            alts_p[alt] = (alts_x[alt]/(alts_x['PT'] + x_w))*alts_p['plats'][alt] if alt!='PT' else alts_x[alt]/(alts_x['PT'] + x_w)
    
    #-----------------------------------------------------------------------------------
    rand_v = random.uniform(0,1)

    if rand_v <= alts_p['PT']:
        return True  # opts for PT
    elif rand_v <= alts_p['PT'] + alts_p['P1']:
        pax.platform_id = 1
        pax.pax.platform = 1  # opts for platform number 1
        pax.platform = pax.sim.plats[pax.pax.platform]
        return False
    else:
        pax.platform_id = 2
        pax.pax.platform = 2  # opts for platform number 2
        pax.platform = pax.sim.plats[pax.pax.platform]
        return False

    
def d2d_kpi_veh(*args,**kwargs):

    """
    calculate vehicle KPIs (global and individual)
    apdates driver expected income
    """
    sim =  kwargs.get('sim', None)
    params = sim.params
    platforms = sim.platforms
    run_id = kwargs.get('run_id', None)
    simrun = sim.runs[run_id]
    # sub_lim = params.simTime*params.VoT
    sub_lim = params.simTime*params.min_wage # 14.4
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
    ret['IDLE_TIME'] = ret.RECEIVES_REQUEST + ret.ENDS_SHIFT
    ret['OUT'] = DECIDES_NOT_TO_DRIVE
    ret['OUT'] = ~ret['OUT'].isnull()
    ret['PICKUP_DIST'] = ret.ARRIVES_AT_PICKUP*(params.speeds.ride/1000)  # in km
    ret['DRIVING_TIME'] = ret.ARRIVES_AT_PICKUP + ret.ARRIVES_AT_DROPOFF
    ret['DRIVING_DIST'] = ret['DRIVING_TIME']*(params.speeds.ride/1000)  #here we assume the speed is constant on the network

    d = df[df['event_s']=='ARRIVES_AT_DROPOFF']
    if len(d) != 0:
        d['TRIP_FARE'] = d.apply(lambda row: max(row['dt'] * (params.speeds.ride/1000) * platforms.loc[sim.vehs[row.veh].platform_id].fare + platforms.loc[sim.vehs[row.veh].platform_id].base_fare, platforms.loc[sim.vehs[row.veh].platform_id].min_fare), axis=1)
        ret['TRIP_FARE'] = d.groupby(['veh']).sum().TRIP_FARE
    else:
        ret['TRIP_FARE'] = 0
    
    ret['platform_id'] = ret.apply(lambda row: sim.vehs[row.name].platform_id if row.OUT==False else 0, axis=1) # zero means PT
    ret['REVENUE'] = ret.apply(lambda row: row.TRIP_FARE*(1-platforms.loc[row.platform_id].comm_rate) if row.platform_id>0 else 0, axis=1)    
    ret['COMMISSION'] = ret['TRIP_FARE']-ret['REVENUE']
    ret['COST'] = ret['DRIVING_DIST'] * (params.d2d.fuel_cost) # Operating Cost (OC)
    ret['PROFIT'] = ret['REVENUE'] - ret['COST']
    
    if params.min_wage_sub:
        ret['MIN_WAGE_SUB'] = ret.apply(lambda row: sub_lim-row.PROFIT if row.platform_id>0 and row.PROFIT<sub_lim else 0, axis=1)
        ret['PROFIT'] = ret['PROFIT'] + ret['MIN_WAGE_SUB']
    else:
        ret['MIN_WAGE_SUB'] = 0
    ret['mu'] = ret.apply(lambda row: 1 if row['OUT'] == False else 0, axis=1)
    ret['nDAYS_WORKED'] = ret['mu'] if run_id == 0 else sim.res[run_id-1].veh_exp.nDAYS_WORKED + ret['mu']
    ret.fillna(0, inplace=True)
    
    # Driver adaptation (learning) ------------------------------------ #
    ret['ACTUAL_INC'] = ret.PROFIT    
    
    #====================================================================
    ret['P1_INFORMED'] = False if run_id == 0 else sim.res[run_id-1].veh_exp.P1_INFORMED
    ret['P2_INFORMED'] = False if run_id == 0 else sim.res[run_id-1].veh_exp.P2_INFORMED
    
    #-------------------------------------------------------
    """ Utility gained through experience"""

    ret['inc_dif'] = ret.apply(lambda row: 0 if row.mu==0 else (params.d2d.res_wage-row['ACTUAL_INC'])/params.d2d.res_wage, axis=1)
    
    # P1-------------------------------
    ret['pre_P1_EXPERIENCE_U'] = params.d2d.Eini_att if run_id == 0 else sim.res[run_id-1].veh_exp.P1_EXPERIENCE_U
    ret['P1_EXPERIENCE_U'] = params.d2d.Eini_att if run_id == 0 else sim.res[run_id-1].veh_exp.P1_EXPERIENCE_U
    ret_p1 = ret[ret['platform_id']==1]
    if len(ret_p1)>0:
        ret_p1['P1_EXPERIENCE_U'] = ret_p1.apply(lambda row: min((1-1e-2), max(1/(1+math.exp(params.d2d.learning_d*(ln((1/row.pre_P1_EXPERIENCE_U)-1)+params.d2d.adj_s*row.inc_dif))), 1e-2)), axis=1)
        ret.update(ret_p1)
    
    # P2-------------------------------
    ret['pre_P2_EXPERIENCE_U'] = params.d2d.Eini_att if run_id == 0 else sim.res[run_id-1].veh_exp.P2_EXPERIENCE_U
    ret['P2_EXPERIENCE_U'] = params.d2d.Eini_att if run_id == 0 else sim.res[run_id-1].veh_exp.P2_EXPERIENCE_U
    ret_p2 = ret[ret['platform_id']==2]
    if len(ret_p2)>0:
        ret_p2['P2_EXPERIENCE_U'] = ret_p2.apply(lambda row: min((1-1e-2), max(1/(1+math.exp(params.d2d.learning_d*(ln((1/row.pre_P2_EXPERIENCE_U)-1)+params.d2d.adj_s*row.inc_dif))), 1e-2)), axis=1)
        ret.update(ret_p2)
    
    #--------------------------------------------------------
    """ Utility gained through marketing"""
    # P1-------------------------------
    ret['pre_P1_MARKETING_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].veh_exp.P1_MARKETING_U
    ret['P1_MARKETING_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].veh_exp.P1_MARKETING_U
    if platforms.daily_marketing[1]==True:
        ret_p1 = ret.sample(int(params.d2d.diffusion_speed*params.nV))
        ret_p1['P1_MARKETING_U'] = ret_p1.apply(lambda row: min((1-1e-2),
                                        max(1/(1+math.exp(params.d2d.learning_d*(ln((1/row.pre_P1_MARKETING_U)-1)+row.pre_P1_MARKETING_U-1))), 1e-2)), axis=1)
        ret_p1['P1_INFORMED'] = True
        ret.update(ret_p1)
    
    # P2-------------------------------
    ret['pre_P2_MARKETING_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].veh_exp.P2_MARKETING_U
    ret['P2_MARKETING_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].veh_exp.P2_MARKETING_U
    if platforms.daily_marketing[2]==True:
        ret_p2 = ret.sample(int(params.d2d.diffusion_speed*params.nV))
        ret_p2['P2_MARKETING_U'] = ret_p2.apply(lambda row: min((1-1e-2),
                                        max(1/(1+math.exp(params.d2d.learning_d*(ln((1/row.pre_P2_MARKETING_U)-1)+row.pre_P2_MARKETING_U-1))), 1e-2)), axis=1)
        ret_p2['P2_INFORMED'] = True
        ret.update(ret_p2)
        
    #--------------------------------------------------------
    """ Utility gained through Word of Mouth (WOM)"""
    
    ret['pre_P1_WOM_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].veh_exp.P1_WOM_U
    ret['P1_WOM_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].veh_exp.P1_WOM_U
    
    ret['pre_P2_WOM_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].veh_exp.P2_WOM_U
    ret['P2_WOM_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].veh_exp.P2_WOM_U
    
    v_list = [v for v in range(1, params.nV+1)]
    selected_v = random.sample(v_list, int(params.d2d.diffusion_speed*params.nV))
    tuples = []
    # for v in range(1, params.nV+1):
    for v in selected_v:
        if v in v_list:
            v_list.remove(v)
            interlocutor = random.choice(v_list)
            v_list.remove(interlocutor)
            tuples.append((v,interlocutor))
    
    for tup in tuples:
        v1 = tup[0]
        v2 = tup[1]
        
        # P1-------------------------------
        ret['P1_WOM_U'].loc[v1] = 1/(1+math.exp(params.d2d.learning_d*(ln((1/ret['pre_P1_WOM_U'].loc[v1])-1)+ret['pre_P1_WOM_U'].loc[v1]-sim.vehs[v2].veh.P1_U)))
        ret['P1_WOM_U'].loc[v2] = 1/(1+math.exp(params.d2d.learning_d*(ln((1/ret['pre_P1_WOM_U'].loc[v2])-1)+ret['pre_P1_WOM_U'].loc[v2]-sim.vehs[v1].veh.P1_U)))
        if (ret['P1_INFORMED'].loc[v1] == False and ret['P1_INFORMED'].loc[v2] == True) | (ret['P1_INFORMED'].loc[v2] == False and ret['P1_INFORMED'].loc[v1] == True):
            ret['P1_INFORMED'].loc[v1] = True
            ret['P1_INFORMED'].loc[v2] = True
            
        # P2------------------------------- 
        ret['P2_WOM_U'].loc[v1] = 1/(1+math.exp(params.d2d.learning_d*(ln((1/ret['pre_P2_WOM_U'].loc[v1])-1)+ret['pre_P2_WOM_U'].loc[v1]-sim.vehs[v2].veh.P2_U)))
        ret['P2_WOM_U'].loc[v2] = 1/(1+math.exp(params.d2d.learning_d*(ln((1/ret['pre_P2_WOM_U'].loc[v2])-1)+ret['pre_P2_WOM_U'].loc[v2]-sim.vehs[v1].veh.P2_U)))
        if (ret['P2_INFORMED'].loc[v1] == False and ret['P2_INFORMED'].loc[v2] == True) | (ret['P2_INFORMED'].loc[v2] == False and ret['P2_INFORMED'].loc[v1] == True):
            ret['P2_INFORMED'].loc[v1] = True
            ret['P2_INFORMED'].loc[v2] = True
            
    # punishment -------------------------------------------------------    
    if run_id==0:
        ret['P1_hate'] = 0; ret['P2_hate'] = 0
    else:
        ret['P1_hate'] = ret.apply(lambda row: determine_p1_hate(row, run_id, sim.res[run_id-1].veh_exp), axis=1)
        ret['P2_hate'] = ret.apply(lambda row: determine_p2_hate(row, run_id, sim.res[run_id-1].veh_exp), axis=1)
    
    ret.fillna(0, inplace=True)
    
    # Lock out strategy -------------------------------------------------
    ret['P1_U'] = float('nan'); ret['P2_U'] = float('nan')
    ret['TOM_OUT'] = float('nan'); ret['TOM_platform_id'] = 1
    
    P1_current_pax = 0; P2_current_pax = 0
    pax_list = list(simrun.trips[simrun.trips.event=='REQUESTS_RIDE']['pax'])
    for traveller in range(0, params.nP):
        if traveller in pax_list and sim.pax[traveller].platform_id == 1:
            P1_current_pax += 1
        if traveller in pax_list and sim.pax[traveller].platform_id == 2:
            P2_current_pax += 1
    P_current_pax = [P1_current_pax, P2_current_pax]
            
    if params.lock_out:
        ret['TOM_OUT'] = ret.apply(lambda row: lock_out_part_1(row, sim, ret), axis=1)
        if run_id > 0:
            ret['TOM_OUT'] = ret.apply(lambda row: lock_out_part_2(row, sim, ret, P_current_pax), axis=1)
    
    #===================================================================
    ret = ret[['nRIDES','nREJECTED', 'nDAYS_WORKED', 'DRIVING_TIME', 'IDLE_TIME', 'PICKUP_DIST',
               'DRIVING_DIST','REVENUE','COST','COMMISSION','TRIP_FARE','ACTUAL_INC','OUT','mu',
               'P1_INFORMED','P2_INFORMED','P1_EXPERIENCE_U','P2_EXPERIENCE_U','P1_MARKETING_U',
               'P2_MARKETING_U', 'P1_WOM_U', 'P2_WOM_U','inc_dif', 'platform_id','P1_hate', 'P2_hate',
               'MIN_WAGE_SUB', 'TOM_OUT', 'TOM_platform_id', 'P1_U', 'P2_U'] + 
              [_.name for _ in driverEvent]]
    ret.index.name = 'veh'
    
    # KPIs
    kpi = ret.agg(['sum', 'mean', 'std'])
    kpi['nV'] = ret.shape[0]
    return {'veh_exp': ret, 'veh_kpi': kpi}
    
##########################################################
def lock_out_part_2(row, sim, ret, P_current_pax):
    
    run_id = sim.run_id
    veh_id = row.name
    pax_per_veh = 10 
        
    if row.TOM_OUT:
        return row.TOM_OUT
    
    else:
        if row.TOM_platform_id==1:
            P1_current_veh = len(ret[(ret.TOM_OUT==False) & (ret.TOM_platform_id==1)])
            P1_required_veh = 1 + (P_current_pax[0]/pax_per_veh)
            
            if P1_current_veh > P1_required_veh:
                return bool(not veh_id in ret.P1_U.sort_values(ascending=False).index[:round(P1_required_veh)])
            else:
                return row.TOM_OUT
        
        if row.TOM_platform_id==2:
            P2_current_veh = len(ret[(ret.TOM_OUT==False) & (ret.TOM_platform_id==2)])
            P2_required_veh = 1 + (P_current_pax[1]/pax_per_veh)
            
            if P2_current_veh > P2_required_veh:
                return bool(not veh_id in ret.P2_U.sort_values(ascending=False).index[:round(P2_required_veh)])
            else:
                return row.TOM_OUT
    

def lock_out_part_1(row, sim, ret):

    params = sim.params
    run_id = sim.run_id
    veh_id = row.name
    veh = sim.vehs[veh_id]
    RW_U = params.d2d.B_Experience*0.5 + params.d2d.B_Marketing*0.5 + params.d2d.B_WOM*0.5
    alts_u = {'RW': RW_U, 'plats':{'P1': 'Nan', 'P2': 'Nan'}}
    alts_x = {'RW': 'Nan', 'P1': 'Nan', 'P2': 'Nan', 'plats':{'P1': 'Nan', 'P2': 'Nan'}}
    alts_p = {'RW': 'Nan', 'plats':{'P1': 'Nan', 'P2': 'Nan'}}
    nPM = 0

    p1_informed = ret.P1_INFORMED.loc[veh_id]
    p2_informed = ret.P2_INFORMED.loc[veh_id]
    
    P1_hate = ret.P1_hate.loc[veh_id]
    P2_hate = ret.P2_hate.loc[veh_id]
    
    p1_informed = P1_hate <= 0 and p1_informed
    p2_informed = P2_hate <= 0 and p2_informed
    
    # P1 utilization-------------------------------------------------------------------
    if p1_informed==True:
        
        P1_EXPERIENCE_U = ret.P1_EXPERIENCE_U.loc[veh_id]
        P1_MARKETING_U = ret.P1_MARKETING_U.loc[veh_id]
        P1_WOM_U = ret.P1_WOM_U.loc[veh_id]
        
        P1_U = params.d2d.B_Experience*P1_EXPERIENCE_U + params.d2d.B_Marketing*P1_MARKETING_U + params.d2d.B_WOM*P1_WOM_U
        alts_u['plats']['P1'] = P1_U
        ret.at[veh_id, 'P1_U'] = P1_U
        nPM += 1
    
    #P2 utilization---------------------------------------------------------------------
    if p2_informed==True:

        P2_EXPERIENCE_U = ret.P2_EXPERIENCE_U.loc[veh_id]
        P2_MARKETING_U = ret.P2_MARKETING_U.loc[veh_id]
        P2_WOM_U = ret.P2_WOM_U.loc[veh_id]
        
        P2_U = params.d2d.B_Experience*P2_EXPERIENCE_U + params.d2d.B_Marketing*P2_MARKETING_U + params.d2d.B_WOM*P2_WOM_U
        alts_u['plats']['P2'] = P2_U
        ret.at[veh_id, 'P2_U'] = P2_U
        nPM += 1
        
    #------------------------------------------------------------------------------------
    
    if nPM == 0: # there is only one choice RW
        return True  
            
    else: # Nested Logit (NL)
        for p in alts_u['plats']: # calculate X in platform nest
            alts_x['plats'][p] = math.exp(params.d2d.mn*alts_u['plats'][p]) if alts_u['plats'][p]!= 'Nan' else 0

        w = (1/params.d2d.mn)*ln(sum(alts_x['plats'].values()))    # RH satisfaction - Logsum
        for p in alts_u['plats']: # calculate probability in platform nest
            alts_p['plats'][p] = alts_x['plats'][p]/sum(alts_x['plats'].values())
            alts_u[p] = w if alts_u['plats'][p]!= 'Nan' else 'Nan'

        alts_x.pop('plats', None)
        for alt in alts_x:
            alts_x[alt] = math.exp(params.d2d.m*alts_u[alt]) if alts_u[alt]!= 'Nan' else 0

        x_w = math.exp(params.d2d.m*w)
        for alt in alts_x:
            alts_p[alt] = (alts_x[alt]/(alts_x['RW'] + x_w))*alts_p['plats'][alt] if alt!='RW' else alts_x[alt]/(alts_x['RW'] + x_w)

    #-----------------------------------------------------------------------------------------
    rand_v = random.uniform(0,1)

    if rand_v <= alts_p['RW']:
        return True  # opts for RW
    elif rand_v <= alts_p['RW'] + alts_p['P1']:
        ret.at[veh_id, 'TOM_platform_id'] = 1
        return False
    else:
        ret.at[veh_id, 'TOM_platform_id'] = 2
        return False


################################################################################################

def d2d_kpi_pax(*args,**kwargs):
    # calculate passenger indicators (global and individual)

    sim = kwargs.get('sim', None)
    params = sim.params
    run_id = kwargs.get('run_id', None)
    retV = kwargs.get('retV', None)
    simrun = sim.runs[run_id]
    platforms = sim.platforms
    paxindex = sim.inData.passengers.index
    df = simrun['trips'].copy()  # results of previous simulation
    unfulfilled_requests = list(df[df['event']=='LOSES_PATIENCE'].pax)
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
    
    ret['fulfilled'] = ret.apply(lambda row: True if row['ARRIVES_AT_DROPOFF']>0 else False, axis=1)
    ret['mu'] = ret.apply(lambda row: 1 if row['fulfilled'] == True else 0, axis=1)
    ret['wu'] = ret.apply(lambda row: 1 if row['OUT'] == False else 0, axis=1)
    ret['nDAYS_HAILED'] = ret['mu'] if run_id == 0 else sim.res[run_id-1].pax_exp.nDAYS_HAILED + ret['mu']
    ret['nDAYS_TRY'] = ret['wu'] if run_id == 0 else sim.res[run_id-1].pax_exp.nDAYS_TRY + ret['wu']
    ret['TRAVEL'] = ret['ARRIVES_AT_DROPOFF']  # time with traveller (paid time)
    ret['ACTUAL_WT'] = (ret['RECEIVES_OFFER'] + ret['MEETS_DRIVER_AT_PICKUP'] + ret.get('LOSES_PATIENCE', 0))/60  #in minute
    ret['MATCHING_T'] = (ret['RECEIVES_OFFER'] + ret.get('LOSES_PATIENCE', 0))/60  #in minute
    ret['OPERATIONS'] = ret['ACCEPTS_OFFER'] + ret['DEPARTS_FROM_PICKUP'] + ret['SETS_OFF_FOR_DEST']
    ret['platform_id'] = ret.apply(lambda row: sim.pax[row.name].platform_id if row.OUT==False else 0, axis=1) # zero means pt  
    #====================================================================
    # Rafal & Farnoud (2022)
    
    ret['plat_revenue'] = float('nan')
    ret['P1_INFORMED'] = False if run_id == 0 else sim.res[run_id-1].pax_exp.P1_INFORMED
    ret['P2_INFORMED'] = False if run_id == 0 else sim.res[run_id-1].pax_exp.P2_INFORMED
    #-------------------------------------------------------
    """ Utility gained through experience"""
    ret['PT_U'] = ret.apply(lambda row: sim.pax[row.name].pax.u_PT, axis=1)
    ret['P_U'] = ret.apply(lambda row: P_U_func(row, sim, unfulfilled_requests, ret), axis=1)
    ret['U_dif'] = ret.apply(lambda row: 0 if row.mu==0 else (row['PT_U']-row['P_U'])/abs(row['PT_U']), axis=1)
    # P1-------------------------------
    ret['pre_P1_EXPERIENCE_U'] = params.d2d.Eini_att if run_id == 0 else sim.res[run_id-1].pax_exp.P1_EXPERIENCE_U
    ret['P1_EXPERIENCE_U'] = params.d2d.Eini_att if run_id == 0 else sim.res[run_id-1].pax_exp.P1_EXPERIENCE_U
    
    ret_p1 = ret[ret['platform_id']==1]
    if len(ret_p1)>0:
        ret_p1['P1_EXPERIENCE_U'] = ret_p1.apply(lambda row: min((1-1e-2), max(1/(1+math.exp(params.d2d.learning_d*(ln((1/row.pre_P1_EXPERIENCE_U)-1)+params.d2d.adj_s*row.U_dif))), 1e-2)), axis=1)
        ret.update(ret_p1)
    
    # P2-------------------------------
    ret['pre_P2_EXPERIENCE_U'] = params.d2d.Eini_att if run_id == 0 else sim.res[run_id-1].pax_exp.P2_EXPERIENCE_U
    ret['P2_EXPERIENCE_U'] = params.d2d.Eini_att if run_id == 0 else sim.res[run_id-1].pax_exp.P2_EXPERIENCE_U
    
    ret_p2 = ret[ret['platform_id']==2]
    if len(ret_p2)>0:
        ret_p2['P2_EXPERIENCE_U'] = ret_p2.apply(lambda row: min((1-1e-2), max(1/(1+math.exp(params.d2d.learning_d*(ln((1/row.pre_P2_EXPERIENCE_U)-1)+params.d2d.adj_s*row.U_dif))), 1e-2)), axis=1)
        ret.update(ret_p2)
    
    #--------------------------------------------------------
    """ Utility gained through marketing"""
    # P1-------------------------------
    ret['pre_P1_MARKETING_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].pax_exp.P1_MARKETING_U
    ret['P1_MARKETING_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].pax_exp.P1_MARKETING_U
    
    if platforms.daily_marketing[1]==True:
        ret_p1 = ret.sample(int(params.d2d.diffusion_speed*params.nP))
        ret_p1['P1_MARKETING_U'] = ret_p1.apply(lambda row: min((1-1e-2), max(1/(1+math.exp(params.d2d.learning_d*(ln((1/row.pre_P1_MARKETING_U)-1)+row.pre_P1_MARKETING_U-1))), 1e-2)), axis=1)
        ret_p1['P1_INFORMED'] = True
        ret.update(ret_p1)
        
    # P2-------------------------------
    ret['pre_P2_MARKETING_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].pax_exp.P2_MARKETING_U
    ret['P2_MARKETING_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].pax_exp.P2_MARKETING_U
    
    if platforms.daily_marketing[2]==True:
        ret_p2 = ret.sample(int(params.d2d.diffusion_speed*params.nP))
        ret_p2['P2_MARKETING_U'] = ret_p2.apply(lambda row: min((1-1e-2), max(1/(1+math.exp(params.d2d.learning_d*(ln((1/row.pre_P2_MARKETING_U)-1)+row.pre_P2_MARKETING_U-1))), 1e-2)), axis=1)
        ret_p2['P2_INFORMED'] = True
        ret.update(ret_p2)
        
    #--------------------------------------------------------
    """ Utility gained through Word of Mouth (WOM)"""
    
    # print(ret.P1_INFORMED)
    ret['pre_P1_WOM_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].pax_exp.P1_WOM_U
    ret['P1_WOM_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].pax_exp.P1_WOM_U
    
    ret['pre_P2_WOM_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].pax_exp.P2_WOM_U
    ret['P2_WOM_U'] = params.d2d.ini_att if run_id == 0 else sim.res[run_id-1].pax_exp.P2_WOM_U

    p_list = [p for p in range(0, params.nP)]
    selected_p = random.sample(p_list, int(params.d2d.diffusion_speed*params.nP))
    tuples = []
    # for p in range(0, params.nP):
    for p in selected_p:
        if p in p_list:
            p_list.remove(p)
            interlocutor = random.choice(p_list)
            p_list.remove(interlocutor)
            tuples.append((p,interlocutor))
    
    for tup in tuples:
        p1 = tup[0]
        p2 = tup[1]
        # P1-------------------------------
        ret['P1_WOM_U'].loc[p1] = 1/(1+math.exp(params.d2d.learning_d*(ln((1/ret['pre_P1_WOM_U'].loc[p1])-1)+ret['pre_P1_WOM_U'].loc[p1]-sim.pax[p2].pax.P1_U)))
        ret['P1_WOM_U'].loc[p2] = 1/(1+math.exp(params.d2d.learning_d*(ln((1/ret['pre_P1_WOM_U'].loc[p2])-1)+ret['pre_P1_WOM_U'].loc[p2]-sim.pax[p1].pax.P1_U)))
        if (ret['P1_INFORMED'].loc[p1] == False and ret['P1_INFORMED'].loc[p2] == True) | (ret['P1_INFORMED'].loc[p2] == False and ret['P1_INFORMED'].loc[p1] == True):
            ret['P1_INFORMED'].loc[p1] = True
            ret['P1_INFORMED'].loc[p2] = True
            
        # P2-------------------------------
        ret['P2_WOM_U'].loc[p1] = 1/(1+math.exp(params.d2d.learning_d*(ln((1/ret['pre_P2_WOM_U'].loc[p1])-1)+ret['pre_P2_WOM_U'].loc[p1]-sim.pax[p2].pax.P2_U)))
        ret['P2_WOM_U'].loc[p2] = 1/(1+math.exp(params.d2d.learning_d*(ln((1/ret['pre_P2_WOM_U'].loc[p2])-1)+ret['pre_P2_WOM_U'].loc[p2]-sim.pax[p1].pax.P2_U)))
        if (ret['P2_INFORMED'].loc[p1] == False and ret['P2_INFORMED'].loc[p2] == True) | (ret['P2_INFORMED'].loc[p2] == False and ret['P2_INFORMED'].loc[p1] == True):
            ret['P2_INFORMED'].loc[p1] = True
            ret['P2_INFORMED'].loc[p2] = True

    # punishment -------------------------------------------------------    
    if run_id==0:
        ret['P1_hate'] = 0; ret['P2_hate'] = 0
    else:
        ret['P1_hate'] = ret.apply(lambda row: determine_p1_hate(row, run_id, sim.res[run_id-1].pax_exp), axis=1)
        ret['P2_hate'] = ret.apply(lambda row: determine_p2_hate(row, run_id, sim.res[run_id-1].pax_exp), axis=1)

    # punishment -------------------------------------------------------
    
    ret.fillna(0, inplace=True)
    
    # Platform ---------------------------------------------------------
    initial_capital = params.initial_capital
    expense_per_day = params.expense_per_day
    df_plat = pd.DataFrame(columns=['platform_id', 'fare', 'revenue', 'profit', 'nP', 'nV']).set_index('platform_id')
    df_plat.at[1, 'revenue'] = ret[ret.platform_id==1].plat_revenue.sum()
    df_plat.at[2, 'revenue'] = ret[ret.platform_id==2].plat_revenue.sum()
    df_plat.at[1, 'min_wage_sub'] = retV[retV.platform_id==1].MIN_WAGE_SUB.sum()
    df_plat.at[2, 'min_wage_sub'] = retV[retV.platform_id==2].MIN_WAGE_SUB.sum()
    df_plat['profit'] = df_plat.revenue-expense_per_day-df_plat.min_wage_sub
    df_plat['remaining_capital'] = initial_capital+df_plat.profit if run_id == 0 else sim.res[run_id-1].plat_exp.remaining_capital+df_plat.profit
    
    df_plat.at[1, 'nP'] = len(ret[ret.platform_id==1])
    df_plat.at[2, 'nP'] = len(ret[ret.platform_id==2])
    df_plat.at[1, 'nV'] = len(retV[retV.platform_id==1])
    df_plat.at[2, 'nV'] = len(retV[retV.platform_id==2])
    df_plat['P_market_share'] = df_plat.nP/params.nP
    df_plat['V_market_share'] = df_plat.nV/params.nV
    df_plat['market_share'] = (df_plat.P_market_share + df_plat.V_market_share)/2
    df_plat.at[1, 'fare'] = sim.platforms.fare[1]
    df_plat.at[2, 'fare'] = sim.platforms.fare[2]
    # ===================================================================================== #

    ret = ret[['P_U','PT_U','ACTUAL_WT', 'U_dif','OUT','mu','wu','nDAYS_HAILED','nDAYS_TRY','P1_EXPERIENCE_U',
               'P2_EXPERIENCE_U','P1_MARKETING_U','P2_MARKETING_U','P1_WOM_U','P2_WOM_U',
               'P1_INFORMED', 'P2_INFORMED', 'platform_id', 'plat_revenue','MATCHING_T','P1_hate', 'P2_hate'] + [_.name for _ in travellerEvent]]
    
    ret.index.name = 'pax'
    kpi = ret.agg(['sum', 'mean', 'std'])
    kpi['nP'] = ret.shape[0]
    return {'pax_exp': ret, 'pax_kpi': kpi, 'plat_exp':df_plat}


def P_U_func(row, sim, unfulfilled_requests, ret):

    params = sim.params
    req = sim.pax[row.name].request
    platform_id = sim.pax[row.name].platform_id
    plat = sim.platforms.loc[platform_id]
    disc = 0
    
    if sim.pax[row.name].pax['P{}_U'.format(platform_id)] < 0.5:
        disc = plat.discount #####
    
    if row.name in unfulfilled_requests:
        hate = 3
    else:
        hate = 0
    
    if plat.fare > 0:
        fare = max(plat.get('base_fare',0) + plat.fare*req.dist/1000, plat.get('min_fare',0))
    else: 
        fare = 0
    disc_fare = (1-disc)*fare
    P_U = -(1+hate)*(disc_fare + (params.VoT/3600)*(params.d2d.B_inveh_time*req.ttrav.total_seconds() + params.d2d.B_exp_time*row.ACTUAL_WT*60))
    
    ret.at[row.name, 'plat_revenue'] = fare*(plat.comm_rate-disc) if ret.mu[row.name]==1 else 0

    return P_U

def PT_U_func(row, sim):

    params = sim.params
    req = sim.pax[row.name].request
    plat = sim.platforms.loc[1]
    inveh_time = (req.dist/params.PT_speed)/3600
    alt_fare = params.PT_fare*req.dist/1000
    PT_U = -params.d2d.B_fare*alt_fare -params.d2d.B_inveh_time*inveh_time- params.d2d.B_exp_time*inveh_time*0.25
    return PT_U

def PT_utility(requests,params):
    if 'walkDistance' in requests.columns:
        requests = requests
        walk_factor = 2
        wait_factor = 2
        transfer_penalty = 500
        requests['PT_fare'] = 1 + requests.transitTime * params.avg_speed/1000 * 0.175
        requests['u_PT'] = requests['PT_fare'] + \
                           requests.VoT * (walk_factor * requests.walkDistance / params.speeds.walk +
                                           wait_factor * requests.waitingTime +
                                           transfer_penalty * requests.transfers + requests.transitTime)
    return requests


def determine_p1_hate(row, run_id, exp):
    hate = 10
    if row['platform_id'] == 1 and run_id>0:
        if row['P1_EXPERIENCE_U'] == 1e-2:
            return hate
        else:
            return 0
    else:
        return max(exp['P1_hate'][row.name]-1, 0)

def determine_p2_hate(row, run_id, exp):
    hate = 10
    if row['platform_id'] == 2 and run_id>0:
        if row['P2_EXPERIENCE_U'] == 1e-2:
            return hate
        else:
            return 0
    else:
        return max(exp['P2_hate'][row.name]-1, 0)
    
def my_function(veh, **kwargs): # user defined function to represent agent decisions
    sim = veh.sim
    if  veh.veh.expected_income < sim.params.d2d.res_wage:
        return True
    else:
        return False
    
    
def S_driver_opt_out(veh, **kwargs): # user defined function to represent agent participation choice
    """
    This function depends on stochasticity and heterogeneity of model
    """
    sim = veh.sim
    params = sim.params
    RW_U = params.d2d.B_Experience*0.5 + params.d2d.B_Marketing*0.5 + params.d2d.B_WOM*0.5
    alts_u = {'RW': RW_U, 'plats':{'P1': 'Nan', 'P2': 'Nan'}}
    alts_x = {'RW': 'Nan', 'P1': 'Nan', 'P2': 'Nan', 'plats':{'P1': 'Nan', 'P2': 'Nan'}}
    alts_p = {'RW': 'Nan', 'plats':{'P1': 'Nan', 'P2': 'Nan'}}
    nPM = 0

    # P1 utilization-------------------------------------------------------------------
    # p1_informed = False if len(sim.res) == 0 else sim.res[len(sim.res)-1].veh_exp.P1_INFORMED.loc[veh.id]
    p1_informed = True
    if p1_informed==True:
        
        P1_EXPERIENCE_U = 0 if len(sim.res) == 0 else sim.res[len(sim.res)-1].veh_exp.P1_EXPERIENCE_U.loc[veh.id]
        P1_MARKETING_U = 0 if len(sim.res) == 0 else sim.res[len(sim.res)-1].veh_exp.P1_MARKETING_U.loc[veh.id]
        P1_WOM_U = 0 if len(sim.res) == 0 else sim.res[len(sim.res)-1].veh_exp.P1_WOM_U.loc[veh.id]
        
        P1_U = params.d2d.B_Experience*P1_EXPERIENCE_U + params.d2d.B_Marketing*P1_MARKETING_U + params.d2d.B_WOM*P1_WOM_U
        alts_u['plats']['P1'] = P1_U
        nPM += 1
        veh.veh.P1_U = P1_U
    
    #P2 utilization---------------------------------------------------------------------
    # p2_informed = False if len(sim.res) == 0 else sim.res[len(sim.res)-1].veh_exp.P2_INFORMED.loc[veh.id]
    p2_informed = True
    if p2_informed==True:

        P2_EXPERIENCE_U = 0 if len(sim.res) == 0 else sim.res[len(sim.res)-1].veh_exp.P2_EXPERIENCE_U.loc[veh.id]
        P2_MARKETING_U = 0 if len(sim.res) == 0 else sim.res[len(sim.res)-1].veh_exp.P2_MARKETING_U.loc[veh.id]
        P2_WOM_U = 0 if len(sim.res) == 0 else sim.res[len(sim.res)-1].veh_exp.P2_WOM_U.loc[veh.id]
        
        P2_U = params.d2d.B_Experience*P2_EXPERIENCE_U + params.d2d.B_Marketing*P2_MARKETING_U + params.d2d.B_WOM*P2_WOM_U
        alts_u['plats']['P2'] = P2_U
        nPM += 1
        veh.veh.P2_U = P2_U
    
    #=====================================================================================
    if nPM == 0: # there is only one choice RW
        return True  
            
    else: # Nested Logit (NL)
        for p in alts_u['plats']: # calculate X in platform nest
            alts_x['plats'][p] = math.exp(params.d2d.mn*alts_u['plats'][p]) if alts_u['plats'][p]!= 'Nan' else 0

        w = (1/params.d2d.mn)*ln(sum(alts_x['plats'].values()))    # RH satisfaction - Logsum
        for p in alts_u['plats']: # calculate probability in platform nest
            alts_p['plats'][p] = alts_x['plats'][p]/sum(alts_x['plats'].values())
            alts_u[p] = w if alts_u['plats'][p]!= 'Nan' else 'Nan'

        alts_x.pop('plats', None)
        for alt in alts_x:
            alts_x[alt] = math.exp(params.d2d.m*alts_u[alt]) if alts_u[alt]!= 'Nan' else 0

        x_w = math.exp(params.d2d.m*w)
        for alt in alts_x:
            alts_p[alt] = (alts_x[alt]/(alts_x['RW'] + x_w))*alts_p['plats'][alt] if alt!='RW' else alts_x[alt]/(alts_x['RW'] + x_w)

    #=====================================================================================
    rand_v = random.uniform(0,1)
    
    if rand_v <= alts_p['RW']:
        return True  # opts for RW
    elif rand_v <= alts_p['RW'] + alts_p['P1']:
        veh.platform_id = 1
        veh.veh.platform = 1  # opts for platform number 1
        veh.platform = veh.sim.plats[veh.veh.platform]
        return False
    else:
        veh.platform_id = 2
        veh.veh.platform = 2  # opts for platform number 2
        veh.platform = veh.sim.plats[veh.veh.platform]
        return False 
    
def S_traveller_opt_out(pax, **kwargs):
    
    sim = pax.sim
    params = sim.params
    PT_U = params.d2d.B_Experience*0.5 + params.d2d.B_Marketing*0.5 + params.d2d.B_WOM*0.5
    
    alts_u = {'PT': PT_U, 'plats':{'P1': 'Nan', 'P2': 'Nan'}}
    alts_x = {'PT': 'Nan', 'P1': 'Nan', 'P2': 'Nan', 'plats':{'P1': 'Nan', 'P2': 'Nan'}}
    alts_p = {'PT': 'Nan', 'plats':{'P1': 'Nan', 'P2': 'Nan'}}
    nPM = 0
    
    P1_hate = 0 if len(sim.res) == 0 else sim.res[len(sim.res)-1].pax_exp.P1_hate.loc[pax.id]
    P2_hate = 0 if len(sim.res) == 0 else sim.res[len(sim.res)-1].pax_exp.P2_hate.loc[pax.id]
    
    p1_informed = P1_hate <= 0
    p2_informed = P2_hate <= 0
    
    #P1 utilization---------------------------------------------------------------------
    # p1_informed = False if len(sim.res) == 0 else sim.res[len(sim.res)-1].pax_exp.P1_INFORMED.loc[pax.id]
    p1_informed = True
    if p1_informed==True:
        
        P1_EXPERIENCE_U = 0 if len(sim.res) == 0 else sim.res[len(sim.res)-1].pax_exp.P1_EXPERIENCE_U.loc[pax.id]    
        P1_MARKETING_U = 0 if len(sim.res) == 0 else sim.res[len(sim.res)-1].pax_exp.P1_MARKETING_U.loc[pax.id]
        P1_WOM_U = 0 if len(sim.res) == 0 else sim.res[len(sim.res)-1].pax_exp.P1_WOM_U.loc[pax.id]
    
        P1_U = params.d2d.B_Experience*P1_EXPERIENCE_U + params.d2d.B_Marketing*P1_MARKETING_U + params.d2d.B_WOM*P1_WOM_U
        alts_u['plats']['P1'] = P1_U
        nPM += 1
        pax.pax.P1_U = P1_U
        
    #P2 utilization---------------------------------------------------------------------
    # p2_informed = False if len(sim.res) == 0 else sim.res[len(sim.res)-1].pax_exp.P2_INFORMED.loc[pax.id]
    p2_informed = True
    if p2_informed==True:
        
        P2_EXPERIENCE_U = 0 if len(sim.res) == 0 else sim.res[len(sim.res)-1].pax_exp.P2_EXPERIENCE_U.loc[pax.id]    
        P2_MARKETING_U = 0 if len(sim.res) == 0 else sim.res[len(sim.res)-1].pax_exp.P2_MARKETING_U.loc[pax.id]
        P2_WOM_U = 0 if len(sim.res) == 0 else sim.res[len(sim.res)-1].pax_exp.P2_WOM_U.loc[pax.id]
    
        P2_U = params.d2d.B_Experience*P2_EXPERIENCE_U + params.d2d.B_Marketing*P2_MARKETING_U + params.d2d.B_WOM*P2_WOM_U
        alts_u['plats']['P2'] = P2_U
        nPM += 1
        pax.pax.P2_U = P2_U
        
    #=====================================================================================
    if nPM == 0: # there is only one choice RW
        return True   
            
    else: # Nested Logit (NL)
        for p in alts_u['plats']: # calculate X in platform nest
            alts_x['plats'][p] = math.exp(params.d2d.mn*alts_u['plats'][p]) if alts_u['plats'][p]!= 'Nan' else 0

        w = (1/params.d2d.mn)*ln(sum(alts_x['plats'].values()))    # RH satisfaction - Logsum
        for p in alts_u['plats']: # calculate probability in platform nest
            alts_p['plats'][p] = alts_x['plats'][p]/sum(alts_x['plats'].values())
            alts_u[p] = w if alts_u['plats'][p]!= 'Nan' else 'Nan'

        alts_x.pop('plats', None)
        for alt in alts_x:
            alts_x[alt] = math.exp(params.d2d.m*alts_u[alt]) if alts_u[alt]!= 'Nan' else 0

        x_w = math.exp(params.d2d.m*w)
        for alt in alts_x:
            alts_p[alt] = (alts_x[alt]/(alts_x['PT'] + x_w))*alts_p['plats'][alt] if alt!='PT' else alts_x[alt]/(alts_x['PT'] + x_w)
    
    #=====================================================================================
    rand_v = random.uniform(0,1)

    if rand_v <= alts_p['PT']:
        return True  # opts for PT
    elif rand_v <= alts_p['PT'] + alts_p['P1']:
        pax.platform_id = 1
        pax.pax.platform = 1  # opts for platform number 1
        pax.platform = pax.sim.plats[pax.pax.platform]
        return False
    else:
        pax.platform_id = 2
        pax.pax.platform = 2  # opts for platform number 2
        pax.platform = pax.sim.plats[pax.pax.platform]
        return False
