from .traveller import travellerEvent
from .driver import driverEvent
import pandas as pd
import numpy as np
from dotmap import DotMap
import math
import random as random

def imposed_delay(sim,veh_id):
    
    delay=len(sim.vehs[veh_id].lDECLINES)*15
    
    for j in sim.vehs[veh_id].lDECLINES:
        pax=pd.DataFrame(sim.pax[j].rides)
        for k in range(len(pax[pax['veh_id']==veh_id])):
            if pax.index[-1]!=pax[pax['veh_id']==veh_id].index[k]:
                delay=delay+pax['t'].loc[pax[pax['veh_id']==veh_id].index[k]+1]-pax['t'].loc[pax[pax['veh_id']==veh_id].index[k]]
        
        
    return delay

def RA_kpi_veh(*args,**kwargs):

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
    #P df = df[~(df.t == df.t_s)]  # filter for positive time periods only
    df['dt'] = df.t_s - df.t  # make time intervals
    ret = df.groupby(['veh', 'event_s'])['dt'].sum().unstack()  # aggreagted by vehicle and event
    ret.columns.name = None
    ret = ret.reindex(vehindex)  # update for vehicles with no record
    ret['nREQUESTS'] = df[df.event == driverEvent.RECEIVES_REQUEST.name].groupby(['veh']).size().reindex(ret.index)
    ret['nRIDES'] = df[df.event == driverEvent.ARRIVES_AT_DROPOFF.name].groupby(['veh']).size().reindex(ret.index)
    ret['nREJECTS'] = df[df.event==driverEvent.REJECTS_REQUEST.name].groupby(['veh']).size().reindex(ret.index)
    for status in driverEvent:
        if status.name not in ret.columns:
            ret[status.name] = 0
    DECIDES_NOT_TO_DRIVE.index = DECIDES_NOT_TO_DRIVE.values
    ret['OUT'] = DECIDES_NOT_TO_DRIVE
    ret['OUT'] = ~ret['OUT'].isnull()
    ret['PICKUP_TIME'] = ret.ARRIVES_AT_PICKUP 
    ret['AVE_PICKUP_DIST'] = (ret.ARRIVES_AT_PICKUP/ret.nRIDES) * (params.speeds.ride/1000)
    ret['TRIP_TIME'] = ret.ARRIVES_AT_DROPOFF
    ret['AVE_TRIP_DIST']= (ret.ARRIVES_AT_DROPOFF/ret.nRIDES) * (params.speeds.ride/1000)
    ret['DRIVING_TIME'] = ret.ARRIVES_AT_PICKUP + ret.ARRIVES_AT_DROPOFF #we assum there is no repositioning
    ret['DRIVING_DIST'] = ret['DRIVING_TIME']*(params.speeds.ride/1000)   #here we assume the speed is constant on the network
    cooling_t = 60*60*2
    ret['AVE_DRIVING_DIST']=ret['DRIVING_DIST']/ret['nRIDES']
    ret['IDLE_TIME'] = ret.apply(lambda row: row['RECEIVES_REQUEST'] if row['ENDS_SHIFT']<cooling_t 
                                 else row['RECEIVES_REQUEST'] + row['ENDS_SHIFT'] - cooling_t, axis=1)
    ret.fillna(0, inplace=True)  
    d = df[df['event_s']=='ARRIVES_AT_DROPOFF']
    
    surge_mps = {}; surge_fees = {}
    for v in range(1, params.nV+1):
        surge_mps = surge_mps | sim.vehs[v].surge_mps
        surge_fees = surge_fees | sim.vehs[v].surge_fees
    d['surge_mps'] = d.apply(lambda row: surge_mps[row.paxes[0]], axis=1)
    d['surge_fees'] = d.apply(lambda row: surge_fees[row.paxes[0]], axis=1)

    
    if len(d) != 0:
        d['REVENUE'] = d.apply(lambda row: (max(row['dt'] * (params.speeds.ride/1000) * params.platforms.fare + params.platforms.base_fare, params.platforms.min_fare))*row['surge_mps'], axis=1)*(1-params.platforms.comm_rate)
        ret['REVENUE'] = d.groupby(['veh']).sum().REVENUE
        ret['SURGE_FEES'] = d.groupby(['veh']).sum().surge_fees
        ret['AVE_SURGE_MP'] = d.groupby(['veh']).mean().surge_mps
    else:
        ret['REVENUE'] = 0
        ret['SURGE_FEES'] = 0 
        ret['AVE_SURGE_MP'] = 0
    # ret['surge_fee'] = d.groupby(['veh']).sum().surge_fee
    ret['REVENUE/hour']=ret['REVENUE']/params.simTime
    ret['COST'] = ret['DRIVING_DIST'] * (params.d2d.fuel_cost) # Operating Cost (OC)
    ret['PROFIT'] = ret['REVENUE'] - ret['COST']
    ret['PROFIT/hour']=ret['PROFIT']/params.simTime
    ret['ACCEPTANCE_RATE'] = (ret['nRIDES']/ret['nREQUESTS'])*100
    # add imposed delay------------------------------------------------
    # trips = simrun['trips']
    # trips = trips.reset_index().drop(["index"], axis=1)
    # i = 0
    # len_ = len(trips)
    # while i < len_:
    #     #print(i)
    #     if trips.loc[i]['event'] != 'IS_REJECTED_BY_VEHICLE':
    #         trips.drop(i, inplace=True)
    #     else:
    #         trips.at[i,'t'] = trips['t'].loc[i+1] - trips['t'].loc[i]
    #         i += 1
    #         trips.drop(i, inplace=True)
    #     i += 1
    # trips = trips.groupby(['veh_id']).sum()
    # ret['IMPOSED_DELAY'] = ret.apply(lambda row: 0 if row['nREJECTS'] ==0 else row['nREJECTS']*15 + 
    #                                  trips.loc[row.name]['t'], axis=1) # add the IMPOSED_DELAY column to ret
    # #--------------------------------------------------
    ret.replace([np.inf, -np.inf], np.nan, inplace=True)
    ret.fillna(0, inplace=True)  

    ret['veh']=ret.index
    ret['IMPOSED_DELAY'] = ret.apply(lambda row: imposed_delay(sim,row['veh']),axis=1)
    # ret['IMPOSED_DELAY'] = 0
    
    ret['AR']=ret['ACCEPTANCE_RATE'].apply(lambda x: '60 or less' if x<=60 else \
                                     '60-70' if x>60 and x<=70  else '70-80' if x>70 and x<=80  else '80-90' if x>80 and x<90 else '90-100')


    ret = ret[['ACCEPTANCE_RATE','PROFIT','PROFIT/hour','IDLE_TIME','nREQUESTS','nRIDES','nREJECTS','DRIVING_DIST',
     'AVE_DRIVING_DIST','DRIVING_TIME','AVE_PICKUP_DIST','AVE_TRIP_DIST','REVENUE',
     'REVENUE/hour','COST','IMPOSED_DELAY','AR','SURGE_FEES','AVE_SURGE_MP']]#+ [_.name for _ in driverEvent]]
    
    ret.index.name = 'veh'

    # KPIs
    kpi = ret.agg(['sum', 'mean', 'std'])
    kpi['nV'] = ret.shape[0]
    AR = ret.groupby(['AR']).describe().T
    #---------------------------------------------------------------------------------------
    plats = sim.platforms
    plats['profit'] = (ret.REVENUE.sum()/(1-params.platforms.comm_rate))*params.platforms.comm_rate
    
    #---------------------------------------------------------------------------------------
    
    return {'veh_exp': ret, 'veh_kpi': kpi, 'veh_AR': AR, 'platform':plats}


def RA_kpi_pax(*args,**kwargs):

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
    df = df[df.pax == df.pax_s]  # filter for the same pax only
    # df = df[~(df.t == df.t_s)]  # filter for positive time periods only
    df['dt'] = df.t_s - df.t  # make time intervals
    ret = df.groupby(['pax', 'event_s'])['dt'].sum().unstack()  # aggreagted by vehicle and event

    ret.columns.name = None
    ret = ret.reindex(paxindex)  # update for pax with no record
    ret.index.name = 'pax'
    ret['nREJECTS'] = df[df.event == travellerEvent.IS_REJECTED_BY_VEHICLE.name].groupby(['pax']).size().reindex(ret.index)
    ret['veh_id'] = df.groupby(['pax']).sum()['veh_id']
    ret['veh_id'].replace(0, 'Unfulfilled', inplace=True)
    ret = ret.fillna(0)

    for status in travellerEvent:
        if status.name not in ret.columns:
            ret[status.name] = 0  # cover all statuses
    PREFERS_OTHER_SERVICE.index = PREFERS_OTHER_SERVICE.values
    ret['OUT'] = PREFERS_OTHER_SERVICE
    ret['OUT'] = ~ret['OUT'].isnull()
    ret['LOST_PATIENCE'] = ret.apply(lambda row: False if row['REJECTS_OFFER']>0 or row['ARRIVES_AT_DROPOFF']>0 else True ,axis=1)  
    ret['TRAVEL_TIME'] = ret['ARRIVES_AT_DROPOFF']  # time with traveller (paid time)
    ret['WAIT_TIME'] = (ret['RECEIVES_OFFER'] + ret['MEETS_DRIVER_AT_PICKUP'] + ret.get('LOSES_PATIENCE', 0) + ret.get('REJECTS_OFFER', 0))
    ret['SURGE_MP'] = 0
    for pax_id in range(0, params.nP):
        try:
            ret.SURGE_MP.iloc[pax_id] = sim.pax[pax_id].offers[1]['surge_mp']
        except:
            ret.SURGE_MP.iloc[pax_id] = 'no_offer'
    ret['OFFER_REJECTED_BY_PAX'] = ret.apply(lambda row: True if row.REJECTS_OFFER>0 else False, axis=1)
    
    
    ret.fillna(0, inplace=True)

    ret = ret[['veh_id','WAIT_TIME','nREJECTS','TRAVEL_TIME','LOST_PATIENCE','OFFER_REJECTED_BY_PAX',
               'SURGE_MP']] #+ [_.name for _ in travellerEvent]]
    ret.index.name = 'pax'

    kpi = ret.agg(['sum', 'mean', 'std'])
    kpi['nP'] = ret.shape[0]
    return {'pax_exp': ret, 'pax_kpi': kpi}


def f_mode(pax, sim):
    surge_mp = pax.offers[1]['surge_mp']
    purchase_rate = sim.inData.surge_dict[surge_mp]
    # print('purchase_rate= ',purchase_rate)
    
    if random.random() > purchase_rate:
        sim.alt_mode_pax.append(pax.id)
        return True
    else:
        return False


def f_decline(veh, **kwargs):
    
    sim = veh.sim
    df = pd.DataFrame(veh.myrides)
    ASC = 1.810   #ASC    
    surge = veh.offers[1]['surge_fee']
    # print('here----',surge)
    # print('veh_id', veh.id)
    # print('surge= ',surge)
    d = veh.offers[1]['request']["origin"]                                                       #pickup_time
    o = veh.veh.pos
    pickup_time = veh.sim.skims.ride[o][d]/60  #minutes
      
    t = df[df['event']=='RECEIVES_REQUEST'].iloc[-1]['t']                                        #waiting_time
    if 'ARRIVES_AT_DROPOFF' in df['event'].unique():
        t0 = df[df['event']=='ARRIVES_AT_DROPOFF'].iloc[-1]['t']
    else:
        t0 = df[df['event']=='OPENS_APP'].iloc[-1]['t']
    waiting_time = (t - t0)/60 #minutes 
        
           
    V = (ASC*1)+ (pickup_time*(-0.050)) + (waiting_time*(-0.017)) + (surge*0.101)
    
    acc_prob = (math.exp(V))/(1+math.exp(V))

    if acc_prob > random.uniform(0, 1):
        return False
    else:
        return True




def f_decline_base(veh, **kwargs):
    
    sim = veh.sim

    df = pd.DataFrame(veh.myrides)
    ASC = 1.810                                                                                   #ASC
    
    working_shift = sim.params.simTime*3600 - veh.veh['shift_start']                               #Time1_loc
    T1 = int(working_shift/3)
    request_time = df[df['event']=='RECEIVES_REQUEST'].iloc[-1]['t']
    
    if  request_time in range(veh.veh['shift_start'], veh.veh['shift_start']+T1):
        Time1 = 1
    else:
        Time1 = 0
        
    if veh.veh['pos'] in sim.inData.stats.central_nodes:
        loc = 1
    else:
        loc = 0
        
        
    d = veh.offers[1]['request']["origin"]                                                       #pickup_time
    o = veh.veh.pos
    pickup_time = veh.sim.skims.ride[o][d]/60  #minutes
      
    t = df[df['event']=='RECEIVES_REQUEST'].iloc[-1]['t']                                        #waiting_time
    
    if 'ARRIVES_AT_DROPOFF' in df['event'].unique():
        t0 = df[df['event']=='ARRIVES_AT_DROPOFF'].iloc[-1]['t']
    else:
        t0 = df[df['event']=='OPENS_APP'].iloc[-1]['t']
    waiting_time = (t - t0)/60 #minutes 
    
    surge_price = 0                                                                               #surge_price
    
    req = 1                         #req                                                          #req_long_rate_dec
    
    if (veh.offers[1]["request"]["dist"]/sim.params.speeds.ride)/60 > 6.5: #long
        long = 1
    else:
        long = 0
        
    rate = sim.pax[veh.offers[1]['pax_id']].pax.get('rate',5)   #rate
    
    if len(veh.declines.index) == 0:          #dec
        last_declined = 'False'
    else:
        last_declined = veh.declines.loc[len(veh.declines.index)-1]['declined']
        
    if last_declined == 'True':
        dec = 1
    else:
        dec = 0
        
           
    V = ((ASC*1) + (Time1*loc*(-0.303)) + (pickup_time*(-0.050)) + (waiting_time*(-0.017)) + 
        ((req*long*rate*dec)*0.091) + (surge_price*0.101))
    
    acc_prob = (math.exp(V))/(1+math.exp(V))

    if acc_prob > random.uniform(0, 1):
        return False
    else:
        return True
    
def f_decline_R50 (veh, **kwargs):
    
    sim = veh.sim  
    veh.nRIDES+= 1
    
    if veh.nDECLINES==0:
        R50 = random.randint(0,1)
        
        if R50 ==0 :
            veh.nDECLINES+=1
            return True
        else:
            return False
    else:
        
        AR = 1-(veh.nDECLINES/veh.nRIDES)
        
        if AR>0.5:
            veh.nDECLINES+=1
            return True
        
        else:
            return False

def f_decline_R75 (veh, **kwargs):
    
    sim = veh.sim  
    veh.nRIDES+= 1
    
    if veh.nDECLINES==0:
        R50 = random.randint(0,1)
        
        if R50 ==0 :
            veh.nDECLINES+=1
            return True
        else:
            return False
    else:
        
        AR = 1-(veh.nDECLINES/veh.nRIDES)
        
        if AR>0.75:
            veh.nDECLINES+=1
            return True
        
        else:
            return False
        
def f_decline_R100 (veh, **kwargs):
    
    sim = veh.sim
    return False

def f_decline_mixed (veh, **kwargs):

    sim= veh.sim
    params = sim.params
    id = veh.id
        
    R = random.random()

    if id < params.nV/3:
        return f_decline_R50(veh, **kwargs)

    if params.nV/3<id<2*params.nV/3:
        return f_decline_R75(veh, **kwargs)

    if 2*params.nV/3<id:
        return f_decline_R100(veh, **kwargs)

    
                    # while self.reject.triggered: 
                
                # if self.reject.triggered:
                #     yield self.sim.timeout(15)
                #     self.update(event=travellerEvent.IS_REJECTED_BY_VEHICLE)
                #     for platform_id in self.platform_ids:
                #         platform = self.sim.plats[platform_id]
                #         platform.appendReq(self.id)

                # else:
                #     # self.reject.fail()
                #     for platform_id in self.platform_ids:
                #         platform = self.sim.plats[platform_id]
                #         platform.appendReq(self.id)
                    
                    
#                 if self.rec_off.triggered:
#                     yield self.sim.timeout(100)
#                     self.update(event=travellerEvent.IS_REJECTED_BY_VEHICLE)



        #             while True:

        #                 yield self.rejects | self.sim.timeout(self.till_end())

        #                 if self.flagrej == True:

        #                     yield self.sim.pax[self.rejected_pax_id].sim.timeout(30)
        #                     self.sim.pax[self.rejected_pax_id].update(event=travellerEvent.IS_REJECTED_BY_VEHICLE)
        #                     self.update(event=driverEvent.REJECTS_REQUEST)


        #                     self.rejects = self.sim.env.event()               
        #                     self.platform.appendReq(self.rejected_pax_id)
        #                     self.platform.appendVeh(self.id)
        #                     self.flagrej = False

        #                 else:
        #                     break