from .traveller import travellerEvent
from .driver import driverEvent
import pandas as pd
import numpy as np
from dotmap import DotMap
import math
import random as random

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
    df = df[~(df.t == df.t_s)]  # filter for positive time periods only
    df['dt'] = df.t_s - df.t  # make time intervals
    ret = df.groupby(['veh', 'event_s'])['dt'].sum().unstack()  # aggreagted by vehicle and event
    ret.columns.name = None
    ret = ret.reindex(vehindex)  # update for vehicles with no record
    ret['nRIDES'] = df[df.event == driverEvent.ARRIVES_AT_DROPOFF.name].groupby(['veh']).size().reindex(ret.index)
    ret['nREJECTS'] = df[df.event==driverEvent.REJECTS_REQUEST.name].groupby(['veh']).size().reindex(ret.index)
    for status in driverEvent:
        if status.name not in ret.columns:
            ret[status.name] = 0
    DECIDES_NOT_TO_DRIVE.index = DECIDES_NOT_TO_DRIVE.values
    ret['OUT'] = DECIDES_NOT_TO_DRIVE
    ret['OUT'] = ~ret['OUT'].isnull()
    ret['DRIVING_TIME'] = ret.ARRIVES_AT_PICKUP + ret.ARRIVES_AT_DROPOFF #we assum there is no repositioning
    ret['DRIVING_DIST'] = ret['DRIVING_TIME']*(params.speeds.ride/1000)   #here we assume the speed is constant on the network
    ret['IDLE_TIME'] = ret['ENDS_SHIFT'] + ret['RECEIVES_REQUEST']
    ret.fillna(0, inplace=True)  
    d = df[df['event_s']=='ARRIVES_AT_DROPOFF']
    if len(d) != 0:
        d['REVENUE'] = d.apply(lambda row: max(row['dt'] * (params.speeds.ride/1000) * params.platforms.fare + params.platforms.base_fare, params.platforms.min_fare), axis=1)*(1-params.platforms.comm_rate)
        ret['REVENUE'] = d.groupby(['veh']).sum().REVENUE
    else:
        ret['REVENUE'] = 0
    ret['COST'] = ret['DRIVING_DIST'] * (params.d2d.fuel_cost) # Operating Cost (OC)
    ret['PROFIT'] = ret['REVENUE'] - ret['COST']
    ret['ACCEPTANCE_RATE'] = (ret['nRIDES']-ret['nREJECTS'])/ret['nRIDES']*100
    ret.replace([np.inf, -np.inf], np.nan, inplace=True)
    ret.fillna(0, inplace=True)  

    ret = ret[['ACCEPTANCE_RATE','PROFIT','IDLE_TIME','nRIDES','nREJECTS','DRIVING_TIME','DRIVING_DIST','REVENUE','COST']]#+ [_.name for _ in driverEvent]]
    ret.index.name = 'veh'
    
    # KPIs
    kpi = ret.agg(['sum', 'mean', 'std'])
    kpi['nV'] = ret.shape[0]
    return {'veh_exp': ret, 'veh_kpi': kpi}


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
    df = df[df.pax == df.pax_s]  # filter for the same vehicles only
    df = df[~(df.t == df.t_s)]  # filter for positive time periods only
    df['dt'] = df.t_s - df.t  # make time intervals
    ret = df.groupby(['pax', 'event_s'])['dt'].sum().unstack()  # aggreagted by vehicle and event

    ret.columns.name = None
    ret = ret.reindex(paxindex)  # update for pax with no record

    ret.index.name = 'pax'
    ret = ret.fillna(0)

    for status in travellerEvent:
        if status.name not in ret.columns:
            ret[status.name] = 0  # cover all statuses
    PREFERS_OTHER_SERVICE.index = PREFERS_OTHER_SERVICE.values
    ret['OUT'] = PREFERS_OTHER_SERVICE
    ret['OUT'] = ~ret['OUT'].isnull()
    ret['LOST_PATIENCE'] = ret.apply(lambda row: False if row['ARRIVES_AT_DROPOFF']>0 else True,axis=1) 
    ret['TRAVEL'] = ret['ARRIVES_AT_DROPOFF']  # time with traveller (paid time)
    ret['WAIT_TIME'] = (ret['RECEIVES_OFFER'] + ret['MEETS_DRIVER_AT_PICKUP'] + ret.get('LOSES_PATIENCE', 0))/60  #in minute
    ret.fillna(0, inplace=True)

    ret = ret[['WAIT_TIME','TRAVEL','LOST_PATIENCE']] #+ [_.name for _ in travellerEvent]]
    ret.index.name = 'pax'

    kpi = ret.agg(['sum', 'mean', 'std'])
    kpi['nP'] = ret.shape[0]
    return {'pax_exp': ret, 'pax_kpi': kpi}




def f_decline(veh, **kwargs):
    
    sim = veh.sim
    df = pd.DataFrame(veh.myrides)
    ASC = 1.810                                                                                   #ASC    
        
    d = veh.offers[1]['request']["origin"]                                                       #pickup_time
    o = veh.veh.pos
    pickup_time = veh.sim.skims.ride[o][d]/60  #minutes
      
    t = df[df['event']=='RECEIVES_REQUEST'].iloc[-1]['t']                                        #waiting_time
    if 'ARRIVES_AT_DROPOFF' in df['event'].unique():
        t0 = df[df['event']=='ARRIVES_AT_DROPOFF'].iloc[-1]['t']
    else:
        t0 = df[df['event']=='OPENS_APP'].iloc[-1]['t']
    waiting_time = (t - t0)/60 #minutes 
        
           
    V = (ASC*1)+ (pickup_time*(-0.050)) + (waiting_time*(-0.017))
    
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
    df = pd.DataFrame(veh.myrides)  

    R50 = random.random()
        

    
    if R50 <0.5 :
        return False
    else:
        return True

def f_decline_R75 (veh, **kwargs):
    
    sim = veh.sim
    df = pd.DataFrame(veh.myrides)  

    R50 = random.random()
        

    
    if R50 <0.75 :
        return False
    else:
        return True

def f_decline_R100 (veh, **kwargs):
    
    sim = veh.sim
    df = pd.DataFrame(veh.myrides)  

    R50 = random.random()
        

    
    if R50 <1 :
        return False
    else:
        return True

def f_decline_mixed (veh, **kwargs):

    sim= veh.sim
    params = sim.params
    id = veh.id
    
    R = random.random()

    if id < params.nV/3:
        if R <0.5 :
            return False
        else:
            return True

    if params.nV/3<id<2*params.nV/3:
        if R <0.75 :
            return False
        else:
            return True

    if 2*params.nV/3<id:
        if R <1 :
            return False
        else:
            return True



def results(sim):
    
    trips = pd.DataFrame()
    requests = pd.DataFrame()
    passengers = pd.DataFrame()
    
    for veh in sim.vehs:
        res = pd.DataFrame(columns=['veh_id', 'pax_id'])
        df = pd.DataFrame(sim.vehs[veh].myrides)
        a = df[df['event']=='IS_ACCEPTED_BY_TRAVELLER'].reset_index()
        b = df[df['event']=='ARRIVES_AT_PICKUP'].reset_index()
        res['pickup_t(min)'] = (b['t']-a['t'])/60
        res['pickup_km'] = res['pickup_t(min)']*sim.vehs[veh].speed*0.06
        
        a = df[df['event']=='DEPARTS_FROM_PICKUP'].reset_index()
        b = df[df['event']=='ARRIVES_AT_DROPOFF'].reset_index()
        res['travel_t_with(min)'] = (b['t']-a['t'])/60
        res['pax_km'] = res['travel_t_with(min)']*sim.vehs[veh].speed*0.06
        res['travel_t(min)'] = res['pickup_t(min)']+res['travel_t_with(min)']
        res['travel_km'] = res['travel_t(min)']*sim.vehs[veh].speed*0.06
        
        dd = df[(df['event']=='RECEIVES_REQUEST') | (df['event']=='ARRIVES_AT_DROPOFF') | (df['event']=='OPENS_APP')]
        dd.reset_index(inplace=True)
        l = list()
        for i in range(0,len(dd)-1):
            if not dd.iloc[i]['event']==dd.iloc[i+1]['event']:
                l.append(i)
        if dd.iloc[-1]['event']=='RECEIVES_REQUEST':
            l.pop()
        dd = dd.iloc[l]
        
        veh_waiting_t = list()
        for i in range(0,len(dd),2):
            x = (dd.iloc[i+1]['t'] - dd.iloc[i]['t'])/60
            veh_waiting_t.append(x)
        
        #print('len res[veh_waiting_t(min)]', len(res))
        #print('len veh_waiting_t', len(veh_waiting_t))

        res['veh_waiting_t(min)'] = veh_waiting_t
        
        cc = df[(df['event']=='OPENS_APP') | (df['event']=='ACCEPTS_REQUEST') | (df['event']=='ARRIVES_AT_DROPOFF')]
        idle_time = []
        for i in range(0,len(cc)-1,2):
            t = (cc.iloc[i+1]['t'] - cc.iloc[i]['t'])/60
            idle_time.append(t)
        res['idle_t(min)'] = idle_time
        res['revenue $'] = res['pax_km']*sim.inData.platforms.iloc[sim.vehs[veh].platform_id]['fare']
        
        req = pd.DataFrame(columns=['veh_id'])
        req = req.append({'veh_id':veh}, ignore_index=True)
        req['n_of_requests'] = sim.vehs[veh].declines['declined'].count()
        req['n_of_accepted'] = sim.vehs[veh].declines['declined'].value_counts().get('False',0)
        req['n_of_declined'] = sim.vehs[veh].declines['declined'].value_counts().get('True',0)
        req['acceptance_rate %'] = (req['n_of_accepted']/req['n_of_requests'])*100
                 
        res.veh_id = veh
        res.pax_id = df[df['event']=='ARRIVES_AT_DROPOFF']['paxes'].apply(lambda x: x[0]).values
        trips = pd.concat([trips,res])
        requests = pd.concat([requests,req])
                 
    for pax in sim.pax:
        ff = pd.DataFrame(sim.pax[pax].rides)
        if 'ACCEPTS_OFFER' in list(ff['event']):
            a = ff[ff['event']=='REQUESTS_RIDE']['t'].values[0]; b = ff[ff['event']=='ACCEPTS_OFFER']['t'].values[0]
            waiting_t = b-a
        else:
            waiting_t = 'Unsuccessful hailing'
        if len(ff['veh_id'].dropna()) > 0:
            vehid = ff['veh_id'].dropna().values[0]
        else:
            vehid = np.NaN
        pax_veh = ff
        data = {'pax_id':[pax], 'veh_id':[vehid], 'waiting_t':[waiting_t]}
        pas = pd.DataFrame(data)
        passengers = pd.concat([passengers,pas])
        
    passengers.reset_index(inplace=True); passengers.drop(['index'], axis=1, inplace=True)
    requests.reset_index(inplace=True); requests.drop(['index'], axis=1, inplace=True)
        
    results = DotMap()
    results.trips = trips
    results.requests = requests
    results.passengers = passengers
                 
    return results
        
    
    
def ResultS(sim):
    
    trips = pd.DataFrame()
    requests = pd.DataFrame()
    passengers = pd.DataFrame()
    declines = pd.DataFrame()
    veh_speed = sim.params.speeds.ride
    
    for veh in sim.vehs:
        df = pd.DataFrame(sim.vehs[veh].myrides)
        if not df.iloc[-1]['event']=='ENDS_SHIFT':  # delete the trips are not complete due to lack of time
            while not df.iloc[-1]['event']=='ARRIVES_AT_DROPOFF':
                df.drop(index=df.index[-1],axis=0,inplace=True)   

        res = pd.DataFrame(columns=['veh_id', 'pax_id'])
        a = df[df['event']=='IS_ACCEPTED_BY_TRAVELLER'].reset_index()
        b = df[df['event']=='ARRIVES_AT_PICKUP'].reset_index()
        res['pickup_t[min]'] = (b['t']-a['t'])/60
        res['pickup_d[km]'] = res['pickup_t[min]']*veh_speed*0.06
        
        
        a = df[df['event']=='DEPARTS_FROM_PICKUP'].reset_index()
        b = df[df['event']=='ARRIVES_AT_DROPOFF'].reset_index()
        res['travel_t_with[min]'] = (b['t']-a['t'])/60
        res['pax_km'] = res['travel_t_with[min]']*veh_speed*0.06
        
        a = df[df['event']=='IS_ACCEPTED_BY_TRAVELLER'].reset_index()
        b = df[df['event']=='ARRIVES_AT_DROPOFF'].reset_index()
        res['travel_t[min]'] = (b['t']-a['t'])/60
        res['travel_d[km]'] = res['travel_t[min]']*veh_speed*0.06
        
        dd = df[(df['event']=='OPENS_APP') | (df['event']=='ARRIVES_AT_DROPOFF') | (df['event']=='RECEIVES_REQUEST')]
        dd.reset_index(inplace=True)
        if 'ARRIVES_AT_DROPOFF' in dd['event'].unique():
            a = []; b = []
            for i in range(0,len(dd)):
                if dd.iloc[i]['event']=='ARRIVES_AT_DROPOFF':
                    a.append(dd.iloc[i]['t'])
                    b.append(dd.iloc[i-1]['t'])
            a.pop(); a.insert(0,dd.iloc[0]['t'])
            veh_waiting_t = np.array(b)-np.array(a)
        else:
            veh_waiting_t = df[df['event']=='ENDS_SHIFT']['t'].values[0] - df[df['event']=='OPENS_APP']['t'].values[0]
        res['veh_waiting_t[sec]'] = veh_waiting_t
        
        res['revenue $'] = res['pax_km']*sim.inData.platforms.iloc[sim.vehs[veh].platform_id]['fare']

        req = pd.DataFrame(columns=['veh_id'])
        req = req.append({'veh_id':veh}, ignore_index=True)
        req['n_of_requests'] = sim.vehs[veh].declines['declined'].count()
        req['n_of_accepted'] = sim.vehs[veh].declines['declined'].value_counts().get('False',0)
        req['n_of_declined'] = sim.vehs[veh].declines['declined'].value_counts().get('True',0)
        req['acceptance_rate %'] = (req['n_of_accepted']/req['n_of_requests'])*100

        if 'ARRIVES_AT_DROPOFF' in dd['event'].unique():
            res.pax_id = df[df['event']=='ARRIVES_AT_DROPOFF']['paxes'].apply(lambda x: x[0]).values
        else:
            res.pax_id = None
        res.veh_id = veh
        trips = pd.concat([trips,res])
        requests = pd.concat([requests,req])
        declines = pd.concat([declines,sim.vehs[veh].declines])
        
        
    for pax in sim.pax:
        ff = pd.DataFrame(sim.pax[pax].rides)
        if 'MEETS_DRIVER_AT_PICKUP' in list(ff['event']):
            a = ff[ff['event']=='REQUESTS_RIDE']['t'].values[0]
            b = ff.iloc[ff[ff['event']=='MEETS_DRIVER_AT_PICKUP'].index]['t'].values[0]
            passenger_waiting_t = b-a
            veh_id = ff['veh_id'].dropna().values[0]
        elif 'ACCEPTS_OFFER' in list(ff['event']):
            a = ff[ff['event']=='REQUESTS_RIDE']['t'].values[0]
            b = ff.iloc[ff[ff['event']=='ACCEPTS_OFFER'].index]['t'].values[0]
            passenger_waiting_t = b-a
            veh_id = ff['veh_id'].dropna().values[0]
        else:
            a = ff[ff['event']=='REQUESTS_RIDE']['t'].values[0]
            b = ff[ff['event']=='LOSES_PATIENCE']['t'].values[0]
            passenger_waiting_t = b-a
            #passenger_waiting_t = 'no hail'
            veh_id = 0 #'null'

        dec = declines[declines['pax_id']==pax]['declined'].value_counts().get('True',0)
            
        pas = pd.DataFrame({'pax_id':[pax], 'veh_id':[veh_id], 'waiting_t[sec]':[passenger_waiting_t],
                            'number of declines':[dec]})
        passengers = pd.concat([passengers,pas])
    passengers.reset_index(drop=True, inplace=True)
    requests.reset_index(drop=True, inplace=True)
    
     
    results = DotMap()
    results.trips = trips
    results.requests = requests
    results.passengers = passengers
    results.declines = declines
    
    
    return results