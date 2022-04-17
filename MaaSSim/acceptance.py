import pandas as pd
import numpy as np
from dotmap import DotMap
import math
import random as random

def f_decline(veh, **kwargs):
    
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