import os
import json
import pickle
import random
import shutil
import pymysql
import datetime
import numpy as np
import pandas as pd
import logging
import logging.config
from multiprocessing import Process, Manager
from logging.handlers import QueueHandler, QueueListener
import queue
import sys
import tslearn
pwd = sys.path[0]
# os.path.abspath(os.getcwd())
if not os.path.exists(pwd+'/logs'):
    os.makedirs(pwd+'/logs')

def save_json(file, name='model_config.json'):
    with open(pwd+'/config/'+name, 'w') as outfile:
        json.dump(file, outfile)
    print('save_json', name)
    
# def save_json(parameters, name, save_version):
#     with open(path + "/model/"+name+"_"+save_version +".json", "w") as json_file:      
#         json.dump(parameters, json_file)
        
def load_json(name):
    with open(pwd+'/config/'+name) as json_file:
        model_config = json.loads(json_file.read())
    print('load_json', name)
    return model_config

def save_obj(obj, path):
    with open(pwd+'/obj/{}.pkl'.format(path), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print('save_obj: {}.pkl'.format(path))

def load_obj(path):
    with open(pwd+'/obj/{}.pkl'.format(path), 'rb') as f:
        print('load_obj: {}.pkl'.format(path))
        return pickle.load(f)
    
    
def detect_mad_outliers(points, threshold= 2.5):  
    median = np.median(points, axis=0)
    deviation = np.abs(points - median)
    med_abs_deviation = np.median(deviation)
    modified_z_score = 0.976 * deviation / med_abs_deviation
    idx = (np.abs(modified_z_score - threshold)).argmin()
    threshold_value = points[idx]
    return modified_z_score, threshold_value

def make_prob_df(data, rate_list):
    rate_df = pd.DataFrame(index = range(100))

    for i in rate_list:
        temp = pd.DataFrame(data = data[i].value_counts()).reset_index(drop = False)
        temp.rename(columns = {i : '{}_count'.format(i), 'index' : i}, inplace = True)
        temp['{}_all'.format(i)] = temp['{}_count'.format(i)].sum()
        temp['{}_rate'.format(i)] = temp['{}_count'.format(i)]/temp['{}_all'.format(i)]
        rate_df[i] = temp[i]
        rate_df['{}_rate'.format(i)] = temp['{}_rate'.format(i)]
        
    rate_df.dropna(how = 'all', inplace = True)
    
    return rate_df


def make_prop_map(data, category, rate_list, ip_list_df, filt_key):
    filt_key = filt_key

    prob_map = pd.DataFrame()

    for i in ip_list_df['ip_list']:
        filt_data = data[data[filt_key] == i]
        if len(filt_data)>0:
            temp_df = 'prob_df_{}_{}'.format(category, ip_list_df[ip_list_df['ip_list'] == i].index.tolist()[0])
            globals()[temp_df] = make_prob_df(filt_data, rate_list)
            globals()[temp_df][filt_key] = i
            prob_map = pd.concat([prob_map, globals()[temp_df]])
            
    prob_map.reset_index(drop = True, inplace = True)
    
    return prob_map

def make_drop_list(data):
    drop_list= []
    for i in data:
        drop_list.append(i+"_rate")
    return drop_list

def multiply_rows(x):
    return np.prod(x)

#config
with open(pwd+'/config/config.json') as f:
    config = json.loads(json.dumps(json.load(f)))

def set_logger(mode, model_id, default=False):
    if default:
        logconf = json.loads(json.dumps(config['logging']).replace('{DIR}', pwd).replace('{MODE}', 'default').replace('{MODEL_ID}', 'ai'))
    else:
        logconf = json.loads(json.dumps(config['logging']).replace('{DIR}', pwd).replace('{MODE}', mode).replace('{MODEL_ID}', str(model_id)))
    logging.config.dictConfig(logconf)
    logger = logging.getLogger()
    logger.propagate = False
    #log queue
    q = queue.Queue(-1) #unlimit
    q_handler = QueueHandler(q)
    q_listener = QueueListener(q, logger.handlers)
    q_listener.start()
    logger.info('logging config : {}'.format(logconf))

    return logger

logger = set_logger(None, None, True)

def isExistModel(model_id):
    conn = pymysql.connect(host = config['mysql']['host'], port = config['mysql']['port'], user = config['mysql']['user'], password = config['mysql']['password'], db = config['mysql']['db'])
    curs = conn.cursor()
    sql = 'select model_name from model_meta where model_id = {}'.format(model_id)
    result = curs.execute(sql)

    if result == 0:
        model_name = None
    else:
        model_name = list(curs.fetchone())[0]

    conn.close()
    return model_name

def getConfig(mode, model_id, model_name, train=True):
    conn = pymysql.connect(host = config['mysql']['host'], port = config['mysql']['port'], user = config['mysql']['user'], password = config['mysql']['password'], db = config['mysql']['db'])
    curs = conn.cursor()
    sql = 'select config from model_meta where model_id = {}'.format(model_id)
    curs.execute(sql)
    result = list(curs.fetchone())[0]
    print(result)
    if train:
        model_config = json.loads(result)['train']
    else:
        model_config = json.loads(result)['predict']
    model_config["common"] = json.loads(result)['common']
    conn.close()

    model_config['now_delta'] = getDelta(model_config['now_delta'])
    model_config['prev_delta'] = getDelta(model_config['prev_delta'])
    logger.info('[{}({})] model config : {}'.format(mode, model_name, model_config))
    return model_config

def getToStartOf(crontab):
    try:
        m, h, d, M, y = crontab.split(' ')
        if m == '*/1' or crontab == '* * * * *':
            return 'toStartOfMinute'
        elif m == '*/5':
            return 'toStartOfFiveMinute'
        elif m == '*/15':
            return 'toStartOfFifteenMinutes'
        elif h == '*/1':
            return 'toStartOfHour'
        elif d == '*/1':
            return 'toStartOfDay'
        elif d == '*/4':
            return 'toStartOfQuarter'
        elif M == '*/1':
            return 'toStartOfMonth'
        elif y == '*/1':
            return 'toStartOfYear'
        else:
            logger.info('[getToStartOf ELSE] crontab: {} => return default value: toStartOfHour'.format(crontab))
            return 'toStartOfHour'
    except:
        logger.error('[getToStartOf ERROR] crontab: {} => return default value: toStartOfHour'.format(crontab))
        return 'toStartOfHour'

def getDelta(delta):
    try:
        unit, num = delta.split('=')[0], int(delta.split('=')[1])
        if unit == 'seconds':
            return datetime.timedelta(seconds=num)
        elif unit == 'minutes':
            return datetime.timedelta(minutes=num)
        elif unit == 'hours':
            return datetime.timedelta(hours=num)
        elif unit == 'days':
            return datetime.timedelta(days=num)
        elif unit == 'weeks':
            return datetime.timedelta(weeks=num)
        else:
            logger.error('[getDelta ELSE] delta: {} => return default value: days=1'.format(delta))
            return datetime.timedelta(days=1)
    except:
        logger.error('[getDelta ERROR] delta: {} => return default value: days=1'.format(delta))
        return datetime.timedelta(days=1)

# def create_ts_data(data, time_steps=1):
#     Xs = []
#     for i in range(time_steps, len(data)):
#         v = data.iloc[(i - time_steps):i].values
#         feats = v.shape[-1]
#         Xs.append(v)

#     return np.concatenate([np.zeros((time_steps, time_steps, np.array(Xs).shape[-1])), np.array(Xs)], 0)
    
def create_ws_data(data, time_steps=0):
    if len(data) > time_steps:
        Xs = []
        for i in range(time_steps, len(data)):
            v = data.iloc[(i - time_steps):i].values
            feats = v.shape[-1]
            Xs.append(v)

        return np.concatenate([np.zeros((time_steps, time_steps, feats)), np.array(Xs)], 0)
    else:
        print('DATA LENGTH : ', len(data))
        print('TIME STEPS : ', time_steps)
        print('LENGTH OF DATA MUST BE BIGGER THAN TIMESTEP')
        return None

def create_output_data(data):
    output_data = data
#     output_data = np.concatenate([output_data, np.ones((1, output_data.shape[-2], output_data.shape[-1]))], 0)
    return output_data

def create_input_data(data):
    input_data = data[:-1]
    input_data = np.concatenate([np.ones((1, input_data.shape[-2], input_data.shape[-1])), input_data], 0)
    return input_data

# def ts_kmeans_fit(data, k, save_version):
#     kmeans = KMeans(n_clusters = k).fit(data)
#     centroids = kmeans.cluster_centers_
#     with open(path + "/model/kmeans_model_"+save_version+".pickle", "wb") as f:
#         pickle.dump(kmeans, f)
#     with open(path + "/data/kmeans_centroid_"+save_version+".pickle", "wb") as f:
#         pickle.dump(centroids, f)
        
# def ts_kmeans_trans(data, save_version):
    
#     with open(path + "/model/kmeans_model_"+save_version+".pickle", 'rb') as f:
#         kmeans = pickle.load(f)
#     return  kmeans.predict(data), kmeans

# def feature_selection_fit(data, save_version):
#     selector = VarianceThreshold(threshold=np.var(data)).fit(data)
#     with open(path + "/model/selector_"+save_version+".pickle", "wb") as f:
#         pickle.dump(selector, f)
        
# def feature_selection_trans(data, save_version):
#     with open(path + "/model/selector_"+save_version+".pickle", 'rb') as f:
#         selector = pickle.load(f)
#     return data.columns[selector.get_support()]

# def smoothListGaussian(list, strippedXs=False, degree=3):
#     window = degree * 2 - 1
#     weight = np.array([1.0] * window)
#     weightGauss = []
#     for i in range(window):
#         i = i - degree + 1
#         frac = i / float(window)
#         gauss = 1 / (np.exp((4 * (frac)) ** 2))
#         weightGauss.append(gauss)
#     weight = np.array(weightGauss) * weight
#     smoothed = [0.0] * (len(list) - window)
#     for i in range(len(smoothed)):
#         smoothed[i] = sum(np.array(list[i:i + window]) * weight) / sum(weight)
#     return smoothed

def get_queries(start_date, end_date, query_type=None):
    conn = pymysql.connect(host = config['mysql']['host'], port = config['mysql']['port'], user = config['mysql']['user'], password = config['mysql']['password'], db = config['mysql']['db'])
    curs = conn.cursor()
    if query_type == 'l':
        sql = 'select ruleId, ruleQuery from motie_rule_single where ruleGubn = 2'
        time = 'loged_time'
    elif query_type == 'p':
        sql = 'select ruleId, ruleQuery from motie_rule_single where ruleGubn = 1'
        time = 'packet_time'
    else:
        print('UNKNOWN RULE TYPE')
        return None
    result = curs.execute(sql)
    if result == 0:
        print('LOG RULE TABLE DOES NOT EXIST...')
        print('PLEASE CHECK CONFIG...')
        return None
    else:
        queries =list(curs.fetchall())
    conn.close()
    log_list = []
    new_where = "where parseDateTimeBestEffort(substring(toString(time),1,14)) >= '{start_date}' and parseDateTimeBestEffort(substring(toString(time),1,14)) <= '{end_date}' and".format(start_date=start_date, end_date=end_date)
    
    for query in queries:
        if query[1].find('WHERE') > 0:
            new_query = query[1].replace('WHERE',  new_where)
        else:
            new_query = query[1] + new_where
        if query_type == 'l':
            new_query = new_query.replace('*', 'toString(sipHash64(*)) as hash, * , {time} as time, {rule} as single_rule'.format(rule=str(query[0]), time=time))
        elif query_type == 'p':
            new_query = new_query.replace('*', 'toString(sipHash64(*)) as hash, * , {time} as time, {rule} as single_rule'.format(rule=str(query[0]), time=time))
        log_list.append(new_query)
    return log_list

def get_corr():
    conn = pymysql.connect(host = config['mysql']['host'], port = config['mysql']['port'], user = config['mysql']['user'], password = config['mysql']['password'], db = config['mysql']['db'])
    curs = conn.cursor()
    log_sql = 'SELECT multiId, ruleId from dti.motie_rule_mapping where ruleGubn = 2'
    packet_sql = 'SELECT multiId, ruleId from dti.motie_rule_mapping where ruleGubn = 1'
    curs.execute(log_sql)
    log_ids =list(curs.fetchall())
    curs.execute(packet_sql)
    packet_ids =list(curs.fetchall())
    conn.close()
    log_df = pd.DataFrame(log_ids, columns=['corr', 'single_rule_log'])
    packet_df = pd.DataFrame(packet_ids, columns=['corr', 'single_rule_packet'])
    corr_df = pd.merge(log_df, packet_df, on = ['corr'], how='outer')
    corr_df.fillna(0, inplace=True)
    corr_df['corr'] = corr_df['corr'].astype('int')
    corr_df.drop_duplicates(inplace = True)

    return corr_df

def packet_save_single_rule(data, type, version):
    single_rule_df = pd.DataFrame(columns = ['time', 'ip', 'type', 'single_rule', 'hash', 'version','id', 'milli_time'])
    single_rule_df[['time','ip','single_rule','hash','id', 'milli_time']] = data[['time','ip','single_rule','hash','id', 'milli_time']]
    single_rule_df['type'] = type
    single_rule_df['version'] = version
    single_rule_df.drop_duplicates(inplace = True)
    single_rule_df.reset_index(drop = True, inplace = True)
    single_rule_df['time'] = pd.to_datetime(single_rule_df['time'], format = '%Y%m%d%H%M%S%f', errors = 'raise')
    single_rule_df['milli_time'] = pd.to_datetime(single_rule_df['time'], format = '%Y%m%d%H%M%S%f', errors = 'raise')
    single_rule_df['milli_time'] = single_rule_df['milli_time'].astype('str')
    single_rule_df.single_rule = single_rule_df.single_rule.astype('uint')
    
    return single_rule_df

def log_save_single_rule(data, type, version):
    single_rule_df = pd.DataFrame(columns = ['time', 'ip', 'type', 'single_rule', 'hash', 'version','id', 'milli_time'])
    single_rule_df[['time','ip','single_rule','hash','id', 'milli_time']] = data[['time','ip','single_rule','hash','id', 'milli_time']]
    single_rule_df['type'] = type
    single_rule_df['version'] = version
    single_rule_df.drop_duplicates(inplace = True)
    single_rule_df.reset_index(drop = True, inplace = True)
    single_rule_df['time'] = pd.to_datetime(single_rule_df['time'], format = '%Y%m%d%H%M%S', errors = 'raise')
    single_rule_df['milli_time'] = pd.to_datetime(single_rule_df['time'], format = '%Y%m%d%H%M%S', errors = 'raise')
    single_rule_df['milli_time'] = single_rule_df['milli_time'].astype('str')
    single_rule_df.single_rule = single_rule_df.single_rule.astype('uint')
    
    return single_rule_df