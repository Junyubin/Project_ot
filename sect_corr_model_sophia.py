import pandas as pd
from sect_model import *
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sect_utils import *
import json
from sect_base_component import *
from optparse import OptionParser
import time
from time import gmtime, strftime
from datetime import datetime, timedelta
import sys
import itertools

sys.path.insert(0, sys.path[0])

class Train1(BaseComponent):
    def init(self):
        pass

    async def run(self, param):
        try:
            start = datetime.now().replace(microsecond=0) + timedelta(hours=9)
            version = start.strftime("%Y%m%d_%H")
            print(version)
            
            if not os.path.exists(pwd+'/sect_model/'+version):
                logger.info('create directory: {}'.format(pwd+'/sect_model/'+version))
                os.makedirs(pwd+'/sect_model/'+version)            
            
            for data_type in ['packet', 'log']:
                if data_type == 'log':
                    data_code = 2
                    model_id = 3
                elif data_type == 'packet':
                    data_code = 1
                    model_id = 4
                ## train data load
                data, meta = execute_ch("select * from dti.motie_rule_single where ruleGubn = '{}' and toString(ruleId) in (select distinct ruleId from dti.sect_rule_mapping)".format(data_code), with_column_types = True)
                feats = [m[0] for m in meta]
                single_rule = pd.DataFrame(data, columns = feats)  
                if len(single_rule) > 0:
                    
                    single_rule = single_rule['ruleId']
                    self.logger.info('********************* TRAIN RULE LOAD DONE *********************')            
                    train_df = list(itertools.product('01', repeat=len(single_rule)))
                    train_df = pd.DataFrame(train_df)
                    train_df.columns = single_rule.values
                    train_df.to_csv(pwd + '/sect_model/{}/{}_train_data.csv'.format(version, data_type))                    

                    ## over sampling
                    for i in range(3):
                        train_df = pd.concat([train_df,train_df])

                    train_df = train_df.astype('int')
                    ############################ 상관분석 모듈 시작 ############################
                    ## CNN Autoencoder            
                    X_data = train_df.copy()                        
                    with open(pwd + '/sect_model/{}/{}_col_list'.format(version, data_type), 'wb') as f:
                        pickle.dump(list(X_data), f, pickle.HIGHEST_PROTOCOL)

                    ae_model = Sequential([
                        Dense(X_data.shape[1], activation = 'relu', input_shape = (X_data.shape[1],)),
                        Dropout(0.4),
                        Dense(16, activation = 'relu'),
                        Dropout(0.3),
                        Dense(32, activation = 'relu'),
                        Dropout(0.4),
                        Dense(X_data.shape[1], activation = 'sigmoid')
                    ])     
                    ae_model.compile(optimizer='adam', loss='mse')
                    early_stop = EarlyStopping(monitor = 'loss',  patience=15, verbose=1, min_delta=0.00001)
                    ai_history = ae_model.fit(X_data, X_data, epochs=500, batch_size=64, shuffle=True, verbose=1, callbacks=[early_stop])
                    ae_model.save(pwd + '/sect_model/{}/{}_sect_model'.format(version, data_type))
                    
                    ############################ 모델 학습 완료 ############################
                    history_df = pd.DataFrame({'loss': ai_history.history['loss']})
                    history_df['model_id'] = model_id
                    history_df['epoch'] = list(range(1, len(history_df)+1))
                    history_df['model_type'] = data_type
                    history_df['data_shape'] = [X_data.shape for i in range(len(history_df))]
                    history_df['version'] = start
                    history_df['train_time'] = start      

                    pred = ae_model.predict(X_data)
                    rmse = sect_rmse(X_data, pred, 1).numpy()            
                    z_scores, threshold = detect_mad_outliers(rmse, threshold= 2.5)
                    print(threshold)
                    print("*"*100)

                    with open(pwd + '/sect_model/{}/{}_threshold'.format(version, data_type), 'wb') as f:
                        pickle.dump(threshold, f, pickle.HIGHEST_PROTOCOL)

                    valid_df = pd.DataFrame(columns = ['rmse'], data = list(sect_rmse(X_data, pred, axis = 1).numpy()))
                    valid_df.drop_duplicates(inplace = True)            
                    self.logger.info(valid_df.value_counts())
                    ############################ test result print ############################            
                    ## test
                    validation_time = datetime.now().replace(microsecond=0) + timedelta(hours=9)
                    history_df['validation_time'] = validation_time
                    history_df = history_df[['model_id', 'model_type', 'train_time', 'validation_time', 'version', 'loss', 'epoch', 'data_shape']]
                    execute_ch("INSERT INTO dti.motie_ai_history VALUES", history_df.to_dict('records'))  
                    
                else:
                    pass
                
            return "OK"

        except Exception as err:
            self.logger.error(err)
            self.logger.error(traceback.print_exc())            
            return None


class Prediction1(BaseComponent):
    def init(self):
        pass

    async def run(self, param):
        try:
            start = datetime.now().replace(microsecond=0) + timedelta(hours=9)
            version = start.strftime("%Y%m%d_%H")            
            new_time = start
            real_time = start                 
            
            for timerange in range(100):
                if not os.path.exists(pwd + '/sect_model/{}'.format(version)):
                    new_time = start - timedelta(hours=timerange+1)
                    version = new_time.strftime("%Y%m%d_%H")     
                else:
                    break
                    
            for timerange in range(100):
                if len(os.listdir(pwd +'/sect_model/{}/'.format(version))) != 8:                         
                    new_time = start - timedelta(hours=timerange+1)
                    version = new_time.strftime("%Y%m%d_%H")     
                    print(version)
                else:
                    break
                    
            self.logger.info('VERSION : ' + version)
            
            ############################ log data load ############################
            data, meta = execute_ch("select ip_address, device_id from dti.kdn_lgsys_L003 where ip_address!='' and device_id!=''", with_column_types = True)
            feats = [m[0] for m in meta]
            map_ip = pd.DataFrame(data, columns = feats)              
            map_ip.drop_duplicates(inplace = True)
            self.logger.info('********************* LOG MAP IP LOAD DONE *********************')   
            log_sql = get_queries('2021-06-13 00:00:00', '2021-06-22 24:00:00', query_type='l')
            globals()['sql_005'], globals()['sql_009'], globals()['sql_011'], log_list = [], [], [], []

            for sql in log_sql:
                if 'motie_lgsys_L005' in sql:
                    sql_005.append(sql)                
                elif 'motie_lgsys_L009' in sql:
                    sql_009.append(sql)
                elif 'motie_lgsys_L011' in sql:
                    sql_011.append(sql)

            if len(sql_005)>0: log_list.append('005')
            if len(sql_009)>0: log_list.append('009')
            if len(sql_011)>0: log_list.append('011')
                
            for num in log_list:  
                globals()['log_{}'.format(num)] = pd.DataFrame()
                for i in range(0, len(globals()['sql_{}'.format(num)])):
                    data, meta = execute_ch(globals()['sql_{}'.format(num)][i], with_column_types = True)
                    feats = [m[0] for m in meta]
                    temp = pd.DataFrame(data, columns = feats)
                    globals()['log_{}'.format(num)] = pd.concat([globals()['log_{}'.format(num)], temp])        
                globals()['log_{}'.format(num)].reset_index(drop = True, inplace = True)
            self.logger.info('********************* LOG DATA LOAD DONE *********************')            
            
            ## log_data concat ( 005, 009 , 011)
            log = pd.DataFrame()                
            for num in log_list:
                print(globals()['log_{}'.format(num)].shape)
                log = pd.concat([log, globals()['log_{}'.format(num)]])

            log.reset_index(drop = True, inplace = True)
            log.drop_duplicates(inplace =True)

            ## log data ip mapping                
            log = pd.merge(log, map_ip, on = 'device_id', how = 'left')
            log = log[log.ip_address != 'nan']
            log = log[log.ip_address.isna() == False]
            log['ip'] = log.ip_address
            log.reset_index(drop = True, inplace = True)
            if len(sql_009)>0: 
                log = log[log.hash.isin(log_009.hash) == False]
                
            log['milli_time'] = pd.to_datetime(log['time'].str[:14], format = '%Y%m%d%H%M%S', errors = 'raise')
            log['time'] = pd.to_datetime(log['time'].str[:14], format = '%Y%m%d%H%M%S', errors = 'raise')
            self.logger.info('********************* PREDICTION LOG DATA SHAPE : {} *********************'.format(log.shape))    
            
            ############################ packet data load ############################
            packet_sql = get_queries('2021-06-13 00:00:00', '2021-06-22 24:00:00', query_type='p')
            packet = pd.DataFrame()
            for i in range(0,len(packet_sql)):
                data, meta = execute_ch(packet_sql[i], with_column_types = True)
                feats = [m[0] for m in meta]
                temp = pd.DataFrame(data, columns = feats)
                packet = pd.concat([packet, temp])   
            packet.reset_index(drop = True, inplace = True)
            packet.drop_duplicates(inplace = True)
            packet = packet[(packet.src_ip != '') | (packet.dst_ip != '')]         
            packet['ip'] = packet.src_ip
            
            packet['milli_time'] = pd.to_datetime(packet['time'].str[:16], format = '%Y%m%d%H%M%S%f', errors = 'raise')
            packet['time'] = packet['time'].str[:14]
            self.logger.info('********************* PACKET DATA LOAD DONE *********************')  
            self.logger.info('********************* PREDICTION PACKET DATA SHAPE : {} *********************'.format(packet.shape))    
            
            # 밀리타임 설정 
            log['milli_time'] = log['milli_time'].astype('str')
            packet['milli_time'] = packet['milli_time'].astype('str')

            ## col name change
            log.rename(columns = {'plant_id' : 'id'}, inplace = True)
            packet.rename(columns = {'make_id' : 'id'}, inplace = True)                        

            ## make pred data format
            log_packet_result = pd.DataFrame()
            type_list = []
            if len(log) > 0: type_list.append('log')
            if len(packet) > 0: type_list.append('packet')
            
            
            for data_type in type_list:            
                if data_type == 'log':
                    data = log.copy()
                    vec_list = ['msg', 'value01', 'value02', 'value03', 'type01']
                    sort_time = 'loged_time'
                    print(data_type)
                    print(data.shape)
                elif data_type == 'packet':
                    data = packet.copy()
                    vec_list = ['protocol_type', 'protocol_detail', 'anomaly_type']
                    sort_time = 'packet_time'
                    print(data_type)
                    print(data.shape)
                with open(pwd + '/sect_model/{}/{}_col_list'.format(version, data_type), 'rb') as f:
                    col_list = pickle.load(f)                      
                
                data['time'] = pd.to_datetime(data['time']).dt.floor('10min')
                
                temp_df = data[['time', 'single_rule']].copy()
                temp_df.sort_values('time', inplace = True)
                temp_df.reset_index(drop = True, inplace = True)
                temp_df = pd.crosstab(index = temp_df.time, columns = temp_df.single_rule, values = temp_df.single_rule, aggfunc='count')

                pred_df = pd.DataFrame(columns = col_list)
                pred_df[list(temp_df)] = temp_df[list(temp_df)]
                pred_df.fillna(0, inplace = True)

                ############################ 상관분석 모듈 시작 ##########################################            
                X_data = pred_df[col_list].copy()

                ## model load
                ae_model =  load_model(pwd + '/sect_model/{}/{}_sect_model'.format(version, data_type))            
                pred = ae_model.predict(X_data)
                rmse = sect_rmse(X_data, pred, 1).numpy()   

                with open(pwd + '/sect_model/{}/{}_threshold'.format(version, data_type), 'rb') as f:
                    threshold  = pickle.load(f)
                        
                result_df = pred_df.copy()
                result_df['ai_rmse'] = rmse
                result_df['ai_label'] = rmse > threshold
                result_true = result_df[result_df['ai_label'] == True]
                raw_result = data[data.time.isin(result_true.index)].copy()
                raw_result.fillna(',', inplace = True)
                raw_result.sort_values(['time'], inplace = True)
                raw_result.reset_index(drop = True, inplace = True)
                
                raw_result['cnt_vec'] = raw_result[vec_list].agg(','.join, axis=1)
                #####################################################################################################
                cnt_vec = CountVectorizer()
                cnt_vec_df = cnt_vec.fit_transform(raw_result['cnt_vec'])
                cnt_vec_df = pd.DataFrame(columns = cnt_vec.get_feature_names(), data = cnt_vec_df.toarray())
                
                result_list = []
                
                new_df = pd.DataFrame()
                for i in result_true.index:    
                    temp = raw_result[raw_result.time == i].index
                    cos_matrix = cosine_similarity(cnt_vec_df.iloc[temp,])
                    cos_matrix = np.triu(cos_matrix, 1)
                    cos_index = np.where(cos_matrix > 0.0)
                    ## result_list.append(list(set([temp[item] for t in cos_index for item in t])))
                    ## result_df col_list
                    f_list = ['f_time', 'f_ip', 'f_type', 'f_single_rule', 'f_hash', 'f_id', 'f_milli_time', 'time']
                    b_list = ['b_time', 'b_ip', 'b_type', 'b_single_rule', 'b_hash', 'b_id', 'b_milli_time']

                    sim_df = pd.DataFrame()
                    sim_df['f'] = temp[cos_index[0]]
                    sim_df['b'] = temp[cos_index[1]]
                    sim_df['sim'] = cos_matrix[cos_index]

                    new_df = pd.concat([new_df, sim_df]).reset_index(drop=True)

                raw_result['single_rule'] = raw_result['single_rule'].astype('int')
                raw_result["message_id"] = np.select([raw_result['message_id'].str.startswith('H'), raw_result['message_id'].str.startswith('L')],  ['packet', 'log'], default=np.nan)
                if 'loged_time' in list(raw_result):
                    new_df[f_list] = raw_result[['loged_time', 'ip', 'message_id','single_rule','hash', 'id', 'milli_time', 'time']].loc[new_df['f']].reset_index(drop=True)
                    new_df[b_list] = raw_result[['loged_time', 'ip', 'message_id','single_rule','hash', 'id', 'milli_time']].loc[new_df['b']].reset_index(drop=True)
                else:
                    new_df[f_list] = raw_result[['packet_time', 'ip', 'message_id','single_rule','hash', 'id', 'milli_time', 'time']].loc[new_df['f']].reset_index(drop=True)
                    new_df[b_list] = raw_result[['packet_time', 'ip', 'message_id','single_rule','hash', 'id', 'milli_time']].loc[new_df['b']].reset_index(drop=True)
                new_df[['ai_rmse','ai_label']] = result_true.loc[new_df['time']][['ai_rmse','ai_label']].reset_index(drop=True)
                new_df['cos_sim'] = new_df['sim']
                new_df.drop(['f','b','sim', 'time'], axis = 1, inplace = True)
                new_df['f_time'] = pd.to_datetime(new_df['f_time'].str[:14])
                new_df['b_time'] = pd.to_datetime(new_df['b_time'].str[:14])
                final_df = new_df.copy()
                final_df = final_df[final_df['f_id'] != final_df['b_id']]
                final_df = final_df[final_df['f_id'].str.split('_').str[0:2] != final_df['b_id'].str.split('_').str[0:2]]
                final_df = final_df[final_df['f_single_rule'] == final_df['b_single_rule']]
                final_df = final_df[final_df['f_type'] == final_df['b_type']]
                log_packet_result = pd.concat([log_packet_result, final_df])
                
            log_packet_result.reset_index(drop = True, inplace = True)

            data, meta = execute_ch("select distinct multiId, ruleId from dti.sect_rule_mapping", with_column_types = True)
            feats = [m[0] for m in meta]
            corr_mapping = pd.DataFrame(data, columns = feats) 
            corr_mapping.columns = ['corr', 'f_single_rule']
            corr_mapping['f_single_rule'] = corr_mapping['f_single_rule'].astype('int')
            
            log_packet_result = pd.merge(log_packet_result, corr_mapping, on = 'f_single_rule', how = 'left')
            log_packet_result['version'] = real_time
            
            threshold = 0.99
            log_packet_result.loc[log_packet_result['cos_sim'] <= threshold, ['ai_label']] = False
            log_packet_result['threshold'] = threshold
            
            log_packet_result.to_csv('test.csv')
            
            ## 상관분석 table insert 형변환
            int_list = ['f_single_rule', 'b_single_rule', 'corr']
            date_list = ['f_time', 'b_time', 'version']
            float_list = ['ai_rmse', 'cos_sim', 'threshold']
            str_list = ['f_type',  'b_type', 'ai_label', 'f_milli_time', 'b_milli_time']

            log_packet_result['corr'].fillna(0, inplace = True)  
            log_packet_result[int_list] = log_packet_result[int_list].astype('uint')
            log_packet_result[float_list] = log_packet_result[float_list].astype('float')
            log_packet_result[str_list] = log_packet_result[str_list].astype('str')
            uint_list = ['f_hash', 'b_hash']
            log_packet_result[uint_list] = log_packet_result[uint_list].astype('str')
            log_packet_result.drop_duplicates(inplace = True)
            
            ## 기존 데이터 중복 체크
            data, meta = execute_ch(""" select * from dti.sect_ai_corr_result """, with_column_types = True) 
            feats = [m[0] for m in meta]
            check_corr_result = pd.DataFrame(data, columns = feats)
            check_corr_result['f_hash'] = check_corr_result['f_hash'].astype('str')
            check_corr_result['b_hash'] = check_corr_result['b_hash'].astype('str')
            check_corr_result['hash_sum'] = check_corr_result['f_hash'].str.cat(check_corr_result['b_hash'], sep = ',')
            
            check_result = log_packet_result.copy()
            check_result['f_hash'] = check_result['f_hash'].astype('str')
            check_result['b_hash'] = check_result['b_hash'].astype('str')
            check_result['hash_sum'] = check_result['f_hash'].str.cat(check_result['b_hash'], sep = ',')
            
            check_result = check_result[check_result.hash_sum.isin(check_corr_result.hash_sum) == False]
            del check_result['hash_sum']
            
            # 밀리타임 설정
#             check_result['f_milli_time'] = check_result['f_milli_time'].astype('str')
#             check_result['b_milli_time'] = check_result['b_milli_time'].astype('str')
    
            execute_ch("INSERT INTO dti.sect_ai_corr_result VALUES", check_result.to_dict('records'))

            return "OK"


        except Exception as err:
            self.logger.error(err)
            self.logger.error(traceback.print_exc())            
            return None


def main(model_id, train, prediction, now=False):
    model_name = isExistModel(model_id)
    if model_name == None:
        sys.exit(1)
    else:
        # nc=NATS()
        loop = asyncio.get_event_loop()

        if train:
            if model_id == 3:
                if now:
                    loop.run_until_complete(Train1(loop, model_id, model_name).test_train())
                    sys.exit()
                else:
                    loop.run_until_complete(Train1(loop, model_id, model_name).start_train())
#                     sys.exit()
            else:
                self.logger.info("[TRAIN({})] model_i is invalid".format(model_id))
                sys.exit()

        if prediction:
            if model_id == 3:
                if now:
                    loop.run_until_complete(Prediction1(loop, model_id, model_name).test_pred())
                    sys.exit()
                else:
                    loop.run_until_complete(Prediction1(loop, model_id, model_name).start_pred())
#                     sys.exit()
            else:
                self.logger.info("[PREDICTION({})] model_id is invalid".format(model_id))
                sys.exit()
        try:
            loop.run_forever()
        finally:
            loop.close()

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        self.logger.info("python3 total_model.py -h or python3 total_model.py --help")
        sys.exit()

    parser = OptionParser(usage="Usage: python3 total_model.py -t -m [model_id] -now or python3 total_model.py -p -m [model_id] -now")
    parser.add_option("-t", action = "store_true", dest = "isTrain", default=False, help = "train")
    parser.add_option("-p", action = "store_true", dest = "isPred", default=False, help = "pred")
    parser.add_option("-n", action = "store_true", dest = "now", default=False, help = "once_run")
    parser.add_option("-m", type = int, dest = "MODEL_ID", help = "model_id")
    options, args = parser.parse_args()

    if options.isTrain == options.isPred:
        self.logger.error("you have to choose train or prediction")
        sys.exit()
    if options.MODEL_ID is None:
        self.logger.error("you have to input model id")
        sys.exit()

    main(options.MODEL_ID, options.isTrain, options.isPred, now=options.now)