import pandas as pd
from model import *
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from utils import *
import json
from base_component import *
from optparse import OptionParser
import time
from time import gmtime, strftime
from datetime import datetime, timedelta
import sys
sys.path.insert(0, sys.path[0])

class Train1(BaseComponent):
    def init(self):
        pass

    async def run(self, param):
        try:
#             param['logtime_s'] = '2021-08-01 00:00:00'
#             param['logtime_e'] = '2021-10-25 00:00:00'
            self.logger.info('********************* MODEL TRAINING START {} ~ {} *********************'.format(param['logtime_s'], param['logtime_e']))            

            """ 모델 버전 세팅 및 디렉토리 생성 """
            start = datetime.now().replace(microsecond=0) + timedelta(hours=9)
            version = start.strftime("%Y%m%d_%H")            
            if not os.path.exists(pwd+'/obj/'+version):
                logger.info('create directory: {}'.format(pwd+'/obj/'))
                os.makedirs(pwd+'/obj/'+version)
            self.logger.info('********************* MODEL VERSION : {} *********************'.format(version))            
            
            """ 상관분석 룰 데이터 수집 """                
            temp_f = pd.DataFrame(columns = ['corr', 'f_single_rule', 'b_single_rule'], data = get_corr().values)
            temp_b = pd.DataFrame(columns = ['corr', 'b_single_rule', 'f_single_rule'], data = get_corr().values)
            corr_temp = pd.concat([temp_f, temp_b])     
            
            """ 로그/패킷 데이터 불러오기 """            
            # 패킷 데이터 불러오기
            packet_sql = get_queries(param['logtime_s'], param['logtime_e'], query_type='p')
            packet = pd.DataFrame()
            for i in range(0,len(packet_sql)):
                data, meta = execute_ch(packet_sql[i], with_column_types = True)
                feats = [m[0] for m in meta]
                temp = pd.DataFrame(data, columns = feats)
                packet = pd.concat([packet, temp])   
            packet.reset_index(drop = True, inplace = True)
            packet = packet[(packet.src_ip != '') | (packet.dst_ip != '')]     
            self.logger.info('********************* PACKET DATA LOAD DONE : {} *********************'.format(packet.shape))  
            
            # 로그 자산정보 불러오기
            data, meta = execute_ch("select ip_address, device_id from dti.kdn_lgsys_L003 where ip_address!='' and device_id!=''", with_column_types = True)
            feats = [m[0] for m in meta]
            map_ip = pd.DataFrame(data, columns = feats)              
            map_ip.drop_duplicates(inplace = True)
            self.logger.info('********************* LOG MAP IP LOAD DONE : {} *********************'.format(map_ip.shape))
            
            # 로그 데이터 불러오기
            log_sql = get_queries(param['logtime_s'], param['logtime_e'], query_type='l')
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
                
            ## 테이블별 로그 데이터 병합
            log = pd.DataFrame()                
            for num in log_list:
                print("num_{}_shape : {}".format(num, globals()['log_{}'.format(num)].shape))                
                log = pd.concat([log, globals()['log_{}'.format(num)]])

            ## 로그 데이터 자산정보 매핑& 분석 데이터 필터링
            log = pd.merge(log, map_ip, on = 'device_id', how = 'left')
            log = log[log.ip_address != 'nan']
            log = log[log.ip_address.isna() == False]
            log['ip'] = log.ip_address
            log.reset_index(drop = True, inplace = True)
            if len(sql_009)>0: 
                log = log[log.hash.isin(log_009.hash) == False]                  
            self.logger.info('********************* LOG DATA LOAD DONE : {} *********************'.format(log.shape))  

            """ 컬럼명 변환 및 기본 처리 """
            log['single_rule_log'] = log.single_rule        
            log.rename(columns = {'plant_id' : 'id'}, inplace = True)
            log.drop_duplicates(inplace = True)
            log.reset_index(drop = True, inplace = True)            
            packet['single_rule_packet'] = packet.single_rule            
            packet.rename(columns = {'make_id' : 'id'}, inplace = True)            
            packet.drop_duplicates(inplace =True)
            packet.reset_index(drop = True, inplace = True)                                
            packet['ip'] = packet.src_ip

            """ 로그/패킷 데이터 병합 """
            con_df = pd.concat([log, packet])
            con_df.sort_values('time', inplace = True)
            self.logger.info('********************* LOG + PACKET COMBINED DATA : {} *********************'.format(log.shape))  

            """ 패턴 다양성을 위한 n skip 진행 """
            ## skip_n
            skip_n = self.model_config["common"]["skip_n"]
            temp_df = con_df.copy()
            if skip_n > 1:
                for x in range(1, skip_n):
                    self.logger.info('********************* SKIP {} *********************'.format(x))  
                    for i in range(len(con_df)-x-1):
                        temp_df = pd.concat([temp_df, con_df.iloc[[i, i+x+1]]])
                con_df = temp_df.copy()                
            del con_df['time']     
            con_df.reset_index(drop = True, inplace = True)
            
            """ 학습 데이터 생성 """
            df = con_df[['message_id','single_rule','hash']].copy()
            df["message_id"] = np.select([df['message_id'].str.startswith('H'), df['message_id'].str.startswith('L')],  ['packet', 'log'], default=np.nan)
            f_list = ['f_type','f_single_rule','f_hash']
            b_list = ['b_type','b_single_rule','b_hash']
            result_col = f_list + b_list
            result_df = pd.DataFrame(columns = result_col)
            result_df[f_list] = df[0:len(df)-1].reset_index(drop = True)
            result_df[b_list] = df[1:len(df)].reset_index(drop = True)  
            result_df = pd.merge(result_df, corr_temp, on = ['f_single_rule', 'b_single_rule'], how = 'left')
            test_df = result_df[(result_df['f_type'] != result_df['b_type']) & (result_df['corr'].isna() == True)].reset_index(drop = True)
            result_df = result_df[result_df['corr'].isna() == False].reset_index(drop = True)
            
            ## upsampling
            temp_result = result_df.copy()
            temp_result[['b_single_rule','f_single_rule']] = temp_result[['f_single_rule','b_single_rule']].astype('str')
            temp_result['rule_sum'] = temp_result['f_single_rule'].str.cat(temp_result['b_single_rule'], sep = ',') ## 룰 패턴 파악
            filter_list = list(set(list(temp_result['rule_sum']))) ## 룰 unique list
            max_cnt = temp_result['rule_sum'].value_counts().max() ## 최대 룰 개수
            temp_df = pd.DataFrame()
            for i in filter_list: ## 최대 룰 개수에 맞춰 upsampling 진행
                temp_data = temp_result[temp_result['rule_sum'].isin([str(i)]) == True]
                if len(temp_data) < max_cnt:
                    temp_data = temp_data.sample(max_cnt, replace = True)
                    temp_df = pd.concat([temp_df, temp_data])
                else:
                    temp_df = pd.concat([temp_df, temp_data])   
            temp_df.reset_index(drop = True, inplace = True)
            result_df = temp_df.copy()                        
            result_col = list(result_df)
            self.logger.info('********************* UPSAMPLING DATA SHAPE : {} *********************'.format(result_df.shape))            
            
            ## columns list for onehotencoding
            log_rate_list = self.model_config["common"]["log_rate_list"].split(', ')
            packet_rate_list = self.model_config["common"]["packet_rate_list"].split(', ')
            
            ## onehotencoding            
            log_onehotencoder = OneHotEncoder(categories= 'auto', handle_unknown='ignore').fit(log[log_rate_list].astype('str').values)
            save_obj(log_onehotencoder, path= version + '/' + "log_"+ self.model_config["common"]["onehotencoder_save"])
            log_onehot = pd.DataFrame(index = log.hash, columns = log_onehotencoder.get_feature_names(log_rate_list), data = log_onehotencoder.transform(log[log_rate_list].astype('str').values).toarray())
            packet_onehotencoder = OneHotEncoder(categories= 'auto', handle_unknown='ignore').fit(packet[packet_rate_list].astype('str').values)            
            save_obj(packet_onehotencoder, path= version + '/' + "packet_"+ self.model_config["common"]["onehotencoder_save"])        
            packet_onehot = pd.DataFrame(index = packet.hash, columns = packet_onehotencoder.get_feature_names(packet_rate_list),data = packet_onehotencoder.transform(packet[packet_rate_list].astype('str').values).toarray())
            
            ## 전처리 데이터 병합
            col_list = result_col + list(log_onehot) + list(packet_onehot)
            log_result = result_df[result_df['f_type'] == 'log'].copy()
            packet_result = result_df[result_df['f_type'] == 'packet'].copy()
            log_result = pd.merge(log_result, log_onehot, left_on = 'f_hash', right_on = 'hash', how = 'left')
            log_result = pd.merge(log_result, packet_onehot, left_on = 'b_hash', right_on = 'hash', how = 'left')
            packet_result = pd.merge(packet_result, packet_onehot, left_on = 'f_hash', right_on = 'hash', how = 'left')
            packet_result = pd.merge(packet_result, log_onehot, left_on = 'b_hash', right_on = 'hash', how = 'left')                                    
            result_df = pd.concat([packet_result, log_result])
            result_df.drop(result_col, axis = 1, inplace = True)
            
            ## oversampling
            result_df = pd.concat([result_df, result_df])
            result_df = pd.concat([result_df, result_df])
            X_data = result_df.copy()
            self.logger.info('********************* OVERSAMPLING DATA SHAPE : {} *********************'.format(X_data.shape))            
            
            """ CNN MODEL """
            ## 모델 세팅
            model_config = self.model_config
            model_config["x_datashape"] = X_data.shape            
            for k in model_config.keys():
                if model_config[k]==None:
                    self.logger.info("MODEL CONFIG {} HAS TO BE SET UP...")                    
            model = corr_model(config=model_config, mode="train", name=self.model_config["common"]["model_name"] + '_' + version)   
            
            ## 모델 FITTING
            self.logger.info('********************* MODEL FITTING START *********************')
            res, ai_history = model.optimize_nn(X=X_data, Y=X_data)                   
            self.logger.info('********************* MODEL FITTING FINISH *********************')
            
            """ CNN PREDICTION """            
            ## 예측 시행            
            pred = model.predict(X_data)
            rmse = model.rmse_custom(X_data, pred, 1).numpy()
            temp_df = pd.DataFrame(rmse)
            z_scores, threshold = detect_mad_outliers(rmse, threshold= 2.5)
            save_obj({'threshold': threshold}, path=version + '/' + self.model_config["common"]["threshold_path"])
            scaler = MinMaxScaler().fit(np.array(rmse).reshape(-1,1))
            save_obj(scaler, path= version + '/' + self.model_config["common"]["minmax_rmse_save"])
            self.logger.info('********************* MODEL RESULT ----- THRESHOLD : {} *********************'.format(threshold))
            self.logger.info('********************* TRAIN RMSE RESULT *********************')
            self.logger.info(temp_df.value_counts())

            ## 성능 검증 예측 시행 (전처리 데이터 병합)
            log_result = test_df[test_df['f_type'] == 'log'].copy()
            packet_result = test_df[test_df['f_type'] == 'packet'].copy()
            log_result = pd.merge(log_result, log_onehot, left_on = 'f_hash', right_on = 'hash', how = 'left')
            log_result = pd.merge(log_result, packet_onehot, left_on = 'b_hash', right_on = 'hash', how = 'left')
            packet_result = pd.merge(packet_result, packet_onehot, left_on = 'f_hash', right_on = 'hash', how = 'left')
            packet_result = pd.merge(packet_result, log_onehot, left_on = 'b_hash', right_on = 'hash', how = 'left')                                    
            test_df = pd.concat([packet_result, log_result])
            result_col.remove('rule_sum')
            test_df.drop(result_col, axis = 1, inplace = True)
            X_test = test_df.copy()            
            pred = model.predict(X_test)
            rmse = model.rmse_custom(X_test, pred, 1).numpy()
            temp_df = pd.DataFrame(rmse)
            self.logger.info('********************* MODEL RESULT ----- THRESHOLD : {} *********************'.format(threshold))
            self.logger.info('********************* TRAIN RMSE RESULT *********************')
            self.logger.info(temp_df.value_counts())            
                
            ## 학습이력 정의
            validation_time = datetime.now().replace(microsecond=0) + timedelta(hours=9)
            history_df = pd.DataFrame({'loss': ai_history.history['loss']})
            history_df['model_id'] = self.model_id
            history_df['epoch'] = list(range(1, len(history_df)+1))
            history_df['model_type'] = 'corr'
            history_df['data_shape'] = [X_data.shape for i in range(len(history_df))]
            history_df['version'] = start
            history_df['train_time'] = start               
            history_df['validation_time'] = validation_time
            history_df = history_df[['model_id', 'model_type', 'train_time', 'validation_time', 'version', 'loss', 'epoch', 'data_shape']]
            execute_ch("INSERT INTO dti.motie_ai_history VALUES", history_df.to_dict('records'))              
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
            param['logtime_s'] = '2021-10-01 00:00:00'
            param['logtime_e'] = '2021-10-14 00:00:00'            
            self.logger.info('********************* MODEL PREDICTION START {} ~ {} *********************'.format(param['logtime_s'], param['logtime_e']))            
            """ 모델 버전 세팅 및 디렉토리 생성 """
            start = datetime.now().replace(microsecond=0) + timedelta(hours=9)
            version = start.strftime("%Y%m%d_%H")
            new_time = start
            real_time = start
            for timerange in range(5000):
                if not os.path.exists(pwd + '/{}/{}'.format(self.model_config["common"]["path"], self.model_config["common"]["model_name"] + '_' + version +'/saved_model.pb')):
                    new_time = start - timedelta(hours=timerange+1)
                    version = new_time.strftime("%Y%m%d_%H")     
                else:
                    break
            for timerange in range(5000):
                if len(os.listdir(pwd +'/obj/{}'.format(version))) != 4:                         
                    new_time = start - timedelta(hours=timerange+1)
                    version = new_time.strftime("%Y%m%d_%H")     
                else:
                    break                                                
            self.logger.info('********************* MODEL VERSION : {} *********************'.format(version))            
            model_name = self.model_config["common"]["model_name"] + '_' + version
            minmax_rmse_save = version + '/' + self.model_config["common"]["minmax_rmse_save"]
            threshold_path = version + '/' + self.model_config["common"]["threshold_path"]       
            
            """ 상관분석 룰 데이터 수집 """                
            temp_f = pd.DataFrame(columns = ['corr', 'f_single_rule', 'b_single_rule'], data = get_corr().values)
            temp_b = pd.DataFrame(columns = ['corr', 'b_single_rule', 'f_single_rule'], data = get_corr().values)
            corr_temp = pd.concat([temp_f, temp_b])               
            corr_temp[['f_single_rule', 'b_single_rule']] = corr_temp[['f_single_rule', 'b_single_rule']].astype('str')

            """ 로그/패킷 데이터 불러오기 """            
            # 패킷 데이터 불러오기
            packet_sql = get_queries(param['logtime_s'], param['logtime_e'], query_type='p')
            packet = pd.DataFrame()
            for i in range(0,len(packet_sql)):
                data, meta = execute_ch(packet_sql[i], with_column_types = True)
                feats = [m[0] for m in meta]
                temp = pd.DataFrame(data, columns = feats)
                packet = pd.concat([packet, temp])   
            packet.reset_index(drop = True, inplace = True)
            packet = packet[(packet.src_ip != '') | (packet.dst_ip != '')]     
            self.logger.info('********************* PACKET DATA LOAD DONE : {} *********************'.format(packet.shape))  
            
            # 로그 자산정보 불러오기
            data, meta = execute_ch("select ip_address, device_id from dti.kdn_lgsys_L003 where ip_address!='' and device_id!=''", with_column_types = True)
            feats = [m[0] for m in meta]
            map_ip = pd.DataFrame(data, columns = feats)              
            map_ip.drop_duplicates(inplace = True)
            self.logger.info('********************* LOG MAP IP LOAD DONE : {} *********************'.format(map_ip.shape))
            
            # 로그 데이터 불러오기
            log_sql = get_queries(param['logtime_s'], param['logtime_e'], query_type='l')
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
            
            ## 테이블별 로그 데이터 병합
            log = pd.DataFrame()                
            for num in log_list:
                print("num_{}_shape : {}".format(num, globals()['log_{}'.format(num)].shape))                
                log = pd.concat([log, globals()['log_{}'.format(num)]])

            ## 로그 데이터 자산정보 매핑& 분석 데이터 필터링
            log = pd.merge(log, map_ip, on = 'device_id', how = 'left')
            log = log[log.ip_address != 'nan']
            log = log[log.ip_address.isna() == False]
            log['ip'] = log.ip_address
            log.reset_index(drop = True, inplace = True)               
            self.logger.info('********************* LOG DATA LOAD DONE : {} *********************'.format(log.shape))                  
                
            """ 컬럼명 변환 및 기본 처리 """
            log['single_rule_log'] = log.single_rule        
            log.rename(columns = {'plant_id' : 'id'}, inplace = True)
            log.drop_duplicates(inplace = True)
            log.reset_index(drop = True, inplace = True)            
            log['time'] = pd.to_datetime(log['time'].str[:14], format = '%Y%m%d%H%M%S', errors = 'raise')
            log['milli_time'] = log['time']

            packet['single_rule_packet'] = packet.single_rule            
            packet.rename(columns = {'make_id' : 'id'}, inplace = True)            
            packet.drop_duplicates(inplace =True)
            packet.reset_index(drop = True, inplace = True)                                
            packet['ip'] = packet.src_ip
            packet['milli_time'] = pd.to_datetime(packet['time'].str[:16], format = '%Y%m%d%H%M%S%f', errors = 'raise')
            packet['time'] = pd.to_datetime(packet['time'].astype('str').str[0:14], format = '%Y%m%d%H%M%S', errors = 'raise')
            packet['manufacturer_name'] = np.where(packet.id.str[13:15] == 'GE','GE','ABB')
                
            """ 로그/패킷 데이터 병합 """
            con_df = pd.concat([log, packet])
            con_df.sort_values('time', inplace = True)
            self.logger.info('********************* LOG + PACKET COMBINED DATA : {} *********************'.format(log.shape))                                                                                  

            """ 이미 저장된 데이터 HASH 불러오기 """
            data, meta = execute_ch(""" 
                select * from dti.motie_ai_single_packet 
                where time between '{start_date}' and '{end_date}' """.replace('{start_date}', param['logtime_s']).replace('{end_date}', param['logtime_e']), with_column_types = True) 
            feats = [m[0] for m in meta]
            check_packet_result = pd.DataFrame(data, columns = feats)

            data, meta = execute_ch("""
                select * from dti.motie_ai_single_log 
                where time between '{start_date}' and '{end_date}'""".replace('{start_date}', param['logtime_s']).replace('{end_date}',  param['logtime_e']), with_column_types = True) 
            feats = [m[0] for m in meta]
            check_log_result = pd.DataFrame(data, columns = feats)
            
            data, meta = execute_ch(""" 
                select * from dti.motie_ai_corr_prep_v2 
                where (parseDateTimeBestEffortOrZero(toString(loged_time)) between '{start_date}' and '{end_date}') or (parseDateTimeBestEffortOrZero(toString(packet_time)) between '{start_date}' and '{end_date}')""".replace('{start_date}', param['logtime_s']).replace('{end_date}',  param['logtime_e']), with_column_types = True) 
            feats = [m[0] for m in meta]
            check_corr_prep = pd.DataFrame(data, columns = feats)

            data, meta = execute_ch("""
                select * from dti.motie_ai_corr_result_v3
                where f_time between '{start_date}' and '{end_date}'""".replace('{start_date}', param['logtime_s']).replace('{end_date}',  param['logtime_e']), with_column_types = True) 
            feats = [m[0] for m in meta]
            check_corr_result = pd.DataFrame(data, columns = feats)
            check_corr_result[['f_hash', 'b_hash']] = check_corr_result[['f_hash', 'b_hash']].astype('str')
            check_corr_result['hash_sum'] = check_corr_result['f_hash'].str.cat(check_corr_result['b_hash'], sep = ',')                        
    
            ## UTILS로 옮겨야됨
#             def batch_input(data, input_bs, type = ['log', 'packet']):
#                 if len(data) > input_bs:
#                     start_num = 0
#                     end_num = 0
#                     for i in range(int(len(data)/input_bs) + 1):
#                         end_num += input_bs
#                         if end_num > len(data):            
#                             end_num = len(data)                        
#                         temp = data[start_num : end_num].copy()
#                         temp_time =  datetime.now().replace(microsecond=0) + timedelta(hours=9)
#                         temp['version'] = temp_time
#                         if type == 'packet':
#                             self.logger.info('********************* PACKET INSERT VERSION : {} *********************'.format(temp_time))
#                             self.logger.info('********************* PACKET INSERT SHAPE : {} *********************'.format(temp.shape))
#                             execute_ch("INSERT INTO dti.motie_ai_single_packet VALUES", temp.to_dict('records'))
#                         elif type == 'log':
#                             self.logger.info('********************* LOG INSERT VERSION : {} *********************'.format(temp_time))
#                             self.logger.info('********************* LOG INSERT SHAPE : {} *********************'.format(temp.shape))                            
#                             execute_ch("INSERT INTO dti.motie_ai_single_log VALUES", temp.to_dict('records'))
#                         print('Insert data ({} : {})'.format(start_num, end_num))
#                         start_num += input_bs
#                     else:
#                         if type == 'packet':
#                             temp_time =  datetime.now().replace(microsecond=0) + timedelta(hours=9)
#                             data['version'] = temp_time
#                             self.logger.info('********************* PACKET INSERT VERSION : {} *********************'.format(temp_time))
#                             self.logger.info('********************* PACKET INSERT SHAPE : {} *********************'.format(data.shape))
#                             execute_ch("INSERT INTO dti.motie_ai_single_packet VALUES", data.to_dict('records'))
#                         elif type == 'log':
#                             temp_time =  datetime.now().replace(microsecond=0) + timedelta(hours=9)
#                             data['version'] = temp_time
#                             self.logger.info('********************* LOG INSERT VERSION : {} *********************'.format(temp_time))
#                             self.logger.info('********************* LOG INSERT SHAPE : {} *********************'.format(data.shape))                            
#                             execute_ch("INSERT INTO dti.motie_ai_single_log VALUES", data.to_dict('records'))

            """ 단일 이벤트 DB 저장 """
            if log.shape[0] == 0:
                single_rule_packet = packet_save_single_rule(packet, 'packet', start)
                check_packet = single_rule_packet.copy()
                check_packet = check_packet[check_packet.hash.isin(check_packet_result.hash) == False] 
                batch_input(check_packet, 1000, 'packet')          

            elif packet.shape[0] == 0:
                single_rule_log = log_save_single_rule(log, 'log', start)
                check_log = single_rule_log.copy()
                check_log = check_log[check_log.hash.isin(check_log_result.hash) == False]
                batch_input(check_log, 1000, 'log')             
            else:             
                single_rule_packet = packet_save_single_rule(packet, 'packet', start)
                check_packet = single_rule_packet.copy()
                check_packet = check_packet[check_packet.hash.isin(check_packet_result.hash) == False]
                temp_time =  datetime.now().replace(microsecond=0) + timedelta(hours=9)
                batch_input(check_packet, 1000, 'packet')
                
                single_rule_log = log_save_single_rule(log, 'log', start)
                check_log = single_rule_log.copy()
                check_log = check_log[check_log.hash.isin(check_log_result.hash) == False]
                temp_time =  datetime.now().replace(microsecond=0) + timedelta(hours=9)
                batch_input(check_log, 1000, 'log')

                """ 탐지용 데이터 필터링 """
                if len(sql_009)>0: 
                    log = log[log.hash.isin(log_009.hash) == False]   
                    
                """ 보안서버 이벤트 제외 """
                rm_list = ['192.168.50.', '172.19.1.152', '172.19.2.152']
                for i in rm_list:
                    log = log[log.ip.str.contains(i) == False]                    
                self.logger.info('********************* LOG REMOVE SECURITY SERVER DATA SHAPE : {} *********************'.format(log.shape))

                """ 과다 & 초과 데이터 병합 """        
                excess_id = (8,   9,  11,  12,  14,  15,  17,  18,  20,  21,  23,  24,  26,
                             27,  29,  30,  32,  33,  34,  36,  37,  41,  42,  44,  45,  47,
                             48,  50,  51,  53,  54,  56,  57,  59,  60,  62,  63,  65,  66,
                             67,  79, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
                             174, 175, 176, 177, 178, 179, 180, 181, 182, 183)
                remainder_log = log[log.single_rule_log.isin(excess_id) != True]
                excess_log = log[log.single_rule_log.isin(excess_id) == True]            
                excess_log.sort_values('hash', inplace = True)
                excess_log.drop_duplicates(['id', 'machine_no', 'manufacturer_name', 'single_rule_log', 'loged_time', 'ip'], keep = 'first', inplace = True)                
                log = pd.concat([remainder_log, excess_log])
                self.logger.info('********************* LOG MERGE DATA SHAPE : {} *********************'.format(log.shape))                                

                
                """ 호기/장비 구분하여 데이터 생성 """
                temp_con_df = pd.concat([log, packet])                
                total_result_df = pd.DataFrame()
                total_final_result = pd.DataFrame()
                
                for i in ['1', '2']:
                    for j in ['ABB', 'GE']:
                        con_df = pd.DataFrame()
                        con_df = pd.concat([con_df, temp_con_df[(temp_con_df.machine_no == '{}호기'.format(i))&(temp_con_df.manufacturer_name == j)]])
                        con_df = pd.concat([con_df, temp_con_df[(temp_con_df.unit_id.str[-1] == i)&(temp_con_df.manufacturer_name == j)]])   
                        con_df.sort_values('time', inplace = True)
                        con_df['milli_time'] = con_df['milli_time'].astype('str')
                        
                        ## 패턴 다양성을 위한 n skip 진행
                        skip_n = self.model_config["common"]["skip_n"]
                        temp_df = con_df.copy()
                        if skip_n > 1:
                            for x in range(1, skip_n):
                                self.logger.info('********************* SKIP {} *********************'.format(x))  
                                for i in range(len(con_df)-x-1):
                                    temp_df = pd.concat([temp_df, con_df.iloc[[i, i+x+1]]])
                            temp_df.reset_index(drop = True, inplace = True)
                            con_df = temp_df.copy()                

                        ## 전처리 데이터 저장 형식 변경
                        con_df[list(con_df.select_dtypes(include = 'object'))] = con_df[list(con_df.select_dtypes(include = 'object'))].astype('str')
                        con_df['single_rule_packet'] = con_df['single_rule_packet'].astype('str')
                        con_df['plant_id'] = con_df['id']
                        con_df['make_id'] = con_df['id']
                        con_df['version'] = real_time
                                                
                        ## 전처리 데이터 중복 INSERT 체크                        
                        check_prep = con_df[con_df.hash.isin(check_corr_prep.hash) == False].copy()
                        check_prep.fillna(' ', inplace = True)
                        check_prep.replace('nan', ' ', inplace = True)
                        execute_ch("INSERT INTO dti.motie_ai_corr_prep_v2 VALUES", check_prep.to_dict('records'))
                        self.logger.info('********************* PREPROCESSING DATA INSERT SHAPE :{} *********************'.format(con_df.shape))
                        
                        """ 예측 데이터 생성 """
                        df = con_df[['message_id','single_rule','hash']].copy()
                        df["message_id"] = np.select([df['message_id'].str.startswith('H'), df['message_id'].str.startswith('L')],  ['packet', 'log'], default=np.nan)
                        f_list = ['f_type','f_single_rule','f_hash']
                        b_list = ['b_type','b_single_rule','b_hash']
                        result_col = f_list + b_list
                        result_df = pd.DataFrame(columns = result_col)
                        result_df[f_list] = df[0:len(df)-1].reset_index(drop = True)
                        result_df[b_list] = df[1:len(df)].reset_index(drop = True)  
                        result_df = pd.merge(result_df, corr_temp, on = ['f_single_rule', 'b_single_rule'], how = 'left')
                        result_df = result_df[result_df['f_type'] != result_df['b_type']]
                        result_col = list(result_df)                        
                        self.logger.info('********************* PREDICTION DATA SHAPE : {} *********************'.format(result_df.shape))            
                        
                        """ 데이터 필터링 및 형변환 """
                        df_all = con_df[['time','ip','message_id','single_rule','hash','id', 'milli_time']].copy()                                
                        df_all["message_id"] = np.select([df_all['message_id'].str.startswith('H'), df_all['message_id'].str.startswith('L')],  ['packet', 'log'], default=np.nan)      

                        f_list = ['f_time','f_ip','f_type','f_single_rule','f_hash','f_id', 'f_milli_time']
                        b_list = ['b_time','b_ip','b_type','b_single_rule','b_hash','b_id', 'b_milli_time']
                        final_result_col = f_list + b_list
                        final_result = pd.DataFrame(columns = final_result_col)
                        final_result[f_list] = df_all[0:len(df_all)-1].reset_index(drop=True)
                        final_result[b_list] = df_all[1:len(df_all)].reset_index(drop=True)
                        final_result = final_result[final_result['f_type'] != final_result['b_type']]
                        
                        """ 호기/장비 데이터 결합"""
                        total_result_df = pd.concat([total_result_df, result_df])
                        total_final_result = pd.concat([total_final_result, final_result])
                        self.logger.info('********************* TOTAL FINAL RESULT SHAPE :{} *********************'.format(final_result.shape))
                                    
                """ 결합된 데이터 copy """
                result_df = total_result_df.drop_duplicates()
                final_result = total_final_result.drop_duplicates() ## 최종 결과 데이터에 필요한 columns 보유
                ## columns list for onehotencoding                
                log_rate_list = self.model_config["common"]["log_rate_list"].split(', ')
                packet_rate_list = self.model_config["common"]["packet_rate_list"].split(', ')

                ## onehotencoding       
                log_onehotencoder = load_obj(path=version + '/' + "log_"+ self.model_config["common"]["onehotencoder_save"])
                log_onehot = pd.DataFrame(index = log.hash, columns = log_onehotencoder.get_feature_names(log_rate_list), data = log_onehotencoder.transform(log[log_rate_list].astype('str').values).toarray())
                packet_onehotencoder = load_obj(path=version + '/' + "packet_"+ self.model_config["common"]["onehotencoder_save"])
                packet_onehot = pd.DataFrame(index = packet.hash, columns = packet_onehotencoder.get_feature_names(packet_rate_list), data = packet_onehotencoder.transform(packet[packet_rate_list].astype('str').values).toarray())
                
                ## 전처리 데이터 병합
                col_list = result_col + list(log_onehot) + list(packet_onehot)
                log_result = result_df[result_df['f_type'] == 'log'].copy()
                packet_result = result_df[result_df['f_type'] == 'packet'].copy()
                log_result = pd.merge(log_result, log_onehot, left_on = 'f_hash', right_on = 'hash', how = 'left')
                log_result = pd.merge(log_result, packet_onehot, left_on = 'b_hash', right_on = 'hash', how = 'left')
                packet_result = pd.merge(packet_result, packet_onehot, left_on = 'f_hash', right_on = 'hash', how = 'left')
                packet_result = pd.merge(packet_result, log_onehot, left_on = 'b_hash', right_on = 'hash', how = 'left')
                result_df = pd.concat([packet_result, log_result])
                result_df.drop_duplicates(inplace = True)
                result_df.reset_index(drop = True, inplace = True)
                result_df.fillna(0, inplace = True)
                train_col = list(result_df)
                self.logger.info('********************* PREDICTION DATA SHAPE :{} *********************'.format(result_df.shape))            

                """ CNN MODEL """
                ## 모델 세팅                                
                for i in result_col:
                    train_col.remove(i)
                X_data = result_df[train_col].copy()                   
                model_config = self.model_config
                model_config["x_datashape"] = X_data.shape
                for k in model_config.keys():
                    if model_config[k]==None:
                        self.logger.info("MODEL CONFIG {} HAS TO BE SET UP...")
                model = corr_model(config=self.model_config, mode="predict", name=model_name)        
                
                ## 예측 시행            
                pred = model.predict(X_data)
                rmse = model.rmse_custom(X_data, pred, 1).numpy()                              
                minmaxscaler = load_obj(path=minmax_rmse_save)                
                rmse_scaled = minmaxscaler.transform(np.array(rmse).reshape(-1,1)) 
                threshold = load_obj(path=threshold_path)['threshold']                                
                result_df['ai_rmse'] = rmse
                result_df['ai_rmse_scaled'] = rmse_scaled
                result_df['ai_label'] = rmse <= threshold
                result_df['version'] = real_time                
                self.logger.info('********************* AI RMSE VALUE COUNTS ********************* \n {}'.format(result_df['ai_rmse'].value_counts()))
                self.logger.info('********************* AI LABEL VALUE COUNTS ********************* \n {}'.format(result_df['ai_label'].value_counts()))
                
                """ 자산 매핑 진행 """
                ## 패킷 데이터 ip 변경 (f일때 dst_ip, b일때 src_ip)                
                corr_result = pd.merge(result_df, final_result, on = list(set(list(result_df)) & set(list(final_result))))
                packet_ip = pd.DataFrame(columns = ['f_hash','b_hash','src_ip','dst_ip'], data = packet[['hash','hash','src_ip','dst_ip']].values)
                packet_f = pd.merge(corr_result,packet_ip[['f_hash','src_ip','dst_ip']], on = ['f_hash'], how = 'left')
                packet_b = pd.merge(corr_result,packet_ip[['b_hash','src_ip','dst_ip']], on = ['b_hash'], how = 'left')
                packet_f['f_ip'] = packet_f['dst_ip']
                packet_b['b_ip'] = packet_b['src_ip']
                corr_result = pd.concat([packet_f, packet_b])
                corr_result.dropna(axis = 0, inplace = True)

                ## 자산 데이터 불러오기
                data, meta = execute_ch("select distinct assetIp, assetNm from dti.view_motie_pwer_asset_info", with_column_types = True)
                feats = [m[0] for m in meta]
                asset_df = pd.DataFrame(data, columns = ['f_ip','f_asset'])
                
                ## 동일 자산에서 발생한 상관분석만 인정
                self.logger.info('********************* BEFORE MAPPING RESULT DATA SHAPE : {} *********************'.format(result_df.shape))
                asset_df['b_ip'] = asset_df['f_ip']
                asset_df['b_asset'] = asset_df['f_asset']
                temp = pd.merge(corr_result, asset_df[['f_ip','f_asset']], on = ['f_ip'], how = 'left')
                temp = pd.merge(temp, asset_df[['b_ip','b_asset']], on = ['b_ip'], how = 'left')
                temp = temp[(temp['corr'] != 0) & (temp['f_asset'] == temp['b_asset'])]
                corr_result = temp.copy()
                if len(corr_result) > 0:
                    self.logger.info('********************* AFTER MAPPING RESULT DATA SHAPE : {} *********************'.format(corr_result.shape))
                    """ 결과 데이터 DB INSERT """
                    ## 데이터 형변환
                    date_list = ['f_time', 'b_time', 'version']
                    int_list = ['f_single_rule', 'b_single_rule', 'corr']
                    float_list = ['ai_rmse', 'ai_rmse_scaled']
                    str_list = [i for i in list(corr_result) if i not in date_list + int_list + float_list]

                    for i in date_list:
                        corr_result[i] = pd.to_datetime(corr_result[i])
                    corr_result[int_list] = corr_result[int_list].astype('uint')
                    corr_result[float_list] = corr_result[float_list].astype('float')
                    corr_result[str_list] = corr_result[str_list].astype('str')

                    ## 중복 데이터 필터링
                    corr_result['hash_sum'] = corr_result['f_hash'].str.cat(corr_result['b_hash'], sep = ',')
                    corr_result = corr_result[corr_result.hash_sum.isin(check_corr_result.hash_sum) == False]
                    
                    if len(corr_result) > 0 :
                        ## 최종 결과 버전 설정 및 데이터 INSERT
                        corr_result['version'] = datetime.now().replace(microsecond=0) + timedelta(hours=9)      
                        print(corr_result.info())
#                         execute_ch("INSERT INTO dti.motie_ai_corr_result_v2 VALUES", corr_result.to_dict('records'))
                        execute_ch("INSERT INTO dti.motie_ai_corr_result_v3 VALUES", corr_result.to_dict('records'))
                        self.logger.info('********************* FINAL RESULT INSERT DONE : {} *********************'.format(corr_result.shape))
                    else :     
                        self.logger.info('********************* FINAL RESULT LENGTH IS ZERO *********************')
                else :
                    self.logger.info('********************* FINAL RESULT LENGTH IS ZERO *********************')

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
            if model_id == 1:
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
            if model_id == 1:
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