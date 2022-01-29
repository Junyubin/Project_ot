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
sys.path.insert(0, sys.path[0])

class Train1(BaseComponent):
    def init(self):
        pass

    async def run(self, param):
        try:
            start = datetime.now().replace(microsecond=0) + timedelta(hours=9)
            version = start.strftime("%Y%m%d_%H")
            print(version)
            
            if not os.path.exists(pwd+'/obj/'+version):
                logger.info('create directory: {}'.format(pwd+'/obj/'))
                os.makedirs(pwd+'/obj/'+version)
                
            corr_temp = get_corr()
            temp = corr_temp.copy()
            temp_2 = corr_temp.copy()
            temp.rename(columns = {"single_rule_log" : 'f_single_rule'}, inplace = True)
            temp_2.rename(columns = {"single_rule_log" : 'b_single_rule'}, inplace = True)
            corr_temp_2 =pd.merge(temp, temp_2, on = ['corr', 'corr'], how = 'outer')
            corr_temp_2.drop_duplicates(inplace = True)
                
            if self.model_config["data_load"] == 1:
                temp_version = version
                for timerange in range(100):
                    if not os.path.exists(pwd + '/{}/{}'.format(self.model_config["common"]["path"], self.model_config["common"]["model_name"] + '_' + temp_version)):
                        temp_new_time = start - timedelta(hours=timerange+1)
                        temp_version = temp_new_time.strftime("%Y%m%d_%H")
                    else:
                        break
                con_df = pd.read_csv(pwd + '/data/{}_save_train_data.csv'.format(temp_version))
                log = pd.read_csv(pwd + '/data/{}_save_train_log_data.csv'.format(temp_version))
                packet = pd.read_csv(pwd + '/data/{}_save_train_packet_data.csv'.format(temp_version))

                self.logger.info('********************* SAVED DATA LOAD DONE *********************')            
            else:
                 ## data load
                data, meta = execute_ch("select ip_address, device_id from dti.kdn_lgsys_L003 where ip_address!='' and device_id!=''", with_column_types = True)
                feats = [m[0] for m in meta]
                map_ip = pd.DataFrame(data, columns = feats)              
                map_ip.drop_duplicates(inplace = True)
                self.logger.info('********************* LOG MAP IP LOAD DONE *********************')            

                log_sql = get_queries(param['logtime_s'], param['logtime_e'], query_type='l')
                globals()['sql_005'], globals()['sql_009'], globals()['sql_011'], log_list = [], [], [], []

                for sql in log_sql:
                    if 'sect_lgsys_L005' in sql:
                        sql_005.append(sql)                
                    elif 'sect_lgsys_L009' in sql:
                        sql_009.append(sql)
                    elif 'sect_lgsys_L011' in sql:
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

                log.rename(columns = {'single_rule' : 'single_rule_log'}, inplace = True)
                log['single_rule'] = log.single_rule_log        
                log.rename(columns = {'plant_id' : 'id'}, inplace = True)
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
                self.logger.info('********************* SINGLE LOG DATA SHAPE : {} *********************'.format(log.shape))       

                con_df = log.copy()
                con_df.sort_values('time', inplace = True)

                ## skip_n
                skip_n = self.model_config["common"]["skip_n"]
                temp_df = con_df.copy()
                for x in range(1, skip_n):
                    print("skip_n : {}".format(x))
                    for i in range(len(con_df)-x-1):
                        temp_df = pd.concat([temp_df, con_df.iloc[[i, i+x+1]]])

                con_df = temp_df.copy()
                
                con_df.to_csv(pwd + '/data/{}_save_train_data.csv'.format(version))
                log.to_csv(pwd + '/data/{}_save_train_log_data.csv'.format(version))
                ###################################################### TRAIN DATA LOAD ######################################################
            
            del con_df['time']     
            con_df.reset_index(drop = True, inplace = True)

            ## make result table
            df = con_df[['message_id','single_rule','hash']].copy()
            df["message_id"] = np.select([df['message_id'].str.startswith('H'), df['message_id'].str.startswith('L')],  ['packet', 'log'], default=np.nan)

            ## result_df col_list
            f_list = ['f_type','f_single_rule','f_hash']
            b_list = ['b_type','b_single_rule','b_hash']
            result_col = f_list + b_list

            ## make result table
            result_df = pd.DataFrame(columns = result_col)
            result_df[[i for i in list(result_df) if i.startswith('f_')]] = df[0:len(df)-1].reset_index(drop=True)
            result_df[[i for i in list(result_df) if i.startswith('b_')]] = df[1:len(df)].reset_index(drop=True)
            result_df = pd.merge(result_df, corr_temp_2, on = ['f_single_rule', 'b_single_rule'], how = 'left')
            

            self.logger.info('********************* ALL DATA SHAPE : {} *********************'.format(result_df.shape))                        
            test_df = result_df[result_df['corr'].isna() == True].reset_index(drop = True)
            self.logger.info('********************* TEST DATA SHAPE : {} *********************'.format(test_df.shape))            
            result_df = result_df[result_df['corr'].isna() == False].reset_index(drop = True)
            
            
            ## oversampling
            temp_result = result_df.copy()
            temp_result['f_single_rule'] = temp_result['f_single_rule'].astype('str')
            temp_result['b_single_rule'] = temp_result['b_single_rule'].astype('str')
            temp_result['rule_sum'] = temp_result['f_single_rule'].str.cat(temp_result['b_single_rule'], sep = ',')
            filter_list = list(set(list(temp_result['rule_sum'])))
            max_cnt = temp_result['rule_sum'].value_counts().max()
            temp_df = pd.DataFrame()


            for i in filter_list:
                temp_data = temp_result[temp_result['rule_sum'].isin([str(i)]) == True]
                if len(temp_data) < max_cnt:
                    temp_data = temp_data.sample(max_cnt, replace = True)
                    temp_df = pd.concat([temp_df, temp_data])
                else:
                    temp_df = pd.concat([temp_df, temp_data])
                    
            result_df = pd.concat([result_df, temp_df])
            del result_df['rule_sum']
            
            self.logger.info('********************* TRAIN DATA SHAPE : {} *********************'.format(result_df.shape))            
            result_col = list(result_df)

            ## rate list
            log_rate_list = self.model_config["common"]["log_rate_list"].split(', ')

            ## onehotencoding            
            log_onehotencoder = OneHotEncoder(categories= 'auto', handle_unknown='ignore')
            log_onehotencoder.fit(log[log_rate_list].astype('str').values)
            save_obj(log_onehotencoder, path= version + '/' + "log_"+ self.model_config["common"]["onehotencoder_save"])
            log_onehot = pd.DataFrame(index = log.hash, columns = log_onehotencoder.get_feature_names(log_rate_list),
                                  data = log_onehotencoder.transform(log[log_rate_list].astype('str').values).toarray())

            
            ## make train data
            col_list = result_col + list(log_onehot)
            log_result = result_df[result_df['f_type'] == 'log'].copy()

            log_result = pd.merge(log_result, log_onehot, left_on = 'f_hash', right_on = 'hash', how = 'left')
            log_result = pd.merge(log_result, log_onehot, left_on = 'b_hash', right_on = 'hash', how = 'left')

            result_df = log_result.copy()
            raw_data = result_df.copy()
            raw_data = pd.concat([raw_data,raw_data])
            raw_data = pd.concat([raw_data,raw_data])
            
            result_df.drop(result_col, axis = 1, inplace = True)
            result_df = pd.concat([result_df, result_df])
            result_df = pd.concat([result_df, result_df])
            X_data = result_df.copy()
            self.logger.info('********************* ONEHOT DATA SHAPE : {} *********************'.format(result_df.shape))            
            ############################ 상관분석 모듈 시작 ##########################################
            ## CNN Autoencoder
            model_config = self.model_config
            model_config["x_datashape"] = X_data.shape
            
            for k in model_config.keys():
                if model_config[k]==None:
                    self.logger.info("MODEL CONFIG {} HAS TO BE SET UP...")
                    
            model = corr_model(config=model_config, mode="train", name=self.model_config["common"]["model_name"] + '_' + version)   
            res, ai_history = model.optimize_nn(X=X_data, Y=X_data)
            history_df = pd.DataFrame({'loss': ai_history.history['loss']})
            history_df['model_id'] = self.model_id
            history_df['epoch'] = list(range(1, len(history_df)+1))
            history_df['model_type'] = 'corr'
            history_df['data_shape'] = [X_data.shape for i in range(len(history_df))]
            history_df['version'] = start
            history_df['train_time'] = start          
                            
            pred = model.predict(X_data)
            rmse = model.rmse_custom(X_data, pred, 1).numpy()            
            z_scores, threshold = detect_mad_outliers(rmse, threshold= 2.5)
            print(threshold)
            print("*"*100)
            save_obj({'threshold': threshold}, path=version + '/' + self.model_config["common"]["threshold_path"])
            scaler = MinMaxScaler().fit(np.array(rmse).reshape(-1,1))
            save_obj(scaler, path= version + '/' + self.model_config["common"]["minmax_rmse_save"])
            
            temp_rmse = list(np.mean(np.power(X_data - pred, 2), axis = 1))
            self.logger.info(np.mean(np.power(X_data - pred, 2), axis = 1).value_counts())
            raw_data['rmse'] = temp_rmse
            raw_data.drop_duplicates(inplace = True)
            raw_data.to_csv('train_rmse.csv')
            self.logger.info(X_data.shape)            
            ############################ test result print #####################################            
            ## test
            log_result = test_df[test_df['f_type'] == 'log'].copy()

            log_result = pd.merge(log_result, log_onehot, left_on = 'f_hash', right_on = 'hash', how = 'left')
            log_result = pd.merge(log_result, log_onehot, left_on = 'b_hash', right_on = 'hash', how = 'left')
            
            test_df = log_result.copy()
            test_df.drop(result_col, axis = 1, inplace = True)
            pred = model.predict(test_df)
            self.logger.info(test_df.shape)
            self.logger.info(np.mean(np.power(test_df - pred, 2), axis = 1).value_counts())
            
            validation_time = datetime.now().replace(microsecond=0) + timedelta(hours=9)
            history_df['validation_time'] = validation_time
            history_df = history_df[['model_id', 'model_type', 'train_time', 'validation_time', 'version', 'loss', 'epoch', 'data_shape']]
#             execute_ch("INSERT INTO dti.motie_ai_history VALUES", history_df.to_dict('records'))  
            
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
                if not os.path.exists(pwd + '/{}/{}'.format(self.model_config["common"]["path"], self.model_config["common"]["model_name"] + '_' + version)):
                    new_time = start - timedelta(hours=timerange+1)
                    version = new_time.strftime("%Y%m%d_%H")
                else:
                    break

            self.logger.info('VERSION : ' + version)
            model_name = self.model_config["common"]["model_name"] + '_' + version
            minmax_rmse_save = version + '/' + self.model_config["common"]["minmax_rmse_save"]
            threshold_path = version + '/' + self.model_config["common"]["threshold_path"]                        
            
             ## data load
            data, meta = execute_ch("select ip_address, device_id from dti.kdn_lgsys_L003 where ip_address!='' and device_id!=''", with_column_types = True)
            feats = [m[0] for m in meta]
            map_ip = pd.DataFrame(data, columns = feats)              
            map_ip.drop_duplicates(inplace = True)
            self.logger.info('********************* LOG MAP IP LOAD DONE *********************')            
#             packet_sql = get_queries(param['logtime_s'], param['logtime_e'], query_type='p')

#             log_sql = get_queries(param['logtime_s'], param['logtime_e'], query_type='l')
            log_sql = get_queries('2021-06-08 00:00:00', '2021-06-08 24:00:00', query_type='l')
            globals()['sql_005'], globals()['sql_009'], globals()['sql_011'], log_list = [], [], [], []
            
            for sql in log_sql:
                if 'sect_lgsys_L005' in sql:
                    sql_005.append(sql)                
                elif 'sect_lgsys_L009' in sql:
                    sql_009.append(sql)
                elif 'sect_lgsys_L011' in sql:
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

            log.rename(columns = {'single_rule' : 'single_rule_log'}, inplace = True)
            log['single_rule'] = log.single_rule_log        
            log.rename(columns = {'plant_id' : 'id'}, inplace = True)
            log.reset_index(drop = True, inplace = True)
            log.drop_duplicates(inplace =True)                   
            
            ## log data ip mapping    
            log = pd.merge(log, map_ip, on = 'device_id', how = 'left')
            log = log[log.ip_address != 'nan']
            log = log[log.ip_address.isna() == False]
            log['ip'] = log.ip_address
            log.reset_index(drop = True, inplace = True)
            self.logger.info('********************* SINGLE LOG DATA SHAPE : {} *********************'.format(log.shape))    

    
    
    ########################################################################################
  
            
#             # TEST
#             data, meta = execute_ch("""
#             select * 
#             from dti.motie_ai_corr_prep_v2
#             where (parseDateTimeBestEffortOrZero(toString(loged_time)) between '{start_date}' and '{end_date}') or (parseDateTimeBestEffortOrZero(toString(packet_time)) between '{start_date}' and '{end_date}')""".replace('{start_date}', param['logtime_s']).replace('{end_date}',  param['logtime_e']), with_column_types = True) 
#             feats = [m[0] for m in meta]
#             check_corr_prep = pd.DataFrame(data, columns = feats)

#             data, meta = execute_ch("""
#             select * 
#             from dti.motie_ai_corr_result_v2
#             where f_time between '{start_date}' and '{end_date}'""".replace('{start_date}', param['logtime_s']).replace('{end_date}',  param['logtime_e']), with_column_types = True) 
#             feats = [m[0] for m in meta]
#             check_corr_result = pd.DataFrame(data, columns = feats)
            
#             check_corr_result['f_hash'] = check_corr_result['f_hash'].astype('str')
#             check_corr_result['b_hash'] = check_corr_result['b_hash'].astype('str')
#             check_corr_result['hash_sum'] = check_corr_result['f_hash'].str.cat(check_corr_result['b_hash'], sep = ',')

    
    

#########################################################################################################
            if len(sql_009)>0: 
                log = log[log.hash.isin(log_009.hash) == False]

            self.logger.info('********************* PREDICTION LOG DATA SHAPE : {} *********************'.format(log.shape))    

            con_df = log.copy()
            con_df.sort_values('time', inplace = True)

            ## skip_n
            skip_n = self.model_config["common"]["skip_n"]
            temp_df = con_df.copy()
            for x in range(1, skip_n):
                print("skip_n : {}".format(x))
                for i in range(len(con_df)-x-1):
                    temp_df = pd.concat([temp_df, con_df.iloc[[i, i+x+1]]])

            temp_df.drop_duplicates(inplace = True)
            temp_df.reset_index(drop = True, inplace = True)
            con_df = temp_df.copy()
            con_df_all = temp_df.copy()

            ## prep data insert
            con_df[list(con_df.select_dtypes(include = 'object'))] = con_df[list(con_df.select_dtypes(include = 'object'))].astype('str')
            con_df['plant_id'] = con_df['id']
            con_df['make_id'] = con_df['id']
            con_df['version'] = real_time

#                 check_prep = con_df.copy()
#                 check_prep = check_prep[check_prep.hash.isin(check_corr_prep.hash) == False]

#                 execute_ch("INSERT INTO dti.motie_ai_corr_prep_v2 VALUES", check_prep.to_dict('records'))
            self.logger.info('********************* con_df shape :{} *********************'.format(con_df.shape))

            del con_df['time']    

            ## make result table
            df = con_df[['message_id','single_rule','hash']].copy()
            df['single_rule'] = df['single_rule'].astype('int')
            df["message_id"] = np.select([df['message_id'].str.startswith('H'), df['message_id'].str.startswith('L')],  ['packet', 'log'], default=np.nan)       

            ## result_df col_list
            f_list = ['f_type','f_single_rule','f_hash']
            b_list = ['b_type','b_single_rule','b_hash']
            result_col = f_list + b_list

            ## make result table
            corr_temp = get_corr()
            temp = corr_temp.copy()
            temp_2 = corr_temp.copy()
            temp.rename(columns = {"single_rule_log" : 'f_single_rule'}, inplace = True)
            temp_2.rename(columns = {"single_rule_log" : 'b_single_rule'}, inplace = True)
            corr_temp_2 =pd.merge(temp, temp_2, on = ['corr', 'corr'], how = 'outer')
            corr_temp_2.drop_duplicates(inplace = True)

            result_df = pd.DataFrame(columns = result_col)
            result_df[[i for i in list(result_df) if i.startswith('f_')]] = df[0:len(df)-1].reset_index(drop=True)
            result_df[[i for i in list(result_df) if i.startswith('b_')]] = df[1:len(df)].reset_index(drop=True)
            result_df = pd.merge(result_df, corr_temp_2, on = ['f_single_rule', 'b_single_rule'], how = 'left')
            result_df['corr'].fillna(0, inplace = True)
            result_df['corr'] = result_df['corr'].astype('uint')
#                 result_df.reset_index(drop = True, inplace = True)
            result_col = list(result_df)
    
            self.logger.info('********************* result_df shape :{} *********************'.format(result_df.shape))

            #########################################  for final result  ###############################################
            ## make result table
            df_all = con_df_all[['time','ip','message_id','single_rule','hash','id']].copy()                                
            df_all['time'] = pd.to_datetime(df_all['time'], format = '%Y%m%d%H%M%S', errors = 'raise')
            df_all["message_id"] = np.select([df_all['message_id'].str.startswith('H'), df_all['message_id'].str.startswith('L')],  ['packet', 'log'], default=np.nan)      

            ## result_df col_list
            f_list = ['f_time','f_ip','f_type','f_single_rule','f_hash','f_id']
            b_list = ['b_time','b_ip','b_type','b_single_rule','b_hash','b_id']
            final_result_col = f_list + b_list
#                 other_result = ['ai_rmse','ai_rmse_scaled','ai_label','version']
#                 final_result_col = f_list + b_list + other_result

            ## make result table
            final_result = pd.DataFrame(columns = final_result_col)
            final_result[[i for i in list(final_result) if i.startswith('f_')]] = df_all[0:len(df_all)-1].reset_index(drop=True)
            final_result[[i for i in list(final_result) if i.startswith('b_')]] = df_all[1:len(df_all)].reset_index(drop=True)
#                 final_result = pd.merge(final_result, corr_temp_2, on = ['f_single_rule', 'b_single_rule'], how = 'left')
#                 final_result.reset_index(drop = True, inplace = True)
            self.logger.info('********************* final_result shape :{} *********************'.format(final_result.shape))

            ############################################################################################################
            ## rate list
            log_rate_list = self.model_config["common"]["log_rate_list"].split(', ')

            ## onehotencoding       
            log_onehotencoder = load_obj(path=version + '/' + "log_"+ self.model_config["common"]["onehotencoder_save"])
            log_onehot = pd.DataFrame(index = log.hash, columns = log_onehotencoder.get_feature_names(log_rate_list),
                                  data = log_onehotencoder.transform(log[log_rate_list].astype('str').values).toarray())

            ## make train data
            col_list = result_col + list(log_onehot)
            log_result = result_df[result_df['f_type'] == 'log'].copy()

            log_result = pd.merge(log_result, log_onehot, left_on = 'f_hash', right_on = 'hash', how = 'left')
            log_result = pd.merge(log_result, log_onehot, left_on = 'b_hash', right_on = 'hash', how = 'left')

            result_df = log_result.copy()
            result_df.drop_duplicates(inplace = True)
            final_result.drop_duplicates(inplace = True)
            self.logger.info('********************* result_df shape :{} *********************'.format(result_df.shape))
            self.logger.info('********************* final_result shape :{} *********************'.format(final_result.shape))
            result_df.reset_index(drop = True, inplace = True)
            result_df.fillna(0, inplace = True)
            train_col = list(result_df)

            for i in result_col:
                train_col.remove(i)


            X_data = result_df[train_col].copy()
            ############################ 상관분석 모듈 시작 ##########################################
            model_config = self.model_config
            model_config["x_datashape"] = X_data.shape

            for k in model_config.keys():
                if model_config[k]==None:
                    self.logger.info("MODEL CONFIG {} HAS TO BE SET UP...")

            model = corr_model(config=self.model_config, mode="predict", name=model_name)        
            pred = model.predict(X_data)
            print(np.mean(np.power(X_data - pred, 2), axis = 1).value_counts())                
            rmse = model.rmse_custom(X_data, pred, 1).numpy()                              
            minmaxscaler = load_obj(path=minmax_rmse_save)                
            rmse_scaled = minmaxscaler.transform(np.array(rmse).reshape(-1,1)) 
            threshold = 0.017396
#                 threshold = load_obj(path=threshold_path)['threshold']
            result_df['ai_rmse'] = rmse
            result_df['ai_rmse_scaled'] = rmse_scaled
            result_df['ai_label'] = rmse <= threshold
                            

            final_result = pd.merge(final_result, result_df, on = ['f_type','f_single_rule','f_hash', 'b_type','b_single_rule','b_hash'], how = 'left')
            final_result['version'] = real_time                
            final_result[['f_single_rule', 'b_single_rule']]= final_result[['f_single_rule', 'b_single_rule']].astype('int')
            final_result.drop_duplicates(inplace = True)
            final_result.reset_index(drop = True, inplace = True)
            final_result.loc[list(final_result[final_result['corr'] != 0].index), 'ai_label'] = 'True'
            final_result.loc[list(final_result[final_result['corr'] == 0].index), 'ai_label'] = 'False'

            corr_result = final_result.copy()

            ## 상관분석 table insert
            int_list = ['f_single_rule', 'b_single_rule', 'corr']
            date_list = ['f_time', 'b_time', 'version']
            float_list = ['ai_rmse', 'ai_rmse_scaled']
            str_list = ['f_type',  'b_type', 'ai_label']

            corr_result['corr'].fillna(0, inplace = True)  
            corr_result[int_list] = corr_result[int_list].astype('uint')
            corr_result[float_list] = corr_result[float_list].astype('float')
            corr_result[str_list] = corr_result[str_list].astype('str')
            uint_list = ['f_hash', 'b_hash']
            corr_result[uint_list] = corr_result[uint_list].astype('str')
            corr_result.drop_duplicates(inplace = True)
            print(corr_result.info())

#             check_result = corr_result.copy()
#             check_result['f_hash'] = check_result['f_hash'].astype('str')
#             check_result['b_hash'] = check_result['b_hash'].astype('str')
#             check_result['hash_sum'] = check_result['f_hash'].str.cat(check_result['b_hash'], sep = ',')

#             check_result = check_result[check_result.hash_sum.isin(check_corr_result.hash_sum) == False]
#             del check_result['hash_sum']

            execute_ch("INSERT INTO dti.sect_ai_corr_result VALUES", corr_result.to_dict('records'))
                                
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