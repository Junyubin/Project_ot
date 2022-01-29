import pandas as pd
from model import *
import random
import numpy as np
import scipy.signal
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
from multiprocessing import Pool
sys.path.insert(0, sys.path[0])

class Train2(BaseComponent):
    def init(self):
        pass

    async def run(self, param):
        try:
            train_list = self.model_config["common"]["model_list"].split(', ')
            self.logger.info(train_list)
            start = datetime.now().replace(microsecond=0) + timedelta(hours=9)
            for m_type in train_list:
                try:
                    self.logger.info(m_type)
                    version = m_type + "_" + start.strftime("%Y%m%d_%H")
                #############################################################################################################            
                    model_name = self.model_config["common"]["model_name"] + '_' + version
                    minmaxscaler_save = version + '/' + self.model_config["common"]["minmaxscaler_save"]
                    minmax_rmse_save = version + '/' + self.model_config["common"]["minmax_rmse_save"]
                    threshold_save = version + '/' + self.model_config["common"]["threshold_path"]
                    
                    if not os.path.exists(pwd+'/obj/'+version):
                        os.makedirs(pwd+'/obj/'+version)                        
                                        
                    tag = pd.read_csv(pwd + "/csv/tag_3.csv")
#                     tag['type'] = tag['type'].replace("Generator\\", 'Generator')
                    param['logtime_s'] = '2021-10-10 00:00:00'
                    param['logtime_e'] = '2021-10-13 00:00:00'
                    print(param['logtime_s'])
                    print(param['logtime_e'])        
            
                    data, meta = execute_ch("""
                    select toString(sipHash64(*)) as hash, *, parseDateTimeBestEffort(toString(date_time)) as date_time
                    from dti.motie_manag_I002 
                    where date_time between '{start_date}' and '{end_date}'
                    and tag_name in {name}""".replace('{name}', str(tuple(tag.tag_name))).replace('{start_date}', param['logtime_s']).replace('{end_date}', param['logtime_e']), with_column_types = True)

                    feats = [m[0] for m in meta]
                    data = pd.DataFrame(data, columns = feats)
                
                    ## 이상 tag value 제거
                    data.tag_value = data.tag_value.replace('False', '0')
                    data.tag_value = data.tag_value.replace('null', '0')
                    data.tag_value = data.tag_value.replace('True', '0')

                    ## data transform
                    data = pd.crosstab(index = data.date_time, columns = data.tag_name, values = data.tag_value.astype('float'), aggfunc = np.sum)
                    data = data.dropna(axis = 0)
                    tag = tag[tag.tag_name.isin(list(data)) == True]
                    model_df = data[set(list(tag[tag.type == m_type].tag_name))]

                    #data dropna
                    model_df = model_df.dropna(axis=0)
                    ## data scale
                    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
                    model_df = model_df.reindex(sorted(model_df.columns), axis=1)
                    scaler.fit(model_df.values)
                    save_obj(scaler, path=minmaxscaler_save)
                    scl_df = pd.DataFrame(columns = list(model_df), data=scaler.transform(model_df))
                    scl_df = scl_df.fillna(0)
                    
                    
                    ## make in/output data
                    input_data = create_ws_data(scl_df, self.model_config["common"]["window_size"])
                    output_data = create_output_data(input_data)
                    input_data = input_data[1:]
                    output_data = output_data[1:]
#                     input_data = scl_df.values
#                     output_data = create_output_data(input_data)
                    
    
                    ## set model config
                    model_config = self.model_config
                    model_config["x_datashape"] = input_data.shape
                    model_config["y_datashape"] = output_data.shape

                    for k in model_config.keys():
                        if model_config[k]==None:
                            self.logger.info("MODEL CONFIG {} HAS TO BE SET UP...")
                            
                    ## model fitting
                    model = lstm_model(config=model_config, mode="train", name=model_name)
                    res, ai_history = model.optimize_nn(X=input_data, Y=output_data)
                    
                    ## model prediction
                    rmse = model.predict_rmse(X=input_data, Y=output_data)
                    z_scores, threshold = detect_mad_outliers(rmse)
                    save_obj({'threshold': threshold}, path=threshold_save)
                    
                    ## rmse scaler
                    minmaxscaler_rmse = MinMaxScaler(copy=True, feature_range=(0, 1))
                    minmaxscaler_rmse.fit(rmse.reshape(-1, 1))
                    save_obj(minmaxscaler_rmse, path=minmax_rmse_save)
                    rmse_scaled = minmaxscaler_rmse.transform(rmse.reshape(-1, 1))
                    
                    
                    ## model validation
                    model_df = model_df[:-1]
                    model_df['ai_rmse'] = rmse
                    model_df['ai_label'] = rmse > threshold
                    model_df['ai_rmse_scaled'] = rmse_scaled
                    model_df['version'] = start   
                    
                    model_df.reset_index(drop = False, inplace = True )
                    model_df = model_df.set_index(['date_time', 'ai_rmse', 'ai_rmse_scaled', 'ai_label', 'version'])
                    data_df = model_df.stack().reset_index()
                    data_df = data_df.rename({0 : 'tag_value'}, axis = 'columns')
                    data_df.sort_values(by=['tag_name', 'date_time'], axis=0, inplace = True)
                    data_df = data_df.reset_index()
                    validation_df = data_df[['date_time', 'tag_name', 'tag_value', 'ai_rmse', 'ai_label', 'version']]
  
                    self.logger.info('*'*100)
                    self.logger.info('True data')
                    self.logger.info(validation_df[validation_df.ai_label == True])
                    self.logger.info('False data')
                    self.logger.info(validation_df[validation_df.ai_label == False])
                    self.logger.info('*'*100)
                    
                    
                    ## save model history
                    history_df = pd.DataFrame({'loss': ai_history.history['loss']})
                    history_df['model_id'] = self.model_id
                    history_df['epoch'] = list(range(1, len(history_df)+1))
                    history_df['model_type'] = m_type
                    history_df['data_shape'] = [input_data.shape for i in range(len(history_df))]
                    history_df['version'] = start
                    history_df['train_time'] = start
                    
                    
                    validation_time = datetime.now().replace(microsecond=0) + timedelta(hours=9)
                    history_df['validation_time'] = validation_time
                    history_df = history_df[['model_id', 'model_type', 'train_time', 'validation_time', 'version', 'loss', 'epoch', 'data_shape']]
                    execute_ch("INSERT INTO dti.motie_ai_history VALUES", history_df.to_dict('records'))

                    self.logger.info(res)
                except Exception as err:
                    self.logger.error(err)
                    self.logger.error(traceback.print_exc())
            
            return "OK"


        except Exception as err:
            self.logger.error(err)
            self.logger.error(traceback.print_exc())
            return None


class Prediction2(BaseComponent):

    def init(self):
        pass
    
    def run_multi(self, param, m_type):
        self.logger.info(m_type)
        start = datetime.now().replace(microsecond=0) + timedelta(hours=9)
        version = m_type + "_" + start.strftime("%Y%m%d_%H")
        new_time = start
        real_time = start
        
        ## version error code
        for timerange in range(500):
            if not os.path.exists(pwd + '/{}/{}'.format(self.model_config["common"]["path"], self.model_config["common"]["model_name"] + '_' + version)):
                new_time = start - timedelta(hours=timerange+1)
                version = m_type + "_" + new_time.strftime("%Y%m%d_%H")
            else:
                break
                
        print(version)

                
        for timerange in range(200):
            try:
                if len(os.listdir(pwd +'/obj/{}'.format(version))) != 3:                         
                    new_time = new_time - timedelta(hours=timerange+1)        
                    version = m_type + "_" + new_time.strftime("%Y%m%d_%H")
                else:
                    break
            except:
                new_time = start - timedelta(hours=timerange+1)
                version = m_type + "_" + new_time.strftime("%Y%m%d_%H")
                
                continue
            
        print('final_version : ' + version)    
        
        model_name = self.model_config["common"]["model_name"] + '_' + version
        minmaxscaler_save = version + '/' + self.model_config["common"]["minmaxscaler_save"]
        minmax_rmse_save = version + '/' + self.model_config["common"]["minmax_rmse_save"]
        threshold_save = version + '/' + self.model_config["common"]["threshold_path"]
        
        
        ## data load
        tag = pd.read_csv(pwd + "/csv/tag_3.csv")
        
        param['logtime_s'] = '2021-10-10 00:00:00'
        param['logtime_e'] = '2021-10-10 04:00:00'
        print(param['logtime_s'])
        print(param['logtime_e'])        
        
        data, meta = execute_ch("""
        select toString(sipHash64(*)) as hash, *, parseDateTimeBestEffort(toString(date_time)) as date_time
        from dti.motie_manag_I002
        where date_time between '{start_date}' and '{end_date}'
        and tag_name in {name}""".replace('{name}', str(tuple(tag.tag_name))).replace('{start_date}', param['logtime_s']).replace('{end_date}', param['logtime_e']), with_column_types = True)

        feats = [m[0] for m in meta]
        data_orig = pd.DataFrame(data, columns = feats)
        data_orig.sort_values(by=['tag_name'], axis=0, inplace = True)
        print(data_orig.shape)
        print('*' *10 + m_type + 'data_orig.shape' + '*' *10)
        
        ## 이상 tag value 제거
        data_orig.tag_value = data_orig.tag_value.replace('False', '0')
        data_orig.tag_value = data_orig.tag_value.replace('null', '0')
        data_orig.tag_value = data_orig.tag_value.replace('True', '0')
        print(data_orig.shape)
        print('*' *10 + m_type + '이상 제거 data_orig.shape' + '*' *10)
        print(m_type)
        
        if m_type == 'Turbine1':
            data_orig = data_orig[data_orig['tag_name'].str.startswith(('T1', 'M1', 'TBN1')) == True]
            data_orig.sort_values(by = ['tag_name', 'date_time'], axis = 0, inplace = True)
            data_orig['unit_id'] = 'EWP_01_UN_01'
            data_orig['operate_info_id'] = 'EWP_01_OI_01'
        elif m_type == 'Turbine2':
            data_orig = data_orig[data_orig['tag_name'].str.startswith(('T2', 'M2', 'TBN2')) == True]
            data_orig.sort_values(by = ['tag_name', 'date_time'], axis = 0, inplace = True)
            data_orig['unit_id'] = 'EWP_01_UN_02'
            data_orig['operate_info_id'] = 'EWP_01_OI_02'            
        elif m_type == 'Boiler1':
            data_orig = data_orig[data_orig['tag_name'].str.startswith(('BLR1')) == True]
            data_orig.sort_values(by = ['tag_name', 'date_time'], axis = 0, inplace = True)
            data_orig['unit_id'] = 'EWP_01_UN_01'
            data_orig['operate_info_id'] = 'EWP_01_OI_01'            
        elif m_type == 'Boiler2':
            data_orig = data_orig[data_orig['tag_name'].str.startswith(('BLR2')) == True]
            data_orig.sort_values(by = ['tag_name', 'date_time'], axis = 0, inplace = True)            
            data_orig['unit_id'] = 'EWP_01_UN_02'
            data_orig['operate_info_id'] = 'EWP_01_OI_02'
            
        data_orig.drop_duplicates(inplace = True)
        data_orig.drop_duplicates(['tag_name','date_time'], inplace = True)

        ## data transform
        data = pd.crosstab(index = data_orig.date_time, columns = data_orig.tag_name, values = data_orig.tag_value.astype('float'), aggfunc = np.sum)
        data = data.dropna(axis = 0)
        tag = tag[tag.tag_name.isin(list(data)) == True]
        model_df = data[set(list(tag[tag.type == m_type].tag_name))]
        print(model_df.shape)
        #data dropna
        model_df = model_df.dropna(axis=0)
        
        ## data scale
        model_df = model_df.reindex(sorted(model_df.columns), axis=1)
        print(model_df.shape)
        scaler = load_obj(path=minmaxscaler_save)
        scl_df = pd.DataFrame(index = model_df.index, columns = list(model_df), data=scaler.transform(model_df))     
        scl_df = scl_df.fillna(0)
        
        ## hash 저장
        hash_df = pd.crosstab(index = data_orig.date_time, columns = data_orig.tag_name, values = data_orig.hash, aggfunc = np.sum)
        new_hash_df = hash_df[set(list(tag[tag.type == m_type].tag_name))]
        new_hash_df = new_hash_df.stack().reset_index()
        new_hash_df = new_hash_df.rename({0 : 'hash'}, axis = 'columns')
        
        ## hash 정렬
        new_hash_df.sort_values(by=['tag_name', 'date_time'], axis=0, inplace = True)
        new_hash_df= new_hash_df.reset_index()
        del new_hash_df['index']
        
        ## tag_time 저장
        tag_df = pd.crosstab(index = data_orig.date_time, columns = data_orig.tag_name, values = data_orig.tag_time, aggfunc = np.sum)
        new_tag_df = tag_df[set(list(tag[tag.type == m_type].tag_name))]
        new_tag_df = new_tag_df.stack().reset_index()
        new_tag_df = new_tag_df.rename({0 : 'tag_time'}, axis = 'columns')
        
        ## tag_time 정렬
        new_tag_df.sort_values(by=['tag_name', 'date_time'], axis=0, inplace = True)
        new_tag_df= new_tag_df.reset_index()
        del new_tag_df['index']
        
        ## send_time 저장
        send_df = pd.crosstab(index = data_orig.date_time, columns = data_orig.tag_name, values = data_orig.send_time, aggfunc = np.sum)
        new_send_df = send_df[set(list(tag[tag.type == m_type].tag_name))]
        new_send_df = new_send_df.stack().reset_index()
        new_send_df = new_send_df.rename({0 : 'send_time'}, axis = 'columns')
        
        ## send_time 정렬
        new_send_df.sort_values(by=['tag_name', 'date_time'], axis=0, inplace = True)
        new_send_df= new_send_df.reset_index()
        del new_send_df['index']
                
        ## 전처리 데이터 저장
        scl_df_save = scl_df.stack().reset_index()
        scl_df_save = scl_df_save.rename({'level_1' : 'tag_name', 0 : 'tag_value'}, axis = 'columns')
        scl_df_save.sort_values(by=['tag_name', 'date_time'], axis=0, inplace = True)
        scl_df_save = scl_df_save.reset_index()
        del scl_df_save['index']
        scl_df_save['hash'] = m_type + '_' + new_hash_df['hash'].astype('str')
        scl_df_save['message_id'] = 'I002'
        scl_df_save['operate_info_id'] = data_orig.operate_info_id.unique()[0]
        scl_df_save['unit_id'] = data_orig.unit_id.unique()[0]
        scl_df_save['send_time'] = new_send_df.send_time
        scl_df_save['send_time'] = scl_df_save['send_time'].astype('str')
        scl_df_save['tag_time'] = new_tag_df.tag_time
        scl_df_save['trans_tag'] = m_type
        scl_df_save['version'] = real_time
        scl_df_save = scl_df_save[['hash', 'message_id', 'operate_info_id', 'send_time', 'unit_id', 'tag_name', 'tag_value', 'tag_time', 'date_time', 'trans_tag', 'version']]
        
        ## save prep data
        scl_df_save[list(scl_df_save.select_dtypes(include = 'float'))] = scl_df_save[list(scl_df_save.select_dtypes(include = 'float'))].astype('float')
        scl_df_save[list(scl_df_save.select_dtypes(include = 'object'))] = scl_df_save[list(scl_df_save.select_dtypes(include = 'object'))].astype('str')
        scl_df_save[list(scl_df_save.select_dtypes(include = 'uint'))] = scl_df_save[list(scl_df_save.select_dtypes(include = 'uint'))].astype('str')
        scl_df_save[list(scl_df_save.select_dtypes(include = 'bool'))] = scl_df_save[list(scl_df_save.select_dtypes(include = 'bool'))].astype('str')
        scl_df_save['tag_time'] = scl_df_save['tag_time'].astype(str)
        scl_df_save['date_time'] = scl_df_save['date_time'].astype(str)
        
        
        ## 중복 제거
        data, meta = execute_ch("""
        select * 
        from dti.motie_ai_op_prep
        where date_time between '{start_date}' and '{end_date}'""".replace('{start_date}', param['logtime_s']).replace('{end_date}',  param['logtime_e']), with_column_types = True) 
        feats = [m[0] for m in meta]
        check_op_prep = pd.DataFrame(data, columns = feats)
        
        check_prep = scl_df_save[scl_df_save.date_time.isin(check_op_prep.date_time) == False]
        
        execute_ch("INSERT INTO dti.motie_ai_op_prep VALUES", check_prep.to_dict('records'))
        
        ## make in/output data
#         output_data = create_ws_data(scl_df, self.model_config["common"]["window_size"])
#         input_data = create_input_data(output_data)
#         input_data = input_data[1:]
#         output_data = output_data[1:]
        input_data = scl_df.values
        output_data = create_output_data(input_data)
        input_data = input_data.reshape(len(input_data),1,len(scl_df.columns))
        output_data = output_data.reshape(len(output_data),1,len(scl_df.columns))
        
        ## set model config
        model_config = self.model_config
        model_config["x_datashape"] = input_data.shape
        model_config["y_datashape"] = output_data.shape
        
        for k in model_config.keys():
            if model_config[k]==None:
                self.logger.info("MODEL CONFIG {} HAS TO BE SET UP...")
        
        ## model load & prediction
        model = lstm_model(config=model_config, mode="predict", name=model_name)
        rmse = model.predict_rmse(X=input_data, Y=output_data)

        ## rmse scaler
        minmaxscaler_rmse = load_obj(path=minmax_rmse_save)
        rmse_scaled = minmaxscaler_rmse.transform(rmse.reshape(-1, 1))
        
        ## load threshold
        threshold = load_obj(path=threshold_save)['threshold']
        print(threshold)
        threshold = threshold * 10
        print(threshold)
        print("*"*100)
        self.logger.info(m_type)
        self.logger.info(rmse.shape)

        ## save result table
        model_df['ai_rmse'] = rmse
        print(model_df.ai_rmse.value_counts())
        model_df['ai_rmse_scaled'] = rmse_scaled
        model_df['ai_label'] = rmse > threshold
        print(model_df.ai_label.value_counts())
        model_df['version'] = real_time                
                    
        model_df.reset_index(drop = False, inplace = True )
        model_df = model_df.set_index(['date_time', 'ai_rmse', 'ai_rmse_scaled', 'ai_label', 'version'])
        data_df = model_df.stack().reset_index()
        data_df = data_df.rename({0 : 'tag_value'}, axis = 'columns')
        data_df.sort_values(by=['tag_name', 'date_time'], axis=0, inplace = True)
        data_df = data_df.reset_index()
        del data_df['index']
        
        data_df['date_time'] = data_df['date_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        data_df['send_time'] = new_send_df.send_time
        data_df['send_time'] = data_df['send_time'].astype('str')
        data_df['hash'] = scl_df_save['hash']
        data_df['message_id'] = 'I002'
        data_df['operate_info_id'] = data_orig.operate_info_id.unique()[0]
        data_df['unit_id'] = data_orig.unit_id.unique()[0]
        data_df['tag_time'] = new_tag_df.tag_time
        data_df['trans_tag'] = m_type
        data_df['time_line'] = data_df.date_time
        data_df['time_line'] = pd.to_datetime(data_df['time_line'], format='%Y-%m-%d %H:%M:%S', errors='raise')
        data_df['ai_threshold'] = threshold
        data_df = data_df[['hash', 'time_line', 'message_id', 'operate_info_id', 'send_time', 'unit_id', 'tag_name', 'tag_value', 'tag_time', 'date_time', 'trans_tag', 'ai_rmse', 'ai_rmse_scaled', 'ai_label', 'ai_threshold', 'version']]        
        
        ## 중복 제거
        data, meta = execute_ch("""
        select * 
        from dti.motie_ai_op_result
        where date_time between '{start_date}' and '{end_date}'""".replace('{start_date}', param['logtime_s']).replace('{end_date}',  param['logtime_e']), with_column_types = True) 
        feats = [m[0] for m in meta]
        check_op_result = pd.DataFrame(data, columns = feats)        
        check_result = data_df[data_df.hash.isin(check_op_result.hash) == False]
        
        ## data 형변환
        check_result[list(check_result.select_dtypes(include = 'float'))] = check_result[list(check_result.select_dtypes(include = 'float'))].astype('float')
        check_result[list(check_result.select_dtypes(include = 'object'))] = check_result[list(check_result.select_dtypes(include = 'object'))].astype('str')
        check_result[list(check_result.select_dtypes(include = 'uint'))] = check_result[list(check_result.select_dtypes(include = 'uint'))].astype('str')
        check_result[list(check_result.select_dtypes(include = 'bool'))] = check_result[list(check_result.select_dtypes(include = 'bool'))].astype('str')
        
        check_result['tag_value'] = check_result['tag_value'].astype(str)
        check_result['tag_time'] = check_result['tag_time'].astype(str)
        check_result['date_time'] = check_result['date_time'].astype(str)
        
        print(check_result[check_result.ai_label == 'True'].shape)
        print(check_result[check_result.ai_label == 'False'].shape)        
        print(m_type, check_result.unit_id.unique(), check_result.operate_info_id.unique())
        if len(check_result) > 0:
            execute_ch('INSERT INTO dti.motie_ai_op_result_test VALUES', check_result.to_dict('records'))
            
        return "OK"

    
    
    async def run(self, param):
        try:
            train_list = self.model_config["common"]["model_list"].split(', ')
            await asyncio.gather(*(self.loop.run_in_executor(None, self.run_multi, param, m_type) for m_type in train_list))
            session.close()
            return "OK"

        except Exception as err:
            self.logger.error(err)
            self.logger.error(traceback.print_exc())
            session.close()
            return None



def main(model_id, train, prediction, now=False):
    model_name = isExistModel(model_id)
    if model_name == None:
        sys.exit(1)
    else:
        # nc=NATS()
        loop = asyncio.get_event_loop()

        if train:
            if model_id == 2:
                if now:
                    loop.run_until_complete(Train2(loop, model_id, model_name).test_train())
                    sys.exit()
                else:
                    loop.run_until_complete(Train2(loop, model_id, model_name).start_train())
#                     sys.exit()
            else:
                self.logger.info("[TRAIN({})] model_id is invalid".format(model_id))
                sys.exit()

        if prediction:
            if model_id == 2:
                if now:
                    loop.run_until_complete(Prediction2(loop, model_id, model_name).test_pred())
                    sys.exit()
                else:
                    loop.run_until_complete(Prediction2(loop, model_id, model_name).start_pred())
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