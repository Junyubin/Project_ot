# ! /usr/bin/env python3
# python3.5+

import sys
import sched
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

import asyncio
# from nats.aio.client import Client as NATS # pip install asyncio-nats-client
# from nats.aio.errors import ErrConnectionClosed, ErrTimeout

import signal
import traceback
import time
import itertools as it

# import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from utils import *
from clickhouse_driver.client import Client

def get_config_file():
    return config

def execute_ch_all(sql, param=None, with_column_types=False, **kwargs):
    cs = config['cs']
    for i in range(len(cs)):
        ch = cs[i]
        try:
            client = Client(ch['host'], port = ch['port'], send_receive_timeout=int(ch['timeout']),
                            settings={'max_threads': int(ch['thread'])})
            client.connection.force_connect()
            if client.connection.connected:
                print('[clickhouse client.execute(sql)] connected to {}'.format(ch))
                result = client.execute(sql, params=param, with_column_types=with_column_types)
                client.disconnect()
                print(ch, result)
                # return ch, result
            else:
                print('[clickhouse client.execute(sql)] cannot connected to {}'.format(ch))
                sys.exit(1)
        except Exception as err:
            logging.error(err, exc_info=1)
            sys.exit(1)


def execute_ch(sql, param=None, with_column_types=True):
    client = check_cs(0)
    if client == None:
        sys.exit(1)
    
    result = client.execute(sql, params=param, with_column_types=with_column_types)

    client.disconnect()
    return result


def check_cs(index):
    cs = config['cs']
    if index >= len(cs):
        logging.error('[clickhouse client ERROR] connect fail')
        return None
    ch = cs[index]
    try:
        client = Client(ch['host'], port=ch['port'], send_receive_timeout=int(ch['timeout']),
                        settings={'max_threads': int(ch['thread'])})
        client.connection.force_connect()
        if client.connection.connected:
            return client
        else:
            return check_cs(index + 1)
    except:
        return check_cs(index + 1)


def is_connected_all():
    """Check Clickhouse Server all Connected
    params
    ------
    None

    return
    ------
    _is_connected_all : int
        return 1 when server is all connected
    """
    cs = config['cs']
    _is_connected_all = 1
    not_connected = []
    for i in range(len(cs)):
        ch = cs[i]
        client = Client(ch['host'], port=ch['port'], send_receive_timeout=int(ch['timeout']),
                        settings={'max_threads': int(ch['thread'])})
        client.connection.force_connect()
        print('{} is connected'.format(cs[i]))
        _is_connected_all *= client.connection.connected
        client.disconnect()
        if not client.connection.connected:
            not_connected.append(cs[i])
    if _is_connected_all:
        print("clickHouse client is all connected")
    else:
        print(not_connected)
    return _is_connected_all



def execute_ch_all_return_df(sql, param=None, with_column_types=False, **kwargs):
    """The results from each server are integrated and return
    params
    ------
    sql: str
        sql
    with_column_types: Boolean
        default False
    
    return
    ------
    df: pd.Dataframe
        Target Dataframe
    """
    if is_connected_all():
        cs = config['cs']
        try:
            df = pd.DataFrame()
            for i in range(len(cs)):
                ch = cs[i]
                client = Client(ch['host'], port = ch['port'], send_receive_timeout=int(ch['timeout']),
                                settings={'max_threads': int(ch['thread'])})
                client.connection.force_connect()

                if client.connection.connected:
                    print('[clickhouse client.execute(sql)] connected to {}'.format(ch))

                    result, meta = client.execute(sql, params=param, with_column_types=True)
                    client.disconnect()
                    feats = [m[0] for m in meta]
                    _df = pd.DataFrame(result, columns=feats)
                    _df['clickhouse_server'] = ch['host']
                    df = df.append(_df)
                else:
                    print("some clickhouse client is not connected")
                    sys.exit(1)
            return df.reset_index(drop=True)
        except Exception as err:
            logging.error(err, exc_info=1)
            sys.exit(1)



def getCols(_table):
    (dp.table) = _table.split('.')
    return np.array(execute_ch("""select name from system.columns where database = '{}' and table = '{}' and default_kind != 'MATERIALIZED'""".format(db,table)))[:,0]

async def execute_async_ch(sql, param=None):
    client = check_async_cs(0)
    if client == None:
        sys.exit(1)
    
    result = await client.execute(sql, param)

    client.disconnect()
    return result

def check_async_cs(index):
    cs = config['cs']
    if index >= len(cs):
        logger.error('[clickhouse async client ERROR] connect fail')
        return None
    ch = cs[index]
    try:
        client = AsyncClient(ch["host"], port=ch["port"], send_receive_timeout=int(ch['timeout']), settings={"max_threads":int(ch['thread'])})
        client.connection.force_connect()
        if client.connection.connected:
            logger.info('[clickhouse async client.execute(sql)] connected to {}'.format(ch))
            return client
        else:
            return check_async_cs(index+1)
    except:
        return check_async_cs(index+1)

async def _run_in_executor(executor, func, *args, **kwargs):
    if kwargs:
        func = partial(func, **kwargs)
    loop = asyncio.get_event_loop()

    return await loop.run_in_executor(executor, func, *args)

class Asyncclient(Client):
    def __init__(self, *args, **kwargs):
        self.executor = ThreadPoolExecuter(max_workers = 1)
        super(AsyncClient, self).__init__(*args, **kwargs)

    async def execute(self, *args, **kwargs):
        return await _run_in_executor(self.executor, super(AsyncClient, self).execute, *args, **kwargs)

def execute_client(select_result, sql):
    try:
        res, _ = execute_ch(sql)
        logger.info(len(res))
        logger.info(res[0][-1])
        select_result.extend(res)
    except Exception as err:
        traceback.print_exc()

class BaseComponent(object):
    # def __init__(self, nc, loop, model_id, model_name):
    def __init__(self, loop, model_id, model_name):
        # self.nc = nc
        self.loop = loop
        
        self.model_id = model_id
        self.model_name = model_name
        self.update_topic = 'update_{}'.format(model_id)

        self.model = None
        self.http_doc_dict = None

        if not os.path.exists(pwd+'/logs/'):
            logger.info('create directory: {}'.format(pwd+'/logs/'))
            os.makedirs(pwd+'/logs/')
        if not os.path.exists(pwd+'/model/'):
            logger.info('create directory: {}'.format(pwd+'/graph/'))
            os.makedirs(pwd+'/graph/')
        if not os.path.exists(pwd+'/obj/'):
            logger.info('create directory: {}'.format(pwd+'/obj/'))
            os.makedirs(pwd+'/obj/')

        # for sig in ('SIGINT', 'SIGTERM'):
        #     self.loop.add_signal_handler(getattr(signal, sig), self.signal_handler)

    def init(self):
        pass

    # async def closed_cb(self):
    #     logging.warning('Connection to NATS is closed.')
    #     await asyncio.sleep(0.1, loop=self.loop)
    #     self.loop.stop()
    #     await self.nc.close()

    # def signal_handler(self):
    #     if self.nc.is_closed:
    #         return
    #     logging.warning('Disconnecting...')
    #     self.loop.create_task(self.nc.close())

    async def run(self, param):
        pass

    async def update(self, msg):
        model_id = msg.data.decode()
        self.model.load()
        logger.info('[UPDATE] model update: {}'.format(model_id))

    async def run_scheduler(self, mode):
        try:
            self.logger = set_logger(mode, self.model_id)
            now = datetime.datetime.now() - self.model_config['now_delta'] + datetime.timedelta(hours=9)
            prev = now - self.model_config['prev_delta']
            logger.info('[{}({})] {} ~ {}'.format(mode, self.model_name, prev, now))
            param = {
                'logdate_s': prev.strftime('%Y-%m-%d'),
                'logdate_e': now.strftime('%Y-%m-%d'),
                'logtime_s': prev.strftime('%Y-%m-%d %H:%M:%S'),
                'logtime_e': now.strftime('%Y-%m-%d %H:%M:%S'),
                'toStartOf': getToStartOf(self.model_config['crontab'])
            }

            logger.info('[{}({})] param: {}'.format(mode, self.model_name, param))
            task = [self.run(param)]
            for f in asyncio.as_completed(task):
                task_result = await f
                logger.info('[{}({})] result: {}'.format(mode, self.model_name, task_result))
                if task_result == None:
                    sys.exit(1)
        except Exception as err:
            logger.error(err)
            logger.error(traceback.print_exc())

    async def start_train(self):
        try:
            # options = {
            #     'servers': ["nats://{}:{}".format(str(config['nats']['host']), str(config['nats']['port']))],
            #     'io_loop':self.loop,
            #     'closed_cb':self.closed_cb
            #     ############# max_payload ###########
            # }
            # await self.nc.connect(**options)
            # logger.info('Connected to NATS at {}...'.format(self.nc.connected_url.netloc))

            self.model_config = getConfig('TRAIN', self.model_id, self.model_name)

            scheduler = AsyncIOScheduler()
            scheduler.add_job(self.run_scheduler, CronTrigger.from_crontab(self.model_config['crontab']), ['TRAIN'], misfire_grace_time=3600, max_instances=20)
            scheduler.start()
        except Exception as err:
            logger.error(err)
            logger.error(traceback.print_exc())

    async def test_train(self):
        try:
            # options = {
            #     'servers': ["nats://{}:{}".format(str(config['nats']['host']), str(config['nats']['port']))],
            #     'io_loop':self.loop,
            #     'closed_cb':self.closed_cb
            #     ############# max_payload ###########
            # }
            # await self.nc.connect(**options)
            # logger.info('Connected to NATS at {}...'.format(self.nc.connected_url.netloc))

            self.model_config = getConfig('TEST_TRAIN', self.model_id, self.model_name)
            await self.run_scheduler('TEST_TRAIN')
        except Exception as err:
            logger.error(err)
            logger.error(traceback.print_exc())

    async def start_pred(self):
        try:
            # options = {
            #     'servers': ["nats://{}:{}".format(str(config['nats']['host']), str(config['nats']['port']))],
            #     'io_loop':self.loop,
            #     'closed_cb':self.closed_cb
            #     ############# max_payload ###########
            # }
            # await self.nc.connect(**options)
            # logger.info('Connected to NATS at {}...'.format(self.nc.connected_url.netloc))

            # await self.nc.subscribe(self.update_topic, cb=self.update)

            self.model_config = getConfig('PREDICTION', self.model_id, self.model_name, False)

            scheduler = AsyncIOScheduler()
            scheduler.add_job(self.run_scheduler, CronTrigger.from_crontab(self.model_config['crontab']), ['PREDICTION'], misfire_grace_time=3600, max_instances=20)
            scheduler.start()
        except Exception as err:
            logger.error(err)
            logger.error(traceback.print_exc())

    async def test_pred(self):
        try:
            # options = {
            #     'servers': ["nats://{}:{}".format(str(config['nats']['host']), str(config['nats']['port']))],
            #     'io_loop':self.loop,
            #     'closed_cb':self.closed_cb
            #     ############# max_payload ###########
            # }
            # await self.nc.connect(**options)
            # logger.info('Connected to NATS at {}...'.format(self.nc.connected_url.netloc))

            self.model_config = getConfig('TEST_PREDICTION', self.model_id, self.model_name, False)

            await self.run_scheduler('TEST_PREDICTION')
        except Exception as err:
            logger.error(err)
            logger.error(traceback.print_exc())