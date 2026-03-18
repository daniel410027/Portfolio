import os
import json
import requests
import time
import pickle
import typing
import logging
import pathlib
import config

logger = logging.Logger(name='Worker')

import wqb_sess as wqb_sess_lib
import alpha as alpha_lib
import alpha_utils as alpha_utils_lib

from concurrent.futures import ThreadPoolExecutor

os.makedirs(config.PENDING_FOLDER, exist_ok=True)
os.makedirs(config.COMPLETE_FOLDER, exist_ok=True)


class Worker:
    def __init__(self):
        self._sess = self.login()

    def login(self):
        return wqb_sess_lib.login()

    def simulate_alpha(self, alpha_path):
        with open(alpha_path, 'r') as f:
            alpha_data = json.load(f)

        response = self._sess.post(f'{config.URL_BASE}/simulations', json=alpha_data['payload'])
        if response.status_code != 201:
            raise Exception("Failed to start simulation: ", response.text)

        simulation_location = response.headers['Location']
        while True:
            response = self._sess.get(simulation_location)
            if response.status_code != 200:
                raise Exception("Failed to get simulation status: ", response.text)

            simulation_status = response.json()
            if 'progress' in simulation_status:
                progress = simulation_status['progress']
                print(f"Simulation progress for {alpha_data['name']}: {progress}%")
            else:
                if simulation_status['status'] == 'ERROR':
                    raise Exception(f"Simulation error for {alpha_data['name']}: ", simulation_status)
                break

            time.sleep(5)

        alpha_id = simulation_status['alpha']
        alpha_response = self._sess.get(f'{config.URL_BASE}/alphas/{alpha_id}')
        if alpha_response.status_code != 200:
            raise Exception(f"Failed to get alpha results for {alpha_data['name']}: ", alpha_response.text)

        alpha_results = alpha_response.json()
        alpha_data['result'] = alpha_results

        complete_path = os.path.join(config.COMPLETE_FOLDER, os.path.basename(alpha_path))
        with open(complete_path, 'w') as f:
            json.dump(alpha_data, f, indent=4)
        print(f"Alpha {alpha_data['name']} results saved to {complete_path}")

    def notify_complete(self, alpha_name):
        """在 complete_notify.json 文件中寫入完成的 alpha_name"""
        notify_file_path = pathlib.Path(config.COMPLETE_NOTIFY_FILE)
        if notify_file_path.exists():
            with open(notify_file_path, 'r') as f:
                complete_names = json.load(f)
        else:
            complete_names = []

        complete_names.append(alpha_name)
        with open(notify_file_path, 'w') as f:
            json.dump(complete_names, f, indent=4)

    def run(self):
        retry_count = 0
        max_retries = 100

        while retry_count < max_retries:
            try:
                running_simulations: typing.Dict[str, str] = {}
                while True:
                    pending_names = [name for name in collect_alpha_names(config.PENDING_FOLDER) if name not in running_simulations.keys()]
                    print(f"Pending names: {pending_names}")  # 添加日志
                    time.sleep(5)
                    while len(running_simulations) < config.MAX_CONCURRENT and len(pending_names):
                        name = pending_names.pop(0)
                        alpha = alpha_lib.Alpha.read_from_db(os.path.join(config.PENDING_FOLDER, name))
                        response = self._sess.post(f'{config.URL_BASE}/simulations', json=alpha.payload)
                        if response.status_code == 401:  # Unauthorized
                            print("Session expired, logging in again...")
                            self._sess = self.login()
                            continue
                        print(response.status_code)
                        print(response.headers)
                        running_simulations[name] = response.headers['Location']
                        print(f'Start simulation on {name}...')

                    for name, location_url in list(running_simulations.items()):  # 使用 list 進行迭代，以便可以在循環內部進行修改
                        response = self._sess.get(location_url)
                        if response.status_code == 401:  # Unauthorized
                            print("Session expired, logging in again...")
                            self._sess = self.login()
                            continue
                        simulation_status = response.json()
                        print(f'Checking {name}...')
                        if 'progress' not in simulation_status:
                            alpha_response = self._sess.get(f'{config.URL_BASE}/alphas/{simulation_status["alpha"]}')
                            if alpha_response.status_code == 401:  # Unauthorized
                                print("Session expired, logging in again...")
                                self._sess = self.login()
                                continue
                            alpha = alpha_lib.Alpha.read_from_db(os.path.join(config.PENDING_FOLDER, name))
                            alpha._raw_js['result'] = alpha_response.json()
                            alpha.update_status(config.COMPLETE_FOLDER)
                            self.notify_complete(name)  # 添加這行，通知 Alpha 完成
                            del running_simulations[name]  # 修正：從字典中刪除完成的名稱
                            logging.info(f'Alpha {name} done!')
                            break  # 修正：在完成處理後退出內部循環
                    if len(running_simulations) == 0:
                        print(f'No Alpha to run...')
                        time.sleep(10)
                break  # 如果成功運行，退出重試循環
            except Exception as e:
                # 清理未完成的模擬
                for name, url in running_simulations.items():
                    self._sess.delete(url)
    
                # 總是進行重試
                retry_count += 1
                print(f"Retrying in 10 minutes... Attempt {retry_count}/{max_retries}")
                print(f"An error occurred: {e}")
                time.sleep(700)  # 等待 10 分鐘

        else:
            print("Max retries reached. Exiting.")

def collect_alpha_names(status: str) -> typing.List[str]:
    files = [file for file in os.listdir(status) if file.endswith('.json')]
    # print(f"Found {len(files)} files in {status}: {files}")
    return files

if __name__ == '__main__':
    worker = Worker()
    worker.run()
