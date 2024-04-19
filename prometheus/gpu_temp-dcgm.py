from prometheus_client import start_http_server, Gauge
import time
import subprocess

def get_power():
   try:
       # cmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv"
       cmd = "curl localhost:9400/metrics"
       res = subprocess.check_output(cmd, shell=True).decode('utf-8')
       gpu1 = res.splitlines()[14].split(" ")[-1]
       gpu2 = res.splitlines()[15].split(" ")[-1]
       # gpu1 = res.splitlines()[1]
       # gpu2 = res.splitlines()[2]
       return float(gpu1), float(gpu2)
   except Exception as e:
       print(f"Fetching power error", e)
       return None, None

gpu_temp = Gauge('gpu_temp_dcgm1', 'GPU temp. consumption in Watts')
gpu_temp_2 = Gauge('gpu_temp_dcgm2', 'GPU temp. consumption in Watts')

if __name__ == '__main__':
   start_http_server(9702)

   while True:
       gpu1, gpu2 = get_power()
       if gpu1 is not None and gpu2 is not None:
           gpu_temp.set(gpu1)
           gpu_temp_2.set(gpu2)
       time.sleep(1)
