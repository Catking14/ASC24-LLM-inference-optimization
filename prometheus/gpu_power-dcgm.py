from prometheus_client import start_http_server, Gauge
import time
import subprocess

def get_power():
   try:
       # cmd = "nvidia-smi --query-gpu=power.draw --format=csv"
       cmd = "curl localhost:9400/metrics"
       res = subprocess.check_output(cmd, shell=True).decode('utf-8')
       gpu1 = res.splitlines()[18].split(" ")[-1]
       gpu2 = res.splitlines()[19].split(" ")[-1]
       # gpu1 = res.splitlines()[1].split(" ")[0]
       # gpu2 = res.splitlines()[2].split(" ")[0]
       print(gpu1, gpu2)
       return float(gpu1), float(gpu2)
   except Exception as e:
       print(f"Fetching power error", e)
       return None, None

gpu_power = Gauge('gpu_power_dcgm1', 'GPU Power consumption in Watts')
gpu_power_2 = Gauge('gpu_power_dcgm2', 'GPU Power consumption in Watts')

if __name__ == '__main__':
   start_http_server(8787)

   while True:
       gpu1, gpu2 = get_power()
       if gpu1 is not None and gpu2 is not None:
           gpu_power.set(gpu1)
           gpu_power_2.set(gpu2)
       time.sleep(1)
