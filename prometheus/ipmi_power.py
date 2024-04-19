from prometheus_client import start_http_server, Gauge
import time
import subprocess

def get_power():
   try:
       cmd = "sudo ipmiutil sensor -i 25 -s | grep Power | cut -d ' ' -f 29"
       res = subprocess.check_output(cmd, shell=True).decode('utf-8')
       # res = res.splitlines()[4].split(' | ')[6].split(' Watts')[0]
       return float(res)
   except Exception as e:
       print(f"Fetching power error", e)
       return None

impi_power = Gauge('ipmi_power', 'Power consumption in Watts')

if __name__ == '__main__':
   start_http_server(9000)

   while True:
       power = get_power()
       if power is not None:
           impi_power.set(power)
       time.sleep(1)
