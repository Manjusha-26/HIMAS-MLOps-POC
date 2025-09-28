import sys
import os
import time
sys.path.append('src')

from src.data.medical_datasets import MedicalDataLoader
from src.federated.flower_client import create_flower_client
import flwr as fl

def run_hospital_client(hospital_index, sleep_time=5):
    time.sleep(sleep_time)
    
    loader = MedicalDataLoader()
    hospital_data = loader.split_for_hospitals('breast_cancer', n_hospitals=3)[hospital_index]
    client = create_flower_client(hospital_data)
    
    fl.client.start_numpy_client(server_address='himas-server:8080', client=client)

if __name__ == "__main__":
    hospital_index = int(sys.argv[1])
    sleep_time = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_hospital_client(hospital_index, sleep_time)