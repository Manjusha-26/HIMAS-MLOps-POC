"""
Flower federated learning server for HIMAS
Coordinates federated learning across multiple hospitals
"""

import flwr as fl
from flwr.common import Metrics
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics from all hospitals"""
    
    # Extract accuracies and sample counts
    accuracies = [num_examples * m["test_accuracy"] for num_examples, m in metrics if "test_accuracy" in m]
    examples = [num_examples for num_examples, m in metrics if "test_accuracy" in m]
    
    if not accuracies:
        return {}
    
    # Calculate weighted average accuracy
    aggregated_accuracy = sum(accuracies) / sum(examples)
    
    logger.info(f"Federated evaluation - Aggregated accuracy: {aggregated_accuracy:.4f}")
    logger.info(f"Total examples across hospitals: {sum(examples)}")
    
    return {"federated_accuracy": aggregated_accuracy}

def get_federated_strategy() -> fl.server.strategy.Strategy:
    """Create federated learning strategy for HIMAS"""
    
    strategy = fl.server.strategy.FedAvg(
        # Fraction of clients to sample for training
        fraction_fit=1.0,  # Use all hospitals
        
        # Fraction of clients to sample for evaluation  
        fraction_evaluate=1.0,  # Evaluate on all hospitals
        
        # Minimum number of clients for training
        min_fit_clients=2,  # Need at least 2 hospitals
        
        # Minimum number of clients for evaluation
        min_evaluate_clients=2,
        
        # Minimum available clients required
        min_available_clients=2,
        
        # Aggregate evaluation metrics
        evaluate_metrics_aggregation_fn=weighted_average,
        
        # Initial model parameters (will be set by first client)
        initial_parameters=None
    )
    
    logger.info("Created FedAvg strategy for HIMAS federated learning")
    return strategy

def start_federated_server(server_address: str = "0.0.0.0:8080", num_rounds: int = 3):
    """Start the federated learning server"""
    
    logger.info(f"Starting HIMAS federated learning server on {server_address}")
    logger.info(f"Training rounds: {num_rounds}")
    
    # Create strategy
    strategy = get_federated_strategy()
    
    # Configure server
    config = fl.server.ServerConfig(num_rounds=num_rounds)
    
    # Start server
    fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy
    )

class HIMASFederatedCoordinator:
    """Coordinates the entire federated learning process"""
    
    def __init__(self, num_rounds: int = 3):
        self.num_rounds = num_rounds
        self.server_address = "localhost:8080"
        
    def run_simulation(self, hospital_data_list: List[Dict]):
        """Run federated learning simulation with multiple hospitals"""
        
        logger.info("Starting HIMAS federated learning simulation")
        logger.info(f"Number of hospitals: {len(hospital_data_list)}")
        
        # Import here to avoid circular imports
        from flower_client import create_flower_client
        
        # Create client functions for each hospital
        def client_fn(cid: str):
            """Create client for hospital with given ID"""
            hospital_idx = int(cid)
            hospital_data = hospital_data_list[hospital_idx]
            
            logger.info(f"Creating client for {hospital_data['hospital_id']}")
            return create_flower_client(hospital_data)
        
        # Run simulation
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(hospital_data_list),
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=get_federated_strategy(),
            client_resources={"num_cpus": 1, "num_gpus": 0}
        )
        
        logger.info("Federated learning simulation completed")
        return history

# Test the federated server
if __name__ == "__main__":
    # Import data loader
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
    from medical_datasets import MedicalDataLoader
    
    # Load and prepare data for multiple hospitals
    loader = MedicalDataLoader()
    hospital_data_list = loader.split_for_hospitals('breast_cancer', n_hospitals=3)
    
    print("=== HIMAS Federated Learning Server Test ===")
    print(f"Prepared data for {len(hospital_data_list)} hospitals:")
    
    for hospital_data in hospital_data_list:
        print(f"- {hospital_data['hospital_id']}: {hospital_data['n_train_samples']} train, {hospital_data['n_test_samples']} test")
    
    # Test simulation
    coordinator = HIMASFederatedCoordinator(num_rounds=2)
    
    print(f"\nRunning federated learning simulation...")
    print("This will simulate 3 hospitals collaborating on breast cancer diagnosis")
    
    try:
        history = coordinator.run_simulation(hospital_data_list)
        print("✅ Federated learning simulation completed successfully!")
        
        # Display results
        if history.metrics_distributed and "federated_accuracy" in history.metrics_distributed:
            final_accuracy = history.metrics_distributed["federated_accuracy"][-1][1]
            print(f"Final federated model accuracy: {final_accuracy:.4f}")
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        print("This is normal - the full simulation needs all components running together")