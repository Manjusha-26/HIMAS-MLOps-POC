"""
Flower federated learning client for HIMAS
Each hospital runs this client to participate in federated learning
Uses Flower 1.11.0 API
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from typing import Dict, List, Tuple
import logging
import sys
import os

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Flower imports
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

logger = logging.getLogger(__name__)

# Global cache for hospital data and models
hospital_data_cache = {}
model_cache = {}


class BreastCancerModel:
    """Logistic Regression model wrapper"""
    
    def __init__(self, max_iter: int = 5000, random_state: int = 42):
        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            warm_start=True,
            solver='lbfgs'
        )
        self.is_initialized = False
    
    def initialize(self, X_sample: np.ndarray, y_sample: np.ndarray) -> None:
        """Initialize model structure"""
        if not self.is_initialized:
            self.model.fit(X_sample, y_sample)
            self.is_initialized = True
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as list of arrays"""
        if hasattr(self.model, 'coef_'):
            return [self.model.coef_.copy(), self.model.intercept_.copy()]
        return []
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from list of arrays"""
        if len(parameters) >= 2:
            self.model.coef_ = parameters[0].copy()
            self.model.intercept_ = parameters[1].copy()
            self.is_initialized = True
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model"""
        self.model.fit(X, y)
        self.is_initialized = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        return self.model.predict_proba(X)


def load_client_data(partition_id: int):
    """Load hospital data for given partition"""
    if partition_id not in hospital_data_cache:
        # Import from data module
        from data.medical_datasets import MedicalDataLoader
        
        loader = MedicalDataLoader()
        all_hospital_data = loader.split_for_hospitals('breast_cancer', n_hospitals=3)
        hospital_data_cache[partition_id] = all_hospital_data[partition_id]
        
        logger.info(f"✓ Loaded data for partition {partition_id}")
    
    return hospital_data_cache[partition_id]


def get_or_create_model(partition_id: int) -> BreastCancerModel:
    """Get or create model for this partition"""
    if partition_id not in model_cache:
        model_cache[partition_id] = BreastCancerModel()
    return model_cache[partition_id]


class HIMASFlowerClient(NumPyClient):
    """Flower client using NumPyClient interface"""
    
    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.hospital_id = f"hospital_{partition_id + 1}"
        
        # Load hospital data
        hospital_data = load_client_data(partition_id)
        self.X_train = hospital_data['X_train']
        self.y_train = hospital_data['y_train']
        self.X_test = hospital_data['X_test']
        self.y_test = hospital_data['y_test']
        
        # Get model
        self.model = get_or_create_model(partition_id)
        
        # Initialize model
        if not self.model.is_initialized:
            sample_size = min(10, len(self.X_train))
            self.model.initialize(self.X_train[:sample_size], self.y_train[:sample_size])
        
        logger.info(f"✓ Initialized {self.hospital_id}")
        logger.info(f"  Train: {len(self.X_train)} samples, Test: {len(self.X_test)} samples")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get model parameters"""
        params = self.model.get_parameters()
        logger.debug(f"{self.hospital_id}: Extracted parameters")
        return params
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters"""
        self.model.set_parameters(parameters)
        logger.debug(f"{self.hospital_id}: Updated parameters")
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model on local hospital data"""
        logger.info(f"\n{'='*60}")
        logger.info(f"{self.hospital_id}: 🔄 Starting training")
        logger.info(f"{'='*60}")
        
        # Update model with global parameters
        self.set_parameters(parameters)
        
        # Train on local data (DATA STAYS LOCAL!)
        logger.info(f"{self.hospital_id}: Training on {len(self.X_train)} samples")
        self.model.fit(self.X_train, self.y_train)
        
        # Calculate training metrics
        train_predictions = self.model.predict(self.X_train)
        train_proba = self.model.predict_proba(self.X_train)
        train_accuracy = accuracy_score(self.y_train, train_predictions)
        train_loss = log_loss(self.y_train, train_proba)
        
        logger.info(f"{self.hospital_id}: ✓ Training complete")
        logger.info(f"{self.hospital_id}:   Accuracy: {train_accuracy:.4f}")
        logger.info(f"{self.hospital_id}:   Loss: {train_loss:.4f}")
        
        # Return updated parameters (only weights, not data!)
        return (
            self.get_parameters(config),
            len(self.X_train),
            {
                'train_accuracy': float(train_accuracy),
                'train_loss': float(train_loss)
            }
        )
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate global model on local test data"""
        logger.info(f"{self.hospital_id}: 📊 Evaluating global model")
        
        # Update model with global parameters
        self.set_parameters(parameters)
        
        # Evaluate on local test data (DATA STAYS LOCAL!)
        test_predictions = self.model.predict(self.X_test)
        test_proba = self.model.predict_proba(self.X_test)
        test_accuracy = accuracy_score(self.y_test, test_predictions)
        test_loss = log_loss(self.y_test, test_proba)
        
        logger.info(f"{self.hospital_id}: ✓ Evaluation complete")
        logger.info(f"{self.hospital_id}:   Accuracy: {test_accuracy:.4f}")
        logger.info(f"{self.hospital_id}:   Loss: {test_loss:.4f}\n")
        
        return (
            float(test_loss),
            len(self.X_test),
            {
                'eval_accuracy': float(test_accuracy),
                'eval_loss': float(test_loss)
            }
        )


def client_fn(context: Context) -> NumPyClient:
    """Create a Flower client for a given partition"""
    partition_id = context.node_config.get("partition-id", 0)
    return HIMASFlowerClient(partition_id).to_client()


# Create Flower ClientApp
app = ClientApp(client_fn=client_fn)


# Test the client
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("="*60)
    print("HIMAS Flower ClientApp (Flower 1.11.0)")
    print("="*60)
    print("\n✓ Client code loaded successfully")
    print("\nTo run federated learning:")
    print("  Method 1: flwr run . (from MLOps_POC directory)")
    print("  Method 2: python -m src.federated.flower_server")
    print("="*60)