"""
Flower federated learning client for HIMAS
Each hospital runs this client to participate in federated learning
"""

import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, List, Tuple, Optional
import logging
import pickle

logger = logging.getLogger(__name__)

class HIMASFlowerClient(fl.client.NumPyClient):
    """Flower client for HIMAS federated learning"""
    
    def __init__(self, hospital_id: str, X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray):
        self.hospital_id = hospital_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test  
        self.y_test = y_test
        
        # Initialize logistic regression model
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        
        logger.info(f"Initialized Flower client for {hospital_id}")
        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get model parameters"""
        if hasattr(self.model, 'coef_'):
            # Model has been trained, return parameters
            params = [self.model.coef_, self.model.intercept_]
            logger.info(f"{self.hospital_id}: Sharing model parameters")
            return params
        else:
            # Model not trained yet, return random parameters
            n_features = self.X_train.shape[1]
            n_classes = len(np.unique(self.y_train))
            if n_classes == 2:
                coef_shape = (1, n_features)
                intercept_shape = (1,)
            else:
                coef_shape = (n_classes, n_features)
                intercept_shape = (n_classes,)
            
            return [
                np.random.normal(0, 0.01, coef_shape),
                np.random.normal(0, 0.01, intercept_shape)
            ]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from federated averaging"""
        if len(parameters) >= 2:
            # Set the model parameters
            coef, intercept = parameters[0], parameters[1]
            
            # Create a new model and set parameters
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            
            # Fit on a small sample first to initialize the model structure
            if len(self.X_train) > 0:
                sample_size = min(10, len(self.X_train))
                self.model.fit(self.X_train[:sample_size], self.y_train[:sample_size])
                
                # Now set the federated parameters
                self.model.coef_ = coef
                self.model.intercept_ = intercept
                
            logger.info(f"{self.hospital_id}: Updated model with federated parameters")
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model on local data"""
        # Set parameters from federation
        self.set_parameters(parameters)
        
        # Train on local hospital data
        self.model.fit(self.X_train, self.y_train)
        
        # Calculate training metrics
        train_predictions = self.model.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, train_predictions)
        
        logger.info(f"{self.hospital_id}: Local training completed")
        logger.info(f"{self.hospital_id}: Training accuracy: {train_accuracy:.4f}")
        
        # Return updated parameters, number of samples, and metrics
        return (
            self.get_parameters(config), 
            len(self.X_train),
            {"train_accuracy": train_accuracy}
        )
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate model on local test data"""
        # Set parameters from federation
        self.set_parameters(parameters)
        
        if not hasattr(self.model, 'coef_'):
            # Model not trained, fit on training data first
            self.model.fit(self.X_train, self.y_train)
        
        # Evaluate on local test data
        test_predictions = self.model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, test_predictions)
        
        # Calculate loss (negative log-likelihood approximation)
        test_loss = 1.0 - test_accuracy
        
        logger.info(f"{self.hospital_id}: Local evaluation completed")
        logger.info(f"{self.hospital_id}: Test accuracy: {test_accuracy:.4f}")
        
        return (
            test_loss,
            len(self.X_test), 
            {"test_accuracy": test_accuracy}
        )

def create_flower_client(hospital_data: Dict) -> HIMASFlowerClient:
    """Create Flower client for a hospital"""
    return HIMASFlowerClient(
        hospital_id=hospital_data['hospital_id'],
        X_train=hospital_data['X_train'],
        y_train=hospital_data['y_train'],
        X_test=hospital_data['X_test'],
        y_test=hospital_data['y_test']
    )

# Test the Flower client
if __name__ == "__main__":
    # Import the data loader
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
    from medical_datasets import MedicalDataLoader
    
    # Load and split data
    loader = MedicalDataLoader()
    hospital_data = loader.split_for_hospitals('breast_cancer', n_hospitals=3)
    
    # Create client for first hospital
    client = create_flower_client(hospital_data[0])
    
    print(f"Created Flower client for {hospital_data[0]['hospital_id']}")
    print(f"Train samples: {hospital_data[0]['n_train_samples']}")
    print(f"Test samples: {hospital_data[0]['n_test_samples']}")
    
    # Test parameter operations
    params = client.get_parameters({})
    print(f"Model parameters shape: coef={params[0].shape}, intercept={params[1].shape}")