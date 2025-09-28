"""
ChromaDB-based medical knowledge storage for HIMAS agents
Stores medical knowledge that can be queried by similarity
"""

import chromadb
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class MedicalKnowledgeStore:
    """ChromaDB setup for storing medical knowledge per agent"""
    
    def __init__(self, agent_id: str, hospital_id: str):
        self.agent_id = agent_id
        self.hospital_id = hospital_id
        
        # Initialize ChromaDB client (local instance as per PoC)
        self.client = chromadb.Client()
        
        # Create unique collection name for this agent at this hospital
        self.collection_name = f"{hospital_id}_{agent_id}_knowledge"
        
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(name=self.collection_name)
            self._populate_medical_knowledge()
            logger.info(f"Created new collection: {self.collection_name}")
    
    def _populate_medical_knowledge(self):
        """Add medical knowledge relevant to our datasets"""
        medical_documents = [
            # Breast cancer knowledge
            "Breast cancer screening mammography recommended annually for women over 40",
            "Breast tumor size and lymph node involvement are key prognostic factors", 
            "HER2-positive breast cancers respond well to targeted therapy with trastuzumab",
            "Triple-negative breast cancers require chemotherapy as primary treatment",
            
            # Diabetes knowledge  
            "Type 2 diabetes diagnosis requires fasting glucose >126 mg/dL or HbA1c >6.5%",
            "Metformin is first-line therapy for type 2 diabetes management",
            "Diabetic patients require annual eye exams to screen for retinopathy",
            "Blood pressure control is crucial in diabetic patients to prevent complications",
            
            # General medical knowledge
            "Age and family history are important risk factors for chronic diseases",
            "Regular exercise and healthy diet prevent many chronic conditions"
        ]
        
        # Add documents with metadata
        metadatas = [
            {"category": "oncology", "disease": "breast_cancer", "urgency": "high"},
            {"category": "oncology", "disease": "breast_cancer", "urgency": "high"},
            {"category": "oncology", "disease": "breast_cancer", "urgency": "medium"},
            {"category": "oncology", "disease": "breast_cancer", "urgency": "high"},
            {"category": "endocrinology", "disease": "diabetes", "urgency": "medium"},
            {"category": "endocrinology", "disease": "diabetes", "urgency": "medium"},
            {"category": "endocrinology", "disease": "diabetes", "urgency": "medium"},
            {"category": "endocrinology", "disease": "diabetes", "urgency": "high"},
            {"category": "general", "disease": "both", "urgency": "low"},
            {"category": "prevention", "disease": "both", "urgency": "low"}
        ]
        
        ids = [f"{self.hospital_id}_{self.agent_id}_knowledge_{i}" 
               for i in range(len(medical_documents))]
        
        self.collection.add(
            documents=medical_documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(medical_documents)} medical documents to {self.collection_name}")
    
    def query_medical_knowledge(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """Query medical knowledge base using similarity search"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count())
            )
            
            logger.info(f"Knowledge query '{query}' returned {len(results['documents'][0])} results")
            return results
            
        except Exception as e:
            logger.error(f"Knowledge query failed: {str(e)}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def get_knowledge_count(self) -> int:
        """Get total number of knowledge documents"""
        return self.collection.count()
    
    def add_knowledge(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add new knowledge to the collection"""
        self.collection.add(
            documents=documents,
            metadatas=metadatas, 
            ids=ids
        )
        logger.info(f"Added {len(documents)} new knowledge documents")

# Test the knowledge store
if __name__ == "__main__":
    # Test knowledge store creation
    knowledge_store = MedicalKnowledgeStore("test_agent", "hospital_1")
    
    print(f"Knowledge store created with {knowledge_store.get_knowledge_count()} documents")
    
    # Test querying
    results = knowledge_store.query_medical_knowledge("breast cancer treatment", n_results=2)
    print(f"\nQuery results for 'breast cancer treatment':")
    for i, doc in enumerate(results['documents'][0]):
        print(f"{i+1}. {doc}")
    
    # Test diabetes query
    results = knowledge_store.query_medical_knowledge("diabetes management", n_results=2)
    print(f"\nQuery results for 'diabetes management':")
    for i, doc in enumerate(results['documents'][0]):
        print(f"{i+1}. {doc}")