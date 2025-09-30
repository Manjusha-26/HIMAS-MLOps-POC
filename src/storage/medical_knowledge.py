"""
ChromaDB-based medical knowledge storage for HIMAS agents
Stores medical knowledge that can be queried by similarity search

MEDICAL DISCLAIMER:
This knowledge base contains general medical information for demonstration 
and educational purposes only. It should NOT be used for actual medical 
diagnosis, treatment decisions, or patient care. Always consult qualified 
healthcare professionals for medical advice. The information provided here 
is simplified and may not reflect the complexity of individual patient cases.
"""


# Just to demonstrate ChromaDB usage for medical knowledge storage


import chromadb
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

MEDICAL_DISCLAIMER = """
IMPORTANT MEDICAL DISCLAIMER:
This system is a proof-of-concept demonstration only.
- NOT intended for clinical use or patient care
- Information is simplified and may be incomplete
- Always consult healthcare professionals for medical decisions
- Individual patient care requires personalized clinical judgment
"""

class MedicalKnowledgeStore:
    """ChromaDB setup for storing medical knowledge per agent"""
    
    def __init__(self, agent_id: str, hospital_id: str):
        self.agent_id = agent_id
        self.hospital_id = hospital_id
        
        # Initialize ChromaDB client (local instance for PoC)
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
        """
        Add breast cancer medical knowledge for demonstration purposes.
        
        Note: This is simplified educational content. Real systems would use
        comprehensive medical databases with citations and evidence levels.
        """
        medical_documents = [
            # Screening and Detection
            "Breast cancer screening with mammography is generally recommended annually for women aged 40 and older, though individual risk factors may modify this recommendation.",
            
            "Clinical breast examination and patient self-awareness complement mammographic screening in early breast cancer detection.",
            
            # Diagnostic Factors
            "Tumor size is a key prognostic factor in breast cancer, with larger tumors generally associated with worse outcomes.",
            
            "Lymph node involvement indicates cancer spread beyond the breast and significantly affects prognosis and treatment planning.",
            
            "Tumor grade reflects how abnormal cancer cells appear under microscopy and helps predict cancer aggressiveness.",
            
            # Molecular Subtypes
            "HER2-positive breast cancers represent approximately 15-20% of cases and may respond to targeted therapies such as trastuzumab.",
            
            "Hormone receptor-positive breast cancers (ER+ and/or PR+) represent the majority of cases and may benefit from endocrine therapy.",
            
            "Triple-negative breast cancers lack ER, PR, and HER2 expression and typically require chemotherapy as primary systemic treatment.",
            
            # Treatment Approaches
            "Treatment decisions in breast cancer depend on multiple factors including stage, molecular subtype, patient age, and overall health status.",
            
            "Multidisciplinary tumor boards involving surgeons, medical oncologists, radiation oncologists, and pathologists help optimize treatment planning.",
            
            # Risk Factors
            "Age is a significant risk factor for breast cancer, with incidence increasing with advancing age.",
            
            "Family history and genetic factors (such as BRCA1/BRCA2 mutations) substantially increase breast cancer risk.",
            
            # Follow-up Care
            "Regular surveillance after breast cancer treatment includes periodic imaging and clinical examinations to detect potential recurrence.",
            
            "Survivorship care addresses both physical and psychosocial needs of breast cancer survivors.",
            
            # General Medical Principles
            "Evidence-based medicine integrates clinical expertise, patient values, and best available research evidence in clinical decision-making.",
            
            "Patient-centered care involves shared decision-making and consideration of individual patient preferences and circumstances."
        ]
        
        # Add metadata with categories and source types
        metadatas = [
            {"category": "screening", "disease": "breast_cancer", "urgency": "medium", "source_type": "clinical_guideline"},
            {"category": "screening", "disease": "breast_cancer", "urgency": "medium", "source_type": "clinical_practice"},
            {"category": "diagnostic", "disease": "breast_cancer", "urgency": "high", "source_type": "prognostic_factor"},
            {"category": "diagnostic", "disease": "breast_cancer", "urgency": "high", "source_type": "prognostic_factor"},
            {"category": "diagnostic", "disease": "breast_cancer", "urgency": "high", "source_type": "prognostic_factor"},
            {"category": "molecular", "disease": "breast_cancer", "urgency": "high", "source_type": "treatment_targeting"},
            {"category": "molecular", "disease": "breast_cancer", "urgency": "high", "source_type": "treatment_targeting"},
            {"category": "molecular", "disease": "breast_cancer", "urgency": "high", "source_type": "treatment_targeting"},
            {"category": "treatment", "disease": "breast_cancer", "urgency": "high", "source_type": "clinical_guideline"},
            {"category": "treatment", "disease": "breast_cancer", "urgency": "high", "source_type": "clinical_practice"},
            {"category": "risk_factors", "disease": "breast_cancer", "urgency": "medium", "source_type": "epidemiology"},
            {"category": "risk_factors", "disease": "breast_cancer", "urgency": "high", "source_type": "genetics"},
            {"category": "follow_up", "disease": "breast_cancer", "urgency": "medium", "source_type": "clinical_guideline"},
            {"category": "follow_up", "disease": "breast_cancer", "urgency": "medium", "source_type": "survivorship"},
            {"category": "general", "disease": "breast_cancer", "urgency": "low", "source_type": "medical_principle"},
            {"category": "general", "disease": "breast_cancer", "urgency": "low", "source_type": "medical_principle"}
        ]
        
        ids = [f"{self.hospital_id}_{self.agent_id}_knowledge_{i}" 
               for i in range(len(medical_documents))]
        
        self.collection.add(
            documents=medical_documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(medical_documents)} breast cancer knowledge documents to {self.collection_name}")
        logger.warning("REMINDER: This is demonstration data only - not for clinical use")
    
    def query_medical_knowledge(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Query medical knowledge base using similarity search.
        
        Returns relevant medical knowledge based on semantic similarity to the query.
        Note: Results are for demonstration purposes only.
        """
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
        """Get total number of knowledge documents in the collection"""
        return self.collection.count()
    
    def add_knowledge(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """
        Add new knowledge to the collection.
        
        In production systems, this would include validation, source verification,
        and medical review processes.
        """
        self.collection.add(
            documents=documents,
            metadatas=metadatas, 
            ids=ids
        )
        logger.info(f"Added {len(documents)} new knowledge documents")
        logger.warning("Ensure all added medical knowledge is properly validated and sourced")
    
    def get_disclaimer(self) -> str:
        """Return the medical disclaimer for this knowledge store"""
        return MEDICAL_DISCLAIMER


# Test the knowledge store
if __name__ == "__main__":
    print(MEDICAL_DISCLAIMER)
    print("\n" + "="*70)
    
    # Test knowledge store creation
    knowledge_store = MedicalKnowledgeStore("test_agent", "hospital_1")
    
    print(f"\nKnowledge store created with {knowledge_store.get_knowledge_count()} documents")
    
    # Test various queries
    test_queries = [
        "breast cancer screening guidelines",
        "HER2 positive treatment options",
        "tumor size prognosis"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: '{query}'")
        print("-"*70)
        
        results = knowledge_store.query_medical_knowledge(query, n_results=2)
        
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            print(f"\n{i+1}. {doc}")
            print(f"   Category: {metadata.get('category', 'N/A')}")
            print(f"   Source Type: {metadata.get('source_type', 'N/A')}")
    
    print("\n" + "="*70)
    print("Test completed")
    print("="*70)