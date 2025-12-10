"""
Memory management: Short-term (conversation) and Long-term (vector store)
"""
import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class MemoryManager:
    """Manages both short-term (conversation) and long-term (vector) memory"""
    
    def __init__(self, persist_directory: str = "./memory_store", use_free_embeddings: bool = True):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embeddings for long-term memory
        # Use free embeddings by default (sentence-transformers)
        if use_free_embeddings:
            self.embeddings = self._create_free_embeddings()
        else:
            # Fallback to OpenAI if API key is available
            try:
                from langchain_openai import OpenAIEmbeddings
                self.embeddings = OpenAIEmbeddings()
            except Exception:
                print("[WARNING] OpenAI embeddings not available, using free embeddings")
                self.embeddings = self._create_free_embeddings()
        
        # Load or create vector store
        self.vector_store = self._load_or_create_vectorstore()
    
    def _create_free_embeddings(self):
        """Create free embeddings using sentence-transformers"""
        try:
            # Try new langchain-huggingface package first
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                # Fallback to deprecated version
                from langchain_community.embeddings import HuggingFaceEmbeddings
            # Use a lightweight, free embedding model
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        except ImportError:
            print("[WARNING] sentence-transformers not installed. Install with: pip install sentence-transformers")
            # Fallback to a simple mock embedding
            return self._create_mock_embeddings()
        except Exception as e:
            print(f"[WARNING] Error loading free embeddings: {e}")
            return self._create_mock_embeddings()
    
    def _create_mock_embeddings(self):
        """Create a simple mock embedding for demonstration"""
        from langchain_core.embeddings import Embeddings
        
        class MockEmbeddings(Embeddings):
            def embed_documents(self, texts):
                # Simple hash-based embeddings for demo
                import hashlib
                return [[float(int(hashlib.md5(t.encode()).hexdigest()[:8], 16)) % 1000 / 1000.0 for _ in range(384)] for t in texts]
            
            def embed_query(self, text):
                return self.embed_documents([text])[0]
        
        return MockEmbeddings()
        
    def _load_or_create_vectorstore(self) -> FAISS:
        """Load existing vector store or create a new one"""
        vector_store_path = os.path.join(self.persist_directory, "faiss_index")
        
        if os.path.exists(vector_store_path):
            try:
                return FAISS.load_local(
                    vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading vector store: {e}. Creating new one.")
        
        # Create new vector store
        return FAISS.from_texts(
            ["Initial memory store"],
            self.embeddings
        )
    
    def save_to_long_term_memory(self, content: str, metadata: Optional[Dict] = None):
        """Store information in long-term memory (vector store)"""
        try:
            doc = Document(page_content=content, metadata=metadata or {})
            self.vector_store.add_documents([doc])
            self._persist_vectorstore()
            return True
        except Exception as e:
            print(f"Error saving to long-term memory: {e}")
            return False
    
    def retrieve_from_long_term_memory(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant information from long-term memory"""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"Error retrieving from long-term memory: {e}")
            return []
    
    def _persist_vectorstore(self):
        """Save vector store to disk"""
        vector_store_path = os.path.join(self.persist_directory, "faiss_index")
        self.vector_store.save_local(vector_store_path)
    
    def get_conversation_history(self, messages: List[Dict]) -> str:
        """Format conversation history for context (short-term memory)"""
        if not messages:
            return "No previous conversation."
        
        history = []
        for msg in messages[-5:]:  # Last 5 messages for context
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history.append(f"{role}: {content}")
        
        return "\n".join(history)

