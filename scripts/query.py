#!/usr/bin/env python3
"""
RAG Query Interface
Query documents stored in Qdrant with LLM integration
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Optional

import yaml
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI


class RAGQuery:
    """Handles querying the RAG system"""
    
    def __init__(self, config_path: str = None):
        """Initialize the query system"""
        
        # Auto-detect config path
        if config_path is None:
            if os.path.exists("/app/config/collections.yaml"):
                config_path = "/app/config/collections.yaml"
            else:
                config_path = "config/collections.yaml"
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Qdrant client
        qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        self.qdrant = QdrantClient(url=qdrant_url)
        print(f"✓ Connected to Qdrant at {qdrant_url}")
        
        # Initialize embeddings model
        settings = self.config.get('settings', {})
        model_name = settings.get('embedding_model', 'BAAI/bge-large-en-v1.5')
        device = settings.get('embedding_device', 'cpu')
        
        # Auto-detect if MPS is available
        import torch
        if device == 'mps' and not torch.backends.mps.is_available():
            print("Warning: MPS requested but not available, falling back to CPU")
            device = 'cpu'
        elif device == 'mps':
            print("✓ Using Metal Performance Shaders (MPS) for acceleration")
        
        print(f"Loading embedding model: {model_name} on device: {device}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"✓ Embedding model loaded")
        
        # Initialize LLM client (LM Studio)
        lm_studio_url = os.getenv('LM_STUDIO_URL', 'http://localhost:1234/v1')
        self.llm = OpenAI(
            base_url=lm_studio_url,
            api_key="lm-studio"
        )
        print(f"✓ Connected to LM Studio at {lm_studio_url}")
    
    def search(self, 
               query: str, 
               collections: List[str] = None,
               top_k: int = 5) -> List[Any]:
        """Search across collections"""
        
        if collections is None:
            # Default to all available collections
            all_collections = self.qdrant.get_collections().collections
            collections = [c.name for c in all_collections]
        
        print(f"\nSearching in collections: {', '.join(collections)}")
        
        # Embed query
        query_vector = self.embeddings.embed_query(query)
        
        # Search each collection
        all_results = []
        for collection in collections:
            try:
                # Qdrant 1.8.0+ uses query_points
                from qdrant_client.models import PointStruct, Filter
                
                search_result = self.qdrant.query_points(
                    collection_name=collection,
                    query=query_vector,
                    limit=top_k,
                    with_payload=True
                )
                
                # Extract points from the result
                if hasattr(search_result, 'points'):
                    all_results.extend(search_result.points)
                else:
                    all_results.extend(search_result)
                    
            except Exception as e:
                print(f"Warning: Could not search collection '{collection}': {e}")
                import traceback
                traceback.print_exc()
        
        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
 
    def query_with_context(self, 
                          question: str,
                          collections: List[str] = None,
                          top_k: int = 5,
                          temperature: float = 0.3,
                          max_tokens: int = 1000) -> Dict[str, Any]:
        """Query LLM with RAG context"""
        
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        # Get relevant context
        results = self.search(question, collections, top_k)
        
        if not results:
            return {
                "answer": "No relevant information found in the knowledge base.",
                "sources": []
            }
        
        print(f"Found {len(results)} relevant chunks")
        
        # Build context
        context_parts = []
        for idx, result in enumerate(results, 1):
            source = result.payload['metadata'].get('source_file', 'Unknown')
            text = result.payload['text']
            score = result.score
            context_parts.append(
                f"[Source {idx}] (Relevance: {score:.3f})\n"
                f"File: {source}\n"
                f"{text}\n"
            )
        
        context = "\n---\n\n".join(context_parts)
        
        # Build prompt
        system_prompt = """You are a helpful assistant with access to a knowledge base. 
Answer questions using ONLY the provided context from the documents. 
If the context doesn't contain enough information to answer the question, say so clearly.
Always cite which source(s) you used (e.g., "According to Source 1...").
Be precise and factual."""

        user_prompt = f"""Context from knowledge base:

{context}

---

Question: {question}

Answer based on the context above:"""
        
        print("\nQuerying LLM...")
        
        # Query LLM
        try:
            response = self.llm.chat.completions.create(
                model="local-model",  # LM Studio uses whatever model is loaded
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content
            
        except Exception as e:
            answer = f"Error querying LLM: {e}\n\nMake sure LM Studio is running with a model loaded."
        
        # Extract source metadata
        sources = []
        for result in results:
            sources.append({
                'file': result.payload['metadata'].get('source_file', 'Unknown'),
                'collection': result.payload['metadata'].get('collection', 'Unknown'),
                'score': result.score,
                'snippet': result.payload['text'][:200] + "..."
            })
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def print_result(self, result: Dict[str, Any]):
        """Pretty print query result"""
        print(f"\n{'='*60}")
        print("ANSWER:")
        print(f"{'='*60}")
        print(result['answer'])
        
        print(f"\n{'='*60}")
        print("SOURCES:")
        print(f"{'='*60}")
        for idx, source in enumerate(result['sources'], 1):
            print(f"\n[{idx}] {source['file']}")
            print(f"    Collection: {source['collection']}")
            print(f"    Relevance: {source['score']:.3f}")
            print(f"    Snippet: {source['snippet']}")


def main():
    parser = argparse.ArgumentParser(
        description='Query the RAG system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query with LLM (default)
  python query.py "How does grappling work in D&D?"
  
  # Query specific collection
  python query.py "Who is Strahd?" --collection dnd_curse_of_strahd
  
  # Query multiple collections
  python query.py "What undead creatures exist?" --collection dnd_core_rules dnd_curse_of_strahd
  
  # Search only (no LLM)
  python query.py "death saving throws" --search-only
  
  # Adjust number of results
  python query.py "spell slots" --top-k 10
        """
    )
    
    parser.add_argument(
        'question',
        type=str,
        help='Question to ask'
    )
    
    parser.add_argument(
        '--collection',
        type=str,
        nargs='+',
        help='Collection(s) to search (default: all)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to retrieve (default: 5)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='LLM temperature (default: 0.3 for factual responses)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1000,
        help='Maximum tokens in LLM response (default: 1000)'
    )
    
    parser.add_argument(
        '--search-only',
        action='store_true',
        help='Only search, do not query LLM'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/collections.yaml',
        help='Path to collections config file'
    )
    
    args = parser.parse_args()
    
    try:
        rag = RAGQuery(config_path=args.config)
        
        if args.search_only:
            # Just search, don't use LLM
            results = rag.search(
                query=args.question,
                collections=args.collection,
                top_k=args.top_k
            )
            
            print(f"\n{'='*60}")
            print(f"SEARCH RESULTS ({len(results)} found):")
            print(f"{'='*60}")
            
            for idx, result in enumerate(results, 1):
                print(f"\n[{idx}] Score: {result.score:.3f}")
                print(f"File: {result.payload['metadata'].get('source_file', 'Unknown')}")
                print(f"Collection: {result.payload['metadata'].get('collection', 'Unknown')}")
                print(f"Text: {result.payload['text'][:300]}...")
        else:
            # Query with LLM
            result = rag.query_with_context(
                question=args.question,
                collections=args.collection,
                top_k=args.top_k,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            rag.print_result(result)
        
    except Exception as e:
        print(f"\n✗ Error during query: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

