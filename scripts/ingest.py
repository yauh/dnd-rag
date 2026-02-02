#!/usr/bin/env python3
"""
Generic RAG Document Ingestion System
Processes documents from collections and stores them in Qdrant
"""

import os
import sys
import json
import hashlib
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import yaml
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


class RAGIngester:
    """Handles document ingestion into Qdrant vector database"""
    
    def __init__(self, config_path: str = None):
        """Initialize the ingester with configuration"""
        
        # Auto-detect config path
        if config_path is None:
            if os.path.exists("/app/config/collections.yaml"):
                config_path = "/app/config/collections.yaml"  # Docker path
            else:
                config_path = "config/collections.yaml"  # Native path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Qdrant client
        qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        self.client = QdrantClient(url=qdrant_url)
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
        
        # State tracking
        self.state_file = settings.get('state_file', '/app/state/processed_files.json')
        # Auto-detect state file path for native execution
        if not os.path.exists(os.path.dirname(self.state_file)):
            self.state_file = '/Volumes/corsair4tb/Users/stephan/docker/rag_state/processed_files.json'
        
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        self.track_state = settings.get('track_processed_files', True)
        self.processed_files = self._load_state()
        
    def _load_state(self) -> Dict[str, Any]:
        """Load processing state from file"""
        if not self.track_state:
            return {}
        
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load state file: {e}")
                return {}
        return {}
    
    def _save_state(self):
        """Save processing state to file"""
        if not self.track_state:
            return
        
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.processed_files, f, indent=2)
    
    def _get_file_hash(self, filepath: str) -> str:
        """Calculate hash of file for change detection"""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _is_file_processed(self, filepath: str, collection_name: str) -> bool:
        """Check if file has already been processed"""
        if not self.track_state:
            return False
        
        file_hash = self._get_file_hash(filepath)
        key = f"{collection_name}:{filepath}"
        
        if key in self.processed_files:
            return self.processed_files[key] == file_hash
        
        return False
    
    def _mark_file_processed(self, filepath: str, collection_name: str):
        """Mark file as processed"""
        if not self.track_state:
            return
        
        file_hash = self._get_file_hash(filepath)
        key = f"{collection_name}:{filepath}"
        self.processed_files[key] = file_hash
    
    def create_collection(self, collection_name: str, vector_size: int = 1024):
        """Create a Qdrant collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            if any(c.name == collection_name for c in collections):
                print(f"✓ Collection '{collection_name}' already exists")
                return
            
            settings = self.config.get('settings', {})
            distance = Distance.COSINE if settings.get('distance_metric') == 'cosine' else Distance.EUCLID
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            print(f"✓ Created collection: {collection_name}")
        except Exception as e:
            print(f"Error creating collection {collection_name}: {e}")
            raise
    
    def load_documents(self, source_path: str, file_patterns: List[str]) -> List[Any]:
        """Load documents from directory"""
        documents = []
        source_path_obj = Path(source_path)
        
        if not source_path_obj.exists():
            print(f"Warning: Path does not exist: {source_path}")
            return documents
        
        # Find all matching files
        all_files = []
        for pattern in file_patterns:
            all_files.extend(source_path_obj.rglob(pattern))
        
        if not all_files:
            print(f"Warning: No files found matching patterns {file_patterns} in {source_path}")
            return documents
        
        print(f"Found {len(all_files)} files to process")
        
        # Load each file
        for filepath in all_files:
            try:
                if filepath.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(filepath))
                elif filepath.suffix.lower() == '.md':
                    loader = UnstructuredMarkdownLoader(str(filepath))
                elif filepath.suffix.lower() == '.txt':
                    loader = TextLoader(str(filepath))
                else:
                    print(f"Skipping unsupported file type: {filepath}")
                    continue
                
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source_file'] = str(filepath)
                documents.extend(docs)
                
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        return documents
    
    def process_collection(self, collection_name: str, force: bool = False):
        """Process a single collection"""
        
        if collection_name not in self.config['collections']:
            print(f"Error: Collection '{collection_name}' not found in config")
            return
        
        collection_config = self.config['collections'][collection_name]
        print(f"\n{'='*60}")
        print(f"Processing collection: {collection_name}")
        print(f"Description: {collection_config['description']}")
        print(f"{'='*60}\n")
        
        # Create collection in Qdrant
        settings = self.config.get('settings', {})
        vector_dim = settings.get('vector_dimension', 1024)
        self.create_collection(collection_name, vector_dim)
        
        # Load documents
        source_path = collection_config['source_path']
        file_patterns = collection_config.get('file_patterns', ['*.pdf'])
        
        documents = self.load_documents(source_path, file_patterns)
        
        if not documents:
            print(f"No documents found for collection '{collection_name}'")
            return
        
        # Filter out already processed files
        if not force:
            original_count = len(documents)
            documents = [
                doc for doc in documents 
                if not self._is_file_processed(doc.metadata['source_file'], collection_name)
            ]
            skipped = original_count - len(documents)
            if skipped > 0:
                print(f"Skipping {skipped} already processed files")
        
        if not documents:
            print(f"All files already processed for '{collection_name}'")
            return
        
        print(f"Processing {len(documents)} documents...")
        
        # Split into chunks
        chunk_size = collection_config.get('chunk_size', 1000)
        chunk_overlap = collection_config.get('chunk_overlap', 200)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Add collection metadata
        collection_metadata = collection_config.get('metadata', {})
        for chunk in chunks:
            chunk.metadata.update(collection_metadata)
            chunk.metadata['collection'] = collection_name
            chunk.metadata['ingested_at'] = datetime.now().isoformat()
        
        # Generate embeddings and upload
        print(f"Generating embeddings and uploading to Qdrant...")
        
        batch_size = settings.get('batch_size', 100)
        points = []
        
        for idx, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            try:
                # Generate unique ID
                chunk_id = hashlib.md5(
                    f"{chunk.page_content}{chunk.metadata}".encode()
                ).hexdigest()
                
                # Embed
                vector = self.embeddings.embed_query(chunk.page_content)
                
                points.append(PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={
                        "text": chunk.page_content,
                        "metadata": chunk.metadata
                    }
                ))
                
                # Batch upload
                if len(points) >= batch_size:
                    self.client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    points = []
                
            except Exception as e:
                print(f"Error processing chunk {idx}: {e}")
        
        # Upload remaining points
        if points:
            self.client.upsert(collection_name=collection_name, points=points)
        
        # Mark files as processed
        processed_files = set(doc.metadata['source_file'] for doc in documents)
        for filepath in processed_files:
            self._mark_file_processed(filepath, collection_name)
        
        self._save_state()
        
        print(f"\n✓ Completed '{collection_name}': {len(chunks)} chunks indexed")
        print(f"✓ Processed {len(processed_files)} files")
    
    def process_all_collections(self, force: bool = False):
        """Process all collections defined in config"""
        collections = self.config['collections'].keys()
        print(f"Processing {len(collections)} collections...")
        
        for collection_name in collections:
            try:
                self.process_collection(collection_name, force)
            except Exception as e:
                print(f"Error processing collection '{collection_name}': {e}")
                continue


def main():
    parser = argparse.ArgumentParser(
        description='Ingest documents into RAG system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a specific collection
  python ingest.py --collection dnd_core_rules
  
  # Process all collections
  python ingest.py --all
  
  # Force re-process (ignore state tracking)
  python ingest.py --collection dnd_core_rules --force
  
  # Process all with force
  python ingest.py --all --force
        """
    )
    
    parser.add_argument(
        '--collection',
        type=str,
        help='Name of collection to process'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all collections'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-processing of all files (ignore state tracking)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/collections.yaml',
        help='Path to collections config file'
    )
    
    args = parser.parse_args()
    
    if not args.collection and not args.all:
        parser.print_help()
        sys.exit(1)
    
    try:
        ingester = RAGIngester(config_path=args.config)
        
        if args.all:
            ingester.process_all_collections(force=args.force)
        else:
            ingester.process_collection(args.collection, force=args.force)
        
        print("\n" + "="*60)
        print("✓ Ingestion completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

