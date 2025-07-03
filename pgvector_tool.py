"""
Custom CrewAI tool for PostgreSQL vector similarity search using pgvector.
"""

import time
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

import redis
import numpy as np
from openai import OpenAI
from sqlalchemy import text, func, and_, or_
from sqlalchemy.orm import Session
from crewai_tools import BaseTool
from pydantic import Field

from config import config
from models import Document, DocumentEmbedding, QueryLog, db_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PGVectorSearchTool(BaseTool):
    """
    Custom CrewAI tool for performing vector similarity search in PostgreSQL with pgvector extension.
    
    Supports:
    - Semantic vector search using OpenAI embeddings
    - Hybrid search combining keyword and vector search
    - Caching with Redis to reduce embedding generation
    - Multiple similarity metrics (cosine, L2)
    - Configurable similarity thresholds and result limits
    """
    
    name: str = "PGVector Search Tool"
    description: str = (
        "Performs semantic and hybrid search using PostgreSQL with pgvector extension. "
        "Can search for similar content using vector embeddings, combine with keyword search, "
        "and return the most relevant documents to enhance chatbot responses."
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.redis_client = None
        if config.enable_cache:
            try:
                self.redis_client = redis.from_url(config.redis_url)
                self.redis_client.ping()
                logger.info("Redis cache connected successfully")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis cache: {e}")
                self.redis_client = None
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for embeddings."""
        content = f"{text}:{model}"
        return f"embedding:{hashlib.md5(content.encode()).hexdigest()}"
    
    def _get_cached_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Retrieve cached embedding if available."""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._get_cache_key(text, model)
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        
        return None
    
    def _cache_embedding(self, text: str, model: str, embedding: List[float]) -> None:
        """Cache embedding for future use."""
        if not self.redis_client:
            return
        
        try:
            cache_key = self._get_cache_key(text, model)
            cached_data = json.dumps(embedding)
            self.redis_client.setex(cache_key, config.cache_ttl, cached_data)
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
    
    def _generate_embedding(self, text: str, model: str = None) -> List[float]:
        """Generate embedding using OpenAI API with caching."""
        if not model:
            model = config.embedding_model
        
        # Check cache first
        cached_embedding = self._get_cached_embedding(text, model)
        if cached_embedding:
            logger.debug("Using cached embedding")
            return cached_embedding
        
        try:
            # Generate new embedding
            response = self.openai_client.embeddings.create(
                model=model,
                input=text.replace("\n", " "),
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            # Cache the embedding
            self._cache_embedding(text, model, embedding)
            
            logger.debug(f"Generated new embedding for text length: {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def _get_similarity_function(self, metric: str) -> str:
        """Get the appropriate similarity function for the metric."""
        if metric.lower() == "cosine":
            return "1 - (embedding <=> %s)"
        elif metric.lower() == "l2":
            return "-(embedding <-> %s)"
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    def _perform_vector_search(
        self,
        session: Session,
        query_embedding: List[float],
        similarity_threshold: float = None,
        max_results: int = None,
        similarity_metric: str = None,
        document_types: List[str] = None,
        metadata_filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        
        if similarity_threshold is None:
            similarity_threshold = config.similarity_threshold
        if max_results is None:
            max_results = config.max_results
        if similarity_metric is None:
            similarity_metric = config.similarity_metric
        
        similarity_func = self._get_similarity_function(similarity_metric)
        
        # Build the query
        query = session.query(
            DocumentEmbedding,
            Document,
            text(similarity_func).label('similarity')
        ).join(
            Document, DocumentEmbedding.document_id == Document.id
        ).filter(
            Document.is_active == True
        )
        
        # Apply document type filters
        if document_types:
            query = query.filter(Document.document_type.in_(document_types))
        
        # Apply metadata filters
        if metadata_filters:
            for key, value in metadata_filters.items():
                query = query.filter(Document.metadata[key].astext == str(value))
        
        # Apply similarity threshold and ordering
        if similarity_metric.lower() == "cosine":
            query = query.filter(text("1 - (embedding <=> %s) >= %s") % (query_embedding, similarity_threshold))
            query = query.order_by(text("embedding <=> %s").params(embedding=query_embedding))
        else:  # L2
            query = query.filter(text("-(embedding <-> %s) >= %s") % (query_embedding, similarity_threshold))
            query = query.order_by(text("embedding <-> %s").params(embedding=query_embedding))
        
        # Limit results
        query = query.limit(max_results)
        
        results = []
        for embedding_row, document_row, similarity in query.all():
            results.append({
                'document_id': str(document_row.id),
                'title': document_row.title,
                'content': document_row.content,
                'chunk_content': embedding_row.chunk_content,
                'chunk_index': embedding_row.chunk_index,
                'source': document_row.source,
                'document_type': document_row.document_type,
                'metadata': document_row.metadata or {},
                'similarity_score': float(similarity),
                'created_at': document_row.created_at.isoformat(),
                'updated_at': document_row.updated_at.isoformat()
            })
        
        return results
    
    def _perform_keyword_search(
        self,
        session: Session,
        query_text: str,
        max_results: int = None,
        document_types: List[str] = None,
        metadata_filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Perform keyword-based search using PostgreSQL full-text search."""
        
        if max_results is None:
            max_results = config.max_results
        
        # Build the query using PostgreSQL full-text search
        query = session.query(Document).filter(
            Document.is_active == True
        )
        
        # Add full-text search
        search_query = query_text.replace(" ", " & ")
        query = query.filter(
            or_(
                func.to_tsvector('english', Document.title).match(search_query),
                func.to_tsvector('english', Document.content).match(search_query)
            )
        )
        
        # Apply document type filters
        if document_types:
            query = query.filter(Document.document_type.in_(document_types))
        
        # Apply metadata filters
        if metadata_filters:
            for key, value in metadata_filters.items():
                query = query.filter(Document.metadata[key].astext == str(value))
        
        # Order by relevance and limit results
        query = query.order_by(
            func.ts_rank(
                func.to_tsvector('english', Document.title + ' ' + Document.content),
                func.plainto_tsquery('english', query_text)
            ).desc()
        ).limit(max_results)
        
        results = []
        for document_row in query.all():
            results.append({
                'document_id': str(document_row.id),
                'title': document_row.title,
                'content': document_row.content,
                'source': document_row.source,
                'document_type': document_row.document_type,
                'metadata': document_row.metadata or {},
                'created_at': document_row.created_at.isoformat(),
                'updated_at': document_row.updated_at.isoformat(),
                'search_type': 'keyword'
            })
        
        return results
    
    def _log_query(
        self,
        session: Session,
        query_text: str,
        query_embedding: List[float],
        similarity_threshold: float,
        max_results: int,
        similarity_metric: str,
        results_count: int,
        execution_time_ms: float,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Log search query for analytics."""
        try:
            query_log = QueryLog(
                query_text=query_text,
                query_embedding=query_embedding,
                similarity_threshold=similarity_threshold,
                max_results=max_results,
                similarity_metric=similarity_metric,
                results_count=results_count,
                execution_time_ms=execution_time_ms,
                metadata=metadata or {}
            )
            session.add(query_log)
            session.commit()
        except Exception as e:
            logger.warning(f"Failed to log query: {e}")
            session.rollback()
    
    def _run(
        self,
        query: str,
        search_type: str = "semantic",
        similarity_threshold: float = None,
        max_results: int = None,
        similarity_metric: str = None,
        document_types: List[str] = None,
        metadata_filters: Dict[str, Any] = None,
        hybrid_weight: float = 0.7
    ) -> str:
        """
        Execute the vector search tool.
        
        Args:
            query: The search query text
            search_type: Type of search - "semantic", "keyword", or "hybrid"
            similarity_threshold: Minimum similarity score for results
            max_results: Maximum number of results to return
            similarity_metric: Similarity metric to use ("cosine" or "l2")
            document_types: List of document types to filter by
            metadata_filters: Dictionary of metadata filters
            hybrid_weight: Weight for semantic vs keyword search in hybrid mode (0-1)
        
        Returns:
            JSON string containing search results
        """
        start_time = time.time()
        
        try:
            # Get database session
            session = db_manager.get_session()
            
            results = []
            query_embedding = None
            
            if search_type in ["semantic", "hybrid"]:
                # Generate embedding for the query
                query_embedding = self._generate_embedding(query)
                
                # Perform vector search
                vector_results = self._perform_vector_search(
                    session=session,
                    query_embedding=query_embedding,
                    similarity_threshold=similarity_threshold,
                    max_results=max_results,
                    similarity_metric=similarity_metric,
                    document_types=document_types,
                    metadata_filters=metadata_filters
                )
                
                if search_type == "semantic":
                    results = vector_results
                else:  # hybrid
                    # Also perform keyword search
                    keyword_results = self._perform_keyword_search(
                        session=session,
                        query_text=query,
                        max_results=max_results,
                        document_types=document_types,
                        metadata_filters=metadata_filters
                    )
                    
                    # Combine and rank results
                    results = self._combine_hybrid_results(
                        vector_results, keyword_results, hybrid_weight
                    )
            
            elif search_type == "keyword":
                # Perform only keyword search
                results = self._perform_keyword_search(
                    session=session,
                    query_text=query,
                    max_results=max_results,
                    document_types=document_types,
                    metadata_filters=metadata_filters
                )
            
            else:
                raise ValueError(f"Unsupported search type: {search_type}")
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Log the query
            self._log_query(
                session=session,
                query_text=query,
                query_embedding=query_embedding,
                similarity_threshold=similarity_threshold or config.similarity_threshold,
                max_results=max_results or config.max_results,
                similarity_metric=similarity_metric or config.similarity_metric,
                results_count=len(results),
                execution_time_ms=execution_time_ms,
                metadata={'search_type': search_type}
            )
            
            session.close()
            
            # Format response
            response = {
                'query': query,
                'search_type': search_type,
                'results_count': len(results),
                'execution_time_ms': execution_time_ms,
                'results': results[:max_results or config.max_results]
            }
            
            logger.info(f"Search completed: {len(results)} results in {execution_time_ms:.2f}ms")
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            error_response = {
                'error': str(e),
                'query': query,
                'search_type': search_type
            }
            return json.dumps(error_response, indent=2)
    
    def _combine_hybrid_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        hybrid_weight: float
    ) -> List[Dict[str, Any]]:
        """Combine vector and keyword search results for hybrid search."""
        
        # Create a dictionary to merge results by document_id
        combined_results = {}
        
        # Add vector results with weighted scores
        for result in vector_results:
            doc_id = result['document_id']
            result['search_type'] = 'vector'
            result['final_score'] = result['similarity_score'] * hybrid_weight
            combined_results[doc_id] = result
        
        # Add keyword results with weighted scores
        for i, result in enumerate(keyword_results):
            doc_id = result['document_id']
            keyword_score = (len(keyword_results) - i) / len(keyword_results)  # Rank-based score
            
            if doc_id in combined_results:
                # Combine scores for documents found in both searches
                combined_results[doc_id]['final_score'] += keyword_score * (1 - hybrid_weight)
                combined_results[doc_id]['search_type'] = 'hybrid'
            else:
                # Add new keyword-only result
                result['search_type'] = 'keyword'
                result['final_score'] = keyword_score * (1 - hybrid_weight)
                combined_results[doc_id] = result
        
        # Sort by final score and return
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )
        
        return sorted_results


# Create the tool instance
pgvector_search_tool = PGVectorSearchTool()