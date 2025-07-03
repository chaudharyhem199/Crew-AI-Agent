"""
Document management utilities for ingesting and processing documents for vector search.
"""

import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import hashlib
import math

from openai import OpenAI
from sqlalchemy.orm import Session
from sqlalchemy import and_

from config import config
from models import Document, DocumentEmbedding, db_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentManager:
    """
    Manages document ingestion, processing, and embedding generation for vector search.
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.max_chunk_size = 1000  # Maximum tokens per chunk
        self.chunk_overlap = 100    # Overlap between chunks
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = None) -> List[str]:
        """
        Split text into overlapping chunks for better embedding coverage.
        
        Args:
            text: Text to split
            max_chunk_size: Maximum size per chunk in characters (approximates tokens)
            
        Returns:
            List of text chunks
        """
        if max_chunk_size is None:
            max_chunk_size = self.max_chunk_size * 4  # Rough approximation: 1 token ≈ 4 chars
        
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # If we're not at the end, try to break at a sentence or paragraph boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + max_chunk_size // 2, start + 1), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI API."""
        try:
            response = self.openai_client.embeddings.create(
                model=config.embedding_model,
                input=text.replace("\n", " "),
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def _estimate_token_count(self, text: str) -> int:
        """Rough estimation of token count."""
        return math.ceil(len(text.split()) * 1.3)  # Rough approximation
    
    def add_document(
        self,
        title: str,
        content: str,
        source: Optional[str] = None,
        document_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_content: bool = True
    ) -> str:
        """
        Add a new document to the vector database.
        
        Args:
            title: Document title
            content: Document content
            source: Source URL or file path
            document_type: Type of document (pdf, web, text, etc.)
            metadata: Additional metadata
            chunk_content: Whether to split content into chunks
            
        Returns:
            Document ID
        """
        
        session = db_manager.get_session()
        
        try:
            # Create document record
            document = Document(
                title=title,
                content=content,
                source=source,
                document_type=document_type,
                metadata=metadata or {}
            )
            
            session.add(document)
            session.flush()  # Get the document ID
            
            document_id = document.id
            logger.info(f"Created document: {document_id}")
            
            # Process content into chunks if requested
            if chunk_content and content:
                chunks = self._split_text_into_chunks(content)
            else:
                chunks = [content] if content else []
            
            # Generate embeddings for each chunk
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                
                try:
                    embedding = self._generate_embedding(chunk)
                    token_count = self._estimate_token_count(chunk)
                    
                    doc_embedding = DocumentEmbedding(
                        document_id=document_id,
                        chunk_index=i,
                        chunk_content=chunk,
                        embedding=embedding,
                        embedding_model=config.embedding_model,
                        token_count=token_count
                    )
                    
                    session.add(doc_embedding)
                    logger.debug(f"Created embedding for chunk {i} of document {document_id}")
                    
                except Exception as e:
                    logger.error(f"Error creating embedding for chunk {i}: {e}")
                    continue
            
            session.commit()
            logger.info(f"Successfully processed document {document_id} with {len(chunks)} chunks")
            
            return str(document_id)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding document: {e}")
            raise
        finally:
            session.close()
    
    def update_document(
        self,
        document_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        source: Optional[str] = None,
        document_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing document and regenerate embeddings if content changed.
        
        Args:
            document_id: ID of document to update
            title: New title (optional)
            content: New content (optional) 
            source: New source (optional)
            document_type: New document type (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if successful, False otherwise
        """
        
        session = db_manager.get_session()
        
        try:
            # Get existing document
            document = session.query(Document).filter(
                Document.id == document_id
            ).first()
            
            if not document:
                logger.error(f"Document {document_id} not found")
                return False
            
            content_changed = False
            
            # Update fields
            if title is not None:
                document.title = title
            if content is not None and content != document.content:
                document.content = content
                content_changed = True
            if source is not None:
                document.source = source
            if document_type is not None:
                document.document_type = document_type
            if metadata is not None:
                document.metadata = metadata
            
            document.updated_at = datetime.utcnow()
            
            # If content changed, regenerate embeddings
            if content_changed:
                # Delete existing embeddings
                session.query(DocumentEmbedding).filter(
                    DocumentEmbedding.document_id == document_id
                ).delete()
                
                # Generate new embeddings
                chunks = self._split_text_into_chunks(content)
                
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    
                    try:
                        embedding = self._generate_embedding(chunk)
                        token_count = self._estimate_token_count(chunk)
                        
                        doc_embedding = DocumentEmbedding(
                            document_id=document_id,
                            chunk_index=i,
                            chunk_content=chunk,
                            embedding=embedding,
                            embedding_model=config.embedding_model,
                            token_count=token_count
                        )
                        
                        session.add(doc_embedding)
                        
                    except Exception as e:
                        logger.error(f"Error creating embedding for chunk {i}: {e}")
                        continue
                
                logger.info(f"Regenerated embeddings for document {document_id}")
            
            session.commit()
            logger.info(f"Successfully updated document {document_id}")
            
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating document {document_id}: {e}")
            return False
        finally:
            session.close()
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and its embeddings.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if successful, False otherwise
        """
        
        session = db_manager.get_session()
        
        try:
            # Delete embeddings first (due to foreign key constraint)
            embedding_count = session.query(DocumentEmbedding).filter(
                DocumentEmbedding.document_id == document_id
            ).count()
            
            session.query(DocumentEmbedding).filter(
                DocumentEmbedding.document_id == document_id
            ).delete()
            
            # Delete document
            document_count = session.query(Document).filter(
                Document.id == document_id
            ).delete()
            
            session.commit()
            
            if document_count > 0:
                logger.info(f"Deleted document {document_id} and {embedding_count} embeddings")
                return True
            else:
                logger.warning(f"Document {document_id} not found")
                return False
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
        finally:
            session.close()
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document details by ID.
        
        Args:
            document_id: ID of document to retrieve
            
        Returns:
            Document details or None if not found
        """
        
        session = db_manager.get_session()
        
        try:
            document = session.query(Document).filter(
                Document.id == document_id
            ).first()
            
            if not document:
                return None
            
            # Get embedding count
            embedding_count = session.query(DocumentEmbedding).filter(
                DocumentEmbedding.document_id == document_id
            ).count()
            
            return {
                'id': str(document.id),
                'title': document.title,
                'content': document.content,
                'source': document.source,
                'document_type': document.document_type,
                'metadata': document.metadata,
                'created_at': document.created_at.isoformat(),
                'updated_at': document.updated_at.isoformat(),
                'is_active': document.is_active,
                'embedding_count': embedding_count
            }
            
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {e}")
            return None
        finally:
            session.close()
    
    def list_documents(
        self,
        document_type: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List documents with optional filtering.
        
        Args:
            document_type: Filter by document type
            source: Filter by source
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of document summaries
        """
        
        session = db_manager.get_session()
        
        try:
            query = session.query(Document).filter(Document.is_active == True)
            
            if document_type:
                query = query.filter(Document.document_type == document_type)
            if source:
                query = query.filter(Document.source == source)
            
            query = query.order_by(Document.created_at.desc())
            query = query.offset(offset).limit(limit)
            
            documents = []
            for doc in query.all():
                # Get embedding count
                embedding_count = session.query(DocumentEmbedding).filter(
                    DocumentEmbedding.document_id == doc.id
                ).count()
                
                documents.append({
                    'id': str(doc.id),
                    'title': doc.title,
                    'source': doc.source,
                    'document_type': doc.document_type,
                    'created_at': doc.created_at.isoformat(),
                    'updated_at': doc.updated_at.isoformat(),
                    'embedding_count': embedding_count,
                    'content_length': len(doc.content) if doc.content else 0
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
        finally:
            session.close()
    
    def bulk_add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> List[str]:
        """
        Add multiple documents in batches.
        
        Args:
            documents: List of document dictionaries with keys: title, content, source, document_type, metadata
            batch_size: Number of documents to process in each batch
            
        Returns:
            List of document IDs created
        """
        
        created_ids = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            for doc_data in batch:
                try:
                    doc_id = self.add_document(
                        title=doc_data.get('title', ''),
                        content=doc_data.get('content', ''),
                        source=doc_data.get('source'),
                        document_type=doc_data.get('document_type'),
                        metadata=doc_data.get('metadata')
                    )
                    created_ids.append(doc_id)
                    
                except Exception as e:
                    logger.error(f"Error adding document in batch: {e}")
                    continue
            
            logger.info(f"Processed batch {i//batch_size + 1}: {len(batch)} documents")
        
        logger.info(f"Bulk add completed: {len(created_ids)} documents created")
        return created_ids


# Global document manager instance
document_manager = DocumentManager()


def initialize_database():
    """Initialize the database with tables and extensions."""
    try:
        db_manager.create_tables()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False


def add_sample_documents():
    """Add some sample documents for testing."""
    sample_docs = [
        {
            'title': 'Introduction to Machine Learning',
            'content': 'Machine learning is a subset of artificial intelligence that focuses on developing algorithms that can learn and make decisions from data. It involves training models on datasets to recognize patterns and make predictions.',
            'document_type': 'educational',
            'metadata': {'category': 'AI/ML', 'difficulty': 'beginner'}
        },
        {
            'title': 'Python Programming Best Practices',
            'content': 'Writing clean, maintainable Python code requires following established conventions. Use meaningful variable names, write docstrings for functions, and follow PEP 8 style guidelines. Consider using type hints to improve code readability.',
            'document_type': 'technical',
            'metadata': {'category': 'programming', 'language': 'python'}
        },
        {
            'title': 'Vector Databases Explained',
            'content': 'Vector databases are specialized databases designed to store and search high-dimensional vectors efficiently. They use techniques like approximate nearest neighbor search to find similar vectors quickly, making them ideal for AI applications.',
            'document_type': 'technical',
            'metadata': {'category': 'databases', 'technology': 'vector search'}
        }
    ]
    
    return document_manager.bulk_add_documents(sample_docs)