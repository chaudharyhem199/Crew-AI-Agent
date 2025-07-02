"""
Database models for vector search system using SQLAlchemy and pgvector.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, func
from pgvector.sqlalchemy import Vector
import uuid

from config import config

Base = declarative_base()


class Document(Base):
    """Model for storing documents and their metadata."""
    
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(255), nullable=True)  # URL, file path, etc.
    document_type = Column(String(50), nullable=True)  # pdf, web, text, etc.
    metadata = Column(JSON, nullable=True)  # Additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Indexes for better performance
    __table_args__ = (
        Index('idx_documents_title', 'title'),
        Index('idx_documents_source', 'source'),
        Index('idx_documents_type', 'document_type'),
        Index('idx_documents_active', 'is_active'),
        Index('idx_documents_created', 'created_at'),
    )


class DocumentEmbedding(Base):
    """Model for storing document embeddings."""
    
    __tablename__ = "document_embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False, default=0)  # For document chunks
    chunk_content = Column(Text, nullable=False)
    embedding = Column(Vector(config.embedding_dimensions), nullable=False)
    embedding_model = Column(String(100), nullable=False)
    token_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes for vector similarity search
    __table_args__ = (
        Index('idx_embeddings_document', 'document_id'),
        Index('idx_embeddings_model', 'embedding_model'),
        Index('idx_embeddings_created', 'created_at'),
        # Vector similarity indexes
        Index('idx_embeddings_cosine', 'embedding', postgresql_using='ivfflat', 
              postgresql_with={'lists': 100}, postgresql_ops={'embedding': 'vector_cosine_ops'}),
        Index('idx_embeddings_l2', 'embedding', postgresql_using='ivfflat',
              postgresql_with={'lists': 100}, postgresql_ops={'embedding': 'vector_l2_ops'}),
    )


class QueryLog(Base):
    """Model for logging search queries and results for analytics."""
    
    __tablename__ = "query_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_text = Column(Text, nullable=False)
    query_embedding = Column(Vector(config.embedding_dimensions), nullable=True)
    similarity_threshold = Column(Float, nullable=False)
    max_results = Column(Integer, nullable=False)
    similarity_metric = Column(String(20), nullable=False)
    results_count = Column(Integer, nullable=False)
    execution_time_ms = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, nullable=True)  # Additional query context
    
    # Indexes for analytics
    __table_args__ = (
        Index('idx_query_logs_timestamp', 'timestamp'),
        Index('idx_query_logs_metric', 'similarity_metric'),
        Index('idx_query_logs_threshold', 'similarity_threshold'),
    )


class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self):
        self.engine = create_engine(
            config.database_url,
            echo=False,  # Set to True for SQL debugging
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all tables and enable pgvector extension."""
        with self.engine.connect() as conn:
            # Enable pgvector extension
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
        
        # Create all tables
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()
    
    def drop_tables(self):
        """Drop all tables (useful for testing)."""
        Base.metadata.drop_all(bind=self.engine)


# Global database manager instance
db_manager = DatabaseManager()


def get_db_session():
    """Dependency for getting database session."""
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()