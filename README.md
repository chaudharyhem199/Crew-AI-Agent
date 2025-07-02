# CrewAI Vector Search System

A comprehensive PostgreSQL pgvector-powered semantic search system for enhancing CrewAI chatbot responses with relevant contextual information.

## 🌟 Features

- **Advanced Vector Search**: Semantic similarity search using OpenAI embeddings and PostgreSQL pgvector
- **Hybrid Search**: Combines vector similarity with keyword search for optimal results
- **CrewAI Integration**: Seamlessly integrates with CrewAI agents and workflows
- **Intelligent Caching**: Redis-based caching to reduce embedding generation costs
- **Multiple Search Strategies**: Semantic, keyword, and hybrid search modes
- **Configurable Parameters**: Customizable similarity thresholds, result limits, and distance metrics
- **Document Management**: Complete document lifecycle management with chunking and metadata
- **Error Handling**: Robust error handling and fallback mechanisms
- **Performance Monitoring**: Query logging and analytics for optimization

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- Redis (optional, for caching)
- OpenAI API key

### Installation

1. **Clone and install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Setup PostgreSQL with pgvector:**
```sql
-- Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create database
CREATE DATABASE vector_search;
```

3. **Configure environment:**
```bash
# Copy environment template
python setup.py --create-env
cp .env.template .env

# Edit .env with your configuration
# Required: OPENAI_API_KEY, POSTGRES_* settings
```

4. **Initialize the system:**
```bash
# Run complete setup
python setup.py --full-setup

# Or step by step:
python setup.py --check-deps
python setup.py --init-db
python setup.py --add-samples
python setup.py --test-search
```

### Basic Usage

```python
from chatbot_integration import VectorEnhancedChatbot

# Initialize chatbot with vector search
chatbot = VectorEnhancedChatbot()

# Ask a question
result = chatbot.process_user_query("What is machine learning?")
print(result['response'])
print(f"Confidence: {result['confidence']}")
```

## 📁 Project Structure

```
├── config.py                 # Configuration management
├── models.py                 # SQLAlchemy database models
├── pgvector_tool.py          # Custom CrewAI vector search tool
├── vector_agent.py           # CrewAI agents for vector search
├── vector_tasks.py           # CrewAI tasks for different search scenarios
├── document_manager.py       # Document ingestion and management
├── chatbot_integration.py    # Complete chatbot integration examples
├── setup.py                  # System setup and testing utilities
├── tools.py                  # Updated tools including vector search
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔧 Configuration

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vector_search
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here

# Vector Search Settings
SIMILARITY_THRESHOLD=0.7
MAX_RESULTS=10
SIMILARITY_METRIC=cosine

# Cache Settings (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
ENABLE_CACHE=true
CACHE_TTL=3600
```

### Database Schema

The system creates three main tables:

- **documents**: Store document content and metadata
- **document_embeddings**: Store vector embeddings with pgvector
- **query_logs**: Analytics and performance monitoring

## 🎯 Usage Examples

### 1. Basic Vector Search

```python
from pgvector_tool import pgvector_search_tool

# Semantic search
result = pgvector_search_tool._run(
    query="machine learning algorithms",
    search_type="semantic",
    max_results=5
)

# Hybrid search (semantic + keyword)
result = pgvector_search_tool._run(
    query="python programming best practices",
    search_type="hybrid",
    similarity_threshold=0.75
)
```

### 2. Document Management

```python
from document_manager import document_manager

# Add a document
doc_id = document_manager.add_document(
    title="Introduction to Neural Networks",
    content="Neural networks are computing systems...",
    document_type="educational",
    metadata={"category": "AI", "difficulty": "intermediate"}
)

# Update document
success = document_manager.update_document(
    document_id=doc_id,
    content="Updated neural networks content..."
)

# Search documents
documents = document_manager.list_documents(
    document_type="educational",
    limit=10
)
```

### 3. CrewAI Integration

```python
from crewai import Crew, Process
from vector_agent import enhanced_vector_agent
from vector_tasks import quick_answer_task

# Create a crew with vector search capabilities
crew = Crew(
    agents=[enhanced_vector_agent],
    tasks=[quick_answer_task],
    process=Process.sequential,
    memory=True,
    cache=True
)

# Execute with vector search enhancement
result = crew.kickoff(inputs={
    'user_question': 'Explain the difference between supervised and unsupervised learning'
})
```

### 4. Specialized Agents

```python
from vector_agent import technical_vector_agent, academic_vector_agent, business_vector_agent

# Technical documentation search
tech_result = technical_vector_agent.search_knowledge_base(
    query="PostgreSQL pgvector configuration",
    search_type="keyword",
    max_results=8
)

# Academic research search
academic_result = academic_vector_agent.search_knowledge_base(
    query="transformer architecture research",
    search_type="hybrid",
    max_results=12
)
```

## 📊 Performance Features

### Caching
- Redis-based embedding cache reduces API calls
- Configurable TTL and cache policies
- Automatic cache invalidation

### Query Analytics
- Execution time tracking
- Result quality metrics
- Search pattern analysis
- Performance optimization insights

### Similarity Metrics
- **Cosine similarity**: Best for semantic search
- **L2 distance**: Good for exact matching
- **Hybrid scoring**: Combines multiple signals

## 🛠️ Advanced Configuration

### Custom Similarity Thresholds

```python
# Configure for different use cases
config = {
    "technical_docs": {"threshold": 0.8, "metric": "cosine"},
    "general_qa": {"threshold": 0.7, "metric": "cosine"},
    "exact_match": {"threshold": 0.9, "metric": "l2"}
}
```

### Document Chunking

```python
# Customize chunking strategy
document_manager.max_chunk_size = 1500  # tokens
document_manager.chunk_overlap = 150    # tokens
```

### Search Optimization

```python
# Optimize for different scenarios
search_configs = {
    "speed": {"max_results": 5, "search_type": "semantic"},
    "quality": {"max_results": 15, "search_type": "hybrid"},
    "precision": {"similarity_threshold": 0.85, "search_type": "keyword"}
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Issues**
   ```bash
   python setup.py --check-env
   ```

2. **Missing pgvector Extension**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. **OpenAI API Errors**
   - Check API key validity
   - Verify rate limits
   - Monitor usage quotas

4. **Performance Issues**
   - Enable Redis caching
   - Adjust similarity thresholds
   - Optimize chunk sizes

### Testing

```bash
# Test individual components
python setup.py --test-search
python setup.py --test-chatbot

# Full system test
python chatbot_integration.py
```

## 🚀 Deployment

### Production Considerations

1. **Database Optimization**
   - Configure pgvector indexes
   - Set appropriate connection pools
   - Monitor query performance

2. **Caching Strategy**
   - Redis cluster for high availability
   - Appropriate TTL settings
   - Cache warming strategies

3. **Security**
   - Secure API key management
   - Database access controls
   - Input validation and sanitization

4. **Monitoring**
   - Query performance metrics
   - Error rate tracking
   - Resource utilization monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- Check the troubleshooting section
- Review the examples
- Open an issue on GitHub

## 📚 Additional Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)