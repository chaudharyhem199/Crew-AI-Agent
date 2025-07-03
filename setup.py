"""
Setup script for the CrewAI Vector Search System.
This script initializes the database, sets up the environment, and provides utilities for system management.
"""

import os
import sys
import logging
import argparse
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'crewai', 'crewai_tools', 'psycopg2', 'sqlalchemy', 'pgvector',
        'openai', 'redis', 'pydantic', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages using: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("All required dependencies are installed")
    return True


def check_environment():
    """Check if all required environment variables are set."""
    required_env_vars = [
        'OPENAI_API_KEY',
        'POSTGRES_HOST',
        'POSTGRES_DB',
        'POSTGRES_USER',
        'POSTGRES_PASSWORD'
    ]
    
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.info("Please set these variables in your .env file or environment")
        return False
    
    logger.info("All required environment variables are set")
    return True


def create_env_template():
    """Create a template .env file."""
    env_template = """# CrewAI Vector Search System Configuration

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4-0125-preview
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# PostgreSQL Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vector_search
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here

# Redis Cache Configuration (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
ENABLE_CACHE=true
CACHE_TTL=3600

# Vector Search Configuration
SIMILARITY_THRESHOLD=0.7
MAX_RESULTS=10
SIMILARITY_METRIC=cosine
"""

    with open('.env.template', 'w') as f:
        f.write(env_template)
    
    logger.info("Created .env.template file. Please copy it to .env and fill in your values.")


def initialize_database():
    """Initialize the database with tables and extensions."""
    try:
        from document_manager import initialize_database
        success = initialize_database()
        
        if success:
            logger.info("Database initialized successfully")
            return True
        else:
            logger.error("Failed to initialize database")
            return False
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False


def add_sample_data():
    """Add sample documents to the database."""
    try:
        from document_manager import add_sample_documents
        doc_ids = add_sample_documents()
        
        if doc_ids:
            logger.info(f"Added {len(doc_ids)} sample documents")
            return True
        else:
            logger.error("Failed to add sample documents")
            return False
            
    except Exception as e:
        logger.error(f"Error adding sample documents: {e}")
        return False


def test_vector_search():
    """Test the vector search functionality."""
    try:
        from pgvector_tool import pgvector_search_tool
        
        # Test search
        result = pgvector_search_tool._run(
            query="machine learning",
            search_type="semantic",
            max_results=3
        )
        
        logger.info("Vector search test completed successfully")
        logger.debug(f"Test result: {result[:200]}...")
        return True
        
    except Exception as e:
        logger.error(f"Vector search test failed: {e}")
        return False


def test_chatbot_integration():
    """Test the chatbot integration."""
    try:
        from chatbot_integration import VectorEnhancedChatbot
        
        chatbot = VectorEnhancedChatbot(initialize_db=False)
        result = chatbot.process_user_query("What is machine learning?")
        
        logger.info("Chatbot integration test completed successfully")
        logger.debug(f"Chatbot response confidence: {result.get('confidence', 'unknown')}")
        return True
        
    except Exception as e:
        logger.error(f"Chatbot integration test failed: {e}")
        return False


def run_full_setup():
    """Run the complete setup process."""
    logger.info("Starting CrewAI Vector Search System setup...")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Setup failed: Missing dependencies")
        return False
    
    # Create env template if .env doesn't exist
    if not os.path.exists('.env'):
        create_env_template()
        logger.warning("Please configure your .env file before proceeding")
        return False
    
    # Check environment
    if not check_environment():
        logger.warning("Some environment variables are missing, but proceeding with defaults")
    
    # Initialize database
    if not initialize_database():
        logger.error("Setup failed: Database initialization failed")
        return False
    
    # Add sample data
    if not add_sample_data():
        logger.warning("Failed to add sample data, but setup can continue")
    
    # Test vector search
    if not test_vector_search():
        logger.warning("Vector search test failed, please check configuration")
    
    # Test chatbot integration
    if not test_chatbot_integration():
        logger.warning("Chatbot integration test failed, please check configuration")
    
    logger.info("Setup completed successfully!")
    return True


def main():
    """Main setup script entry point."""
    parser = argparse.ArgumentParser(description='CrewAI Vector Search System Setup')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies only')
    parser.add_argument('--check-env', action='store_true', help='Check environment variables only')
    parser.add_argument('--create-env', action='store_true', help='Create .env template only')
    parser.add_argument('--init-db', action='store_true', help='Initialize database only')
    parser.add_argument('--add-samples', action='store_true', help='Add sample documents only')
    parser.add_argument('--test-search', action='store_true', help='Test vector search only')
    parser.add_argument('--test-chatbot', action='store_true', help='Test chatbot integration only')
    parser.add_argument('--full-setup', action='store_true', help='Run complete setup (default)')
    
    args = parser.parse_args()
    
    if args.check_deps:
        return check_dependencies()
    elif args.check_env:
        return check_environment()
    elif args.create_env:
        create_env_template()
        return True
    elif args.init_db:
        return initialize_database()
    elif args.add_samples:
        return add_sample_data()
    elif args.test_search:
        return test_vector_search()
    elif args.test_chatbot:
        return test_chatbot_integration()
    else:
        # Default: run full setup
        return run_full_setup()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)