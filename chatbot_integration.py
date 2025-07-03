"""
Example integration of the vector search system with a CrewAI chatbot.
This demonstrates how to enhance chatbot responses with vector similarity search.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from crewai import Crew, Process
from dotenv import load_dotenv

# Import our custom components
from vector_agent import enhanced_vector_agent, vector_search_agent_instance
from vector_tasks import vector_search_task, quick_answer_task, contextual_search_task
from pgvector_tool import pgvector_search_tool
from document_manager import document_manager, initialize_database, add_sample_documents
from config import config

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorEnhancedChatbot:
    """
    A chatbot that uses vector similarity search to enhance responses with relevant information.
    """
    
    def __init__(self, initialize_db: bool = True):
        """
        Initialize the vector-enhanced chatbot.
        
        Args:
            initialize_db: Whether to initialize the database on startup
        """
        self.vector_agent = vector_search_agent_instance
        
        if initialize_db:
            self._setup_database()
    
    def _setup_database(self):
        """Initialize database and add sample documents if needed."""
        try:
            logger.info("Initializing database...")
            success = initialize_database()
            
            if success:
                # Check if we have any documents
                docs = document_manager.list_documents(limit=1)
                if not docs:
                    logger.info("Adding sample documents...")
                    add_sample_documents()
                    logger.info("Sample documents added successfully")
                
                logger.info("Database setup completed")
            else:
                logger.error("Failed to initialize database")
                
        except Exception as e:
            logger.error(f"Database setup error: {e}")
    
    def search_knowledge_base(
        self,
        query: str,
        search_type: str = "hybrid",
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: The search query
            search_type: Type of search ("semantic", "keyword", or "hybrid")
            max_results: Maximum number of results to return
            
        Returns:
            Search results and metadata
        """
        
        return self.vector_agent.search_knowledge_base(
            query=query,
            search_type=search_type,
            max_results=max_results
        )
    
    def enhance_response(
        self,
        user_query: str,
        base_response: Optional[str] = None,
        search_type: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Enhance a chatbot response with relevant information from the knowledge base.
        
        Args:
            user_query: The user's original query
            base_response: The chatbot's base response (optional)
            search_type: Type of search to perform
            
        Returns:
            Enhanced response with supporting information
        """
        
        try:
            # Search for relevant information
            search_result = self.search_knowledge_base(
                query=user_query,
                search_type=search_type,
                max_results=3
            )
            
            if search_result['success']:
                import json
                search_data = json.loads(search_result['result'])
                
                enhancement = {
                    'original_query': user_query,
                    'base_response': base_response,
                    'search_results': search_data.get('results', []),
                    'search_metadata': {
                        'results_count': search_data.get('results_count', 0),
                        'execution_time_ms': search_data.get('execution_time_ms', 0),
                        'search_type': search_data.get('search_type', search_type)
                    },
                    'enhancement_status': 'success',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Extract key information for enhancement
                if search_data.get('results'):
                    enhancement['key_insights'] = []
                    enhancement['supporting_sources'] = []
                    
                    for result in search_data['results'][:3]:
                        enhancement['key_insights'].append({
                            'content': result.get('chunk_content', '')[:200] + '...',
                            'relevance_score': result.get('similarity_score', 0),
                            'source': result.get('source', 'Unknown')
                        })
                        
                        if result.get('source'):
                            enhancement['supporting_sources'].append(result['source'])
                
                return enhancement
                
            else:
                return {
                    'original_query': user_query,
                    'base_response': base_response,
                    'enhancement_status': 'failed',
                    'error': search_result.get('error', 'Unknown error'),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error enhancing response: {e}")
            return {
                'original_query': user_query,
                'base_response': base_response,
                'enhancement_status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def process_user_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process a user query and return an enhanced response.
        
        Args:
            user_query: The user's query
            
        Returns:
            Complete response with enhancements
        """
        
        logger.info(f"Processing user query: {user_query}")
        
        # For this example, we'll use the vector search to provide the primary response
        enhancement = self.enhance_response(user_query)
        
        if enhancement['enhancement_status'] == 'success' and enhancement.get('key_insights'):
            # Build a comprehensive response
            response_parts = []
            
            # Add relevant information from search results
            for insight in enhancement['key_insights'][:2]:
                response_parts.append(f"Based on relevant information: {insight['content']}")
            
            # Combine into final response
            final_response = "\n\n".join(response_parts)
            
            return {
                'query': user_query,
                'response': final_response,
                'enhancement_data': enhancement,
                'confidence': 'high' if len(enhancement['key_insights']) > 1 else 'medium',
                'sources': enhancement.get('supporting_sources', [])
            }
        
        else:
            # Fallback response
            return {
                'query': user_query,
                'response': "I don't have specific information about that topic in my knowledge base. Could you provide more details or try rephrasing your question?",
                'enhancement_data': enhancement,
                'confidence': 'low',
                'sources': []
            }


class CrewAIVectorChatbot:
    """
    A comprehensive CrewAI chatbot with vector search capabilities.
    """
    
    def __init__(self):
        """Initialize the CrewAI vector chatbot."""
        self.enhanced_chatbot = VectorEnhancedChatbot()
        self.crew = self._create_crew()
    
    def _create_crew(self):
        """Create a CrewAI crew with vector search capabilities."""
        
        return Crew(
            agents=[enhanced_vector_agent],
            tasks=[quick_answer_task],
            process=Process.sequential,
            memory=True,
            cache=True,
            max_rpm=100,
            share_crew=True
        )
    
    def ask_question(self, question: str) -> str:
        """
        Ask a question and get an enhanced response using CrewAI and vector search.
        
        Args:
            question: The user's question
            
        Returns:
            Enhanced response
        """
        
        try:
            # Execute the CrewAI workflow
            result = self.crew.kickoff(inputs={'user_question': question})
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Error in CrewAI execution: {e}")
            
            # Fallback to direct vector search
            fallback_result = self.enhanced_chatbot.process_user_query(question)
            return fallback_result['response']


def main():
    """
    Example usage of the vector-enhanced chatbot system.
    """
    
    # Example 1: Direct vector search enhancement
    print("=== Direct Vector Search Enhancement ===")
    chatbot = VectorEnhancedChatbot()
    
    test_queries = [
        "What is machine learning?",
        "How do I write better Python code?",
        "Explain vector databases"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = chatbot.process_user_query(query)
        print(f"Response: {result['response']}")
        print(f"Confidence: {result['confidence']}")
        if result['sources']:
            print(f"Sources: {', '.join(result['sources'])}")
        print("-" * 50)
    
    # Example 2: CrewAI integration
    print("\n=== CrewAI Vector Chatbot ===")
    crew_chatbot = CrewAIVectorChatbot()
    
    question = "What are the key principles of machine learning?"
    print(f"\nQuestion: {question}")
    answer = crew_chatbot.ask_question(question)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()