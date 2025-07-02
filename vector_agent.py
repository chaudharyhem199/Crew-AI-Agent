"""
CrewAI agent for vector similarity search to enhance chatbot responses.
"""

import os
import logging
from typing import List, Dict, Any
from crewai import Agent
from dotenv import load_dotenv

from pgvector_tool import pgvector_search_tool
from config import config

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure OpenAI API key is set
if not config.openai_api_key:
    logger.warning("OpenAI API key not found in configuration")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# Set OpenAI model
os.environ["OPENAI_MODEL_NAME"] = os.getenv("OPENAI_MODEL_NAME", "gpt-4-0125-preview")


class VectorSearchAgent:
    """
    CrewAI agent specialized in vector similarity search for chatbot enhancement.
    
    This agent uses PostgreSQL with pgvector extension to find relevant information
    from a knowledge base and enhance chatbot responses with contextual data.
    """
    
    def __init__(self):
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create and configure the vector search agent."""
        
        return Agent(
            role='Knowledge Base Search Specialist',
            goal=(
                'Find the most relevant information from the knowledge base using vector similarity search '
                'to enhance chatbot responses with accurate and contextual information for the query: {query}'
            ),
            verbose=True,
            memory=True,
            backstory=(
                "You are an expert information retrieval specialist with deep knowledge of semantic search "
                "and vector similarity matching. Your expertise lies in understanding user queries and finding "
                "the most relevant information from large knowledge bases using advanced vector search techniques. "
                "You excel at:\n"
                "- Analyzing user queries to determine search intent\n"
                "- Choosing the optimal search strategy (semantic, keyword, or hybrid)\n"
                "- Filtering and ranking search results by relevance\n"
                "- Providing contextual information to enhance chatbot responses\n"
                "- Adapting search parameters based on query complexity and domain\n\n"
                "You always strive to provide the most accurate and helpful information while being mindful "
                "of search performance and result quality. You understand the nuances of different similarity "
                "metrics and know when to apply various search strategies for optimal results."
            ),
            tools=[pgvector_search_tool],
            allow_delegation=False,
            step_callback=self._log_step,
            max_iter=3,
            max_execution_time=30
        )
    
    def _log_step(self, step):
        """Log agent execution steps for debugging."""
        logger.debug(f"Agent step: {step}")
    
    def search_knowledge_base(
        self,
        query: str,
        search_type: str = "semantic",
        max_results: int = None,
        similarity_threshold: float = None,
        document_types: List[str] = None,
        metadata_filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Search the knowledge base using the vector search agent.
        
        Args:
            query: The search query
            search_type: Type of search ("semantic", "keyword", or "hybrid")
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score for results
            document_types: List of document types to filter by
            metadata_filters: Dictionary of metadata filters
            
        Returns:
            Dictionary containing search results and metadata
        """
        
        try:
            # Prepare the search parameters
            search_params = {
                'query': query,
                'search_type': search_type
            }
            
            if max_results is not None:
                search_params['max_results'] = max_results
            if similarity_threshold is not None:
                search_params['similarity_threshold'] = similarity_threshold
            if document_types is not None:
                search_params['document_types'] = document_types
            if metadata_filters is not None:
                search_params['metadata_filters'] = metadata_filters
            
            logger.info(f"Searching knowledge base with query: '{query}' using {search_type} search")
            
            # Execute the search using the vector search tool
            result = pgvector_search_tool._run(**search_params)
            
            logger.info(f"Search completed successfully")
            return {
                'success': True,
                'result': result,
                'query': query,
                'search_type': search_type
            }
            
        except Exception as e:
            logger.error(f"Error in knowledge base search: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'search_type': search_type
            }


# Enhanced vector search agent with additional capabilities
enhanced_vector_agent = Agent(
    role='Senior Knowledge Retrieval & Context Enhancement Specialist',
    goal=(
        'Provide comprehensive and contextually relevant information by performing advanced vector similarity '
        'searches and intelligently analyzing results to enhance chatbot responses for the topic: {topic}'
    ),
    verbose=True,
    memory=True,
    backstory=(
        "You are a highly skilled Senior Knowledge Retrieval Specialist with expertise in advanced information "
        "retrieval, semantic search, and contextual analysis. Your role is crucial in enhancing AI chatbot "
        "responses by finding and synthesizing the most relevant information from vast knowledge bases.\n\n"
        
        "Your core competencies include:\n"
        "• Advanced Vector Search: Mastery of semantic similarity search using state-of-the-art embedding models\n"
        "• Hybrid Search Strategies: Expertly combining vector and keyword search for optimal results\n"
        "• Query Analysis: Deep understanding of user intent and context to formulate optimal search strategies\n"
        "• Information Synthesis: Ability to distill and combine multiple search results into coherent insights\n"
        "• Domain Adaptation: Adjusting search parameters and strategies based on subject matter and context\n"
        "• Quality Assessment: Evaluating search result relevance and reliability\n\n"
        
        "Your approach is methodical and intelligent:\n"
        "1. Analyze the user's query to understand intent, complexity, and domain\n"
        "2. Determine the optimal search strategy (semantic for conceptual queries, keyword for specific terms, hybrid for complex needs)\n"
        "3. Execute targeted searches with appropriate parameters\n"
        "4. Evaluate and rank results based on relevance, recency, and reliability\n"
        "5. Synthesize findings to provide comprehensive, contextual information\n"
        "6. Adapt search strategies based on initial results to fill information gaps\n\n"
        
        "You excel at handling diverse query types:\n"
        "• Technical questions requiring precise documentation\n"
        "• Conceptual inquiries needing broad contextual understanding\n"
        "• Complex multi-faceted topics requiring information synthesis\n"
        "• Time-sensitive queries where recency matters\n"
        "• Domain-specific searches requiring specialized knowledge\n\n"
        
        "Your responses always prioritize accuracy, relevance, and usefulness while being mindful of "
        "performance constraints and user experience. You understand that your work directly impacts "
        "the quality and helpfulness of chatbot interactions."
    ),
    tools=[pgvector_search_tool],
    allow_delegation=False,
    max_iter=5,
    max_execution_time=45
)


# Create vector search agent instance
vector_search_agent_instance = VectorSearchAgent()


def create_adaptive_vector_agent(
    specialization: str = "general",
    max_results: int = None,
    similarity_threshold: float = None,
    preferred_search_type: str = "hybrid"
) -> Agent:
    """
    Create a specialized vector search agent with custom configuration.
    
    Args:
        specialization: Agent specialization ("technical", "academic", "business", "general")
        max_results: Default maximum results for searches
        similarity_threshold: Default similarity threshold
        preferred_search_type: Default search type preference
        
    Returns:
        Configured CrewAI Agent instance
    """
    
    specialization_configs = {
        "technical": {
            "role": "Technical Documentation Search Specialist",
            "focus": "technical documentation, API references, code examples, and implementation guides",
            "search_strategy": "Prefers semantic search for conceptual queries and keyword search for specific function/method names"
        },
        "academic": {
            "role": "Academic Research & Literature Search Specialist", 
            "focus": "research papers, academic articles, scholarly content, and educational materials",
            "search_strategy": "Emphasizes comprehensive hybrid search to capture both conceptual depth and specific terminology"
        },
        "business": {
            "role": "Business Intelligence & Market Research Specialist",
            "focus": "business insights, market analysis, industry reports, and strategic information",
            "search_strategy": "Balances keyword precision for specific metrics with semantic understanding of business concepts"
        },
        "general": {
            "role": "General Knowledge Retrieval Specialist",
            "focus": "diverse topics and general information across multiple domains",
            "search_strategy": "Adaptively chooses search strategies based on query characteristics and domain"
        }
    }
    
    config_info = specialization_configs.get(specialization, specialization_configs["general"])
    
    return Agent(
        role=config_info["role"],
        goal=(
            f'Find highly relevant information focused on {config_info["focus"]} using advanced vector '
            f'similarity search to provide comprehensive context for the query: {{query}}'
        ),
        verbose=True,
        memory=True,
        backstory=(
            f"You are a specialized {config_info['role']} with deep expertise in information retrieval "
            f"for {config_info['focus']}. Your search methodology: {config_info['search_strategy']}.\n\n"
            
            f"Default Configuration:\n"
            f"• Max Results: {max_results or config.max_results}\n"
            f"• Similarity Threshold: {similarity_threshold or config.similarity_threshold}\n"
            f"• Preferred Search Type: {preferred_search_type}\n\n"
            
            "You excel at understanding domain-specific terminology, context, and user needs. "
            "Your goal is to provide the most relevant and useful information while maintaining "
            "high standards for accuracy and completeness."
        ),
        tools=[pgvector_search_tool],
        allow_delegation=False,
        max_iter=4,
        max_execution_time=35
    )


# Pre-configured specialized agents
technical_vector_agent = create_adaptive_vector_agent("technical", max_results=8, similarity_threshold=0.75)
academic_vector_agent = create_adaptive_vector_agent("academic", max_results=12, similarity_threshold=0.70)
business_vector_agent = create_adaptive_vector_agent("business", max_results=10, similarity_threshold=0.72)