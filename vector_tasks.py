"""
CrewAI tasks for vector similarity search and knowledge retrieval.
"""

from crewai import Task
from vector_agent import enhanced_vector_agent, technical_vector_agent, academic_vector_agent, business_vector_agent


# Basic vector search task
vector_search_task = Task(
    description=(
        "Search the knowledge base using vector similarity search to find the most relevant information "
        "for the given query: '{query}'. Use semantic search to understand the context and intent, "
        "and return comprehensive information that can enhance chatbot responses.\n\n"
        "Instructions:\n"
        "1. Analyze the query to understand its intent and complexity\n"
        "2. Choose the appropriate search strategy (semantic, keyword, or hybrid)\n"
        "3. Execute the search with optimal parameters\n"
        "4. Evaluate and rank results by relevance\n"
        "5. Provide a summary of findings with key insights\n\n"
        "Focus on finding accurate, relevant, and contextually appropriate information."
    ),
    expected_output=(
        "A comprehensive search report containing:\n"
        "- Search strategy used and reasoning\n"
        "- Number of relevant results found\n"
        "- Top 3-5 most relevant pieces of information\n"
        "- Key insights and context that can enhance chatbot responses\n"
        "- Confidence scores and relevance metrics\n"
        "- Recommendations for follow-up searches if needed"
    ),
    agent=enhanced_vector_agent,
    output_file="vector_search_results.md"
)


# Enhanced contextual search task
contextual_search_task = Task(
    description=(
        "Perform advanced contextual search for the topic: '{topic}' to provide comprehensive "
        "background information and context for AI chatbot enhancement.\n\n"
        "This task requires:\n"
        "1. Multi-faceted query analysis to identify different aspects of the topic\n"
        "2. Strategic use of both semantic and hybrid search approaches\n"
        "3. Information synthesis from multiple relevant sources\n"
        "4. Quality assessment of retrieved information\n"
        "5. Contextual organization of findings\n\n"
        "Search parameters:\n"
        "- Use hybrid search for comprehensive coverage\n"
        "- Prioritize recent and authoritative information\n"
        "- Consider multiple document types and sources\n"
        "- Apply domain-specific filtering when appropriate"
    ),
    expected_output=(
        "An in-depth contextual analysis report including:\n"
        "- Executive summary of the topic\n"
        "- Key concepts and definitions\n"
        "- Multiple perspectives and viewpoints\n"
        "- Current trends and developments\n"
        "- Practical applications and examples\n"
        "- Related topics and connections\n"
        "- Source quality assessment\n"
        "- Actionable insights for chatbot enhancement"
    ),
    agent=enhanced_vector_agent,
    output_file="contextual_analysis.md"
)


# Technical documentation search task
technical_search_task = Task(
    description=(
        "Search for technical documentation and implementation details related to: '{technical_query}'\n\n"
        "Requirements:\n"
        "1. Focus on technical accuracy and implementation details\n"
        "2. Prioritize official documentation, API references, and code examples\n"
        "3. Use precise keyword matching for technical terms\n"
        "4. Validate information currency and version compatibility\n"
        "5. Organize findings by complexity and use case\n\n"
        "Search strategy:\n"
        "- Use semantic search for conceptual understanding\n"
        "- Use keyword search for specific functions/methods\n"
        "- Filter for technical document types\n"
        "- Prioritize recent updates and stable versions"
    ),
    expected_output=(
        "Technical documentation report containing:\n"
        "- Overview of technical concepts\n"
        "- Implementation details and code examples\n"
        "- API references and usage patterns\n"
        "- Best practices and common pitfalls\n"
        "- Version information and compatibility notes\n"
        "- Related tools and dependencies\n"
        "- Step-by-step implementation guides\n"
        "- Troubleshooting information"
    ),
    agent=technical_vector_agent,
    output_file="technical_documentation.md"
)


# Academic research task
academic_research_task = Task(
    description=(
        "Conduct comprehensive academic research on: '{research_topic}' using scholarly sources "
        "and academic literature.\n\n"
        "Research methodology:\n"
        "1. Identify key academic concepts and terminology\n"
        "2. Search for peer-reviewed papers and scholarly articles\n"
        "3. Analyze research trends and methodologies\n"
        "4. Synthesize findings from multiple sources\n"
        "5. Evaluate research quality and credibility\n\n"
        "Focus areas:\n"
        "- Theoretical foundations\n"
        "- Recent research developments\n"
        "- Methodological approaches\n"
        "- Empirical evidence and findings\n"
        "- Future research directions"
    ),
    expected_output=(
        "Academic research synthesis including:\n"
        "- Literature review summary\n"
        "- Key theoretical frameworks\n"
        "- Major research findings\n"
        "- Methodological insights\n"
        "- Current research gaps\n"
        "- Leading researchers and institutions\n"
        "- Citation network analysis\n"
        "- Future research opportunities"
    ),
    agent=academic_vector_agent,
    output_file="academic_research.md"
)


# Business intelligence search task
business_intelligence_task = Task(
    description=(
        "Analyze business intelligence and market information for: '{business_query}'\n\n"
        "Analysis framework:\n"
        "1. Market landscape and competitive analysis\n"
        "2. Industry trends and developments\n"
        "3. Business metrics and performance indicators\n"
        "4. Strategic insights and opportunities\n"
        "5. Risk assessment and market dynamics\n\n"
        "Information sources:\n"
        "- Industry reports and market analysis\n"
        "- Business case studies\n"
        "- Financial data and metrics\n"
        "- Strategic planning documents\n"
        "- Competitive intelligence"
    ),
    expected_output=(
        "Business intelligence report featuring:\n"
        "- Market overview and size\n"
        "- Competitive landscape analysis\n"
        "- Industry trends and drivers\n"
        "- Key performance metrics\n"
        "- Strategic opportunities\n"
        "- Risk factors and challenges\n"
        "- Actionable business insights\n"
        "- Recommendations for decision-making"
    ),
    agent=business_vector_agent,
    output_file="business_intelligence.md"
)


# Multi-agent collaborative search task
collaborative_search_task = Task(
    description=(
        "Coordinate a comprehensive multi-perspective search on: '{comprehensive_topic}' "
        "leveraging different specialized search approaches.\n\n"
        "Collaboration strategy:\n"
        "1. Technical perspective: Implementation and technical details\n"
        "2. Academic perspective: Research and theoretical foundations\n"
        "3. Business perspective: Market applications and value proposition\n"
        "4. General perspective: Broad context and accessibility\n\n"
        "Integration requirements:\n"
        "- Synthesize findings from multiple viewpoints\n"
        "- Identify common themes and contradictions\n"
        "- Provide balanced and comprehensive coverage\n"
        "- Highlight unique insights from each perspective"
    ),
    expected_output=(
        "Comprehensive multi-perspective analysis including:\n"
        "- Executive summary with key findings\n"
        "- Technical implementation insights\n"
        "- Academic research foundations\n"
        "- Business applications and value\n"
        "- Cross-perspective synthesis\n"
        "- Identified knowledge gaps\n"
        "- Recommendations for different audiences\n"
        "- Action items and next steps"
    ),
    agent=enhanced_vector_agent,
    output_file="comprehensive_analysis.md"
)


# Quick answer search task for chatbot enhancement
quick_answer_task = Task(
    description=(
        "Quickly find the most relevant information to answer the user question: '{user_question}'\n\n"
        "Optimization for speed and relevance:\n"
        "1. Rapid query analysis and intent recognition\n"
        "2. Focused semantic search with tight parameters\n"
        "3. Fast result ranking and filtering\n"
        "4. Concise but comprehensive answer extraction\n"
        "5. Quality verification of top results\n\n"
        "Performance targets:\n"
        "- Response time under 10 seconds\n"
        "- High precision and relevance\n"
        "- Clear and actionable information\n"
        "- Confidence scoring for reliability"
    ),
    expected_output=(
        "Quick answer response containing:\n"
        "- Direct answer to the user question\n"
        "- Supporting evidence and context\n"
        "- Confidence score and reliability assessment\n"
        "- Source information and references\n"
        "- Follow-up questions or related topics\n"
        "- Alternative viewpoints if relevant"
    ),
    agent=enhanced_vector_agent,
    output_file="quick_answer.md"
)