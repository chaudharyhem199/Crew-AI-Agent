from crewai_tools import YoutubeChannelSearchTool
from pgvector_tool import pgvector_search_tool

# Initialize the tool with a specific Youtube channel handle to target your search
yt_tool = YoutubeChannelSearchTool(youtube_channel_handle='@krishnaik06')

# Vector search tool for enhanced information retrieval
vector_search_tool = pgvector_search_tool

# List of all available tools for easy import
all_tools = [yt_tool, vector_search_tool]

