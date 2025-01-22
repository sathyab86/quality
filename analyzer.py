import streamlit as st
import os
import requests
import nltk
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
import pandas as pd
import plotly.express as px

# Comprehensive NLTK Resource Download Function
def download_nltk_resources():
    """
    Download necessary NLTK resources with error handling
    """
    # List of resources to download
    resources = [
        'punkt',
        'wordnet',
        'averaged_perceptron_tagger',
    ]
    
    for resource in resources:
        try:
            # Attempt to download the resource
            nltk.download(resource, quiet=True)
        except Exception as e:
            st.warning(f"Could not download NLTK resource {resource}: {e}")

# Attempt to download resources
download_nltk_resources()

class QualityBenchmarkingAgent:
    def __init__(self):
        """
        Initialize the Quality Benchmarking Agent
        """
        # Define capabilities and industries
        self.capabilities = [
            'QMS (Quality Management System)',
            'SPC (Statistical Process Control)', 
            'Manufacturing Quality', 
            'Quality Risk Monitoring', 
            'Supplier Quality', 
            'Analytics & Insights', 
            'Culture of Quality', 
            'Design for Manufacturing'
        ]
        
        self.industries = [
            'Automotive', 
            'Healthcare', 
            'Consumer Packaged Goods (CPG)', 
            'Steel', 
            'Oil and Gas', 
            'Retail', 
            'Life Sciences'
        ]

    def web_search(self, query, num_results=5):
        """
        Perform web search using SerpAPI
        
        Args:
            query (str): Search query
            num_results (int): Number of search results
        
        Returns:
            List[Dict]: List of search results
        """
        # Get API key from environment variable or Streamlit secrets
        api_key = st.secrets.get('SERPAPI_KEY') if hasattr(st.secrets, 'get') else os.getenv('SERPAPI_KEY')
        
        if not api_key:
            st.error("""
            SerpAPI key is required for web search. 
            Please set up your API key:
            1. Sign up at https://serpapi.com
            2. Add key to environment variables or Streamlit secrets
            3. Set SERPAPI_KEY=your_api_key
            """)
            return []

        # Construct search parameters
        params = {
            'engine': 'google',
            'q': query,
            'api_key': api_key,
            'num': num_results
        }

        try:
            # Perform search request
            response = requests.get('https://serpapi.com/search', params=params)
            
            # Check if request was successful
            if response.status_code != 200:
                st.error(f"Search API error: {response.status_code}")
                return []

            # Parse search results
            data = response.json()
            
            # Extract organic results
            organic_results = data.get('organic_results', [])
            
            return [
                {
                    'title': result.get('title', ''),
                    'link': result.get('link', ''),
                    'snippet': result.get('snippet', '')
                }
                for result in organic_results
            ]

        except Exception as e:
            st.error(f"Web search error: {e}")
            return []

    def analyze_text(self, text):
        """
        Perform basic text analysis
        
        Args:
            text (str): Text to analyze
        
        Returns:
            Dict: Text analysis results
        """
        try:
            # Ensure text is a string
            text = str(text)
            
            # Sentiment Analysis
            blob = TextBlob(text)
            
            # Sentence tokenization with fallback
            try:
                sentences = sent_tokenize(text)
            except Exception:
                # Fallback to simple splitting if tokenization fails
                sentences = text.split('.')
            
            return {
                'sentiment': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'word_count': len(text.split()),
                'sentences': len(sentences)
            }
        except Exception as e:
            st.warning(f"Text analysis error: {e}")
            return {}

def main():
    """
    Main Streamlit application
    """
    st.set_page_config(
        page_title="Quality Benchmarking Agent", 
        page_icon="📊", 
        layout="wide"
    )
    
    # Initialize agent
    agent = QualityBenchmarkingAgent()
    
    # Main title
    st.title("Quality Metrics Benchmarking Agent")
    
    # Sidebar configuration
    st.sidebar.header("Research Parameters")
    
    # Industry and Capability Selection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_industry = st.selectbox(
            "Select Industry", 
            agent.industries
        )
    
    with col2:
        selected_capability = st.selectbox(
            "Select Capability", 
            agent.capabilities
        )
    
    # API Key Input
    st.sidebar.header("Search Configuration")
    serpapi_key = st.sidebar.text_input(
        "SerpAPI Key", 
        type="password", 
        help="Get your free API key at serpapi.com"
    )

    # Search button
    if st.button("Perform Research"):
        # Store API key if provided
        if serpapi_key:
            os.environ['SERPAPI_KEY'] = serpapi_key
        
        # Construct search query
        search_query = f"{selected_industry} {selected_capability} best practices"
        
        # Perform web search
        st.subheader(f"Research Results for {selected_industry} - {selected_capability}")
        
        # Perform search
        search_results = agent.web_search(search_query)
        
        # Check if results were found
        if not search_results:
            st.warning("No search results found. Please check your API key and internet connection.")
            st.stop()
        
        # Create DataFrame
        df_results = pd.DataFrame(search_results)
        
        # Display search results
        st.subheader("Search Findings")
        
        # Display results in an expandable format
        for _, result in df_results.iterrows():
            with st.expander(result['title']):
                st.write(f"**URL:** {result['link']}")
                st.write(f"**Snippet:** {result['snippet']}")
                
                # Optional: Basic text analysis
                analysis = agent.analyze_text(result['snippet'])
                if analysis:
                    st.write("**Analysis:**")
                    st.json(analysis)
        
        # Additional insights
        st.subheader("Insights Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution of snippets
            sentiment_scores = [agent.analyze_text(snippet)['sentiment'] 
                                for snippet in df_results['snippet']]
            
            # Sentiment Distribution
            fig_sentiment = px.histogram(
                x=sentiment_scores, 
                title='Sentiment Distribution',
                labels={'x': 'Sentiment Score'}
            )
            st.plotly_chart(fig_sentiment)
        
        with col2:
            # Display some basic statistics
            st.metric("Total Results", len(df_results))
            avg_sentiment = pd.Series(sentiment_scores).mean()
            st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
    
    # About section
    st.sidebar.header("About")
    st.sidebar.info("""
    Automated Quality Benchmarking Tool
    - Powered by SerpAPI
    - Industry-specific insights
    - Advanced search capabilities
    """)

    # NLTK Resource Check
    st.sidebar.header("NLTK Resources")
    try:
        # Check punkt resource
        nltk.data.find('tokenizers/punkt')
        st.sidebar.success("NLTK Punkt Tokenizer ✅")
    except LookupError:
        st.sidebar.warning("NLTK Punkt Tokenizer ⚠️")
        if st.sidebar.button("Download NLTK Resources"):
            download_nltk_resources()
            st.experimental_rerun()

if __name__ == '__main__':
    main()
