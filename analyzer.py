import streamlit as st
import requests
from bs4 import BeautifulSoup
import urllib.parse
import nltk
from textblob import TextBlob
import pandas as pd
import plotly.express as px

# Attempt to download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    st.warning(f"Could not download NLTK resources: {e}")

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
        Perform web search using Google Search API
        
        Args:
            query (str): Search query
            num_results (int): Number of search results
        
        Returns:
            List[str]: List of search result URLs
        """
        try:
            # Encode the query for URL
            encoded_query = urllib.parse.quote(query)
            
            # Use Google Search URL
            search_url = f"https://www.google.com/search?q={encoded_query}"
            
            # Headers to mimic browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Perform the search request
            response = requests.get(search_url, headers=headers)
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract search result links
            results = []
            for link in soup.find_all('div', class_='yuRUbf'):
                a_tag = link.find('a')
                if a_tag and len(results) < num_results:
                    results.append(a_tag['href'])
            
            return results
        except Exception as e:
            st.error(f"Web search error: {e}")
            return []

    def extract_content(self, url):
        """
        Extract text content from a URL
        
        Args:
            url (str): URL to extract content from
        
        Returns:
            Dict: Extracted content details
        """
        try:
            # Fetch webpage content
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            
            # Basic text analysis
            analysis = self.analyze_text(text)
            
            return {
                'url': url,
                'text': text[:1000],  # Limit text length
                'analysis': analysis
            }
        except Exception as e:
            return {
                'url': url,
                'text': f"Error extracting content: {e}",
                'analysis': {}
            }

    def analyze_text(self, text):
        """
        Perform basic text analysis
        
        Args:
            text (str): Text to analyze
        
        Returns:
            Dict: Text analysis results
        """
        # Sentiment Analysis
        blob = TextBlob(text)
        
        try:
            return {
                'sentiment': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'word_count': len(text.split()),
                'sentences': len(nltk.sent_tokenize(text))
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
    
    # Search button
    if st.button("Perform Research"):
        # Construct search query
        search_query = f"{selected_industry} {selected_capability} best practices"
        
        # Perform web search
        st.subheader(f"Research Results for {selected_industry} - {selected_capability}")
        
        # Perform search
        search_results = agent.web_search(search_query)
        
        # Check if results were found
        if not search_results:
            st.warning("No search results found. Please try a different query or check your internet connection.")
            st.stop()
        
        # Analyze and display results
        research_data = []
        
        # Progress bar
        progress_bar = st.progress(0)
        
        for i, url in enumerate(search_results):
            # Update progress
            progress_bar.progress((i + 1) / len(search_results))
            
            # Extract content
            content = agent.extract_content(url)
            
            if content and content['analysis']:
                research_data.append({
                    'url': content['url'],
                    'sentiment': content['analysis'].get('sentiment', 0),
                    'word_count': content['analysis'].get('word_count', 0),
                    'sentences': content['analysis'].get('sentences', 0)
                })
        
        # Clear progress bar
        progress_bar.empty()
        
        # Create DataFrame
        if research_data:
            df_results = pd.DataFrame(research_data)
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment Distribution
                fig_sentiment = px.histogram(
                    df_results, 
                    x='sentiment', 
                    title='Sentiment Distribution',
                    labels={'sentiment': 'Sentiment Score'}
                )
                st.plotly_chart(fig_sentiment)
            
            with col2:
                # Word Count vs Sentiment Scatter
                fig_scatter = px.scatter(
                    df_results, 
                    x='word_count', 
                    y='sentiment', 
                    hover_data=['url'],
                    title='Word Count vs Sentiment',
                    labels={'word_count': 'Document Length', 'sentiment': 'Sentiment Score'}
                )
                st.plotly_chart(fig_scatter)
            
            # Detailed Results Table
            st.subheader("Detailed Research Findings")
            st.dataframe(df_results)
        else:
            st.warning("No analyzable content found in search results.")
    
    # About section
    st.sidebar.header("About")
    st.sidebar.info("""
    Automated Quality Benchmarking Tool
    - Advanced web research
    - Basic text analysis
    - Comprehensive industry insights
    """)

if __name__ == '__main__':
    main()
