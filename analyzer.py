import streamlit as st
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import spacy
import nltk
from textblob import TextBlob
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    st.warning("Could not download NLTK resources automatically")

class QualityBenchmarkingAgent:
    def __init__(self):
        """
        Initialize the Quality Benchmarking Agent
        """
        # Load SpaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            st.error("SpaCy model not found. Please download 'en_core_web_sm'.")
            self.nlp = None

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
        Perform web search and return URLs
        
        Args:
            query (str): Search query
            num_results (int): Number of search results
        
        Returns:
            List[str]: List of search result URLs
        """
        try:
            return list(search(query, num_results=num_results))
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
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text
            text = soup.get_text()
            
            # NLP Analysis
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
        Perform NLP analysis on text
        
        Args:
            text (str): Text to analyze
        
        Returns:
            Dict: NLP analysis results
        """
        if not self.nlp:
            return {}

        # Perform NLP analysis
        doc = self.nlp(text)
        
        # Sentiment Analysis
        blob = TextBlob(text)
        
        return {
            'sentiment': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'entities': [
                {'text': ent.text, 'label': ent.label_} 
                for ent in doc.ents
            ],
            'word_count': len(text.split()),
            'key_phrases': self.extract_key_phrases(doc)
        }

    def extract_key_phrases(self, doc, top_n=5):
        """
        Extract key phrases from document
        
        Args:
            doc (spacy.tokens.Doc): Processed document
            top_n (int): Number of top phrases to return
        
        Returns:
            List[str]: Top key phrases
        """
        # Extract noun chunks as potential key phrases
        phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Remove duplicates while preserving order
        unique_phrases = []
        for phrase in phrases:
            if phrase not in unique_phrases:
                unique_phrases.append(phrase)
        
        return unique_phrases[:top_n]

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
        
        # Analyze and display results
        research_data = []
        
        for url in search_results:
            content = agent.extract_content(url)
            
            if content:
                research_data.append({
                    'url': content['url'],
                    'sentiment': content['analysis'].get('sentiment', 0),
                    'word_count': content['analysis'].get('word_count', 0),
                    'key_phrases': ', '.join(content['analysis'].get('key_phrases', []))
                })
        
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
                    hover_data=['url', 'key_phrases'],
                    title='Word Count vs Sentiment',
                    labels={'word_count': 'Document Length', 'sentiment': 'Sentiment Score'}
                )
                st.plotly_chart(fig_scatter)
            
            # Detailed Results Table
            st.subheader("Detailed Research Findings")
            st.dataframe(df_results)
        else:
            st.warning("No research results found.")
    
    # About section
    st.sidebar.header("About")
    st.sidebar.info("""
    Automated Quality Benchmarking Tool
    - Advanced web research
    - NLP-powered insights
    - Comprehensive industry analysis
    """)

if __name__ == '__main__':
    main()
