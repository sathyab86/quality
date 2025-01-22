import streamlit as st
import subprocess
import sys
import os
import json
import logging
from typing import List, Dict, Any

# NLP and ML Libraries
import spacy
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
from textblob import TextBlob

# Web Scraping
import requests
from bs4 import BeautifulSoup
from googlesearch import search

# Logging Configuration
logging.basicConfig(level=logging.INFO)

class NLPResourceManager:
    """
    Manages NLP resource checking, downloading, and verification
    """
    @staticmethod
    def check_spacy():
        """Check if spaCy model is installed and loadable"""
        try:
            spacy.load('en_core_web_sm')
            return True
        except (ImportError, OSError):
            return False

    @staticmethod
    def check_nltk_punkt():
        """Check if NLTK Punkt tokenizer is installed"""
        try:
            nltk.data.find('tokenizers/punkt')
            return True
        except LookupError:
            return False

    @staticmethod
    def check_nltk_wordnet():
        """Check if NLTK WordNet is installed"""
        try:
            nltk.data.find('corpora/wordnet')
            return True
        except LookupError:
            return False

    @classmethod
    def get_nlp_readiness(cls):
        """
        Comprehensive check for NLP resource readiness
        
        Returns:
            bool: True if all resources are available, False otherwise
        """
        return all([
            cls.check_spacy(),
            cls.check_nltk_punkt(),
            cls.check_nltk_wordnet()
        ])

    @staticmethod
    def download_nlp_resources(resources):
        """
        Download selected NLP resources
        
        Args:
            resources (List[str]): List of resources to download
        
        Returns:
            Dict[str, bool]: Download status for each resource
        """
        download_status = {}
        
        for resource in resources:
            try:
                if resource == "spaCy English Model":
                    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                    download_status[resource] = True
                elif resource == "NLTK Punkt Tokenizer":
                    subprocess.run([sys.executable, "-m", "nltk.downloader", "punkt"], check=True)
                    download_status[resource] = True
                elif resource == "NLTK WordNet":
                    subprocess.run([sys.executable, "-m", "nltk.downloader", "wordnet"], check=True)
                    download_status[resource] = True
            except subprocess.CalledProcessError:
                download_status[resource] = False
        
        return download_status

class QualityBenchmarkingAgent:
    """
    Main application for Quality Benchmarking
    """
    def __init__(self):
        """
        Initialize the Quality Benchmarking Agent
        """
        # Core Configuration
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
        
        # Initialize NLP components
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            st.warning("SpaCy model not loaded. Please download resources.")

    def web_search(self, query: str, num_results: int = 10) -> List[str]:
        """
        Perform web search for a given query
        
        Args:
            query (str): Search query
            num_results (int): Number of search results
        
        Returns:
            List[str]: List of URLs
        """
        try:
            return list(search(query, num_results=num_results))
        except Exception as e:
            st.error(f"Web search error: {e}")
            return []

    def extract_web_content(self, url: str) -> str:
        """
        Extract text content from a given URL
        
        Args:
            url (str): URL to extract content from
        
        Returns:
            str: Extracted text content
        """
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.get_text()
        except Exception as e:
            st.error(f"Content extraction error for {url}: {e}")
            return ""

    def analyze_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze text content
        
        Args:
            text (str): Text to analyze
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Basic NLP processing
        doc = self.nlp(text)
        
        # Sentiment Analysis
        blob = TextBlob(text)
        
        return {
            'sentiment': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'text_length': len(text.split()),
        }

def render_nlp_resource_management():
    """
    Streamlit page for NLP resource management
    """
    st.header("🧠 NLP Resource Management")
    
    # Resource configuration
    resources = [
        "spaCy English Model",
        "NLTK Punkt Tokenizer", 
        "NLTK WordNet"
    ]
    
    # Resource status display
    st.subheader("Resource Status")
    status_cols = st.columns(3)
    
    resource_status = {
        "spaCy English Model": NLPResourceManager.check_spacy(),
        "NLTK Punkt Tokenizer": NLPResourceManager.check_nltk_punkt(),
        "NLTK WordNet": NLPResourceManager.check_nltk_wordnet()
    }
    
    for i, (resource, status) in enumerate(resource_status.items()):
        with status_cols[i]:
            if status:
                st.success(f"{resource} ✅")
            else:
                st.warning(f"{resource} ⚠️")
    
    # Resource download section
    st.subheader("Download Resources")
    selected_resources = st.multiselect(
        "Select Resources to Download", 
        resources,
        default=[r for r, status in resource_status.items() if not status]
    )
    
    # Download button
    if st.button("Download Selected Resources"):
        if selected_resources:
            with st.spinner("Downloading resources..."):
                download_status = NLPResourceManager.download_nlp_resources(selected_resources)
            
            # Display download results
            for resource, success in download_status.items():
                if success:
                    st.success(f"Successfully downloaded {resource}")
                else:
                    st.error(f"Failed to download {resource}")
            
            # Rerun to refresh status
            st.experimental_rerun()
        else:
            st.warning("No resources selected")

def main():
    """
    Main Streamlit application
    """
    st.set_page_config(
        page_title="Quality Benchmarking Agent", 
        page_icon="📊", 
        layout="wide"
    )
    
    # Check NLP resource readiness
    if not NLPResourceManager.get_nlp_readiness():
        st.warning("⚠️ Some NLP resources are missing")
        render_nlp_resource_management()
        return
    
    # Main application navigation
    menu = ["Research", "NLP Resources", "About"]
    choice = st.sidebar.selectbox("Navigation", menu)
    
    # Initialize agent
    agent = QualityBenchmarkingAgent()
    
    if choice == "Research":
        st.title("Quality Metrics Research")
        
        # Industry and Capability Selection
        col1, col2 = st.columns(2)
        with col1:
            selected_industry = st.selectbox("Select Industry", agent.industries)
        with col2:
            selected_capability = st.selectbox("Select Capability", agent.capabilities)
        
        # Search button
        if st.button("Perform Research"):
            # Construct search query
            search_query = f"{selected_industry} {selected_capability} best practices"
            
            # Perform web search
            urls = agent.web_search(search_query)
            
            # Display search results
            st.subheader("Search Results")
            research_results = []
            
            for url in urls[:5]:  # Limit to 5 results
                content = agent.extract_web_content(url)
                analysis = agent.analyze_content(content)
                
                research_results.append({
                    'url': url,
                    'sentiment': analysis['sentiment'],
                    'text_length': analysis['text_length'],
                    'entities': analysis['entities']
                })
            
            # Create DataFrame for results
            results_df = pd.DataFrame(research_results)
            
            # Visualization
            if not results_df.empty:
                fig = px.scatter(
                    results_df, 
                    x='text_length', 
                    y='sentiment', 
                    hover_data=['url'],
                    title="Research Results Analysis"
                )
                st.plotly_chart(fig)
                
                # Detailed results table
                st.dataframe(results_df)
    
    elif choice == "NLP Resources":
        render_nlp_resource_management()
    
    elif choice == "About":
        st.title("About Quality Benchmarking Agent")
        st.markdown("""
        ## 🔍 Automated Quality Research Platform
        
        ### Key Features:
        - Industry-specific quality research
        - Advanced NLP-powered analysis
        - Comprehensive capability assessment
        
        ### Supported Capabilities:
        - Quality Management System
        - Statistical Process Control
        - Manufacturing Quality
        - Quality Risk Monitoring
        - And more...
        """)

if __name__ == '__main__':
    main()
