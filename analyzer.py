import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import sqlite3
import logging
from typing import List, Dict, Any
from loguru import logger
import plotly.express as px
import plotly.graph_objs as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.add("quality_benchmark.log", rotation="500 MB")

class QualityBenchmarkingAgent:
    def __init__(self, db_path='quality_benchmarking.db'):
        """
        Initialize the Quality Benchmarking Agent
        
        Args:
            db_path (str): Path to the SQLite database
        """
        # Database setup
        self.db_path = db_path
        self.conn = self._create_connection()
        self._create_tables()
        
        # Quality capabilities and industries
        self.quality_capabilities = [
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
    
    def _create_connection(self):
        """
        Create a database connection
        
        Returns:
            sqlite3.Connection: Database connection
        """
        try:
            conn = sqlite3.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def _create_tables(self):
        """Create necessary database tables"""
        try:
            cursor = self.conn.cursor()
            
            # Companies table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                industry TEXT,
                website TEXT,
                description TEXT
            )
            ''')
            
            # Quality scores table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER,
                capability TEXT,
                score REAL,
                year INTEGER,
                FOREIGN KEY(company_id) REFERENCES companies(id)
            )
            ''')
            
            self.conn.commit()
            logger.info("Database tables created successfully")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            self.conn.rollback()
    
    def web_scrape_companies(self, industry: str) -> List[Dict[str, str]]:
        """
        Web scraping method to find leading companies in an industry
        
        Args:
            industry (str): Industry to search for companies
        
        Returns:
            List[Dict[str, str]]: List of companies with their details
        """
        companies = []
        
        try:
            # Industry-specific hardcoded data (placeholder for actual web scraping)
            industry_companies = {
                'Automotive': [
                    {'name': 'Toyota Motor Corporation', 'website': 'toyota.com', 'description': 'Leading global automotive manufacturer'},
                    {'name': 'Tesla, Inc.', 'website': 'tesla.com', 'description': 'Electric vehicle and clean energy company'},
                    {'name': 'Ford Motor Company', 'website': 'ford.com', 'description': 'Multinational automaker'}
                ],
                'Healthcare': [
                    {'name': 'Johnson & Johnson', 'website': 'jnj.com', 'description': 'Multinational medical devices, pharmaceutical, and consumer packaged goods manufacturer'},
                    {'name': 'Pfizer Inc.', 'website': 'pfizer.com', 'description': 'Multinational pharmaceutical corporation'},
                    {'name': 'Medtronic plc', 'website': 'medtronic.com', 'description': 'Medical technology company'}
                ],
                # Add more industries as needed
            }
            
            companies = industry_companies.get(industry, [])
            
            # Save companies to database
            if companies:
                self._save_companies(companies, industry)
            
            logger.info(f"Found {len(companies)} companies in {industry}")
        except Exception as e:
            logger.error(f"Error scraping companies for {industry}: {e}")
        
        return companies
    
    def _save_companies(self, companies: List[Dict[str, str]], industry: str):
        """
        Save companies to database
        
        Args:
            companies (List[Dict[str, str]]): List of companies to save
            industry (str): Industry of the companies
        """
        try:
            cursor = self.conn.cursor()
            
            for company in companies:
                cursor.execute('''
                INSERT OR REPLACE INTO companies (name, industry, website, description) 
                VALUES (?, ?, ?, ?)
                ''', (company['name'], industry, company['website'], company.get('description', '')))
            
            self.conn.commit()
            logger.info(f"Saved {len(companies)} companies to database")
        except sqlite3.Error as e:
            logger.error(f"Error saving companies: {e}")
            self.conn.rollback()
    
    def generate_quality_score(self) -> float:
        """
        Generate a simulated quality score with more nuanced logic
        
        Returns:
            float: Quality score between 60 and 95
        """
        # More sophisticated scoring logic
        base_score = np.random.uniform(60, 95)
        variability = np.random.normal(0, 5)  # Add some normal distribution variability
        return max(60, min(95, base_score + variability))
    
    def score_company_capabilities(self, company_name: str, industry: str):
        """
        Score a company across different quality capabilities
        
        Args:
            company_name (str): Name of the company
            industry (str): Industry of the company
        """
        try:
            cursor = self.conn.cursor()
            
            # Get company ID
            cursor.execute('SELECT id FROM companies WHERE name = ? AND industry = ?', (company_name, industry))
            company_record = cursor.fetchone()
            
            if not company_record:
                logger.warning(f"Company {company_name} not found in database")
                return None
            
            company_id = company_record[0]
            current_year = pd.Timestamp.now().year
            
            # Generate and save scores for each capability
            for capability in self.quality_capabilities:
                score = self.generate_quality_score()
                
                cursor.execute('''
                INSERT OR REPLACE INTO quality_scores 
                (company_id, capability, score, year)
                VALUES (?, ?, ?, ?)
                ''', (company_id, capability, score, current_year))
            
            self.conn.commit()
            logger.info(f"Scored capabilities for {company_name}")
        except sqlite3.Error as e:
            logger.error(f"Error scoring company capabilities: {e}")
            self.conn.rollback()
    
    def get_company_scores(self, company_name: str, industry: str) -> List[Dict[str, Any]]:
        """
        Retrieve quality scores for a specific company
        
        Args:
            company_name (str): Name of the company
            industry (str): Industry of the company
        
        Returns:
            List[Dict[str, Any]]: List of company scores
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            SELECT c.name, c.industry, qs.capability, qs.score, qs.year
            FROM companies c
            JOIN quality_scores qs ON c.id = qs.company_id
            WHERE c.name = ? AND c.industry = ?
            ''', (company_name, industry))
            
            scores = [
                {
                    'name': row[0], 
                    'industry': row[1], 
                    'capability': row[2], 
                    'score': row[3],
                    'year': row[4]
                } 
                for row in cursor.fetchall()
            ]
            
            return scores
        except sqlite3.Error as e:
            logger.error(f"Error retrieving company scores: {e}")
            return []

def create_capability_chart(scores: List[Dict[str, Any]]) -> go.Figure:
    """
    Create an interactive Plotly chart for capability scores
    
    Args:
        scores (List[Dict[str, Any]]): List of company scores
    
    Returns:
        go.Figure: Plotly bar chart
    """
    if not scores:
        return go.Figure()
    
    df_scores = pd.DataFrame(scores)
    
    fig = px.bar(
        df_scores, 
        x='capability', 
        y='score', 
        title=f'Quality Capabilities Scores for {scores[0]["name"]}',
        labels={'capability': 'Quality Capabilities', 'score': 'Score'},
        color='score',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=600,
        width=800,
        title_x=0.5
    )
    
    return fig

def main():
    """
    Main Streamlit application function
    """
    st.set_page_config(
        page_title="Quality Benchmarking Agent",
        page_icon="🔍",
        layout="wide"
    )
    
    # Initialize the agent
    agent = QualityBenchmarkingAgent()
    
    # Title and description
    st.title('Quality Metrics Benchmarking Agent')
    st.markdown("""
    ### Comprehensive Quality Performance Analysis
    Analyze and benchmark quality metrics across different industries and companies.
    """)
    
    # Sidebar for industry and company selection
    with st.sidebar:
        st.header('Research Parameters')
        
        # Industry Selection
        selected_industry = st.selectbox(
            'Select Industry', 
            agent.industries, 
            index=0
        )
        
        # Discover Companies Button
        if st.button('Discover Leading Companies'):
            try:
                companies = agent.web_scrape_companies(selected_industry)
                if companies:
                    st.success(f'Discovered {len(companies)} companies in {selected_industry}')
                else:
                    st.warning('No companies found. Try another industry.')
            except Exception as e:
                st.error(f"Error discovering companies: {e}")
        
        # Company Selection
        cursor = agent.conn.cursor()
        cursor.execute('SELECT name FROM companies WHERE industry = ?', (selected_industry,))
        companies = [row[0] for row in cursor.fetchall()]
        
        selected_company = st.selectbox(
            'Select Company', 
            companies if companies else ['No companies found']
        )
        
        # Score Company Button
        if st.button('Score Company Capabilities') and selected_company != 'No companies found':
            try:
                agent.score_company_capabilities(selected_company, selected_industry)
                st.success(f'Scored {selected_company}')
            except Exception as e:
                st.error(f"Error scoring company: {e}")
    
    # Main Content Area
    if selected_company and selected_company != 'No companies found':
        st.header(f'Quality Metrics for {selected_company}')
        
        # Retrieve and Display Scores
        try:
            scores = agent.get_company_scores(selected_company, selected_industry)
            
            if scores:
                # Interactive Plotly Chart
                fig = create_capability_chart(scores)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed Scores Table
                st.subheader('Detailed Capability Scores')
                scores_df = pd.DataFrame(scores)
                st.dataframe(
                    scores_df[['capability', 'score']], 
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info('No scores available. Please score the company first.')
        
        except Exception as e:
            st.error(f"Error displaying company scores: {e}")

if __name__ == '__main__':
    main()
