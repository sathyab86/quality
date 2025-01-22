import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Any
from loguru import logger
import plotly.express as px
import plotly.graph_objs as go

# SQLAlchemy Imports
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.add("quality_benchmark.log", rotation="500 MB")

# SQLAlchemy Base and Session
Base = declarative_base()

class Company(Base):
    """SQLAlchemy model for companies"""
    __tablename__ = 'companies'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    industry = Column(String, nullable=False)
    website = Column(String)
    description = Column(String)
    
    # Relationship to quality scores
    quality_scores = relationship("QualityScore", back_populates="company")

class QualityScore(Base):
    """SQLAlchemy model for quality scores"""
    __tablename__ = 'quality_scores'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey('companies.id'))
    capability = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    year = Column(Integer, nullable=False)
    
    # Relationship back to company
    company = relationship("Company", back_populates="quality_scores")

class QualityBenchmarkingAgent:
    def __init__(self, db_path='sqlite:///quality_benchmarking.db'):
        """
        Initialize the Quality Benchmarking Agent
        
        Args:
            db_path (str): SQLAlchemy database connection string
        """
        try:
            # Create engine and create tables
            self.engine = create_engine(db_path)
            Base.metadata.create_all(self.engine)
            
            # Create session factory
            self.SessionLocal = sessionmaker(bind=self.engine)
            
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
            
            logger.info("Database initialized successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
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
        session = self.SessionLocal()
        try:
            for company_data in companies:
                # Check if company already exists
                existing_company = session.query(Company).filter_by(name=company_data['name']).first()
                
                if not existing_company:
                    # Create new company
                    new_company = Company(
                        name=company_data['name'],
                        industry=industry,
                        website=company_data.get('website', ''),
                        description=company_data.get('description', '')
                    )
                    session.add(new_company)
            
            session.commit()
            logger.info(f"Saved {len(companies)} companies to database")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error saving companies: {e}")
        finally:
            session.close()
    
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
        session = self.SessionLocal()
        try:
            # Find the company
            company = session.query(Company).filter_by(name=company_name, industry=industry).first()
            
            if not company:
                logger.warning(f"Company {company_name} not found in database")
                return None
            
            current_year = pd.Timestamp.now().year
            
            # Generate and save scores for each capability
            for capability in self.quality_capabilities:
                score = self.generate_quality_score()
                
                # Create new quality score
                quality_score = QualityScore(
                    company_id=company.id,
                    capability=capability,
                    score=score,
                    year=current_year
                )
                session.add(quality_score)
            
            session.commit()
            logger.info(f"Scored capabilities for {company_name}")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error scoring company capabilities: {e}")
        finally:
            session.close()
    
    def get_company_scores(self, company_name: str, industry: str) -> List[Dict[str, Any]]:
        """
        Retrieve quality scores for a specific company
        
        Args:
            company_name (str): Name of the company
            industry (str): Industry of the company
        
        Returns:
            List[Dict[str, Any]]: List of company scores
        """
        session = self.SessionLocal()
        try:
            # Find the company and its scores
            company = session.query(Company).filter_by(name=company_name, industry=industry).first()
            
            if not company:
                logger.warning(f"Company {company_name} not found")
                return []
            
            # Convert SQLAlchemy objects to dictionaries
            scores = [
                {
                    'name': company.name, 
                    'industry': company.industry, 
                    'capability': score.capability, 
                    'score': score.score,
                    'year': score.year
                } 
                for score in company.quality_scores
            ]
            
            return scores
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving company scores: {e}")
            return []
        finally:
            session.close()

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
        session = agent.SessionLocal()
        companies = session.query(Company).filter_by(industry=selected_industry).all()
        session.close()
        
        company_names = [company.name for company in companies]
        
        selected_company = st.selectbox(
            'Select Company', 
            company_names if company_names else ['No companies found']
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
