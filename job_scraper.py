import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import concurrent.futures
from urllib.parse import quote
import re
from fake_useragent import UserAgent
import json
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import plotly.express as px
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from huggingface_hub import login

login(token='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')

class JobScraper:
    def __init__(self, job_role, max_jobs=20):
        self.job_role = job_role
        self.max_jobs = max_jobs
        self.results = []
        self.headers = self._get_random_headers()
        self.proxies = self._get_proxies()
        self.progress_callback = None
        
    def _get_random_headers(self):
        """Generate random user-agent and headers to avoid detection"""
        ua = UserAgent()
        headers = {
            'User-Agent': ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
        return headers
    
    def _get_proxies(self):
        """Get a list of free proxies (for demonstration - in production use paid proxies)"""
        # This is a simplified version - in production you would use a paid proxy service
        proxies = []
        try:
            response = requests.get('https://free-proxy-list.net/')
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', id='proxylisttable')
            for row in table.tbody.find_all('tr'):
                cols = row.find_all('td')
                if cols[4].text == 'elite proxy' and cols[6].text == 'yes':
                    proxy = f"https://{cols[0].text}:{cols[1].text}"
                    proxies.append({'https': proxy})
        except Exception as e:
            print(f"Error fetching proxies: {e}")
        
        # Fallback to no proxy if none found
        if not proxies:
            proxies = [None]
        return proxies
    
    def _random_delay(self):
        """Add random delay between requests to mimic human behavior"""
        delay = random.uniform(3, 7)
        time.sleep(delay)
    
    def _make_request(self, url, params=None):
        """Make a request with random proxy and headers"""
        self._random_delay()
        proxy = random.choice(self.proxies)
        headers = self._get_random_headers()
        
        try:
            response = requests.get(
                url, 
                headers=headers, 
                proxies=proxy, 
                params=params,
                timeout=15
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
    
    def _clean_text(self, text):
        """Clean extracted text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def _extract_skills(self, description):
        """Extract skills from job description based on common patterns"""
        # Using Llama model if available, otherwise fallback to keyword extraction
        if hasattr(self, 'skill_extractor') and self.skill_extractor:
            return self.skill_extractor.extract_skills(description)
        
        # Fallback to keyword-based extraction
        common_skills = [
            'python', 'java', 'javascript', 'html', 'css', 'react', 'angular', 
            'node.js', 'sql', 'nosql', 'mongodb', 'aws', 'azure', 'docker', 
            'kubernetes', 'devops', 'machine learning', 'ai', 'data science',
            'hadoop', 'spark', 'tableau', 'power bi', 'excel', 'communication',
            'leadership', 'project management', 'agile', 'scrum', 'jira',
            'git', 'github', 'jenkins', 'ci/cd', 'testing', 'qa', 'mobile',
            'android', 'ios', 'swift', 'kotlin', 'flutter', 'react native'
        ]
        
        found_skills = []
        desc_lower = description.lower()
        
        for skill in common_skills:
            if skill in desc_lower:
                found_skills.append(skill)
                
        return found_skills
    
    def set_skill_extractor(self, extractor):
        """Set the skill extractor to use"""
        self.skill_extractor = extractor
    
    def _parse_naukri(self, max_jobs=None):
        """Scrape job data from Naukri.com"""
        base_url = "https://www.naukri.com/"
        encoded_job = quote(self.job_role)
        
        params = {
            'noOfResults': min(max_jobs or self.max_jobs, 20),
            'urlType': 'search_by_keyword',
            'searchType': 'adv',
            'keyword': self.job_role,
            'k': self.job_role,
            'seoKey': encoded_job,
            'src': 'jobsearchDesk',
            'latLong': ''
        }
        
        response = self._make_request(base_url, params)
        if not response:
            return []
        
        try:
            data = response.json()
            jobs = data.get('jobDetails', [])
            
            results = []
            for job in jobs[:max_jobs or self.max_jobs]:
                job_title = job.get('title', '')
                company = job.get('companyName', '')
                job_description = job.get('jobDescription', '')
                location = job.get('placeholders', [{}])[0].get('label', '')
                experience = job.get('placeholders', [{}])[1].get('label', '')
                salary = job.get('placeholders', [{}])[2].get('label', '')
                
                skills = self._extract_skills(job_description)
                
                results.append({
                    'portal': 'Naukri',
                    'title': job_title,
                    'company': company,
                    'location': location,
                    'experience': experience,
                    'salary': salary,
                    'skills': ', '.join(skills),
                    'description': self._clean_text(job_description),
                    'url': job.get('jdURL', '')
                })
                
            return results
        except Exception as e:
            print(f"Error parsing Naukri data: {e}")
            return []
    
    def _parse_indeed(self, max_jobs=None):
        """Scrape job data from Indeed"""
        base_url = "https://in.indeed.com/jobs"
        encoded_job = quote(self.job_role)
        
        params = {
            'q': self.job_role,
            'l': 'India',
            'fromage': '14',
            'sort': 'date'
        }
        
        response = self._make_request(base_url, params)
        if not response:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        job_cards = soup.find_all('div', class_=re.compile(r'job_.*'))
        
        results = []
        for job in job_cards[:max_jobs or self.max_jobs]:
            try:
                title_elem = job.find('h2', class_=re.compile(r'jobTitle'))
                title = title_elem.text.strip() if title_elem else ''
                
                company_elem = job.find('span', class_=re.compile(r'companyName'))
                company = company_elem.text.strip() if company_elem else ''
                
                location_elem = job.find('div', class_=re.compile(r'companyLocation'))
                location = location_elem.text.strip() if location_elem else ''
                
                # Get job URL
                job_url = ''
                if title_elem and title_elem.find('a'):
                    job_url = 'https://in.indeed.com' + title_elem.find('a').get('href', '')
                
                # Get full description by visiting the job page
                description = ''
                if job_url:
                    job_response = self._make_request(job_url)
                    if job_response:
                        job_soup = BeautifulSoup(job_response.text, 'html.parser')
                        desc_div = job_soup.find('div', id=re.compile(r'jobDescriptionText'))
                        if desc_div:
                            description = desc_div.text.strip()
                
                skills = self._extract_skills(description)
                
                salary_elem = job.find('div', class_=re.compile(r'salary-snippet'))
                salary = salary_elem.text.strip() if salary_elem else 'Not mentioned'
                
                results.append({
                    'portal': 'Indeed',
                    'title': title,
                    'company': company,
                    'location': location,
                    'experience': 'Not specified',
                    'salary': salary,
                    'skills': ', '.join(skills),
                    'description': self._clean_text(description),
                    'url': job_url
                })
            except Exception as e:
                print(f"Error parsing Indeed job: {e}")
                continue
                
        return results
    
    def _parse_linkedin(self, max_jobs=None):
        """Scrape job data from LinkedIn"""
        base_url = "https://www.linkedin.com/jobs/search"
        encoded_job = quote(self.job_role)
        
        params = {
            'keywords': self.job_role,
            'location': 'India',
            'geoId': '102713980',
            'f_TPR': 'r2592000',
            'position': '1',
            'pageNum': '0'
        }
        
        response = self._make_request(base_url, params)
        if not response:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        job_cards = soup.find_all('div', class_='base-card')
        
        results = []
        for job in job_cards[:max_jobs or self.max_jobs]:
            try:
                title_elem = job.find('h3', class_='base-search-card__title')
                title = title_elem.text.strip() if title_elem else ''
                
                company_elem = job.find('h4', class_='base-search-card__subtitle')
                company = company_elem.text.strip() if company_elem else ''
                
                location_elem = job.find('span', class_='job-search-card__location')
                location = location_elem.text.strip() if location_elem else ''
                
                job_url = job.find('a', class_='base-card__full-link').get('href') if job.find('a', class_='base-card__full-link') else ''
                
                # Get full description by visiting the job page
                description = ''
                if job_url:
                    job_response = self._make_request(job_url)
                    if job_response:
                        job_soup = BeautifulSoup(job_response.text, 'html.parser')
                        desc_div = job_soup.find('div', class_='show-more-less-html__markup')
                        if desc_div:
                            description = desc_div.text.strip()
                
                skills = self._extract_skills(description)
                
                results.append({
                    'portal': 'LinkedIn',
                    'title': title,
                    'company': company,
                    'location': location,
                    'experience': 'Not specified',
                    'salary': 'Not mentioned',
                    'skills': ', '.join(skills),
                    'description': self._clean_text(description),
                    'url': job_url
                })
            except Exception as e:
                print(f"Error parsing LinkedIn job: {e}")
                continue
                
        return results
    
    def _parse_monster(self, max_jobs=None):
        """Scrape job data from Monster India"""
        base_url = "https://www.monsterindia.com/srp/results"
        encoded_job = quote(self.job_role)
        
        params = {
            'query': self.job_role,
            'locations': 'India'
        }
        
        response = self._make_request(base_url, params)
        if not response:
            return []
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            job_cards = soup.find_all('div', class_='card-body')
            
            results = []
            for job in job_cards[:max_jobs or self.max_jobs]:
                try:
                    title_elem = job.find('h3', class_='medium')
                    title = title_elem.text.strip() if title_elem else ''
                    
                    company_elem = job.find('span', class_='company-name')
                    company = company_elem.text.strip() if company_elem else ''
                    
                    location_elem = job.find('span', class_='location-name')
                    location = location_elem.text.strip() if location_elem else ''
                    
                    exp_elem = job.find('span', class_='exp-name')
                    experience = exp_elem.text.strip() if exp_elem else 'Not specified'
                    
                    # Get job URL
                    job_url = ''
                    if title_elem and title_elem.find('a'):
                        job_url = title_elem.find('a').get('href', '')
                        if not job_url.startswith('http'):
                            job_url = 'https://www.monsterindia.com' + job_url
                    
                    # Get full description by visiting the job page
                    description = ''
                    if job_url:
                        job_response = self._make_request(job_url)
                        if job_response:
                            job_soup = BeautifulSoup(job_response.text, 'html.parser')
                            desc_div = job_soup.find('div', class_='job-description')
                            if desc_div:
                                description = desc_div.text.strip()
                    
                    skills = self._extract_skills(description)
                    
                    results.append({
                        'portal': 'Monster',
                        'title': title,
                        'company': company,
                        'location': location,
                        'experience': experience,
                        'salary': 'Not mentioned',
                        'skills': ', '.join(skills),
                        'description': self._clean_text(description),
                        'url': job_url
                    })
                except Exception as e:
                    print(f"Error parsing Monster job: {e}")
                    continue
                    
            return results
        except Exception as e:
            print(f"Error parsing Monster data: {e}")
            return []
    
    def _parse_timesjobs(self, max_jobs=None):
        """Scrape job data from TimesJobs"""
        base_url = "https://www.timesjobs.com/candidate/job-search.html"
        encoded_job = quote(self.job_role)
        
        params = {
            'searchType': 'personalizedSearch',
            'from': 'submit',
            'txtKeywords': self.job_role,
            'txtLocation': 'India'
        }
        
        response = self._make_request(base_url, params)
        if not response:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        job_cards = soup.find_all('li', class_='clearfix job-bx wht-shd-bx')
        
        results = []
        for job in job_cards[:max_jobs or self.max_jobs]:
            try:
                title_elem = job.find('h2')
                title = title_elem.text.strip() if title_elem else ''
                
                company_elem = job.find('h3', class_='joblist-comp-name')
                company = company_elem.text.strip() if company_elem else ''
                
                location_elem = job.find('ul', class_='top-jd-dtl').find('li')
                location = location_elem.text.strip() if location_elem else ''
                
                exp_elem = job.find('ul', class_='top-jd-dtl').find_all('li')[1]
                experience = exp_elem.text.strip() if exp_elem else 'Not specified'
                
                # Get job URL
                job_url = ''
                if title_elem and title_elem.find('a'):
                    job_url = title_elem.find('a').get('href', '')
                
                # Get full description by visiting the job page
                description = ''
                if job_url:
                    job_response = self._make_request(job_url)
                    if job_response:
                        job_soup = BeautifulSoup(job_response.text, 'html.parser')
                        desc_div = job_soup.find('div', class_='jd-desc')
                        if desc_div:
                            description = desc_div.text.strip()
                
                skills_elem = job.find('span', class_='srp-skills')
                skills_text = skills_elem.text.strip() if skills_elem else ''
                skills = [s.strip() for s in skills_text.split(',') if s.strip()]
                
                results.append({
                    'portal': 'TimesJobs',
                    'title': title,
                    'company': company,
                    'location': location,
                    'experience': experience,
                    'salary': 'Not mentioned',
                    'skills': ', '.join(skills),
                    'description': self._clean_text(description),
                    'url': job_url
                })
            except Exception as e:
                print(f"Error parsing TimesJobs job: {e}")
                continue
                
        return results
    
    def set_progress_callback(self, callback):
        """Set a callback function to report progress"""
        self.progress_callback = callback
    
    def scrape_jobs(self):
        """Scrape jobs from all portals in parallel"""
        portals = {
            'naukri': self._parse_naukri,
            'indeed': self._parse_indeed,
            'linkedin': self._parse_linkedin,
            'monster': self._parse_monster,
            'timesjobs': self._parse_timesjobs
        }
        
        max_jobs_per_portal = max(2, self.max_jobs // len(portals))
        
        results = []
        portal_count = len(portals)
        completed = 0
        
        # Update progress
        if self.progress_callback:
            self.progress_callback(0, f"Starting job scraping from {portal_count} portals")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(portals)) as executor:
            future_to_portal = {executor.submit(func, max_jobs_per_portal): portal for portal, func in portals.items()}
            for future in concurrent.futures.as_completed(future_to_portal):
                portal = future_to_portal[future]
                try:
                    portal_results = future.result()
                    results.extend(portal_results)
                    completed += 1
                    
                    # Update progress
                    if self.progress_callback:
                        progress = completed / portal_count
                        self.progress_callback(progress, f"Scraped {len(portal_results)} jobs from {portal}")
                    else:
                        print(f"Successfully scraped {len(portal_results)} jobs from {portal}")
                except Exception as e:
                    completed += 1
                    if self.progress_callback:
                        progress = completed / portal_count
                        self.progress_callback(progress, f"Error scraping from {portal}: {e}")
                    else:
                        print(f"Error scraping from {portal}: {e}")
        
        self.results = results
        
        # Final progress update
        if self.progress_callback:
            self.progress_callback(1.0, f"Completed! Total jobs scraped: {len(results)}")
        
        return results
    
    def save_to_csv(self, filename='job_results.csv'):
        """Save results to CSV file"""
        if not self.results:
            print("No results to save")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    def save_to_excel(self, filename='job_results.xlsx'):
        """Save results to Excel file"""
        if not self.results:
            print("No results to save")
            return
        
        df = pd.DataFrame(self.results)
        df.to_excel(filename, index=False)
        print(f"Results saved to {filename}")
    
    def display_results(self):
        """Display results in a formatted table"""
        if not self.results:
            print("No results to display")
            return
        
        df = pd.DataFrame(self.results)
        
        # Display a summary table with key columns
        summary_df = df[['portal', 'title', 'company', 'location', 'experience', 'skills']]
        pd.set_option('display.max_colwidth', 30)
        pd.set_option('display.width', 1000)
        print(summary_df)
        
        return df


class LlamaSkillExtractor:
    def __init__(self):
        # Initialize Llama model
        try:
            # Download NLTK resources if not already downloaded
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            try:
                # Try to load the model and tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
                self.model = AutoModelForCausalLM.from_pretrained(
                    "meta-llama/Llama-2-7b-chat-hf", 
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                self.model_loaded = True
                print("Llama model loaded successfully!")
            except Exception as e:
                print(f"Could not load Llama model: {e}")
                print("Using fallback keyword-based extraction.")
                self.model_loaded = False
        except Exception as e:
            print(f"Error initializing skill extractor: {e}")
            self.model_loaded = False
    
    def extract_skills(self, job_description):
        """Extract skills from job description using Llama model"""
        if not self.model_loaded:
            # Fallback to keyword-based extraction if model fails to load
            return self._keyword_extract_skills(job_description)
        
        try:
            # Prepare prompt for Llama
            prompt = f"""
            You are a helpful AI assistant specialized in HR and recruitment. 
            Extract the key technical skills, soft skills, and qualifications required from the following job description.
            Format your answer as a comma-separated list of skills only.
            
            Job Description:
            {job_description}
            
            Skills:
            """
            
            # Tokenize and generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                top_k=40,
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the skills part
            skills_text = response.split("Skills:")[-1].strip()
            
            # Clean and format the skills
            skills = [skill.strip() for skill in skills_text.split(',') if skill.strip()]
            
            return skills
        except Exception as e:
            print(f"Error in Llama extraction: {e}")
            # Fallback to keyword-based extraction
            return self._keyword_extract_skills(job_description)
    
    def _keyword_extract_skills(self, description):
        """Fallback method for skill extraction using keyword matching"""
        common_skills = [
            'python', 'java', 'javascript', 'html', 'css', 'react', 'angular', 
            'node.js', 'sql', 'nosql', 'mongodb', 'aws', 'azure', 'docker', 
            'kubernetes', 'devops', 'machine learning', 'ai', 'data science',
            'hadoop', 'spark', 'tableau', 'power bi', 'excel', 'communication',
            'leadership', 'project management', 'agile', 'scrum', 'jira',
            'git', 'github', 'jenkins', 'ci/cd', 'testing', 'qa', 'mobile',
            'android', 'ios', 'swift', 'kotlin', 'flutter', 'react native',
            'problem solving', 'teamwork', 'collaboration', 'analytical',
            'critical thinking', 'attention to detail', 'time management'
        ]
        
        found_skills = []
        desc_lower = description.lower()
        
        for skill in common_skills:
            if skill in desc_lower:
                found_skills.append(skill)
                
        return found_skills
    
    def recommend_role(self, user_skills, job_data):
        """Recommend job roles based on user skills and preferences"""
        if not self.model_loaded:
            # Fallback to simple matching if model fails to load
            return self._simple_role_recommendation(user_skills, job_data)
        
        try:
            # Create a context for the model with user skills and available jobs
            job_summaries = []
            for i, job in enumerate(job_data[:10]):  # Limit to 10 jobs for context size
                job_summary = f"Job {i+1}: {job['title']} at {job['company']} requires {job['skills']}"
                job_summaries.append(job_summary)
            
            job_context = "\n".join(job_summaries)
            
            prompt = f"""
            You are a career advisor AI. Based on a candidate's skills and the available job listings, recommend the top 3 most suitable job roles.
            For each recommendation, explain why it's a good fit and what additional skills might help the candidate succeed.
            
            Candidate skills: {', '.join(user_skills)}
            
            Available jobs:
            {job_context}
            
            Provide your recommendations in this format:
            1. [Job Title] - [Company]: Why this is a good match and any skill gaps
            2. [Job Title] - [Company]: Why this is a good match and any skill gaps
            3. [Job Title] - [Company]: Why this is a good match and any skill gaps
            """
            
            # Tokenize and generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=40,
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the recommendations part
            recommendations = response.split("Available jobs:")[-1].strip()
            
            return recommendations
        except Exception as e:
            print(f"Error in Llama recommendation: {e}")
            # Fallback to simple recommendation
            return self._simple_role_recommendation(user_skills, job_data)
    
    def _simple_role_recommendation(self, user_skills, job_data):
        """Simple matching algorithm for job recommendations"""
        job_matches = []
        
        for job in job_data:
            job_skills = job['skills'].lower().split(', ')
            match_count = 0
            for skill in user_skills:
                if skill.lower() in job['skills'].lower():
                    match_count += 1
            
            match_percentage = match_count / max(len(job_skills), 1) * 100
            job_matches.append({
                'title': job['title'],
                'company': job['company'],
                'match_percentage': match_percentage,
                'missing_skills': [s for s in job_skills if s not in [us.lower() for us in user_skills]]
            })
        
        # Sort by match percentage
        job_matches.sort(key=lambda x: x['match_percentage'], reverse=True)
        
        result = "Top 3 Job Recommendations:\n\n"
        for i, job in enumerate(job_matches[:3]):
            result += f"{i+1}. {job['title']} - {job['company']}: {job['match_percentage']:.1f}% match\n"
            result += f"   Missing skills: {', '.join(job['missing_skills'][:5])}\n\n"
        
        return result

def load_css():
    """Load custom CSS for Streamlit app"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #666;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #1E88E5;
    }
    .stProgress .st-eb {
        background-color: #1E88E5;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Streamlit app entry point"""
    st.set_page_config(
        page_title="Job Role Explorer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_css()
    
    st.markdown('<h1 class="main-header">Job Role Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Find the perfect job by analyzing skills and requirements from popular Indian job portals</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'job_data' not in st.session_state:
        st.session_state.job_data = None
    if 'skill_extractor' not in st.session_state:
        st.session_state.skill_extractor = None
    if 'user_skills' not in st.session_state:
        st.session_state.user_skills = []
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    
    # Sidebar for user input
    with st.sidebar:
        st.markdown('<h2 class="sub-header">Search Settings</h2>', unsafe_allow_html=True)
        
        job_role = st.text_input("Job Role", value="Data Scientist")
        max_jobs = st.slider("Maximum Jobs to Scrape", min_value=5, max_value=50, value=15)
        
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("**Anti-Detection Settings**")
        use_proxy = st.checkbox("Use Proxy Rotation", value=True)
        random_delay = st.checkbox("Add Random Delays", value=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Skill extraction method
        st.markdown('<h3 class="sub-header">AI Settings</h3>', unsafe_allow_html=True)
        use_llama = st.checkbox("Use LLaMA for Skill Extraction", value=False, 
                                help="Uses AI to extract skills more accurately. Requires more computing resources.")
        
        if st.button("Start Job Search"):
            # Initialize progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress, status):
                progress_bar.progress(progress)
                status_text.text(status)
            
            # Initialize skill extractor
            if use_llama and st.session_state.skill_extractor is None:
                status_text.text("Initializing LLaMA model for skill extraction...")
                st.session_state.skill_extractor = LlamaSkillExtractor()
            
            # Initialize scraper
            scraper = JobScraper(job_role, max_jobs)
            
            # Set progress callback
            scraper.set_progress_callback(update_progress)
            
            # Set skill extractor
            if use_llama and st.session_state.skill_extractor is not None:
                scraper.set_skill_extractor(st.session_state.skill_extractor)
            
            # Disable anti-detection if needed
            if not use_proxy:
                scraper.proxies = [None]
            if not random_delay:
                scraper._random_delay = lambda: None
            
            # Scrape jobs
            with st.spinner("Scraping job portals... This may take a few minutes"):
                scraper.scrape_jobs()
            
            # Process results
            if scraper.results:
                st.session_state.job_data = scraper.results
                st.success(f"Successfully scraped {len(scraper.results)} jobs!")
            else:
                st.error("No jobs found. Try different settings or job role.")
            
            # Clear progress bar
            progress_bar.empty()
            status_text.empty()
    
    # Main content area
    if st.session_state.job_data:
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Job Listings", "Skills Analysis", "Career Guidance"])
        
        # Tab 1: Job Listings
        with tab1:
            st.markdown('<h2 class="sub-header">Job Listings</h2>', unsafe_allow_html=True)
            
            # Create a DataFrame from the scraped data
            df = pd.DataFrame(st.session_state.job_data)
            
            # Allow filtering by portal
            portals = df['portal'].unique().tolist()
            selected_portals = st.multiselect("Filter by Job Portal", options=portals, default=portals)
            
            # Filter data
            if selected_portals:
                filtered_df = df[df['portal'].isin(selected_portals)]
            else:
                filtered_df = df
            
            # Create expandable sections for each job
            for i, job in filtered_df.iterrows():
                with st.expander(f"{job['title']} - {job['company']} ({job['portal']})"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Company:** {job['company']}")
                        st.markdown(f"**Location:** {job['location']}")
                        st.markdown(f"**Experience:** {job['experience']}")
                        if job['salary'] != 'Not mentioned':
                            st.markdown(f"**Salary:** {job['salary']}")
                    
                    with col2:
                        st.markdown("**Skills Required:**")
                        skills_list = job['skills'].split(', ')
                        for skill in skills_list:
                            st.markdown(f"- {skill}")
                    
                    st.markdown("**Job Description:**")
                    st.text_area("", job['description'], height=150, key=f"desc_{i}")
                    
                    st.markdown(f"**Source:** [{job['portal']}]({job['url']})")
            
            # Download options
            st.markdown('<h3 class="sub-header">Download Data</h3>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="job_results.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create Excel file in memory
                output = pd.ExcelWriter('job_results.xlsx', engine='xlsxwriter')
                df.to_excel(output, index=False)
                output.close()
                
                with open('job_results.xlsx', 'rb') as f:
                    excel_data = f.read()
                
                st.download_button(
                    label="Download as Excel",
                    data=excel_data,
                    file_name="job_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        # Tab 2: Skills Analysis
        with tab2:
            st.markdown('<h2 class="sub-header">Skills Analysis</h2>', unsafe_allow_html=True)
            
            # Extract all skills
            all_skills = []
            for job in st.session_state.job_data:
                skills = job['skills'].split(', ')
                all_skills.extend(skills)
            
            # Count skill occurrences
            skill_counts = {}
            for skill in all_skills:
                if skill in skill_counts:
                    skill_counts[skill] += 1
                else:
                    skill_counts[skill] = 1
            
            # Create a DataFrame for visualization
            skill_df = pd.DataFrame({
                'Skill': list(skill_counts.keys()),
                'Count': list(skill_counts.values())
            })
            
            # Sort by count
            skill_df = skill_df.sort_values(by='Count', ascending=False)
            
            # Create a bar chart
            fig = px.bar(
                skill_df.head(15), 
                x='Skill', 
                y='Count',
                title='Top 15 Most In-Demand Skills',
                color='Count',
                color_continuous_scale='blues'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Skill categories (this is a simplified version)
            technical_skills = [
                'python', 'java', 'javascript', 'html', 'css', 'sql', 'aws', 
                'azure', 'docker', 'kubernetes', 'hadoop', 'spark', 'react', 'angular',
                'node.js', 'tensorflow', 'pytorch', 'machine learning', 'deep learning'
            ]
            
            soft_skills = [
                'communication', 'teamwork', 'leadership', 'problem solving',
                'critical thinking', 'time management', 'creativity', 'adaptability',
                'collaboration', 'presentation'
            ]
            
            # Categorize skills
            tech_count = sum(1 for skill in all_skills if any(tech in skill.lower() for tech in technical_skills))
            soft_count = sum(1 for skill in all_skills if any(soft in skill.lower() for soft in soft_skills))
            other_count = len(all_skills) - tech_count - soft_count
            
            # Create pie chart
            category_df = pd.DataFrame({
                'Category': ['Technical Skills', 'Soft Skills', 'Other Skills'],
                'Count': [tech_count, soft_count, other_count]
            })
            
            fig2 = px.pie(
                category_df,
                values='Count',
                names='Category',
                title='Skills Distribution by Category',
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Display skill trends by job portal
            st.markdown('<h3 class="sub-header">Skill Trends by Job Portal</h3>', unsafe_allow_html=True)
            
            # Group skills by portal
            portal_skills = {}
            for job in st.session_state.job_data:
                portal = job['portal']
                skills = job['skills'].split(', ')
                
                if portal not in portal_skills:
                    portal_skills[portal] = []
                
                portal_skills[portal].extend(skills)
            
            # Create tabs for each portal
            portal_tabs = st.tabs(list(portal_skills.keys()))
            
            for i, portal in enumerate(portal_skills.keys()):
                with portal_tabs[i]:
                    # Count skills for this portal
                    portal_skill_counts = {}
                    for skill in portal_skills[portal]:
                        if skill in portal_skill_counts:
                            portal_skill_counts[skill] += 1
                        else:
                            portal_skill_counts[skill] = 1
                    
                    # Create DataFrame
                    portal_skill_df = pd.DataFrame({
                        'Skill': list(portal_skill_counts.keys()),
                        'Count': list(portal_skill_counts.values())
                    }).sort_values(by='Count', ascending=False)
                    
                    # Create bar chart
                    fig3 = px.bar(
                        portal_skill_df.head(10),
                        x='Skill',
                        y='Count',
                        title=f'Top 10 Skills on {portal}',
                        color='Count',
                        color_continuous_scale='blues'
                    )
                    
                    st.plotly_chart(fig3, use_container_width=True)
        
        # Tab 3: Career Guidance
        with tab3:
            st.markdown('<h2 class="sub-header">Career Guidance</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <p class="info-text">Tell us about your skills and get personalized job role recommendations!</p>
            """, unsafe_allow_html=True)
            
            # Get user skills
            user_skills_input = st.text_area(
                "Enter your skills (comma-separated)",
                value="python, data analysis, machine learning, sql, communication" if not st.session_state.user_skills else ", ".join(st.session_state.user_skills),
                help="Example: python, data analysis, machine learning, sql, communication"
            )
            
            # Years of experience
            experience = st.slider("Years of Experience", min_value=0, max_value=20, value=2)
            
            # Preferred location
            locations = list(set([job['location'].split(',')[0].strip() for job in st.session_state.job_data]))
            preferred_location = st.selectbox("Preferred Location", options=['Any'] + sorted(locations))
            
            # Career goals
            career_goals = st.text_area(
                "Career Goals (optional)",
                value="",
                help="Describe your career goals and what you're looking for in your next role"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Get AI Recommendations", key="ai_recs"):
                    if st.session_state.skill_extractor is None or not st.session_state.skill_extractor.model_loaded:
                        with st.spinner("Initializing LLaMA model..."):
                            st.session_state.skill_extractor = LlamaSkillExtractor()
                    
                    if st.session_state.skill_extractor.model_loaded:
                        with st.spinner("Analyzing your profile with AI..."):
                            # Parse user skills
                            st.session_state.user_skills = [skill.strip() for skill in user_skills_input.split(',') if skill.strip()]
                            
                            # Filter jobs by location if specified
                            if preferred_location != 'Any':
                                filtered_jobs = [job for job in st.session_state.job_data if preferred_location in job['location']]
                            else:
                                filtered_jobs = st.session_state.job_data
                            
                            # Get recommendations
                            st.session_state.recommendations = st.session_state.skill_extractor.recommend_role(
                                st.session_state.user_skills, 
                                filtered_jobs
                            )
                    else:
                        st.error("LLaMA model could not be loaded. Using fallback recommendation method.")
                        # Use fallback method
                        with st.spinner("Analyzing your profile..."):
                            st.session_state.user_skills = [skill.strip() for skill in user_skills_input.split(',') if skill.strip()]
                            if preferred_location != 'Any':
                                filtered_jobs = [job for job in st.session_state.job_data if preferred_location in job['location']]
                            else:
                                filtered_jobs = st.session_state.job_data
                            st.session_state.recommendations = st.session_state.skill_extractor._simple_role_recommendation(
                                st.session_state.user_skills, 
                                filtered_jobs
                            )
            
            with col2:
                if st.button("Get Simple Recommendations", key="simple_recs"):
                    with st.spinner("Analyzing your profile..."):
                        # Initialize skill extractor if needed
                        if st.session_state.skill_extractor is None:
                            st.session_state.skill_extractor = LlamaSkillExtractor()
                        
                        # Parse user skills
                        st.session_state.user_skills = [skill.strip() for skill in user_skills_input.split(',') if skill.strip()]
                        
                        # Filter jobs by location if specified
                        if preferred_location != 'Any':
                            filtered_jobs = [job for job in st.session_state.job_data if preferred_location in job['location']]
                        else:
                            filtered_jobs = st.session_state.job_data
                        
                        # Get recommendations (simple method)
                        st.session_state.recommendations = st.session_state.skill_extractor._simple_role_recommendation(
                            st.session_state.user_skills, 
                            filtered_jobs
                        )
            
            # Display recommendations
            if st.session_state.recommendations:
                st.markdown('<div class="highlight">', unsafe_allow_html=True)
                st.markdown("### Your Personalized Job Recommendations")
                st.markdown(st.session_state.recommendations)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("### Skill Gap Analysis")
                
                # Identify skill gaps
                all_required_skills = set()
                for job in st.session_state.job_data:
                    skills = job['skills'].split(', ')
                    all_required_skills.update(skills)
                
                user_skills_set = set(st.session_state.user_skills)
                missing_skills = all_required_skills - user_skills_set
                
                # Count how many jobs require each missing skill
                missing_skill_counts = {}
                for skill in missing_skills:
                    count = sum(1 for job in st.session_state.job_data if skill in job['skills'])
                    missing_skill_counts[skill] = count
                
                # Sort by count
                sorted_missing_skills = sorted(missing_skill_counts.items(), key=lambda x: x[1], reverse=True)
                
                # Create DataFrame
                missing_df = pd.DataFrame({
                    'Skill': [item[0] for item in sorted_missing_skills[:10]],
                    'Jobs Requiring': [item[1] for item in sorted_missing_skills[:10]]
                })
                
                # Create bar chart
                fig4 = px.bar(
                    missing_df,
                    x='Skill',
                    y='Jobs Requiring',
                    title='Top 10 Skills to Consider Learning',
                    color='Jobs Requiring',
                    color_continuous_scale='blues'
                )
                
                st.plotly_chart(fig4, use_container_width=True)
                
                st.markdown("""
                <p class="info-text">These are the most in-demand skills you might want to acquire to increase your job prospects.</p>
                """, unsafe_allow_html=True)
                
                # Learning resources suggestion
                st.markdown("### Learning Resources")
                st.markdown("""
                Here are some suggested resources to learn the skills you're missing:
                
                - **Technical Skills**: Coursera, Udemy, edX, or freeCodeCamp
                - **Data Science**: Kaggle, DataCamp
                - **Programming**: LeetCode, HackerRank
                - **Cloud**: AWS Training, Microsoft Learn
                - **Soft Skills**: LinkedIn Learning, Toastmasters
                
                Remember that building a portfolio of projects is also essential for demonstrating your skills!
                """)
    else:
        # Display welcome message if no data yet
        st.markdown("""
        <div class="highlight">
        <h3>Welcome to Job Role Explorer!</h3>
        <p>This tool helps you:</p>
        <ul>
            <li>Scrape job listings from major Indian job portals</li>
            <li>Analyze required skills and qualifications</li>
            <li>Get personalized career guidance</li>
            <li>Identify skill gaps and learning opportunities</li>
        </ul>
        <p>Get started by entering a job role in the sidebar and clicking "Start Job Search".</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show a sample visualization
        st.markdown('<h3 class="sub-header">Sample Skills Analysis</h3>', unsafe_allow_html=True)
        
        # Sample data
        sample_skills = {
            'Python': 24, 'SQL': 21, 'Machine Learning': 18, 'Data Analysis': 16,
            'Java': 14, 'AWS': 11, 'JavaScript': 10, 'Communication': 9,
            'Docker': 8, 'Git': 7
        }
        
        sample_df = pd.DataFrame({
            'Skill': list(sample_skills.keys()),
            'Count': list(sample_skills.values())
        })
        
        fig = px.bar(
            sample_df,
            x='Skill',
            y='Count',
            title='Example: Most In-Demand Skills for Data Scientists',
            color='Count',
            color_continuous_scale='blues'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<p class="info-text">This is a sample visualization. Start your job search to see real data!</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
