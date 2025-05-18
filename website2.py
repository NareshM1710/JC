import streamlit as st
import pandas as pd
from PIL import Image
import base64
import requests
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components

# Page Configuration
st.set_page_config(
    page_title="Naresh M - Data Analytics Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
def add_custom_css():
    st.markdown("""
    <style>
    /* Main Page Styling */
    .main {
        padding: 1.5rem;
        background-color: #f8f9fa;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #0a2540;
    }
    
    h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    h2 {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    
    h3 {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #0083B8;
    }
    
    /* Card Styling */
    .card {
        border-radius: 10px;
        box-shadow: 0 6px 10px rgba(0,0,0,0.08);
        padding: 20px;
        margin-bottom: 20px;
        background-color: white;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.12);
    }
    
    /* Project Card */
    .project-card {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background-color: white;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .project-img {
        width: 100%;
        height: 180px;
        object-fit: cover;
    }
    
    .project-content {
        padding: 15px;
    }
    
    .project-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0083B8;
        margin-bottom: 0.5rem;
    }
    
    .project-desc {
        font-size: 0.9rem;
        color: #555;
        margin-bottom: 1rem;
    }
    
    /* Button Styling */
    .custom-button {
        background-color: #0083B8;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    
    .custom-button:hover {
        background-color: #006491;
    }
    
    /* Skill Badge */
    .skill-badge {
        background-color: #e9f5f9;
        color: #0083B8;
        border-radius: 15px;
        padding: 5px 12px;
        margin: 3px;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Experience Timeline */
    .timeline-item {
        padding-left: 20px;
        border-left: 2px solid #0083B8;
        margin-bottom: 20px;
        position: relative;
    }
    
    .timeline-dot {
        width: 12px;
        height: 12px;
        background-color: #0083B8;
        border-radius: 50%;
        position: absolute;
        left: -7px;
        top: 0;
    }
    
    .timeline-date {
        font-size: 0.9rem;
        color: #0083B8;
        font-weight: 600;
    }
    
    /* Navigation Menu */
    div[data-testid="stHorizontalBlock"] div[data-testid="column"] {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #0083B8;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #006491;
    }
    
    /* For better mobile responsiveness */
    @media (max-width: 768px) {
        .card {
            padding: 15px;
        }
        
        h1 {
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Apply custom CSS
add_custom_css()

# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animations
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
lottie_data = load_lottieurl("https://assets5.lottiefiles.com/private_files/lf30_8npirptd.json")
lottie_contact = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_u25cckyh.json")

# Custom HTML components
def create_skill_badge(skill_name):
    return f'<span class="skill-badge">{skill_name}</span>'

def create_timeline_item(role, company, period, description_points):
    timeline_html = f'''
    <div class="timeline-item">
        <div class="timeline-dot"></div>
        <div class="timeline-date">{period}</div>
        <h3>{role} | {company}</h3>
        <ul>
    '''
    
    for point in description_points:
        timeline_html += f'<li>{point}</li>'
    
    timeline_html += '''
        </ul>
    </div>
    '''
    
    return timeline_html

def create_project_card(title, description, image_path, tags=None):
    card_html = f'''
    <div class="project-card">
        <img src="{image_path}" class="project-img" alt="{title}">
        <div class="project-content">
            <div class="project-title">{title}</div>
            <div class="project-desc">{description}</div>
    '''
    
    if tags:
        card_html += '<div>'
        for tag in tags:
            card_html += f'<span class="skill-badge">{tag}</span> '
        card_html += '</div>'
    
    card_html += '''
        </div>
    </div>
    '''
    
    return card_html

# Navigation Menu with enhanced styling
selected = option_menu(
    menu_title=None,
    options=["About", "Experience", "Projects", "Skills", "Contact"],
    icons=["person-circle", "briefcase", "laptop", "graph-up", "envelope"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#ffffff", "border-radius": "10px", "box-shadow": "0 2px 5px rgba(0,0,0,0.05)"},
        "icon": {"color": "#0083B8", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#e0f7fa",
            "color": "#333",
            "font-weight": "500",
            "padding": "15px",
        },
        "nav-link-selected": {"background-color": "#0083B8", "color": "white", "font-weight": "600"},
    }
)

# About Me Section
if selected == "About":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Header with animation
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h1>Hello, I'm Naresh M</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Data Storyteller | Analytics Expert | Problem Solver</h3>", unsafe_allow_html=True)
    
    with col2:
        st_lottie(lottie_data, height=150, key="header_animation")
    
    # About me content
    st.markdown("""
    <p style="font-size: 18px; line-height: 1.6;">
    I'm a passionate Data Analyst with 3+ years of experience transforming complex data into actionable insights.
    My journey in data analytics has been driven by curiosity and a desire to solve real-world problems through data.
    </p>
    """, unsafe_allow_html=True)
    
    # What I do section
    st.markdown("### What I Do")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
            <h4 style="color: #0083B8;">📊 Data Transformation</h4>
            <p>Converting raw data into compelling, actionable stories that drive business decisions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="padding: 15px; background-color: #f8f9fa; border-radius: 8px; margin-top: 15px;">
            <h4 style="color: #0083B8;">🔍 Insight Discovery</h4>
            <p>Uncovering hidden patterns and trends in complex datasets to provide strategic advantages.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
            <h4 style="color: #0083B8;">💡 Dashboard Development</h4>
            <p>Creating interactive dashboards and reports that make data accessible and actionable.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="padding: 15px; background-color: #f8f9fa; border-radius: 8px; margin-top: 15px;">
            <h4 style="color: #0083B8;">📈 Business Intelligence</h4>
            <p>Supporting strategic decision-making with comprehensive data analysis and visualization.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Experience Section
elif selected == "Experience":
    st.markdown("<h1 style='text-align: center;'>Professional Experience</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 30px;'>My journey in the world of data analytics</p>", unsafe_allow_html=True)
    
    # Experience timeline
    experience_col1, experience_col2 = st.columns([2, 1])
    
    with experience_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(create_timeline_item(
            "Data Analyst", 
            "Mitsogo", 
            "Jul 2023 - Present",
            [
                "Implemented data validation processes resulting in a 95% proficiency rate, ensuring accurate analytics on lead conversion rates.",
                "Designed and created dashboards from HubSpot CRM and Zendesk ticket data, delivering insights to the SEM and sales teams.",
                "Provided daily analytics reports to the CMO, showcasing key metrics and actionable insights for informed decision-making.",
                "Utilized Power Query (DAX) and Excel for advanced data visualization, creating visually appealing dashboards.",
                "Automated tasks using Excel VBA Macros, streamlining workflows and enhancing accuracy in data processing."
            ]
        ), unsafe_allow_html=True)
        
        st.markdown(create_timeline_item(
            "Data Processing Analyst", 
            "Nielsen IQ", 
            "Feb 2022 - Jul 2023",
            [
                "Led key projects and trained teams to enhance competency, excelling as an OGRDS Specialist in MS Excel reporting.",
                "Achieved 90-95% accuracy in data management, earning awards from major clients.",
                "Conducted market research, delivered sales forecasts, and prepared daily, weekly, and monthly MIS reports.",
                "Developed and implemented processes in Python, optimizing data processing efficiency by 95%.",
                "Demonstrated automation skills with Python, automating tasks and developing text classifiers and Web Automation bots."
            ]
        ), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with experience_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Education</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <p style="font-weight: 600; margin-bottom: 5px;">Bachelor of Technology</p>
            <p style="color: #666; font-size: 0.9rem;">Computer Science Engineering</p>
            <p style="color: #0083B8; font-size: 0.9rem;">2018 - 2022</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3>Certifications</h3>", unsafe_allow_html=True)
        
        certifications = [
            "Power BI Data Analyst Associate",
            "Microsoft Excel Expert",
            "SQL Database Administration",
            "Python for Data Analysis"
        ]
        
        for cert in certifications:
            st.markdown(f"""
            <div style="margin-bottom: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                <p style="margin: 0; font-weight: 500;">{cert}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Projects Section
elif selected == "Projects":
    st.markdown("<h1 style='text-align: center;'>Featured Projects</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 30px;'>Showcasing my expertise in data analytics and visualization</p>", unsafe_allow_html=True)
    
    # Power BI Projects
    st.subheader("Power BI Dashboards")
    
    powerbi_col1, powerbi_col2, powerbi_col3 = st.columns(3)
    
    # Project 1
    with powerbi_col1:
        st.markdown(create_project_card(
            "Airline Loyalty Program",
            "Interactive dashboard tracking customer loyalty metrics, flight patterns, and reward redemption rates for a major airline.",
            "Airline Loyalty Program.jpg",
            ["Power BI", "DAX", "Customer Analytics"]
        ), unsafe_allow_html=True)
    
    # Project 2
    with powerbi_col2:
        st.markdown(create_project_card(
            "Toy Store KPI Report",
            "Comprehensive dashboard tracking marketing ROI, sales performance, and inventory metrics for Maven Toy Store.",
            "Maven Toy Dashboard.jpg",
            ["Power BI", "Retail Analytics", "KPI Tracking"]
        ), unsafe_allow_html=True)
    
    # Project 3
    with powerbi_col3:
        st.markdown(create_project_card(
            "Survey Dashboard",
            "Analysis of demographics and career insights from a survey of Data Professional roles across industries.",
            "Survey Dashboard.jpg",
            ["Power BI", "Survey Analysis", "Data Visualization"]
        ), unsafe_allow_html=True)
    
    powerbi_col4, powerbi_col5 = st.columns(2)
    
    # Project 4
    with powerbi_col4:
        st.markdown(create_project_card(
            "Uber Analytics Dashboard",
            "In-depth analysis of ride durations, peak hours, popular routes, and driver performance metrics.",
            "Uber Analytics.jpg",
            ["Power BI", "Transportation Analytics", "Geospatial"]
        ), unsafe_allow_html=True)
    
    # Project 5
    with powerbi_col5:
        st.markdown(create_project_card(
            "Plant Co. Performance Dashboard",
            "Executive dashboard analyzing sales performance, market penetration, and growth opportunities for Plant Co.",
            "Plant Co. Dashboard.jpg",
            ["Power BI", "Executive Dashboard", "Performance Metrics"]
        ), unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
    
    # Excel Projects
    st.subheader("Excel Projects")
    
    excel_col1, excel_col2 = st.columns(2)
    
    # Excel Project 1
    with excel_col1:
        st.markdown(create_project_card(
            "Call Center Dashboard",
            "Comprehensive Excel dashboard tracking call volumes, resolution times, customer satisfaction, and agent performance.",
            "Excel Dashboard.jpg",
            ["Excel", "VBA", "Dashboard Design"]
        ), unsafe_allow_html=True)
    
    # Excel Project 2
    with excel_col2:
        st.markdown(create_project_card(
            "Power Pivot Analysis",
            "Advanced data modeling using Power Pivot and Power Query to create a multi-dimensional analysis of business metrics.",
            "Excel PowerPivot.jpg",
            ["Excel", "Power Pivot", "Power Query", "ETL"]
        ), unsafe_allow_html=True)

# Skills Section
elif selected == "Skills":
    st.markdown("<h1 style='text-align: center;'>Skills & Expertise</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 30px;'>My technical and professional competencies</p>", unsafe_allow_html=True)
    
    # Two columns layout
    skill_col1, skill_col2 = st.columns(2)
    
    # Technical Skills
    with skill_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Technical Skills</h3>", unsafe_allow_html=True)
        
        # Data Visualization
        st.markdown("<h4>Data Visualization</h4>", unsafe_allow_html=True)
        skills_viz = ["Power BI", "Advanced Excel Charts", "Dashboard Design", "Data Storytelling", "Interactive Reports"]
        viz_html = ""
        for skill in skills_viz:
            viz_html += create_skill_badge(skill) + " "
        st.markdown(viz_html, unsafe_allow_html=True)
        
        # Data Analysis
        st.markdown("<h4>Data Analysis</h4>", unsafe_allow_html=True)
        skills_analysis = ["SQL", "Power Query", "DAX", "Statistical Analysis", "Trend Analysis", "Forecasting"]
        analysis_html = ""
        for skill in skills_analysis:
            analysis_html += create_skill_badge(skill) + " "
        st.markdown(analysis_html, unsafe_allow_html=True)
        
        # Programming
        st.markdown("<h4>Programming</h4>", unsafe_allow_html=True)
        skills_prog = ["Python", "VBA", "Excel Macros", "ETL Processing", "Data Cleaning"]
        prog_html = ""
        for skill in skills_prog:
            prog_html += create_skill_badge(skill) + " "
        st.markdown(prog_html, unsafe_allow_html=True)
        
        # Database
        st.markdown("<h4>Database</h4>", unsafe_allow_html=True)
        skills_db = ["MySQL", "PostgreSQL", "Data Modeling", "Query Optimization"]
        db_html = ""
        for skill in skills_db:
            db_html += create_skill_badge(skill) + " "
        st.markdown(db_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Professional Skills
    with skill_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Professional Skills</h3>", unsafe_allow_html=True)
        
        # Skill proficiency chart
        skills = {
            "Problem Solving": 90,
            "Data Storytelling": 95,
            "Team Collaboration": 85,
            "Project Management": 80,
            "Critical Thinking": 92,
            "Business Acumen": 85,
            "Communication": 88
        }
        
        for skill, proficiency in skills.items():
            st.markdown(f"""
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>{skill}</span>
                    <span>{proficiency}%</span>
                </div>
                <div style="background-color: #e0e0e0; border-radius: 10px; height: 10px;">
                    <div style="background-color: #0083B8; width: {proficiency}%; height: 10px; border-radius: 10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Tools & Software
        st.markdown("<h4>Tools & Software</h4>", unsafe_allow_html=True)
        tools = ["Microsoft Office Suite", "HubSpot CRM", "Zendesk", "Google Analytics", "Tableau", "JIRA"]
        tools_html = ""
        for tool in tools:
            tools_html += create_skill_badge(tool) + " "
        st.markdown(tools_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Contact Section
elif selected == "Contact":
    st.markdown("<h1 style='text-align: center;'>Let's Connect!</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 30px;'>I'm always open to discussing new projects, opportunities, and collaborations</p>", unsafe_allow_html=True)
    
    contact_col1, contact_col2 = st.columns([1, 1])
    
    with contact_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Get In Touch</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <h4 style="margin-bottom: 10px;">📧 Email</h4>
            <p style="font-size: 16px;">nareshdharun17@gmail.com</p>
        </div>
        
        <div style="margin-bottom: 20px;">
            <h4 style="margin-bottom: 10px;">🔗 Professional Profiles</h4>
            <a href="http://www.linkedin.com/in/naresh-m-90796a141" target="_blank" style="text-decoration: none;">
                <div style="display: flex; align-items: center; margin-bottom: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                    <span style="margin-left: 10px; color: #0077B5; font-weight: 500;">LinkedIn</span>
                </div>
            </a>
            
            <a href="https://github.com/naresh" target="_blank" style="text-decoration: none;">
                <div style="display: flex; align-items: center; margin-bottom: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                    <span style="margin-left: 10px; color: #333; font-weight: 500;">GitHub</span>
                </div>
            </a>
        </div>
        
        <div>
            <h4 style="margin-bottom: 10px;">📝 Resume</h4>
            <a href="#" style="text-decoration: none;">
                <div style="display: flex; align-items: center; padding: 10px; background-color: #0083B8; border-radius: 5px; color: white; justify-content: center;">
                    <span style="font-weight: 500;">Download Resume</span>
                </div>
            </a>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with contact_col2:
        st_lottie(lottie_contact, height=400, key="contact_animation")
    
# Add page footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; color: #666;">
    <p>© 2025 Naresh M | Data Analyst Portfolio</p>
</div>
""", unsafe_allow_html=True)
