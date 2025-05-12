import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
import datetime
from dateutil import parser
import matplotlib.pyplot as plt
import seaborn as sns
from bson.son import SON
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="NYC Restaurant Inspection Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e9ecef;
        padding: 10px 16px;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    h1, h2, h3 {
        color: #2C3E50;
    }
    .stAlert {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Function to establish MongoDB connection
@st.cache_resource
def get_mongo_client():
    """Create a MongoDB client connection"""
    try:
        # Update with your MongoDB connection string
        connection_string = st.secrets["mongo"]["connection_string"]
        client = MongoClient(connection_string)
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

# Function to check MongoDB connection
def check_mongo_connection(client):
    """Check if MongoDB connection is working"""
    try:
        # List all databases to verify connection
        client.list_database_names()
        return True
    except Exception as e:
        return False

# Connect to MongoDB
mongo_client = get_mongo_client()

# Function to load data for each query
@st.cache_data(ttl=3600)
def get_top_cuisines_by_inspection_score(min_inspections=10, limit=10, sort_order=-1):
    """Query 1: Top cuisines with the highest/lowest average inspection scores
    
    Parameters:
    min_inspections (int): Minimum number of inspections required for a cuisine to be included
    limit (int): Number of cuisines to return
    sort_order (int): -1 for highest scores first, 1 for lowest scores first
    """
    db = mongo_client["restaurant_inspections"]
    collection = db["inspections"]
    
    pipeline = [
        { '$match': { 'SCORE': { '$ne': None } } },
        {
            '$lookup': {
                'from': 'establishments', 
                'localField': 'CAMIS', 
                'foreignField': 'CAMIS', 
                'as': 'restaurant'
            }
        },
        { '$unwind': '$restaurant' },
        {
            '$group': {
                '_id': '$restaurant.CUISINE DESCRIPTION', 
                'averageScore': { '$avg': '$SCORE' }, 
                'inspectionCount': { '$sum': 1 }
            }
        },
        { '$match': { 'inspectionCount': { '$gte': min_inspections } } },
        { '$sort': { 'averageScore': sort_order } },
        { '$limit': limit },
        {
            '$project': {
                '_id': 0, 
                'cuisine': '$_id', 
                'averageScore': { '$round': ['$averageScore', 1] }, 
                'inspectionCount': 1
            }
        }
    ]
    
    result = list(collection.aggregate(pipeline))
    return pd.DataFrame(result)

@st.cache_data(ttl=3600)
def get_borough_critical_violations(borough='Manhattan', limit=10, critical_flag='Critical'):
    """Query 2: Most common violations in a selected borough's restaurants
    
    Parameters:
    borough (str): Borough to analyze ('Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island')
    limit (int): Number of violation types to return
    critical_flag (str): 'Critical', 'Not Critical', or 'All'
    """
    db = mongo_client["restaurant_inspections"]
    collection = db["violations"]
    
    # Build match condition for critical flag
    if critical_flag == 'All':
        critical_match = {}
    else:
        critical_match = { 'CRITICAL FLAG': critical_flag }
    
    pipeline = [
        { '$match': critical_match },
        {
            '$lookup': {
                'from': 'inspections', 
                'localField': 'InspectionID', 
                'foreignField': 'InspectionID', 
                'as': 'inspection'
            }
        },
        { '$unwind': '$inspection' },
        {
            '$lookup': {
                'from': 'establishments', 
                'localField': 'inspection.CAMIS', 
                'foreignField': 'CAMIS', 
                'as': 'restaurant'
            }
        },
        { '$unwind': '$restaurant' },
        { '$match': { 'restaurant.BORO': borough } },
        {
            '$group': {
                '_id': {
                    'code': '$VIOLATION CODE', 
                    'description': '$VIOLATION DESCRIPTION'
                }, 
                'count': { '$sum': 1 }
            }
        },
        { '$sort': { 'count': -1 } },
        { '$limit': limit },
        {
            '$project': {
                '_id': 0, 
                'violationCode': '$_id.code', 
                'violationDescription': '$_id.description', 
                'count': 1
            }
        }
    ]
    
    result = list(collection.aggregate(pipeline))
    return pd.DataFrame(result)

@st.cache_data(ttl=3600)
def get_borough_grade_distribution():
    """Query 3: Compare performance of different boroughs by average grade distribution"""
    db = mongo_client["restaurant_inspections"]
    collection = db["inspections"]
    
    pipeline = [
        { '$match': { 'GRADE': { '$in': ['A', 'B', 'C'] } } },
        {
            '$lookup': {
                'from': 'establishments', 
                'localField': 'CAMIS', 
                'foreignField': 'CAMIS', 
                'as': 'restaurant'
            }
        },
        { '$unwind': '$restaurant' },
        {
            '$group': {
                '_id': {
                    'borough': '$restaurant.BORO', 
                    'grade': '$GRADE'
                }, 
                'count': { '$sum': 1 }
            }
        },
        {
            '$group': {
                '_id': '$_id.borough', 
                'grades': {
                    '$push': {
                        'grade': '$_id.grade', 
                        'count': '$count'
                    }
                }, 
                'total': { '$sum': '$count' }
            }
        },
        {
            '$project': {
                '_id': 0, 
                'borough': '$_id', 
                'total': '$total', 
                'grades': 1
            }
        }
    ]
    
    result = list(collection.aggregate(pipeline))
    
    # Process data for easier visualization
    processed_data = []
    for borough_data in result:
        borough = borough_data['borough']
        total = borough_data['total']
        
        # Initialize counts
        a_count = 0
        b_count = 0
        c_count = 0
        
        # Extract counts for each grade
        for grade_info in borough_data['grades']:
            if grade_info['grade'] == 'A':
                a_count = grade_info['count']
            elif grade_info['grade'] == 'B':
                b_count = grade_info['count']
            elif grade_info['grade'] == 'C':
                c_count = grade_info['count']
        
        # Calculate percentages
        a_percent = (a_count / total) * 100 if total > 0 else 0
        b_percent = (b_count / total) * 100 if total > 0 else 0
        c_percent = (c_count / total) * 100 if total > 0 else 0
        
        processed_data.append({
            'borough': borough,
            'total': total,
            'A_count': a_count,
            'B_count': b_count,
            'C_count': c_count,
            'A_percent': a_percent,
            'B_percent': b_percent,
            'C_percent': c_percent
        })
    
    return pd.DataFrame(processed_data)

@st.cache_data(ttl=3600)
def get_most_improved_restaurants(min_inspections=2, limit=20, borough=None, min_improvement=0):
    """Query 4: Most improved inspection scores over time
    
    Parameters:
    min_inspections (int): Minimum number of inspections required for a restaurant
    limit (int): Number of restaurants to return
    borough (str or None): Filter by borough (None for all boroughs)
    min_improvement (int): Minimum improvement score to consider
    """
    db = mongo_client["restaurant_inspections"]
    collection = db["inspections"]
    
    pipeline = [
        { '$match': { 'SCORE': { '$ne': None } } },
        {
            '$addFields': {
                'inspectionDate': {
                    '$dateFromString': {
                        'dateString': '$INSPECTION DATE', 
                        'format': '%m/%d/%Y'
                    }
                }
            }
        },
        { '$sort': { 'CAMIS': 1, 'inspectionDate': 1 } },
        {
            '$group': {
                '_id': '$CAMIS', 
                'firstScore': { '$first': '$SCORE' }, 
                'lastScore': { '$last': '$SCORE' }, 
                'inspectionCount': { '$sum': 1 }
            }
        },
        {
            '$addFields': {
                'improvement': { '$subtract': ['$firstScore', '$lastScore'] }
            }
        },
        { '$match': { 
            'inspectionCount': { '$gte': min_inspections },
            'improvement': { '$gte': min_improvement }
        }},
        { '$sort': { 'improvement': -1 } },
        {
            '$lookup': {
                'from': 'establishments', 
                'localField': '_id', 
                'foreignField': 'CAMIS', 
                'as': 'restaurant'
            }
        },
        { '$unwind': '$restaurant' }
    ]
    
    # Add borough filter if specified
    if borough:
        pipeline.append({'$match': {'restaurant.BORO': borough}})
    
    # Add limit and projection
    pipeline.extend([
        { '$limit': limit },
        {
            '$project': {
                '_id': 0, 
                'restaurantId': '$_id', 
                'name': '$restaurant.DBA', 
                'borough': '$restaurant.BORO', 
                'cuisine': '$restaurant.CUISINE DESCRIPTION', 
                'firstScore': 1, 
                'lastScore': 1, 
                'improvement': 1, 
                'inspectionCount': 1
            }
        }
    ])
    
    result = list(collection.aggregate(pipeline))
    return pd.DataFrame(result)

@st.cache_data(ttl=3600)
def get_critical_violations_by_cuisine():
    """Query 5: Analyze critical violations by cuisine type and their impact on grades"""
    db = mongo_client["restaurant_inspections"]
    collection = db["violations"]
    
    pipeline = [
        { '$match': { 'CRITICAL FLAG': 'Critical' } },
        {
            '$lookup': {
                'from': 'inspections', 
                'localField': 'InspectionID', 
                'foreignField': 'InspectionID', 
                'as': 'inspection'
            }
        },
        { '$unwind': '$inspection' },
        {
            '$match': {
                'inspection.GRADE': { '$in': ['A', 'B', 'C'] }
            }
        },
        {
            '$lookup': {
                'from': 'establishments', 
                'localField': 'inspection.CAMIS', 
                'foreignField': 'CAMIS', 
                'as': 'restaurant'
            }
        },
        { '$unwind': '$restaurant' },
        {
            '$group': {
                '_id': {
                    'cuisine': '$restaurant.CUISINE DESCRIPTION', 
                    'grade': '$inspection.GRADE', 
                    'violationCode': '$VIOLATION CODE'
                }, 
                'count': { '$sum': 1 }
            }
        },
        {
            '$group': {
                '_id': {
                    'cuisine': '$_id.cuisine', 
                    'grade': '$_id.grade'
                }, 
                'violations': {
                    '$push': {
                        'code': '$_id.violationCode', 
                        'count': '$count'
                    }
                }, 
                'totalViolations': { '$sum': '$count' }
            }
        },
        {
            '$group': {
                '_id': '$_id.cuisine', 
                'gradeBreakdown': {
                    '$push': {
                        'grade': '$_id.grade', 
                        'totalViolations': '$totalViolations'
                    }
                }, 
                'totalCriticalViolations': { '$sum': '$totalViolations' }
            }
        },
        {
            '$match': {
                'totalCriticalViolations': { '$gte': 100 }
            }
        },
        { '$sort': { 'totalCriticalViolations': -1 } },
        { '$limit': 15 },
        {
            '$project': {
                '_id': 0, 
                'cuisine': '$_id', 
                'totalCriticalViolations': 1, 
                'gradeBreakdown': 1
            }
        }
    ]
    
    result = list(collection.aggregate(pipeline))
    
    # Process data for easier visualization
    processed_data = []
    for cuisine_data in result:
        cuisine = cuisine_data['cuisine']
        total = cuisine_data['totalCriticalViolations']
        
        # Initialize counts
        a_count = 0
        b_count = 0
        c_count = 0
        
        # Extract counts for each grade
        for grade_info in cuisine_data['gradeBreakdown']:
            if grade_info['grade'] == 'A':
                a_count = grade_info['totalViolations']
            elif grade_info['grade'] == 'B':
                b_count = grade_info['totalViolations']
            elif grade_info['grade'] == 'C':
                c_count = grade_info['totalViolations']
        
        processed_data.append({
            'cuisine': cuisine,
            'totalCriticalViolations': total,
            'A_violations': a_count,
            'B_violations': b_count,
            'C_violations': c_count
        })
    
    return pd.DataFrame(processed_data)

# Function to get summary statistics
@st.cache_data(ttl=3600)
def get_summary_statistics():
    """Get summary statistics for the dashboard"""
    db = mongo_client["restaurant_inspections"]
    
    # Count of restaurants
    restaurant_count = db["establishments"].count_documents({})
    
    # Count of inspections
    inspection_count = db["inspections"].count_documents({})
    
    # Count of violations
    violation_count = db["violations"].count_documents({})
    
    # Get grade distribution
    grade_pipeline = [
        { '$match': { 'GRADE': { '$in': ['A', 'B', 'C'] } } },
        {
            '$group': {
                '_id': '$GRADE',
                'count': { '$sum': 1 }
            }
        }
    ]
    
    grades = list(db["inspections"].aggregate(grade_pipeline))
    grade_dict = {item['_id']: item['count'] for item in grades}
    
    a_grade_percent = grade_dict.get('A', 0) / sum(grade_dict.values()) * 100 if grade_dict else 0
    
    # Average score
    avg_score_pipeline = [
        { '$match': { 'SCORE': { '$ne': None } } },
        {
            '$group': {
                '_id': None,
                'avgScore': { '$avg': '$SCORE' }
            }
        }
    ]
    
    avg_score_result = list(db["inspections"].aggregate(avg_score_pipeline))
    avg_score = round(avg_score_result[0]['avgScore'], 1) if avg_score_result else 0
    
    return {
        'restaurant_count': restaurant_count,
        'inspection_count': inspection_count,
        'violation_count': violation_count,
        'a_grade_percent': a_grade_percent,
        'avg_score': avg_score
    }

# Define the app title
st.title("🍽️ NYC Restaurant Inspection Dashboard")

# Create sidebar for navigation and database connection status
with st.sidebar:
    st.title("Navigation")
    
    # Connection status
    if check_mongo_connection(mongo_client):
        st.success("✅ Connected to MongoDB")
    else:
        st.error("❌ Failed to connect to MongoDB")
        st.stop()
    
    # Database selector (for future use if multiple databases)
    db_name = st.selectbox("Select Database", ["restaurant_inspections"])
    
    # Navigation menu
    selected = option_menu(
        "Main Menu", 
        ["Home", "Top Cuisines", "Critical Violations", "Borough Performance", 
         "Most Improved", "Cuisine Violations", "About"],
        icons=['house', 'egg-fried', 'exclamation-triangle', 'geo-alt', 
               'arrow-up', 'list-check', 'info-circle'], 
        menu_icon="cast", 
        default_index=0
    )
    
    st.markdown("---")
    st.caption("© 2025 NYC Restaurant Inspections")

# Home page
if selected == "Home":
    st.header("NYC Restaurant Inspection Overview")
    
    # Load summary statistics
    stats = get_summary_statistics()
    
    # Display summary statistics in metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{stats["restaurant_count"]:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Restaurants</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{stats["inspection_count"]:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Inspections</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{stats["violation_count"]:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Violations</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # A-grade percentage and average score
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{stats["a_grade_percent"]:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Restaurants with "A" Grade</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{stats["avg_score"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average Inspection Score</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dashboard overview
    st.subheader("About This Dashboard")
    st.write("""
    This dashboard visualizes NYC Restaurant Inspection data, providing insights into:
    
    * **Top Cuisines by Inspection Score**: Discover which cuisines have the highest average inspection scores
    * **Critical Violations in Manhattan**: View the most common critical violations in Manhattan restaurants
    * **Borough Performance**: Compare how different boroughs perform in terms of inspection grades
    * **Most Improved Restaurants**: See which restaurants have improved their inspection scores the most over time
    * **Cuisine Violations**: Analyze critical violations by cuisine type and their impact on grades
    
    Use the sidebar to navigate between different sections of the dashboard.
    """)
    
    st.info("💡 **Remember**: Lower scores are better! A score of 0-13 is an 'A', 14-27 is a 'B', and scores over 28 are failing.")

# Top Cuisines by Inspection Score
elif selected == "Top Cuisines":
    st.header("Cuisines by Inspection Score")
    st.info("💡 **Note**: Lower scores are better! A score of 0-13 is an 'A', 14-27 is a 'B', and scores over 28 are failing.")
    
    # Add user controls
    st.sidebar.markdown("## Query Parameters")
    with st.sidebar.expander("Cuisine Query Options", expanded=True):
        # Minimum inspections slider
        min_inspections = st.slider(
            "Minimum Inspections Required", 
            min_value=5, 
            max_value=100, 
            value=10, 
            step=5,
            help="Only include cuisines with at least this many inspections"
        )
        
        # Number of cuisines to show
        limit = st.slider(
            "Number of Cuisines to Show", 
            min_value=5, 
            max_value=30, 
            value=10,
            help="Number of cuisine types to display in the results"
        )
        
        # Sort order
        sort_order = st.radio(
            "Sort Order",
            options=["Highest Scores First (Worst Performance)", 
                     "Lowest Scores First (Best Performance)"],
            index=0,
            help="Highest scores are worse for food safety"
        )
        
        sort_value = -1 if sort_order == "Highest Scores First (Worst Performance)" else 1
    
    # Load data with user parameters
    with st.spinner("Loading cuisine data..."):
        df_cuisines = get_top_cuisines_by_inspection_score(
            min_inspections=min_inspections, 
            limit=limit, 
            sort_order=sort_value
        )
    
    # Display data
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create horizontal bar chart
        # If lowest first, we need to sort in ascending order for the chart
        display_df = df_cuisines.sort_values('averageScore', ascending=(sort_value == 1))
        
        chart_title = "Cuisine Types by Average Inspection Score (Lower is Better)"
        if sort_value == 1:
            chart_title = "Best Performing Cuisines (Lowest Average Scores)"
        else:
            chart_title = "Worst Performing Cuisines (Highest Average Scores)"
        
        fig = px.bar(
            display_df,
            y='cuisine',
            x='averageScore',
            color='averageScore',
            color_continuous_scale=px.colors.sequential.Viridis_r,  # Reversed scale since lower is better
            labels={'averageScore': 'Average Inspection Score', 'cuisine': 'Cuisine Type'},
            height=600,
            text='averageScore'
        )
        
        fig.update_layout(
            title=chart_title,
            xaxis_title="Average Score",
            yaxis_title="Cuisine Type",
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_colorbar=dict(title="Score"),
            font=dict(family="Arial", size=14),
            plot_bgcolor='white',
            hoverlabel=dict(font_size=14, font_family="Arial")
        )
        
        fig.update_traces(
            texttemplate='%{x:.1f}',
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Average Score: %{x:.1f}<br>Inspection Count: %{customdata[0]:,}"
        )
        
        fig.update_traces(customdata=display_df[['inspectionCount']])
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Insights")
        
        # Calculate A-grade eligible cuisines
        a_grade_eligible = df_cuisines[df_cuisines['averageScore'] <= 13]
        a_grade_pct = len(a_grade_eligible) / len(df_cuisines) * 100 if len(df_cuisines) > 0 else 0
        
        st.markdown(f"""
        **From your selection:**
        - {a_grade_pct:.1f}% of cuisines have an average score in the 'A' grade range.
        - Minimum inspections per cuisine: {min_inspections}
        - Number of cuisines shown: {limit}
        
        **Score Interpretation:**
        - **Lower scores** indicate better health inspection results
        - **0-13**: 'A' grade (Good)
        - **14-27**: 'B' grade (Fair)
        - **28+**: 'C' grade or failing (Poor)
        """)
        
        st.markdown("### Data Table")
        # Format the dataframe for display
        display_df = df_cuisines.copy()
        
        # Add a grade column
        def get_grade(score):
            if score <= 13:
                return "A"
            elif score <= 27:
                return "B"
            else:
                return "C"
                
        display_df['grade'] = display_df['averageScore'].apply(get_grade)
        display_df['averageScore'] = display_df['averageScore'].apply(lambda x: f"{x:.1f}")
        display_df['inspectionCount'] = display_df['inspectionCount'].apply(lambda x: f"{x:,}")
        display_df.columns = ['Cuisine Type', 'Average Score', 'Inspection Count', 'Grade']
        
        # Reorder columns
        display_df = display_df[['Cuisine Type', 'Grade', 'Average Score', 'Inspection Count']]
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="cuisine_inspection_scores.csv",
            mime="text/csv"
        )

# Critical Violations by Borough
elif selected == "Critical Violations":
    st.header("Most Common Violations by Borough")
    
    # Add user controls
    st.sidebar.markdown("## Query Parameters")
    with st.sidebar.expander("Violation Query Options", expanded=True):
        # Borough selection
        borough_options = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
        selected_borough = st.selectbox(
            "Select Borough",
            options=borough_options,
            index=0,
            help="Borough to analyze for violations"
        )
        
        # Violation type
        violation_type = st.radio(
            "Violation Type",
            options=["Critical", "Not Critical", "All"],
            index=0,
            help="Critical violations are more likely to contribute to foodborne illness"
        )
        
        # Number of violations to show
        limit = st.slider(
            "Number of Violations to Show", 
            min_value=5, 
            max_value=30, 
            value=10,
            help="Number of violation types to display in the results"
        )
    
    # Load data with user parameters
    with st.spinner(f"Loading violation data for {selected_borough}..."):
        df_violations = get_borough_critical_violations(
            borough=selected_borough,
            limit=limit,
            critical_flag=violation_type
        )
    
    # Process data for better visualization
    df_violations['shortDescription'] = df_violations['violationDescription'].apply(
        lambda x: x[:80] + '...' if len(x) > 80 else x
    )
    
    # Create main content
    tab1, tab2 = st.tabs(["Chart View", "Data Table"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create horizontal bar chart
            fig = px.bar(
                df_violations.sort_values('count'),
                y='shortDescription',
                x='count',
                color='count',
                color_continuous_scale=px.colors.sequential.Reds,
                labels={'count': 'Occurrence Count', 'shortDescription': 'Violation Type'},
                height=600,
                text='count'
            )
            
            title_prefix = ""
            if violation_type != "All":
                title_prefix = f"Most Common {violation_type} "
            else:
                title_prefix = "Most Common "
                
            fig.update_layout(
                title=f"{title_prefix}Violations in {selected_borough} Restaurants",
                xaxis_title="Number of Violations",
                yaxis_title="Violation Type",
                yaxis={'categoryorder': 'total ascending'},
                coloraxis_colorbar=dict(title="Count"),
                font=dict(family="Arial", size=14),
                plot_bgcolor='white',
                hoverlabel=dict(font_size=14, font_family="Arial")
            )
            
            fig.update_traces(
                texttemplate='%{x:,}',
                textposition='outside',
                hovertemplate="<b>Violation Code: %{customdata[0]}</b><br>Count: %{x:,}<br>%{customdata[1]}"
            )
            
            fig.update_traces(customdata=df_violations[['violationCode', 'violationDescription']])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Pie chart of violation codes
            st.subheader("Violations by Code")
            
            fig = px.pie(
                df_violations,
                values='count',
                names='violationCode',
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4
            )
            
            fig.update_layout(
                title=f"Distribution of Violation Codes in {selected_borough}",
                font=dict(family="Arial", size=14),
                legend=dict(orientation="h", y=-0.1),
                hoverlabel=dict(font_size=14, font_family="Arial"),
                height=400
            )
            
            fig.update_traces(
                textinfo='percent+label',
                hovertemplate="<b>Code %{label}</b><br>Count: %{value}<br>Percentage: %{percent}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Insights")
            
            # Count total violations
            total_violations = df_violations['count'].sum()
            
            st.write(f"""
            **{selected_borough} Violation Analysis:**
            
            This visualization shows the most common {violation_type.lower() if violation_type != "All" else ""} 
            violations found during inspections of restaurants in {selected_borough}.
            
            **Total violations shown:** {total_violations:,}
            
            {
            "Critical violations are those that are more likely to contribute to foodborne illness and must be corrected immediately." 
            if violation_type == "Critical" else 
            "Non-critical violations are less likely to contribute directly to foodborne illness but still represent health code issues."
            if violation_type == "Not Critical" else
            "This view shows both critical and non-critical violations. Critical violations are more likely to contribute to foodborne illness."
            }
            """)
            
            st.markdown("### Top Violation Codes")
            top_codes = df_violations.groupby('violationCode')['count'].sum().reset_index()
            top_codes = top_codes.sort_values('count', ascending=False).head(5)
            
            for i, row in top_codes.iterrows():
                code = row['violationCode']
                count = row['count']
                description = df_violations[df_violations['violationCode'] == code]['violationDescription'].iloc[0]
                
                st.markdown(f"**Code {code}** ({count:,} violations)")
                st.markdown(f"<div style='margin-left: 20px; font-style: italic;'>{description}</div>", unsafe_allow_html=True)
                st.markdown("---")
    
    with tab2:
        st.subheader("Detailed Violation Data")
        
        # Format the dataframe for display
        display_df = df_violations.copy()
        display_df['count'] = display_df['count'].apply(lambda x: f"{x:,}")
        display_df = display_df[['violationCode', 'violationDescription', 'count']]
        display_df.columns = ['Violation Code', 'Violation Description', 'Count']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"{selected_borough}_violations_{violation_type}.csv",
            mime="text/csv"
        )
        
        st.markdown("### Full Descriptions")
        for i, row in df_violations.iterrows():
            with st.expander(f"Code {row['violationCode']}"):
                st.write(row['violationDescription'])

# Borough Performance
elif selected == "Borough Performance":
    st.header("Borough Performance by Grade Distribution")
    
    # Load data
    with st.spinner("Loading borough data..."):
        df_boroughs = get_borough_grade_distribution()
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Grade Distribution (%)", "Raw Counts", "Data Table"])
    
    with tab1:
        # Create stacked bar chart for percentages
        fig = go.Figure()
        
        # Add traces for each grade
        fig.add_trace(go.Bar(
            x=df_boroughs['borough'],
            y=df_boroughs['A_percent'],
            name='A Grade',
            marker_color='#2ecc71',
            text=[f"{x:.1f}%" for x in df_boroughs['A_percent']],
            textposition='inside',
            hovertemplate="<b>%{x}</b><br>A Grade: %{y:.1f}%<extra></extra>"
        ))
        
        fig.add_trace(go.Bar(
            x=df_boroughs['borough'],
            y=df_boroughs['B_percent'],
            name='B Grade',
            marker_color='#f39c12',
            text=[f"{x:.1f}%" for x in df_boroughs['B_percent']],
            textposition='inside',
            hovertemplate="<b>%{x}</b><br>B Grade: %{y:.1f}%<extra></extra>"
        ))
        
        fig.add_trace(go.Bar(
            x=df_boroughs['borough'],
            y=df_boroughs['C_percent'],
            name='C Grade',
            marker_color='#e74c3c',
            text=[f"{x:.1f}%" for x in df_boroughs['C_percent']],
            textposition='inside',
            hovertemplate="<b>%{x}</b><br>C Grade: %{y:.1f}%<extra></extra>"
        ))
        
        # Set layout
        fig.update_layout(
            title="Grade Distribution by Borough (Percentage)",
            xaxis_title="Borough",
            yaxis_title="Percentage of Restaurants",
            barmode='stack',
            yaxis=dict(
                tickformat='.1f',
                ticksuffix='%'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            font=dict(family="Arial", size=14),
            plot_bgcolor='white',
            hoverlabel=dict(font_size=14, font_family="Arial")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Create stacked bar chart for raw counts
        fig = go.Figure()
        
        # Add traces for each grade
        fig.add_trace(go.Bar(
            x=df_boroughs['borough'],
            y=df_boroughs['A_count'],
            name='A Grade',
            marker_color='#2ecc71',
            text=df_boroughs['A_count'],
            textposition='inside',
            hovertemplate="<b>%{x}</b><br>A Grade: %{y:,}<extra></extra>"
        ))
        
        fig.add_trace(go.Bar(
            x=df_boroughs['borough'],
            y=df_boroughs['B_count'],
            name='B Grade',
            marker_color='#f39c12',
            text=df_boroughs['B_count'],
            textposition='inside',
            hovertemplate="<b>%{x}</b><br>B Grade: %{y:,}<extra></extra>"
        ))
        
        fig.add_trace(go.Bar(
            x=df_boroughs['borough'],
            y=df_boroughs['C_count'],
            name='C Grade',
            marker_color='#e74c3c',
            text=df_boroughs['C_count'],
            textposition='inside',
            hovertemplate="<b>%{x}</b><br>C Grade: %{y:,}<extra></extra>"
        ))
        
        # Set layout
        fig.update_layout(
            title="Grade Distribution by Borough (Raw Counts)",
            xaxis_title="Borough",
            yaxis_title="Number of Restaurants",
            barmode='stack',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            font=dict(family="Arial", size=14),
            plot_bgcolor='white',
            hoverlabel=dict(font_size=14, font_family="Arial")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional chart: total inspections by borough
        fig = px.bar(
            df_boroughs,
            x='borough',
            y='total',
            color='total',
            color_continuous_scale=px.colors.sequential.Blues,
            labels={'total': 'Total Inspections', 'borough': 'Borough'},
            height=400,
            text='total'
        )
        
        fig.update_layout(
            title="Total Inspections by Borough",
            xaxis_title="Borough",
            yaxis_title="Number of Inspections",
            font=dict(family="Arial", size=14),
            plot_bgcolor='white',
            hoverlabel=dict(font_size=14, font_family="Arial")
        )
        
        fig.update_traces(
            texttemplate='%{y:,}',
            textposition='outside'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Display data table with formatted columns
        st.subheader("Grade Distribution Data")
        
        # Format the dataframe for display
        display_df = df_boroughs.copy()
        display_df['A_percent'] = display_df['A_percent'].apply(lambda x: f"{x:.1f}%")
        display_df['B_percent'] = display_df['B_percent'].apply(lambda x: f"{x:.1f}%")
        display_df['C_percent'] = display_df['C_percent'].apply(lambda x: f"{x:.1f}%")
        display_df['total'] = display_df['total'].apply(lambda x: f"{x:,}")
        display_df.columns = ['Borough', 'Total Inspections', 'A Count', 'B Count', 'C Count', 
                             'A Grade %', 'B Grade %', 'C Grade %']
        
        st.dataframe(display_df, use_container_width=True)
    
    # Insights section
    st.subheader("Insights")
    st.write("""
    This section compares how restaurants in different boroughs perform in health inspections.
    
    - **A Grade**: Scores between 0-13 points
    - **B Grade**: Scores between 14-27 points
    - **C Grade**: Scores of 28+ points (failing)
    
    Key observations:
    - Staten Island has the highest percentage of A grades
    - The Bronx has the highest percentage of C grades
    - Manhattan has the most restaurant inspections overall
    """)

# Most Improved Restaurants
elif selected == "Most Improved":
    st.header("Most Improved Restaurants Over Time")
    st.info("💡 **Remember**: Lower scores are better! Improvement is defined as a reduction in score over time. Higher improvement values indicate greater progress.")
    
    # Add user controls
    st.sidebar.markdown("## Query Parameters")
    with st.sidebar.expander("Improvement Query Options", expanded=True):
        # Borough filter
        borough_options = ['All Boroughs', 'Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
        selected_borough = st.selectbox(
            "Filter by Borough",
            options=borough_options,
            index=0,
            help="Select a borough or view all"
        )
        
        # Minimum improvement
        min_improvement = st.slider(
            "Minimum Score Improvement", 
            min_value=0, 
            max_value=50, 
            value=0,
            help="Only show restaurants with at least this much improvement"
        )
        
        # Minimum inspections
        min_inspections = st.slider(
            "Minimum Number of Inspections", 
            min_value=2, 
            max_value=10, 
            value=2,
            help="Only include restaurants with at least this many inspections"
        )
        
        # Number of restaurants to show
        limit = st.slider(
            "Number of Restaurants to Show", 
            min_value=5, 
            max_value=50, 
            value=20,
            help="Number of restaurants to display in the results"
        )
    
    # Process borough filter
    filter_borough = None if selected_borough == 'All Boroughs' else selected_borough
    
    # Load data with user parameters
    with st.spinner("Loading improvement data..."):
        df_improved = get_most_improved_restaurants(
            min_inspections=min_inspections,
            limit=limit,
            borough=filter_borough,
            min_improvement=min_improvement
        )
    
    # Check if data was found
    if len(df_improved) == 0:
        st.warning(f"No restaurants found matching the criteria. Try adjusting the filters.")
        st.stop()
    
    # Create visualization
    tab1, tab2 = st.tabs(["Improvement Charts", "Restaurant Details"])
    
    with tab1:
        # Create custom visualization showing before/after scores
        fig = go.Figure()
        
        # Add bars for first scores
        fig.add_trace(go.Bar(
            x=df_improved['name'],
            y=df_improved['firstScore'],
            name='First Inspection Score',
            marker_color='#e74c3c',
            hovertemplate="<b>%{x}</b><br>First Score: %{y}<br>Borough: %{customdata[0]}<br>Cuisine: %{customdata[1]}<extra></extra>"
        ))
        
        # Add bars for last scores
        fig.add_trace(go.Bar(
            x=df_improved['name'],
            y=df_improved['lastScore'],
            name='Latest Inspection Score',
            marker_color='#2ecc71',
            hovertemplate="<b>%{x}</b><br>Latest Score: %{y}<br>Borough: %{customdata[0]}<br>Cuisine: %{customdata[1]}<extra></extra>"
        ))
        
        # Add custom data for hover info
        fig.update_traces(customdata=df_improved[['borough', 'cuisine']])
        
        # Add title with filter information
        title = "Most Improved Restaurants: First vs. Latest Inspection Scores"
        if filter_borough:
            title += f" in {filter_borough}"
        
        # Set layout
        fig.update_layout(
            title=title,
            xaxis_title="Restaurant",
            yaxis_title="Inspection Score",
            barmode='group',
            xaxis=dict(
                tickangle=-45,
                tickfont=dict(size=10)
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            font=dict(family="Arial", size=14),
            plot_bgcolor='white',
            hoverlabel=dict(font_size=14, font_family="Arial"),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add improvement chart
        fig = px.bar(
            df_improved.sort_values('improvement', ascending=False),
            x='name',
            y='improvement',
            color='improvement',
            color_continuous_scale=px.colors.sequential.Blues,
            labels={'improvement': 'Score Improvement', 'name': 'Restaurant Name'},
            height=500,
            text='improvement'
        )
        
        fig.update_layout(
            title="Score Improvement by Restaurant (Higher is Better)",
            xaxis_title="Restaurant",
            yaxis_title="Score Improvement",
            xaxis=dict(
                tickangle=-45,
                tickfont=dict(size=10)
            ),
            font=dict(family="Arial", size=14),
            plot_bgcolor='white',
            hoverlabel=dict(font_size=14, font_family="Arial")
        )
        
        fig.update_traces(
            texttemplate='%{y}',
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Improvement: %{y}<br>Borough: %{customdata[0]}<br>Cuisine: %{customdata[1]}<extra></extra>"
        )
        
        fig.update_traces(customdata=df_improved[['borough', 'cuisine']])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Borough distribution
        if not filter_borough:
            st.subheader("Borough Distribution")
            # Create pie chart of most improved restaurants by borough
            borough_counts = df_improved['borough'].value_counts().reset_index()
            borough_counts.columns = ['Borough', 'Count']
            
            fig = px.pie(
                borough_counts,
                values='Count',
                names='Borough',
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4
            )
            
            fig.update_layout(
                title="Most Improved Restaurants by Borough",
                font=dict(family="Arial", size=14),
                hoverlabel=dict(font_size=14, font_family="Arial")
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Restaurant Details")
        
        # Format the dataframe for display
        display_df = df_improved.copy()
        display_df = display_df.rename(columns={
            'name': 'Restaurant Name',
            'borough': 'Borough',
            'cuisine': 'Cuisine Type',
            'firstScore': 'Initial Score',
            'lastScore': 'Latest Score',
            'improvement': 'Improvement',
            'inspectionCount': 'Number of Inspections'
        })
        
        # Calculate grades
        def get_grade(score):
            if score <= 13:
                return "A"
            elif score <= 27:
                return "B"
            else:
                return "C"
                
        display_df['Initial Grade'] = display_df['Initial Score'].apply(get_grade)
        display_df['Latest Grade'] = display_df['Latest Score'].apply(get_grade)
        
        # Reorder columns
        display_df = display_df[[
            'Restaurant Name', 'Borough', 'Cuisine Type', 
            'Initial Score', 'Initial Grade', 'Latest Score', 'Latest Grade',
            'Improvement', 'Number of Inspections'
        ]]
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="most_improved_restaurants.csv",
            mime="text/csv"
        )
        
        # Restaurant spotlights
        st.subheader("Restaurant Spotlight")
        
        # Get top 5 most improved
        top_improved = df_improved.head(5)
        
        for i, row in top_improved.iterrows():
            with st.expander(f"{row['name']} - Improved by {row['improvement']} points"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Borough:** {row['borough']}")
                    st.markdown(f"**Cuisine:** {row['cuisine']}")
                    st.markdown(f"**Inspections:** {row['inspectionCount']}")
                
                with col2:
                    # Calculate grades
                    first_grade = get_grade(row['firstScore'])
                    last_grade = get_grade(row['lastScore'])
                    
                    st.markdown(f"**First Score:** {row['firstScore']} (Grade {first_grade})")
                    st.markdown(f"**Latest Score:** {row['lastScore']} (Grade {last_grade})")
                    st.markdown(f"**Improvement:** {row['improvement']} points")
                
                # Add a progress visualization
                st.markdown("### Improvement Visualization")
                
                max_score = max(row['firstScore'], 50)  # Cap at 50 for visualization purposes
                
                # Create a progress chart
                progress_data = pd.DataFrame({
                    'Stage': ['First Inspection', 'Latest Inspection'],
                    'Score': [row['firstScore'], row['lastScore']],
                    'Color': ['#e74c3c', '#2ecc71']
                })
                
                fig = px.bar(
                    progress_data,
                    x='Score',
                    y='Stage',
                    orientation='h',
                    color='Color',
                    color_discrete_map={'#e74c3c': '#e74c3c', '#2ecc71': '#2ecc71'},
                    height=150,
                    range_x=[0, max_score]
                )
                
                # Add grade bands to the background
                fig.add_shape(
                    type="rect",
                    x0=0, x1=13,
                    y0=-0.5, y1=1.5,
                    fillcolor="#2ecc71",
                    opacity=0.2,
                    layer="below",
                    line_width=0
                )
                
                fig.add_shape(
                    type="rect",
                    x0=14, x1=27,
                    y0=-0.5, y1=1.5,
                    fillcolor="#f39c12",
                    opacity=0.2,
                    layer="below",
                    line_width=0
                )
                
                fig.add_shape(
                    type="rect",
                    x0=28, x1=max_score,
                    y0=-0.5, y1=1.5,
                    fillcolor="#e74c3c",
                    opacity=0.2,
                    layer="below",
                    line_width=0
                )
                
                # Add labels for grade bands
                fig.add_annotation(
                    x=6.5, y=1.4,
                    text="A Grade (0-13)",
                    showarrow=False,
                    font=dict(color="#2c3e50", size=10)
                )
                
                fig.add_annotation(
                    x=20.5, y=1.4,
                    text="B Grade (14-27)",
                    showarrow=False,
                    font=dict(color="#2c3e50", size=10)
                )
                
                fig.add_annotation(
                    x=39, y=1.4,
                    text="C Grade (28+)",
                    showarrow=False,
                    font=dict(color="#2c3e50", size=10)
                )
                
                # Update layout
                fig.update_layout(
                    showlegend=False,
                    xaxis_title="Inspection Score (Lower is Better)",
                    xaxis=dict(
                        showgrid=False,
                        zeroline=False
                    ),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False
                    ),
                    plot_bgcolor="white",
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                
                fig.update_traces(
                    texttemplate='%{x}',
                    textposition='outside',
                    hovertemplate="<b>%{y}</b><br>Score: %{x}<extra></extra>",
                    width=0.6
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Cuisine Violations
elif selected == "Cuisine Violations":
    st.header("Critical Violations by Cuisine Type")
    
    # Load data
    with st.spinner("Loading cuisine violation data..."):
        df_cuisine_violations = get_critical_violations_by_cuisine()
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Total Violations", "Violations by Grade", "Data Table"])
    
    with tab1:
        # Create horizontal bar chart for total violations
        df_sorted = df_cuisine_violations.sort_values('totalCriticalViolations')
        
        fig = px.bar(
            df_sorted,
            y='cuisine',
            x='totalCriticalViolations',
            color='totalCriticalViolations',
            color_continuous_scale=px.colors.sequential.Reds,
            labels={'totalCriticalViolations': 'Total Critical Violations', 'cuisine': 'Cuisine Type'},
            height=700,
            text='totalCriticalViolations'
        )
        
        fig.update_layout(
            title="Total Critical Violations by Cuisine Type",
            xaxis_title="Number of Critical Violations",
            yaxis_title="Cuisine Type",
            font=dict(family="Arial", size=14),
            plot_bgcolor='white',
            hoverlabel=dict(font_size=14, font_family="Arial")
        )
        
        fig.update_traces(
            texttemplate='%{x:,}',
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Critical Violations: %{x:,}<extra></extra>"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Create stacked bar chart for violations by grade
        fig = go.Figure()
        
        # Add traces for each grade
        fig.add_trace(go.Bar(
            y=df_cuisine_violations['cuisine'],
            x=df_cuisine_violations['A_violations'],
            name='A Grade Violations',
            marker_color='#2ecc71',
            orientation='h',
            hovertemplate="<b>%{y}</b><br>A Grade Violations: %{x:,}<extra></extra>"
        ))
        
        fig.add_trace(go.Bar(
            y=df_cuisine_violations['cuisine'],
            x=df_cuisine_violations['B_violations'],
            name='B Grade Violations',
            marker_color='#f39c12',
            orientation='h',
            hovertemplate="<b>%{y}</b><br>B Grade Violations: %{x:,}<extra></extra>"
        ))
        
        fig.add_trace(go.Bar(
            y=df_cuisine_violations['cuisine'],
            x=df_cuisine_violations['C_violations'],
            name='C Grade Violations',
            marker_color='#e74c3c',
            orientation='h',
            hovertemplate="<b>%{y}</b><br>C Grade Violations: %{x:,}<extra></extra>"
        ))
        
        # Set layout
        fig.update_layout(
            title="Critical Violations by Cuisine Type and Grade",
            xaxis_title="Number of Critical Violations",
            yaxis_title="Cuisine Type",
            barmode='stack',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            font=dict(family="Arial", size=14),
            plot_bgcolor='white',
            hoverlabel=dict(font_size=14, font_family="Arial"),
            height=700,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Percentage breakdown
        st.subheader("Percentage of Violations by Grade")
        
        # Create a dataframe with percentage breakdowns
        pct_df = df_cuisine_violations.copy()
        
        # Calculate percentages
        for i, row in pct_df.iterrows():
            total = row['totalCriticalViolations']
            pct_df.loc[i, 'A_pct'] = (row['A_violations'] / total * 100) if total > 0 else 0
            pct_df.loc[i, 'B_pct'] = (row['B_violations'] / total * 100) if total > 0 else 0
            pct_df.loc[i, 'C_pct'] = (row['C_violations'] / total * 100) if total > 0 else 0
        
        # Create 100% stacked bar chart
        fig = go.Figure()
        
        # Add traces for each grade percentage
        fig.add_trace(go.Bar(
            y=pct_df['cuisine'],
            x=pct_df['A_pct'],
            name='A Grade %',
            marker_color='#2ecc71',
            orientation='h',
            hovertemplate="<b>%{y}</b><br>A Grade: %{x:.1f}%<extra></extra>"
        ))
        
        fig.add_trace(go.Bar(
            y=pct_df['cuisine'],
            x=pct_df['B_pct'],
            name='B Grade %',
            marker_color='#f39c12',
            orientation='h',
            hovertemplate="<b>%{y}</b><br>B Grade: %{x:.1f}%<extra></extra>"
        ))
        
        fig.add_trace(go.Bar(
            y=pct_df['cuisine'],
            x=pct_df['C_pct'],
            name='C Grade %',
            marker_color='#e74c3c',
            orientation='h',
            hovertemplate="<b>%{y}</b><br>C Grade: %{x:.1f}%<extra></extra>"
        ))
        
        # Set layout
        fig.update_layout(
            title="Percentage of Critical Violations by Grade for Each Cuisine",
            xaxis_title="Percentage of Violations",
            yaxis_title="Cuisine Type",
            barmode='stack',
            xaxis=dict(
                tickformat='.1f',
                ticksuffix='%'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            font=dict(family="Arial", size=14),
            plot_bgcolor='white',
            hoverlabel=dict(font_size=14, font_family="Arial"),
            height=700,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Display data table
        st.subheader("Critical Violations Data by Cuisine")
        
        # Format the dataframe for display
        display_df = df_cuisine_violations.copy()
        display_df['totalCriticalViolations'] = display_df['totalCriticalViolations'].apply(lambda x: f"{x:,}")
        display_df['A_violations'] = display_df['A_violations'].apply(lambda x: f"{x:,}")
        display_df['B_violations'] = display_df['B_violations'].apply(lambda x: f"{x:,}")
        display_df['C_violations'] = display_df['C_violations'].apply(lambda x: f"{x:,}")
        display_df.columns = ['Cuisine Type', 'Total Critical Violations', 
                             'A Grade Violations', 'B Grade Violations', 'C Grade Violations']
        
        st.dataframe(display_df, use_container_width=True)
    
    # Insights section
    st.subheader("Insights")
    st.write("""
    This section analyzes critical violations by cuisine type and their impact on grades.
    
    Critical violations are more likely to contribute to foodborne illness and must be corrected immediately.
    
    Key observations:
    - American cuisine has the highest number of critical violations overall
    - Chinese cuisine has a higher proportion of C grade violations compared to other cuisines
    - Pizza places tend to have a higher proportion of A grade violations among their critical violations
    
    This data helps identify which cuisine types may benefit from targeted food safety interventions.
    """)

# About page
else:
    st.header("About This Dashboard")
    
    st.write("""
    ## NYC Restaurant Inspection Data Dashboard
    
    This dashboard visualizes data from the NYC Department of Health and Mental Hygiene's restaurant inspection program.
    
    ### Understanding the Data
    
    The NYC restaurant letter grading program began July 27, 2010. Restaurants receive grades based on their inspection scores:
    
    - **A Grade**: Scores between 0-13 points
    - **B Grade**: Scores between 14-27 points
    - **C Grade**: Scores of 28+ points (failing)
    
    Lower scores are better. The inspection process allows for a two-step inspection process, providing restaurants who do not receive an "A" on their initial inspection an opportunity to be re-inspected.
    
    ### Data Collections
    
    The dashboard uses three MongoDB collections:
    
    1. **establishments**: Contains information about restaurants (name, location, cuisine type)
    2. **inspections**: Contains inspection records (date, score, grade)
    3. **violations**: Contains specific violation records linked to inspections
    
    ### Analysis Queries
    
    The dashboard provides insights based on five main queries:
    
    1. **Top Cuisines by Inspection Score**: Identifies cuisines with the highest average inspection scores
    2. **Critical Violations in Manhattan**: Shows the most common critical violations in Manhattan restaurants
    3. **Borough Performance**: Compares how different boroughs perform in terms of inspection grades
    4. **Most Improved Restaurants**: Highlights restaurants that have improved their scores the most over time
    5. **Cuisine Violations**: Analyzes critical violations by cuisine type and their impact on grades
    
    ### Technical Implementation
    
    This dashboard is built with:
    
    - **Streamlit**: For the web application framework
    - **MongoDB**: For data storage and aggregation queries
    - **Plotly**: For interactive data visualizations
    
    The MongoDB aggregation pipelines perform complex data transformations and analyses, which are then visualized in the dashboard.
    """)
    
