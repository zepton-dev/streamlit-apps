import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="NPI Survey Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #1E1E2E;
        color: #E0E0E0;
        font-family: 'Inter', sans-serif;
    }

    /* Main header */
    .main-header {
        font-size: 2.5rem;
        color: #60A5FA;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        color: #34D399;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    /* Metric cards */
    .metric-container {
        background-color: #2A2A3C;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        color: #E0E0E0;
        border: 1px solid #3B3B4F;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #252537;
        color: #E0E0E0;
    }

    /* File uploader */
    .stFileUploader {
        background-color: #2A2A3C;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #3B3B4F;
    }

    /* Buttons */
    .stButton>button {
        background-color: #60A5FA;
        color: #FFFFFF;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3B82F6;
        transform: translateY(-2px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    /* Dataframe styling */
    .stDataFrame {
        background-color: #2A2A3C;
        border-radius: 8px;
        border: 1px solid #3B3B4F;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab"] {
        background-color: #2A2A3C;
        color: #E0E0E0;
        border-radius: 8px 8px 0 0;
        margin-right: 4px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #60A5FA;
        color: #FFFFFF;
        font-weight: 500;
    }

    /* Info and error messages */
    .stAlert {
        background-color: #2A2A3C;
        color: #E0E0E0;
        border-radius: 8px;
        border: 1px solid #3B3B4F;
    }

    /* Download button */
    .stDownloadButton>button {
        background-color: #34D399;
        color: #FFFFFF;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stDownloadButton>button:hover {
        background-color: #2BB678;
        transform: translateY(-2px);
    }

    /* Spinner */
    .stSpinner {
        color: #60A5FA;
    }

    /* Ensure text visibility */
    p, h1, h2, h3, h4, h5, h6, div, span {
        color: #E0E0E0 !important;
    }

    /* Input fields */
    .stSelectbox, .stTimeInput {
        background-color: #2A2A3C;
        border-radius: 8px;
        border: 1px solid #3B3B4F;
        color: #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)

# Functions from your notebook
@st.cache_data
def preprocess(npi_file, survey_file):
    try:
        npi = pd.read_excel(npi_file)
        survey = pd.read_excel(survey_file)
        npi.dropna(inplace=True)
        survey.dropna(inplace=True)
        survey['attempt_hour'] = survey['Attempt Time'].apply(lambda t: t.hour)
        survey['attempt_minute'] = survey['Attempt Time'].apply(lambda t: t.minute)
        survey.drop('Attempt Time', axis=1, inplace=True)
        npi['Login Date'] = npi['Login Time'].dt.date
        npi['Login Hour'] = npi['Login Time'].dt.hour
        npi['Login Minute'] = npi['Login Time'].dt.minute
        npi['Logout Date'] = npi['Logout Time'].dt.date
        npi['Logout Hour'] = npi['Logout Time'].dt.hour
        npi['Logout Minute'] = npi['Logout Time'].dt.minute
        npi.drop(["Login Time", "Logout Time"], inplace=True, axis=1)
        return npi, survey
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return None, None

def get_active_npis_for_survey(npi_df, survey_df, survey_id, time_str):
    try:
        target_hour = int(time_str.split(':')[0])
        survey_npis = survey_df[survey_df['Survey ID'] == survey_id]['NPI'].unique()
        if len(survey_npis) == 0:
            return pd.DataFrame(), f"No NPIs found for Survey ID {survey_id}"
        survey_participants = npi_df[npi_df['NPI'].isin(survey_npis)].copy()
        active_npis = []
        for _, row in survey_participants.iterrows():
            login_hour = row['Login Hour']
            logout_hour = row['Logout Hour']
            if login_hour <= logout_hour:
                if login_hour <= target_hour <= logout_hour:
                    active_npis.append(row)
            else:
                if target_hour >= login_hour or target_hour <= logout_hour:
                    active_npis.append(row)
        if not active_npis:
            return pd.DataFrame(), f"No active NPIs found for Survey ID {survey_id} at {time_str}"
        result_df = pd.DataFrame(active_npis)
        display_cols = ['NPI', 'Login Date', 'Login Hour', 'Login Minute', 
                       'Logout Date', 'Logout Hour', 'Logout Minute',
                       'Speciality', 'Region', 'Usage Time (mins)', 'State']
        result_df = result_df[display_cols]
        success_msg = f"Found {len(result_df)} active NPIs for Survey ID {survey_id} during hour {target_hour}:00-{target_hour+1}:00"
        return result_df, success_msg
    except Exception as e:
        return pd.DataFrame(), f"Error: {str(e)}"

def workflow(npi_file, survey_file, survey_id, time_str):
    npi_df, survey_df = preprocess(npi_file, survey_file)
    if npi_df is None or survey_df is None:
        return None, None, "Error in preprocessing"
    result_df, message = get_active_npis_for_survey(npi_df, survey_df, survey_id, time_str)
    return result_df, (npi_df, survey_df), message


def create_state_distribution_plot(npi_df):
    state_counts = npi_df['State'].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.set_facecolor('#1E1E2E')
    
    # Modern color palette
    colors = ['#60A5FA', '#34D399', '#F472B6', '#FBBF24', '#A78BFA', '#FB7185', '#38BDF8', '#4ADE80', '#F59E0B', '#8B5CF6']
    
    # Bar chart with enhanced styling
    bars = ax1.bar(state_counts.head(10).index, state_counts.head(10).values, 
                   color=colors[:len(state_counts.head(10))], edgecolor='white', linewidth=2, alpha=0.8)
    ax1.set_title('State Distribution (Top 10)', fontsize=16, fontweight='bold', color='white', pad=20)
    ax1.set_xlabel('State', fontsize=12, color='white')
    ax1.set_ylabel('Number of NPIs', fontsize=12, color='white')
    ax1.tick_params(axis='x', rotation=45, labelsize=10, colors='white')
    ax1.tick_params(axis='y', labelsize=10, colors='white')
    ax1.set_facecolor('#2A2A3C')
    ax1.grid(axis='y', alpha=0.3, color='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', 
                fontsize=10, color='white')
    
    # Pie chart with enhanced styling
    wedges, texts, autotexts = ax2.pie(state_counts.head(8).values, labels=state_counts.head(8).index,
                                      autopct='%1.1f%%', startangle=90, 
                                      colors=colors[:len(state_counts.head(8))],
                                      wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2),
                                      textprops={'fontsize': 10, 'color': 'white', 'fontweight': 'bold'})
    ax2.set_title('State Distribution (Top 8)', fontsize=16, fontweight='bold', color='white', pad=20)
    ax2.set_facecolor('#2A2A3C')
    
    # Style the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    return fig

def create_region_distribution_plot(npi_df):
    region_counts = npi_df['Region'].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.set_facecolor('#1E1E2E')
    
    # Modern gradient colors
    colors = ['#60A5FA', '#34D399', '#F472B6', '#FBBF24', '#A78BFA'][:len(region_counts)]
    
    # Enhanced bar chart
    bars = ax1.bar(region_counts.index, region_counts.values, color=colors, 
                   edgecolor='white', linewidth=2, alpha=0.8)
    ax1.set_title('NPI Distribution by Region', fontsize=16, fontweight='bold', color='white', pad=20)
    ax1.set_xlabel('Region', fontsize=12, color='white')
    ax1.set_ylabel('Number of NPIs', fontsize=12, color='white')
    ax1.tick_params(axis='x', rotation=45, labelsize=11, colors='white')
    ax1.tick_params(axis='y', labelsize=10, colors='white')
    ax1.set_facecolor('#2A2A3C')
    ax1.grid(axis='y', alpha=0.3, color='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add value labels with enhanced styling
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', 
                fontsize=12, color='white',                bbox=dict(boxstyle='round,pad=0.3', facecolor=(1, 1, 1, 0.1), 
                         edgecolor='none'))
    
    # Enhanced donut chart
    wedges, texts, autotexts = ax2.pie(region_counts.values, labels=region_counts.index, 
                                      autopct='%1.1f%%', startangle=90, 
                                      colors=colors, wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2),
                                      textprops={'fontsize': 12, 'color': 'white', 'fontweight': 'bold'})
    ax2.set_title('Regional Distribution (Donut Chart)', fontsize=16, fontweight='bold', color='white', pad=20)
    ax2.set_facecolor('#2A2A3C')
    
    # Style the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    plt.tight_layout()
    return fig

def create_specialty_distribution_plot(npi_df):
    specialty_counts = npi_df['Speciality'].value_counts().head(15)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    fig.set_facecolor('#1E1E2E')
    
    # Modern color palette
    colors = ['#60A5FA', '#34D399', '#F472B6', '#FBBF24', '#A78BFA', '#FB7185', '#38BDF8', 
              '#4ADE80', '#F59E0B', '#8B5CF6', '#EC4899', '#10B981', '#3B82F6', '#EF4444', '#6366F1']
    
    # Enhanced horizontal bar chart
    bars = ax1.barh(range(len(specialty_counts)), specialty_counts.values, 
                   color=colors[:len(specialty_counts)], edgecolor='white', linewidth=1.5, alpha=0.8)
    ax1.set_yticks(range(len(specialty_counts)))
    ax1.set_yticklabels([label[:35] + '...' if len(label) > 35 else label 
                        for label in specialty_counts.index], fontsize=10, color='white')
    ax1.set_title('Top 15 Specialties by NPI Count', fontsize=16, fontweight='bold', color='white', pad=20)
    ax1.set_xlabel('Number of NPIs', fontsize=12, color='white')
    ax1.tick_params(axis='x', labelsize=10, colors='white')
    ax1.set_facecolor('#2A2A3C')
    ax1.grid(axis='x', alpha=0.3, color='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add value labels with enhanced styling
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                f'{int(width)}', ha='left', va='center', fontweight='bold', 
                fontsize=10, color='white',                bbox=dict(boxstyle='round,pad=0.2', facecolor=(1, 1, 1, 0.1), 
                         edgecolor='none'))
    
    # Enhanced pie chart for top 10
    top_10_specialties = specialty_counts.head(10)
    wedges, texts, autotexts = ax2.pie(top_10_specialties.values, 
                                      labels=[label[:25] + '...' if len(label) > 25 else label 
                                             for label in top_10_specialties.index], 
                                      autopct='%1.1f%%', startangle=90, 
                                      colors=colors[:len(top_10_specialties)],
                                      wedgeprops=dict(width=0.7, edgecolor='white', linewidth=1.5),
                                      textprops={'fontsize': 9, 'color': 'white', 'fontweight': 'bold'})
    ax2.set_title('Top 10 Specialties Distribution', fontsize=16, fontweight='bold', color='white', pad=20)
    ax2.set_facecolor('#2A2A3C')
    
    # Style the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    plt.tight_layout()
    return fig

def create_hourly_activity_plot(npi_df):
    hourly_activity = []
    for hour in range(24):
        active_count = 0
        for _, row in npi_df.iterrows():
            login_hour = row['Login Hour']
            logout_hour = row['Logout Hour']
            if login_hour <= logout_hour:
                if login_hour <= hour <= logout_hour:
                    active_count += 1
            else:
                if hour >= login_hour or hour <= logout_hour:
                    active_count += 1
        hourly_activity.append(active_count)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.set_facecolor('#1E1E2E')
    
    # Enhanced line plot with area fill
    ax1.plot(range(24), hourly_activity, marker='o', linewidth=3, 
            markersize=8, color='#60A5FA', markerfacecolor='#34D399', 
            markeredgecolor='white', markeredgewidth=2)
    ax1.fill_between(range(24), hourly_activity, alpha=0.3, color='#60A5FA')
    ax1.set_title('Active NPIs Throughout the Day', fontsize=18, fontweight='bold', color='white', pad=20)
    ax1.set_xlabel('Hour of Day', fontsize=12, color='white')
    ax1.set_ylabel('Number of Active NPIs', fontsize=12, color='white')
    ax1.set_xticks(range(24))
    ax1.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, fontsize=10, color='white')
    ax1.tick_params(axis='y', labelsize=10, colors='white')
    ax1.set_facecolor('#2A2A3C')
    ax1.grid(True, alpha=0.3, color='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Enhanced peak annotation
    max_activity = max(hourly_activity)
    peak_hour = hourly_activity.index(max_activity)
    ax1.annotate(f'Peak Activity\n{max_activity} NPIs at {peak_hour:02d}:00', 
                xy=(peak_hour, max_activity), xytext=(peak_hour+3, max_activity+50),
                arrowprops=dict(arrowstyle='->', color='#F472B6', lw=2),
                fontsize=11, fontweight='bold', color='#F472B6',                bbox=dict(boxstyle='round,pad=0.5', facecolor=(244/255, 114/255, 182/255, 0.2), 
                         edgecolor='#F472B6', linewidth=1))
    
    # Enhanced bar chart with gradient colors
    colors = ['#FF6B6B' if activity == max_activity else '#34D399' if activity > np.mean(hourly_activity) 
              else '#60A5FA' for activity in hourly_activity]
    bars = ax2.bar(range(24), hourly_activity, color=colors, edgecolor='white', 
                   linewidth=1.5, alpha=0.8)
    ax2.set_title('Active NPIs by Hour (Enhanced Bar Chart)', fontsize=18, fontweight='bold', color='white', pad=20)
    ax2.set_xlabel('Hour of Day', fontsize=12, color='white')
    ax2.set_ylabel('Number of Active NPIs', fontsize=12, color='white')
    ax2.set_xticks(range(24))
    ax2.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, fontsize=10, color='white')
    ax2.tick_params(axis='y', labelsize=10, colors='white')
    ax2.set_facecolor('#2A2A3C')
    ax2.grid(axis='y', alpha=0.3, color='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add value labels on bars for peak hours
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > np.mean(hourly_activity):  # Only show labels for above-average hours
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', 
                    fontsize=9, color='white')
    
    plt.tight_layout()
    return fig

def create_survey_participation_plot(survey_df):
    survey_participation = survey_df.groupby('Survey ID')['NPI'].nunique().reset_index()
    survey_participation.columns = ['Survey_ID', 'Unique_NPIs']
    survey_participation = survey_participation.sort_values('Unique_NPIs', ascending=False)
    most_participated = survey_participation.head(6)
    least_participated = survey_participation.tail(6)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.set_facecolor('#1E1E2E')
    
    # Enhanced colors
    colors_high = ['#34D399', '#10B981', '#059669', '#047857', '#065F46', '#064E3B']
    colors_low = ['#EF4444', '#DC2626', '#B91C1C', '#991B1B', '#7F1D1D', '#6F1B1B']
    
    # Top participated surveys - vertical bars
    bars1 = ax1.bar(range(len(most_participated)), most_participated['Unique_NPIs'], 
                   color=colors_high, alpha=0.8, edgecolor='white', linewidth=2)
    ax1.set_title('Top 6 Most Participated Surveys', fontsize=14, fontweight='bold', color='white', pad=15)
    ax1.set_xlabel('Survey ID', fontsize=11, color='white')
    ax1.set_ylabel('Number of Unique NPIs', fontsize=11, color='white')
    ax1.set_xticks(range(len(most_participated)))
    ax1.set_xticklabels(most_participated['Survey_ID'], rotation=45, fontsize=10, color='white')
    ax1.tick_params(axis='y', labelsize=10, colors='white')
    ax1.set_facecolor('#2A2A3C')
    ax1.grid(axis='y', alpha=0.3, color='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', 
                fontsize=10, color='white',                bbox=dict(boxstyle='round,pad=0.2', facecolor=(52/255, 211/255, 153/255, 0.2), 
                         edgecolor='none'))
    
    # Top participated surveys - horizontal bars
    bars2 = ax2.barh(range(len(most_participated)), most_participated['Unique_NPIs'], 
                    color=colors_high, alpha=0.8, edgecolor='white', linewidth=2)
    ax2.set_title('Top 6 Most Participated (Horizontal)', fontsize=14, fontweight='bold', color='white', pad=15)
    ax2.set_xlabel('Number of Unique NPIs', fontsize=11, color='white')
    ax2.set_ylabel('Survey ID', fontsize=11, color='white')
    ax2.set_yticks(range(len(most_participated)))
    ax2.set_yticklabels(most_participated['Survey_ID'], fontsize=10, color='white')
    ax2.tick_params(axis='x', labelsize=10, colors='white')
    ax2.set_facecolor('#2A2A3C')
    ax2.grid(axis='x', alpha=0.3, color='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Least participated surveys
    bars3 = ax3.bar(range(len(least_participated)), least_participated['Unique_NPIs'], 
                   color=colors_low, alpha=0.8, edgecolor='white', linewidth=2)
    ax3.set_title('Bottom 6 Least Participated Surveys', fontsize=14, fontweight='bold', color='white', pad=15)
    ax3.set_xlabel('Survey ID', fontsize=11, color='white')
    ax3.set_ylabel('Number of Unique NPIs', fontsize=11, color='white')
    ax3.set_xticks(range(len(least_participated)))
    ax3.set_xticklabels(least_participated['Survey_ID'], rotation=45, fontsize=10, color='white')
    ax3.tick_params(axis='y', labelsize=10, colors='white')
    ax3.set_facecolor('#2A2A3C')
    ax3.grid(axis='y', alpha=0.3, color='white')
    ax3.spines['bottom'].set_color('white')
    ax3.spines['left'].set_color('white')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', 
                fontsize=10, color='white',                bbox=dict(boxstyle='round,pad=0.2', facecolor=(239/255, 68/255, 68/255, 0.2), 
                         edgecolor='none'))
    
    # Comparison scatter plot
    comparison_data = pd.concat([
        most_participated.head(3).assign(Category='Most Participated'),
        least_participated.head(3).assign(Category='Least Participated')
    ])
    
    for category, color in [('Most Participated', '#34D399'), ('Least Participated', '#EF4444')]:
        data = comparison_data[comparison_data['Category'] == category]
        ax4.scatter(data['Survey_ID'], data['Unique_NPIs'], 
                   label=category, s=100, alpha=0.8, color=color, 
                   edgecolors='white', linewidth=2)
    
    ax4.set_title('Most vs Least Participated Surveys', fontsize=14, fontweight='bold', color='white', pad=15)
    ax4.set_xlabel('Survey ID', fontsize=11, color='white')
    ax4.set_ylabel('Number of Unique NPIs', fontsize=11, color='white')
    ax4.tick_params(axis='both', labelsize=10, colors='white')
    ax4.set_facecolor('#2A2A3C')
    ax4.grid(True, alpha=0.3, color='white')
    ax4.spines['bottom'].set_color('white')
    ax4.spines['left'].set_color('white')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Enhanced legend
    legend = ax4.legend(fontsize=10, fancybox=True, shadow=True, 
                       facecolor='#2A2A3C', edgecolor='white', 
                       labelcolor='white')
    legend.get_frame().set_alpha(0.8)
    
    plt.tight_layout()
    return fig

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">📊 NPI Survey Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📁 File Upload", divider='grey')
    
    # File upload section
    npi_file = st.sidebar.file_uploader("Upload NPI Excel File", type=['xlsx', 'xls'], help="Upload an Excel file containing NPI data")
    survey_file = st.sidebar.file_uploader("Upload Survey Excel File", type=['xlsx', 'xls'], help="Upload an Excel file containing survey data")

    # Show sample data section
    if st.sidebar.button("Show Sample Data Schema", type="secondary"):
        st.markdown('<h2 class="section-header">📋 Sample Data Schema</h2>', unsafe_allow_html=True)
        with st.spinner("Loading sample data..."):
            loading_placeholder = st.empty()
            loading_placeholder.image("https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExZDExc2dzaG95Y25pZGJqd3RhaXl5N2Y1bHk3NnhrOWJyYnVmZ2llYiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/RgzryV9nRCMHPVVXPV/giphy.gif", caption="Loading sample data...", use_container_width=True)
            col1, col2 = st.columns(2)
            npi_data = {
                'NPI': [1000000000, 1000000001, 1000000002, 1000000003, 1000000004],
                'State': ['NY', 'MI', 'CA', 'TX', 'GA'],
                'Login Time': ['2025-03-08 06:09:00', '2025-03-08 12:28:00', '2025-03-08 15:11:00', '2025-03-08 14:17:00', '2025-03-08 15:59:00'],
                'Logout Time': ['2025-03-08 06:28:00', '2025-03-08 13:10:00', '2025-03-08 15:37:00', '2025-03-08 15:36:00', '2025-03-08 17:37:00'],
                'Usage Time (mins)': [19, 42, 26, 79, 98],
                'Region': ['Northeast', 'Midwest', 'West', 'Northeast', 'West'],
                'Speciality': ['Cardiology', 'Oncology', 'Oncology', 'Orthopedics', 'Oncology'],
                'Count of Survey Attempts': [3, 5, 8, 9, 0]
            }
            sample_npi = pd.DataFrame(npi_data)
            survey_data = {
                'Survey ID': [100005, 100099, 100010, 100016, 100052],
                'NPI': [1000038890, 1000019931, 1000008000, 1000037213, 1000017141],
                'Attempt Date': ['2025-01-01', '2025-01-01', '2025-01-01', '2025-01-01', '2025-01-01'],
                'Attempt Time': ['1900-01-01 07:13:23', '1900-01-01 17:04:42', '1900-01-01 19:52:55', '1900-01-01 21:53:43', '1900-01-01 21:16:38']
            }
            sample_survey = pd.DataFrame(survey_data)
            loading_placeholder.empty()
            with col1:
                st.subheader("📊 NPI Data Sample", divider='grey')
                st.dataframe(sample_npi, use_container_width=True)
            with col2:
                st.subheader("📋 Survey Data Sample", divider='grey')
                st.dataframe(sample_survey, use_container_width=True)

    # Main analysis section
    if npi_file is not None and survey_file is not None:
        st.sidebar.success("✅ Files uploaded successfully!", icon="✅")
        
        # Create a placeholder for the initializing message
        init_message = st.info("🔄 Files uploaded! Initializing data processing...", icon="🔄")
        
        st.sidebar.header("🔍 Analysis Parameters", divider='grey')
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
          # Show loading GIF instead of spinner
        loading_gif_placeholder = st.empty()
        loading_gif_placeholder.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2rem;">
            <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExNng4YTR0OTduNWRrMzExMzRodnQwNTFyOWlndjhheHVjaXc1bmg1YiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/4EFt4UAegpqTy3nVce/giphy.gif" 
                 width="200" height="200" style="border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
            <h3 style="color: #60A5FA; margin-top: 1rem; text-align: center;">🔄 Processing Excel Files...</h3>
            <p style="color: #E0E0E0; text-align: center;">Please wait while we prepare your data for analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        progress_placeholder.progress(0)
        status_placeholder.info("📋 Reading Excel files...")
        npi_df, survey_df = preprocess(npi_file, survey_file)
        progress_placeholder.progress(100)
        status_placeholder.success("✅ Data preprocessing completed!")
        
        # Clear all loading elements
        loading_gif_placeholder.empty()
        progress_placeholder.empty()
        status_placeholder.empty()
        
        # Clear the initializing message after preprocessing
        init_message.empty()
        
        if npi_df is not None and survey_df is not None:
            available_surveys = sorted(survey_df['Survey ID'].unique())
            survey_id = st.sidebar.selectbox(
                "Select Survey ID",
                options=available_surveys,
                help="Choose a survey ID from the available options"
            )
            time_input = st.sidebar.time_input(
                "Select Time",
                value=None,
                help="Select the time to check for active NPIs"
            )
            if st.sidebar.button("🚀 Run Analysis", type="primary"):
                if time_input is not None:
                    time_str = time_input.strftime("%H:%M")
                    with st.spinner("🔄 Running comprehensive analysis... This may take a few moments."):
                        result_df, processed_data, message = workflow(npi_file, survey_file, survey_id, time_str)
                        npi_df, survey_df = processed_data
                    
                    # Create two main tabs
                    main_tab1, main_tab2 = st.tabs(["📋 Analysis Results", "📊 Data Visualizations"])
                    
                    with main_tab1:
                        st.markdown('<h2 class="section-header">🎯 Analysis Results</h2>', unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total NPIs", len(npi_df), delta=None, delta_color="normal")
                        with col2:
                            st.metric("Total Survey Attempts", len(survey_df), delta=None, delta_color="normal")
                        with col3:
                            st.metric("Active NPIs Found", len(result_df), delta=None, delta_color="normal")
                        with col4:
                            st.metric("Survey ID", survey_id, delta=None, delta_color="normal")
                        if len(result_df) > 0:
                            st.success(message, icon="✅")
                        else:
                            st.warning(message, icon="⚠️")
                        if len(result_df) > 0:
                            st.subheader("📋 Active NPIs Details", divider='grey')
                            st.dataframe(result_df, use_container_width=True)
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"active_npis_survey_{survey_id}_{time_str.replace(':', '')}.csv",
                                mime="text/csv"                            )
                    
                    with main_tab2:
                        st.markdown('<h2 class="section-header">📊 Comprehensive Data Visualizations</h2>', unsafe_allow_html=True)
                        
                        # Sub-tabs for visualizations with enhanced styling
                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            "🗺️ State Analysis", 
                            "🌍 Regional Insights", 
                            "🏥 Specialty Breakdown",
                            "📊 Survey Metrics",
                            "⏰ Activity Patterns"
                        ])
                        
                        with tab1:
                            st.markdown("### 🗺️ State Distribution Analysis")
                            st.markdown("*Explore how NPIs are distributed across different states*")
                            
                            # Show loading GIF
                            state_loading_placeholder = st.empty()
                            state_loading_placeholder.markdown("""
                            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 1.5rem;">
                                <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExNng4YTR0OTduNWRrMzExMzRodnQwNTFyOWlndjhheHVjaXc1bmg1YiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/4EFt4UAegpqTy3nVce/giphy.gif" 
                                     width="150" height="150" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                                <h5 style="color: #60A5FA; margin-top: 0.8rem; text-align: center;">🗺️ Analyzing States...</h5>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            fig = create_state_distribution_plot(npi_df)
                            state_loading_placeholder.empty()
                            st.pyplot(fig, use_container_width=True)
                            
                            # Add some insights
                            state_counts = npi_df['State'].value_counts()
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Top State", state_counts.index[0], f"{state_counts.iloc[0]} NPIs")
                            with col2:
                                st.metric("Total States", len(state_counts))
                            with col3:
                                st.metric("Average per State", f"{state_counts.mean():.1f}")
                        
                        with tab2:
                            st.markdown("### 🌍 Regional Distribution Analysis")
                            st.markdown("*Understand regional patterns and distributions*")
                            
                            # Show loading GIF
                            region_loading_placeholder = st.empty()
                            region_loading_placeholder.markdown("""
                            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 1.5rem;">
                                <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExNng4YTR0OTduNWRrMzExMzRodnQwNTFyOWlndjhheHVjaXc1bmg1YiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/4EFt4UAegpqTy3nVce/giphy.gif" 
                                     width="150" height="150" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                                <h5 style="color: #60A5FA; margin-top: 0.8rem; text-align: center;">🌍 Analyzing Regions...</h5>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            fig = create_region_distribution_plot(npi_df)
                            region_loading_placeholder.empty()
                            st.pyplot(fig, use_container_width=True)
                            
                            # Add regional insights
                            region_counts = npi_df['Region'].value_counts()
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Top Region", region_counts.index[0], f"{region_counts.iloc[0]} NPIs")
                            with col2:
                                st.metric("Total Regions", len(region_counts))
                            with col3:
                                st.metric("Region Diversity", f"{(region_counts.std()/region_counts.mean()*100):.1f}% CV")
                        
                        with tab3:
                            st.markdown("### 🏥 Medical Specialty Distribution")
                            st.markdown("*Analyze the diversity of medical specialties*")
                            
                            # Show loading GIF
                            specialty_loading_placeholder = st.empty()
                            specialty_loading_placeholder.markdown("""
                            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 1.5rem;">
                                <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExNng4YTR0OTduNWRrMzExMzRodnQwNTFyOWlndjhheHVjaXc1bmg1YiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/4EFt4UAegpqTy3nVce/giphy.gif" 
                                     width="150" height="150" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                                <h5 style="color: #60A5FA; margin-top: 0.8rem; text-align: center;">🏥 Analyzing Specialties...</h5>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            fig = create_specialty_distribution_plot(npi_df)
                            specialty_loading_placeholder.empty()
                            st.pyplot(fig, use_container_width=True)
                            
                            # Add specialty insights
                            specialty_counts = npi_df['Speciality'].value_counts()
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Top Specialty", specialty_counts.index[0], f"{specialty_counts.iloc[0]} NPIs")
                            with col2:
                                st.metric("Total Specialties", len(specialty_counts))
                            with col3:
                                st.metric("Specialty Diversity", f"{len(specialty_counts[specialty_counts >= 10])}")
                        
                        with tab4:
                            st.markdown("### 📊 Survey Participation Analysis")
                            st.markdown("*Examine survey participation patterns and trends*")
                            
                            # Show loading GIF
                            survey_loading_placeholder = st.empty()
                            survey_loading_placeholder.markdown("""
                            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 1.5rem;">
                                <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExNng4YTR0OTduNWRrMzExMzRodnQwNTFyOWlndjhheHVjaXc1bmg1YiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/4EFt4UAegpqTy3nVce/giphy.gif" 
                                     width="150" height="150" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                                <h5 style="color: #60A5FA; margin-top: 0.8rem; text-align: center;">📊 Analyzing Surveys...</h5>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            fig = create_survey_participation_plot(survey_df)
                            survey_loading_placeholder.empty()
                            st.pyplot(fig, use_container_width=True)
                            
                            # Add survey insights
                            survey_participation = survey_df.groupby('Survey ID')['NPI'].nunique()
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Most Participated Survey", survey_participation.idxmax(), f"{survey_participation.max()} NPIs")
                            with col2:
                                st.metric("Total Surveys", len(survey_participation))
                            with col3:
                                st.metric("Avg Participation", f"{survey_participation.mean():.1f}")
                        
                        with tab5:
                            st.markdown("### ⏰ Hourly Activity Patterns")
                            st.markdown("*Discover when NPIs are most active throughout the day*")
                            
                            # Show loading GIF for hourly activity
                            activity_loading_placeholder = st.empty()
                            activity_loading_placeholder.markdown("""
                            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2rem;">
                                <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExNng4YTR0OTduNWRrMzExMzRodnQwNTFyOWlndjhheHVjaXc1bmg1YiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/4EFt4UAegpqTy3nVce/giphy.gif" 
                                     width="180" height="180" style="border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                                <h4 style="color: #60A5FA; margin-top: 1rem; text-align: center;">⏰ Generating Activity Patterns...</h4>
                                <p style="color: #E0E0E0; text-align: center; font-size: 0.9rem;">Analyzing hourly activity data</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Generate the plot
                            fig = create_hourly_activity_plot(npi_df)
                            
                            # Clear loading and show plot
                            activity_loading_placeholder.empty()
                            st.pyplot(fig, use_container_width=True)
                            
                            # Add activity insights
                            hourly_activity = []
                            for hour in range(24):
                                active_count = 0
                                for _, row in npi_df.iterrows():
                                    login_hour = row['Login Hour']
                                    logout_hour = row['Logout Hour']
                                    if login_hour <= logout_hour:
                                        if login_hour <= hour <= logout_hour:
                                            active_count += 1
                                    else:
                                        if hour >= login_hour or hour <= logout_hour:
                                            active_count += 1
                                hourly_activity.append(active_count)
                            
                            peak_hour = hourly_activity.index(max(hourly_activity))
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Peak Activity Hour", f"{peak_hour:02d}:00", f"{max(hourly_activity)} NPIs")
                            with col2:
                                st.metric("Lowest Activity", f"{min(hourly_activity)} NPIs")
                            with col3:
                                st.metric("Average Activity", f"{np.mean(hourly_activity):.1f}")
                    
     

                else:
                    st.sidebar.error("Please select a time!", icon="⚠️")
        else:
            st.error("Error processing the uploaded files. Please check the file format and try again.", icon="❌")
    
    else:
        st.info("👆 Please upload both NPI and Survey Excel files to begin the analysis.", icon="ℹ️")
        st.markdown("""
        ### 📖 How to use this dashboard:
        
        1. **Upload Files**: Upload your NPI and Survey Excel files using the sidebar
        2. **View Sample**: Click "Show Sample Data Schema" to see the expected data format
        3. **Select Parameters**: Choose a Survey ID and time for analysis
        4. **Run Analysis**: Click "Run Analysis" to process the data and generate insights
        5. **Explore Results**: View active NPIs in the 'Analysis Results' tab and visualizations in the 'Data Visualizations' tab
        
        ### 📊 Features:
        - **Data Preprocessing**: Automatic cleaning and transformation of your data
        - **Active NPI Detection**: Find NPIs active at specific times for specific surveys
        - **Interactive Visualizations**: Comprehensive charts and graphs in dedicated tabs
        - **Data Export**: Download results as CSV files
        - **Responsive Design**: Works on desktop and mobile devices
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()