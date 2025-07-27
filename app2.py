import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

# Helper function to safely prepare data for Plotly
def prepare_for_plotly(df):
    """Convert pandas dataframe to a format that works well with Plotly"""
    if df is not None and not df.empty:
        return df.reset_index(drop=True).to_dict('records')
    return []

# Custom CSS for professional dark theme
st.markdown("""
    <style>
    /* Main app background and text */
    .stApp {
        background-color: #1E2A44;
        color: #E0E7FF;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #A5B4FC;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Cards and containers */
    .stMarkdown, .stText, .stDataFrame, .stMetric, .stExpander {
        background-color: #2A3756;
        border-radius: 8px;
        padding: 15px;
        color: #CCFFFF;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background-color: CCFFFF;
        color: #FFFFFF;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #6366F1;
        color: #FFFFFF;
    }
    
    /* Selectbox and File Uploader */
    .stSelectbox, .stFileUploader {
        background-color: #2A3756;
        border-radius: 6px;
        color: #E0E7FF;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #A5B4FC;
        background-color: #2A3756;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #4F46E5;
        color: #FFFFFF;
    }
    
    /* Expander */
    .stExpander {
        border: 1px solid #4F46E5;
    }
    .stExpander > div > div > div {
        color: #A5B4FC;
    }
    
    /* Metric cards */
    .stMetric {
        border: 1px solid #4F46E5;
    }
    .stMetric > div > div > div {
        color: #E0E7FF !important;
    }
    .stMetric > div > div > label {
        color: #A5B4FC !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background-color: #2A3756;
        border: 1px solid #4F46E5;
    }
    
    /* Remove default Streamlit branding */
    footer {visibility: hidden; display: none;}
    .stToolbar {visibility: hidden; display: none;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #2A3756;
    }
    ::-webkit-scrollbar-thumb {
        background: #4F46E5;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Set page configuration
st.set_page_config(
    page_title="NPI Finder PRO",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add logo and title
col1, col2 = st.columns([1, 5])
with col1:
    # Placeholder for logo
    logo_placeholder = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQYV2NgAAIAAAUAAarVyFEAAAAASUVORK5CYII="
    st.image(logo_placeholder, width=80)
with col2:
    st.title("NPI Finder PRO")
    st.markdown("""
    **Explore advanced NPI data with interactive visualizations across specialties and regions.**  
    Upload your Excel file to generate detailed analytics.  
    """)

# Data upload section
st.header("1. Upload Your Data")
uploaded_file = st.file_uploader(
    label="Upload your Excel file containing NPI provider data (XLSX format)",
    type=["xlsx"],
    help="Ensure your file contains columns: NPI, Specialty, State, Region, Usage Time (mins)"
)

# Show expected data schema
with st.expander("View Expected Data Schema"):
    st.markdown("""
    ### Expected Data Format
    Your Excel file should contain:  
    - **NPI**: National Provider Identifier  
    - **Specialty**: Medical specialty  
    - **State**: US state code (e.g., 'NY') or name  
    - **Region**: Geographic region (e.g., 'Northeast')  
    - **Usage Time (mins)**: Provider usage time in minutes  

    Example:
    """)
    sample_data = {
        'NPI': ['1234567890', '2345678901', '3456789012', '4567890123', '5678901234'],
        'Specialty': ['Cardiology', 'Pediatrics', 'Oncology', 'Cardiology', 'Neurology'],
        'State': ['NY', 'CA', 'TX', 'FL', 'IL'],
        'Region': ['Northeast', 'West', 'South', 'South', 'Midwest'],
        'Usage Time (mins)': [45.2, 32.7, 58.1, 41.5, 37.8]
    }
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True)

# Create a mapping of states to regions
region_mapping = {
    'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
    'Midwest': ['OH', 'MI', 'IN', 'IL', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
    'South': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
    'West': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI']
}

# State codes to names mapping
state_code_to_name = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut',
    'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois',
    'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine',
    'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan',
    'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico',
    'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota',
    'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
    'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas',
    'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia',
    'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin',
    'WY': 'Wyoming', 'DC': 'District of Columbia'
}

# Reverse mapping
state_name_to_code = {v: k for k, v in state_code_to_name.items()}

# Dark theme for Plotly
plotly_template = {
    'layout': {
        'paper_bgcolor': '#2A3756',
        'plot_bgcolor': '#2A3756',
        'font': {'color': '#E0E7FF'},
        'xaxis': {
            'gridcolor': '#4B5563',
            'zerolinecolor': '#4B5563',
            'tickfont': {'color': '#E0E7FF'},
            'title_font': {'color': '#E0E7FF'}
        },
        'yaxis': {
            'gridcolor': '#4B5563',
            'zerolinecolor': '#4B5563',
            'tickfont': {'color': '#E0E7FF'},
            'title_font': {'color': '#E0E7FF'}
        }
    }
}

# Process the uploaded data
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        # Normalize column name to 'Specialty'
        if 'Speciality' in df.columns:
            df = df.rename(columns={'Speciality': 'Specialty'})
        st.success("Data loaded successfully!", icon="✅")
        
        # Display filters
        st.header("2. Select Filters")
        col1, col2, col3 = st.columns([2, 2, 1])
        
        specialties = sorted(df['Specialty'].unique())
        regions = sorted(df['Region'].unique())
        
        with col1:
            selected_specialty = st.selectbox("Select Specialty (Required)", specialties)
        
        with col2:
            selected_region = st.selectbox("Select Region", ["All Regions"] + regions)
        
        if selected_region == "All Regions":
            selected_region = None
        
        # Apply filters
        if selected_specialty:
            filtered_df = df[df['Specialty'] == selected_specialty]
            if selected_region:
                filtered_df = filtered_df[filtered_df['Region'] == selected_region]
                st.header(f"{selected_specialty} Providers in {selected_region} Region")
            else:
                st.header(f"{selected_specialty} Providers Across All Regions")
        else:
            st.warning("Please select a specialty to continue.", icon="⚠️")
            st.stop()
        
        # Display filtered dataset
        st.subheader("Filtered Dataset")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download buttons
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_data = buffer.getvalue()
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Data as CSV",
                data=csv_data,
                file_name=f"{selected_specialty}_{'_' + selected_region if selected_region else ''}_data.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                label="Download Data as Excel",
                data=excel_data,
                file_name=f"{selected_specialty}_{'_' + selected_region if selected_region else ''}_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Basic stats
        with st.expander("Dataset Overview"):
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            total_providers = len(filtered_df)
            total_states = filtered_df['State'].nunique()
            avg_usage = filtered_df['Usage Time (mins)'].mean()
            
            stat_col1.metric("Total Providers", total_providers)
            stat_col2.metric("States Represented", total_states)
            stat_col3.metric("Avg Usage Time (mins)", f"{avg_usage:.2f}")
            
            st.write("### Sample Data")
            st.dataframe(filtered_df.head(10), use_container_width=True)
        
        # Standardize state codes
        def standardize_state_codes(df_to_process):
            data_df = df_to_process.copy()
            valid_states = df_to_process['State'].value_counts().index.tolist()
            
            if len(data_df) > 0:
                if len(data_df['State'].iloc[0]) > 2:
                    data_df['StateCode'] = data_df['State'].map(lambda x: state_name_to_code.get(x, x))
                else:
                    data_df['StateCode'] = data_df['State']
                    data_df['StateName'] = data_df['State'].map(lambda x: state_code_to_name.get(x, x))
                
                data_df = data_df[data_df['StateCode'].isin(valid_states) | data_df['State'].isin(valid_states)]
                
                def get_region(state_code):
                    for r, states in region_mapping.items():
                        if state_code in states:
                            return r
                    return "Unknown"
                
                data_df['MappedRegion'] = data_df['StateCode'].apply(get_region)
            
            return data_df
        
        processed_df = standardize_state_codes(filtered_df)
        
        # Geographic Distribution
        st.header("Geographic Distribution")
        tab1, tab2 = st.tabs(["Choropleth Map", "State Distribution"])
        
        with tab1:
            state_counts = processed_df.groupby('StateCode').size().reset_index(name='NPI_Count')
            state_counts['name'] = state_counts['StateCode'].map(lambda x: state_code_to_name.get(x, x))
            state_counts['Region'] = state_counts['StateCode'].apply(
                lambda x: next((r for r, states in region_mapping.items() if x in states), "Unknown")
            )
            
            if selected_region:
                valid_states = region_mapping.get(selected_region, [])
                state_counts = state_counts[state_counts['StateCode'].isin(valid_states)]
                title_suffix = f" in {selected_region} Region"
            else:
                title_suffix = ""
            
            all_states_df = pd.DataFrame({
                'State': state_counts['StateCode'],
                'StateName': state_counts['name'],
                'NPI_Count': state_counts['NPI_Count'],
                'Region': state_counts['Region']
            })
            map_data = prepare_for_plotly(all_states_df)
            
            fig = go.Figure(data=go.Choropleth(
                locations=[d['State'] for d in map_data],
                z=[d['NPI_Count'] for d in map_data],
                locationmode='USA-states',
                colorscale='Blues',
                text=[f"State: {d['StateName']}<br>Region: {d['Region']}<br>NPI Count: {d['NPI_Count']}" for d in map_data],
                colorbar_title="NPI Count",
                marker_line_color='#FFFFFF',  # White borders for visibility
                marker_line_width=1.5  # Increased width for clearer borders
            ))
            
            fig.update_layout(
                title_text=f'Distribution of {selected_specialty} NPIs Across US States{title_suffix}',
                geo=dict(
                    scope='usa',
                    showlakes=True,
                    lakecolor='#2A3756',
                    bgcolor='#808080'
                ),
                margin={"r":0,"t":50,"l":0,"b":0},
                height=600,
                **plotly_template['layout']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.download_button(
                label="Download Map as HTML",
                data=fig.to_html(),
                file_name=f"{selected_specialty}_{'_' + selected_region if selected_region else ''}_map.html",
                mime="text/html"
            )
        
        with tab2:
            state_bar_data = prepare_for_plotly(state_counts.sort_values('NPI_Count', ascending=False))
            state_bar = go.Figure(data=go.Bar(
                x=[d['name'] for d in state_bar_data],
                y=[d['NPI_Count'] for d in state_bar_data],
                marker=dict(
                    color=[d['NPI_Count'] for d in state_bar_data],
                    colorscale='Blues'
                ),
                text=[d['NPI_Count'] for d in state_bar_data],
                textposition='auto'
            ))
            
            state_bar.update_layout(
                title=f'Number of {selected_specialty} NPIs by State',
                xaxis_title='State',
                yaxis_title='Number of NPIs',
                height=500,
                **plotly_template['layout']
            )
            
            st.plotly_chart(state_bar, use_container_width=True)
        
        # Regional Analysis
        st.header("Regional Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if selected_region is None:
                region_counts = processed_df.groupby('MappedRegion').size().reset_index(name='NPI_Count')
                region_counts['Percentage'] = (region_counts['NPI_Count'] / region_counts['NPI_Count'].sum() * 100).round(1)
                region_pie_data = prepare_for_plotly(region_counts)
                
                region_pie = go.Figure(data=go.Pie(
                    values=[d['NPI_Count'] for d in region_pie_data],
                    labels=[d['MappedRegion'] for d in region_pie_data],
                    hole=0.3,
                    textinfo='percent+label+value',
                    textposition='inside',
                    marker=dict(colors=px.colors.sequential.Blues)
                ))
                
                region_pie.update_layout(
                    title=f'Regional Distribution of {selected_specialty} NPIs',
                    **plotly_template['layout']
                )
                
                st.plotly_chart(region_pie, use_container_width=True)
            else:
                region_states = processed_df.groupby('StateCode').size().reset_index(name='NPI_Count')
                region_states['name'] = region_states['StateCode'].map(lambda x: state_code_to_name.get(x, x))
                state_pie_data = prepare_for_plotly(region_states)
                
                state_pie = go.Figure(data=go.Pie(
                    values=[d['NPI_Count'] for d in state_pie_data],
                    labels=[d['name'] for d in state_pie_data],
                    hole=0.3,
                    textinfo='percent+label+value',
                    textposition='inside',
                    marker=dict(colors=px.colors.sequential.Blues)
                ))
                
                state_pie.update_layout(
                    title=f'Distribution of {selected_specialty} NPIs in {selected_region} by State',
                    **plotly_template['layout']
                )
                
                st.plotly_chart(state_pie, use_container_width=True)
        
        with col2:
            if 'Usage Time (mins)' in processed_df.columns:
                if selected_region is None:
                    usage_by_region = processed_df.groupby('MappedRegion')['Usage Time (mins)'].mean().reset_index()
                    usage_by_region['Usage Time (mins)'] = usage_by_region['Usage Time (mins)'].round(1)
                    usage_data = prepare_for_plotly(usage_by_region.sort_values('Usage Time (mins)', ascending=False))
                    
                    usage_fig = go.Figure(data=go.Bar(
                        x=[d['MappedRegion'] for d in usage_data],
                        y=[d['Usage Time (mins)'] for d in usage_data],
                        marker=dict(
                            color=[d['Usage Time (mins)'] for d in usage_data],
                            colorscale='Blues'
                        ),
                        text=[f"{d['Usage Time (mins)']:.1f}" for d in usage_data],
                        textposition='auto'
                    ))
                    
                    usage_fig.update_layout(
                        title=f'Average Usage Time for {selected_specialty} by Region',
                        xaxis_title='Region',
                        yaxis_title='Usage Time (mins)',
                        height=400,
                        **plotly_template['layout']
                    )
                else:
                    usage_by_state = processed_df.groupby('StateCode')['Usage Time (mins)'].mean().reset_index()
                    usage_by_state['StateName'] = usage_by_state['StateCode'].map(lambda x: state_code_to_name.get(x, x))
                    usage_by_state['Usage Time (mins)'] = usage_by_state['Usage Time (mins)'].round(1)
                    state_usage_data = prepare_for_plotly(usage_by_state.sort_values('Usage Time (mins)', ascending=False))
                    
                    usage_fig = go.Figure(data=go.Bar(
                        x=[d['StateName'] for d in state_usage_data],
                        y=[d['Usage Time (mins)'] for d in state_usage_data],
                        marker=dict(
                            color=[d['Usage Time (mins)'] for d in state_usage_data],
                            colorscale='Blues'
                        ),
                        text=[f"{d['Usage Time (mins)']:.1f}" for d in state_usage_data],
                        textposition='auto'
                    ))
                    
                    usage_fig.update_layout(
                        title=f'Average Usage Time for {selected_specialty} by State in {selected_region}',
                        xaxis_title='State',
                        yaxis_title='Usage Time (mins)',
                        height=400,
                        **plotly_template['layout']
                    )
                
                st.plotly_chart(usage_fig, use_container_width=True)
        
        # Advanced Analytics
        st.header("Advanced Analytics")
        col1, col2 = st.columns(2)
        
        with col1:
            hist_data = processed_df['Usage Time (mins)'].dropna().tolist()
            fig = go.Figure(data=go.Histogram(
                x=hist_data,
                nbinsx=20,
                marker_color='#60A5FA'
            ))
            
            fig.update_layout(
                title=f'Distribution of Usage Time for {selected_specialty}',
                xaxis_title='Usage Time (minutes)',
                yaxis_title='Number of Providers',
                showlegend=False,
                **plotly_template['layout']
            )
            
            fig.add_vline(
                x=processed_df['Usage Time (mins)'].mean(),
                line_dash="dash",
                line_color="#F87171",
                annotation_text=f"Mean: {processed_df['Usage Time (mins)'].mean():.1f} mins",
                annotation_position="top right",
                annotation_font_color="#E0E7FF"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if selected_region is None:
                regions = processed_df['Region'].unique().tolist()
                box_fig = go.Figure()
                for region in regions:
                    region_data = processed_df[processed_df['Region'] == region]['Usage Time (mins)'].dropna().tolist()
                    if region_data:
                        box_fig.add_trace(go.Box(
                            y=region_data,
                            name=region,
                            boxpoints='all',
                            marker_color='#60A5FA'
                        ))
                
                box_fig.update_layout(
                    title=f'Usage Time Distribution by Region for {selected_specialty}',
                    yaxis_title='Usage Time (minutes)',
                    showlegend=True,
                    **plotly_template['layout']
                )
            else:
                states = processed_df['State'].unique().tolist()
                box_fig = go.Figure()
                for state in states:
                    state_data = processed_df[processed_df['State'] == state]['Usage Time (mins)'].dropna().tolist()
                    if state_data:
                        box_fig.add_trace(go.Box(
                            y=state_data,
                            name=state,
                            boxpoints='all',
                            marker_color='#60A5FA'
                        ))
                
                box_fig.update_layout(
                    title=f'Usage Time Distribution by State for {selected_specialty} in {selected_region}',
                    yaxis_title='Usage Time (minutes)',
                    showlegend=True,
                    **plotly_template['layout']
                )
            
            st.plotly_chart(box_fig, use_container_width=True)
        
        # Additional Insights
        st.header("Additional Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            top_states = processed_df.groupby(['StateCode', 'State']).size().reset_index(name='Count')
            top_states['State Name'] = top_states['StateCode'].map(lambda x: state_code_to_name.get(x, x))
            top_states = top_states.sort_values('Count', ascending=False).head(10)
            
            st.subheader(f"Top 10 States for {selected_specialty} Providers")
            st.dataframe(
                top_states[['State Name', 'Count']].rename(columns={'Count': 'Number of Providers'}),
                use_container_width=True
            )
        
        with col2:
            st.subheader("Usage Time Statistics (minutes)")
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Minimum', 'Maximum', 'Standard Deviation'],
                'Value': [
                    f"{processed_df['Usage Time (mins)'].mean():.2f}",
                    f"{processed_df['Usage Time (mins)'].median():.2f}",
                    f"{processed_df['Usage Time (mins)'].min():.2f}",
                    f"{processed_df['Usage Time (mins)'].max():.2f}",
                    f"{processed_df['Usage Time (mins)'].std():.2f}"
                ]
            })
            
            st.dataframe(stats_df, use_container_width=True)
            
            overall_avg = df['Usage Time (mins)'].mean()
            specialty_avg = processed_df['Usage Time (mins)'].mean()
            
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=specialty_avg,
                title={'text': "Average Usage Time", 'font': {'color': '#E0E7FF'}},
                delta={'reference': overall_avg, 'relative': False, 'font': {'color': '#E0E7FF'}},
                gauge={
                    'axis': {'range': [0, df['Usage Time (mins)'].max() * 1.2], 'tickfont': {'color': '#E0E7FF'}},
                    'bar': {'color': "#4F46E5"},
                    'steps': [
                        {'range': [0, overall_avg * 0.7], 'color': "#3B82F6"},
                        {'range': [overall_avg * 0.7, overall_avg * 1.3], 'color': "#60A5FA"},
                        {'range': [overall_avg * 1.3, df['Usage Time (mins)'].max() * 1.2], 'color': "#93C5FD"}
                    ],
                    'threshold': {
                        'line': {'color': "#F87171", 'width': 4},
                        'thickness': 0.75,
                        'value': overall_avg
                    },
                    'bgcolor': '#2A3756'
                }
            ))
            
            gauge.update_layout(height=250, **plotly_template['layout'])
            st.plotly_chart(gauge, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error processing data: {e}", icon="❌")
        st.stop()
else:
    st.info("👆 Upload an Excel file to begin analyzing your NPI provider data.", icon="ℹ️")
    st.header("Dashboard Preview")
    st.markdown("This section shows example visualizations using sample data.")
    st.subheader("Example: Geographic Distribution")
    st.download_button(
        label="Download Sample Report (CSV)",
        data="NPI,Specialty,State,Region,Usage Time (mins)\n1234567890,Cardiology,NY,Northeast,45.2\n2345678901,Pediatrics,CA,West,32.7",
        file_name="sample_npi_report.csv",
        mime="text/csv",
        disabled=True
    )

# Footer
st.markdown("---")
st.markdown("NPI Provider Visualization Dashboard | Powered by Streamlit & Plotly", unsafe_allow_html=True)