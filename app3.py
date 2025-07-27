import os
import pandas as pd
import numpy as np
import statistics
import re
from datetime import datetime
import google.generativeai as genai
import streamlit.components.v1 as components
import streamlit as st
import time
from io import StringIO
import uuid

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Medical Data Analyst",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for dark theme and professional UI
st.markdown("""
<style>
/* Global styles */
body {
    background-color: #1E2A44 !important;
    color: #E6E9EF !important;
    font-family: 'Arial', sans-serif;
}

/* Main container */
.stApp {
    background-color: #1E2A44;
    color: #E6E9EF;
}

/* Card styling */
.card {
    background-color: #2A3B5E;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

/* Upload section */
.upload-section {
    background-color: #2A3B5E;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}

/* Chat container */
.chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 10px;
    background-color: #2A3B5E;
    border-radius: 10px;
}

/* Chat message styling */
.stChatMessage {
    background-color: #3B4C7A !important;
    border-radius: 8px !important;
    padding: 10px !important;
    margin-bottom: 10px !important;
    color: #E6E9EF !important;
}
.stChatMessage.user {
    background-color: #4A5C8C !important;
}

/* Input fields */
.stTextInput > div > input {
    background-color: #3B4C7A;
    color: #E6E9EF;
    border: 1px solid #4A5C8C;
    border-radius: 5px;
}
.stTextArea > div > textarea {
    background-color: #3B4C7A;
    color: #E6E9EF;
    border: 1px solid #4A5C8C;
    border-radius: 5px;
}

/* Buttons */
.stButton > button {
    background-color: #4A90E2;
    color: #FFFFFF;
    border-radius: 5px;
    padding: 8px 16px;
    transition: background-color 0.3s;
}
.stButton > button:hover {
    background-color: #357ABD;
}

/* Metrics */
.stMetric {
    background-color: #2A3B5E;
    border-radius: 5px;
    padding: 10px;
}

/* Selectbox and other inputs */
.stSelectbox > div > select {
    background-color: #3B4C7A;
    color: #E6E9EF;
    border: 1px solid #4A5C8C;
    border-radius: 5px;
}

/* Expander */
.stExpander {
    background-color: #2A3B5E;
    border-radius: 5px;
    color: #E6E9EF;
}

/* Table */
.stTable {
    background-color: #2A3B5E;
    color: #E6E9EF;
    border-radius: 5px;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #E6E9EF !important;
}

/* Error and success messages */
.stAlert {
    background-color: #3B4C7A;
    color: #E6E9EF;
    border-radius: 5px;
}

/* Download button */
.stDownloadButton > button {
    background-color: #4A90E2;
    color: #FFFFFF;
    border-radius: 5px;
}
.stDownloadButton > button:hover {
    background-color: #357ABD;
}

/* NPI result display */
.npi-result {
    background-color: #3B4C7A;
    border-radius: 5px;
    padding: 15px;
    margin-top: 10px;
    color: #E6E9EF;
}
</style>
""", unsafe_allow_html=True)

# Initialize API key from environment variables
GEMINI_API_KEY = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, AttributeError):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        st.write("API Key loaded successfully!")
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
        GEMINI_API_KEY = None
else:
    st.warning("Gemini API key not found. Some features will be limited.")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'df' not in st.session_state:
    st.session_state.df = None

if 'summary' not in st.session_state:
    st.session_state.summary = None

# Function definitions (unchanged from original)
def validate_df_columns(df):
    required_columns = ['NPI', 'State', 'Region', 'Speciality', 
                       'Usage Time (mins)', 'Count of Survey Attempts',
                       'Login Date', 'Login Time', 'Logout Date', 'Logout Time']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, missing_columns
    return True, []

def process_uploaded_data(df):
    try:
        is_valid, missing_columns = validate_df_columns(df)
        if not is_valid:
            return None, f"CSV file is missing required columns: {', '.join(missing_columns)}"
        df['Login DateTime'] = pd.to_datetime(df['Login Date'] + ' ' + df['Login Time'])
        df['Logout DateTime'] = pd.to_datetime(df['Logout Date'] + ' ' + df['Logout Time'])
        calculated_usage = (df['Logout DateTime'] - df['Login DateTime']).dt.total_seconds() / 60
        if np.abs((calculated_usage - df['Usage Time (mins)']).mean()) > 1:
            df['Usage Time (mins)'] = calculated_usage
        return df, None
    except Exception as e:
        return None, f"Error processing data: {str(e)}"

def generate_data_summary(df):
    df['Login Hour'] = df['Login DateTime'].dt.hour
    frequent_login_hour = df['Login Hour'].mode()[0]
    summary = {
        "total_records": len(df),
        "date_range": {
            "min": df['Login Date'].min(),
            "max": df['Login Date'].max()
        },
        "specialties": df['Speciality'].unique().tolist(),
        "regions": df['Region'].unique().tolist(),
        "states": df['State'].unique().tolist(),
        "usage_time": {
            "mean": df['Usage Time (mins)'].mean(),
            "median": df['Usage Time (mins)'].median(),
            "min": df['Usage Time (mins)'].min(),
            "max": df['Usage Time (mins)'].max()
        },
        "survey_attempts": {
            "mean": df['Count of Survey Attempts'].mean(),
            "median": df['Count of Survey Attempts'].median(),
            "min": df['Count of Survey Attempts'].min(),
            "max": df['Count of Survey Attempts'].max()
        },
        "frequent_login_hour": frequent_login_hour
    }
    return summary

def analyze_data(df, column, operation, filter_dict=None):
    if filter_dict:
        for key, value in filter_dict.items():
            if key in df.columns:
                df = df[df[key] == value]
    if operation == 'mean':
        return df[column].mean()
    elif operation == 'median':
        return df[column].median()
    elif operation == 'mode':
        try:
            return statistics.mode(df[column])
        except:
            return "No unique mode found"
    elif operation == 'min':
        return df[column].min()
    elif operation == 'max':
        return df[column].max()
    elif operation == 'count':
        return len(df)
    elif operation == 'sum':
        return df[column].sum()
    elif operation == 'std':
        return df[column].std()
    elif operation == 'unique':
        return df[column].unique().tolist()
    else:
        return f"Operation {operation} not supported"

def get_npi_details(df, npi, attribute=None):
    try:
        npi = int(npi)
        record = df[df['NPI'] == npi]
        if record.empty:
            return f"No record found for NPI {npi}"
        record = record.iloc[0]
        if attribute:
            attribute_map = {
                'state': 'State',
                'region': 'Region',
                'speciality': 'Speciality',
                'specialty': 'Speciality',
                'usage time': 'Usage Time (mins)',
                'survey attempts': 'Count of Survey Attempts',
                'login datetime': 'Login DateTime',
                'logout datetime': 'Logout DateTime'
            }
            attribute = attribute.lower()
            if attribute in attribute_map:
                column = attribute_map[attribute]
                value = record[column]
                if column == 'Login DateTime':
                    value = f"{record['Login Date']} {record['Login Time']}"
                elif column == 'Logout DateTime':
                    value = f"{record['Logout Date']} {record['Logout Time']}"
                return f"{attribute.capitalize()} for NPI {npi}: {value}"
            return f"Attribute '{attribute}' not recognized"
        details = f"""
        NPI: {record['NPI']}
        State: {record['State']}
        Region: {record['Region']}
        Speciality: {record['Speciality']}
        Usage Time (mins): {record['Usage Time (mins)']}
        Count of Survey Attempts: {record['Count of Survey Attempts']}
        Login DateTime: {record['Login Date']} {record['Login Time']}
        Logout DateTime: {record['Logout Date']} {record['Logout Time']}
        """
        return details.strip()
    except Exception as e:
        return f"Error retrieving details for NPI {npi}: {e}"

def list_available_models():
    try:
        models = genai.list_models()
        return [model.name for model in models if 'generateContent' in model.supported_generation_methods]
    except Exception as e:
        return f"Error listing models: {e}"

def get_column_synonyms(column_name):
    synonyms = {
        'Usage Time (mins)': ['time', 'usage', 'duration', 'minutes', 'login time', 'logout time'],
        'Count of Survey Attempts': ['survey', 'attempts', 'surveys', 'tries'],
        'Speciality': ['specialty', 'specialization', 'field', 'specialities', 'specialties'],
        'Region': ['area', 'zone', 'location', 'regions'],
        'State': ['state', 'province', 'territory', 'states'],
        'NPI': ['npi', 'provider', 'providers', 'professional', 'professionals']
    }
    return synonyms.get(column_name, [])

def filter_dataframe_from_question(df, question_lower):
    filters = {}
    regions = df['Region'].unique()
    for region in regions:
        if region.lower() in question_lower:
            filters['Region'] = region
    states = df['State'].unique()
    for state in states:
        if state.lower() in question_lower:
            filters['State'] = state
    specialties = df['Speciality'].unique()
    for specialty in specialties:
        if specialty.lower() in question_lower:
            filters['Speciality'] = specialty
    if filters:
        filtered_df = df.copy()
        for key, value in filters.items():
            filtered_df = filtered_df[filtered_df[key] == value]
        return filtered_df
    return df

def process_question(question, df, summary, chat_history):
    question_lower = question.lower()
    conversation_context = ""
    if chat_history:
        recent_history = chat_history[-5:]
        for i, exchange in enumerate(recent_history):
            conversation_context += f"Q{len(chat_history)-len(recent_history)+i+1}: {exchange['question']}\n"
            conversation_context += f"A{len(chat_history)-len(recent_history)+i+1}: {exchange['answer']}\n\n"
    
    if 'specialities' in question_lower or 'specialties' in question_lower:
        specialties = summary['specialties']
        return f"Here's the list of specialties in the dataset: {', '.join(specialties)}."
    
    if 'npi range' in question_lower or 'npi ranges' in question_lower:
        npi_min = df['NPI'].min()
        npi_max = df['NPI'].max()
        return f"The NPIs in the dataset range from {npi_min} to {npi_max}."
    
    if 'frequent login time' in question_lower:
        df['Login Hour'] = df['Login DateTime'].dt.hour
        frequent_hour = df['Login Hour'].mode()[0]
        return f"The most frequent login hour is {frequent_hour}:00."
    
    if 'total count of npi' in question_lower or 'how many npi' in question_lower:
        return f"The total number of NPIs in the dataset is {summary['total_records']}."

    npi_match = re.search(r'\b\d{10}\b', question)
    if npi_match:
        npi = npi_match.group(0)
        attributes = ['state', 'region', 'speciality', 'specialty', 'usage time', 
                      'survey attempts', 'login datetime', 'logout datetime']
        for attr in attributes:
            if attr in question_lower:
                return get_npi_details(df, npi, attribute=attr)
        return f"Here's everything I have for NPI {npi}:\n\n{get_npi_details(df, npi)}"
    
    if any(keyword in question_lower for keyword in ['most', 'highest', 'top']) and \
       any(npi_syn in question_lower for npi_syn in get_column_synonyms('NPI')):
        category_map = {
            'State': ['state', 'states', 'province', 'territory'],
            'Region': ['region', 'area', 'zone', 'location', 'regions'],
            'Speciality': ['specialty', 'specialities', 'specialization', 'field', 'specialties']
        }
        target_category = None
        for category, synonyms in category_map.items():
            if any(synonym in question_lower for synonym in synonyms):
                target_category = category
                break
        if target_category and target_category in df.columns:
            filtered_df = filter_dataframe_from_question(df, question_lower)
            if filtered_df.empty:
                return "Looks like there’s no data matching your filters. Try tweaking your question!"
            counts = filtered_df[target_category].value_counts()
            if counts.empty:
                return f"No {target_category.lower()} found in the filtered data."
            max_count = counts.max()
            top_categories = counts[counts == max_count].index.tolist()
            if len(top_categories) == 1:
                return f"The {target_category.lower()} with the most NPIs is {top_categories[0]} with {max_count} providers."
            else:
                return f"Multiple {target_category.lower()}s are tied for the most NPIs ({max_count} providers each): {', '.join(top_categories)}."
    
    stat_keywords = ['max', 'maximum', 'highest', 'min', 'minimum', 'lowest']
    if any(keyword in question_lower for keyword in stat_keywords):
        operation = 'max' if any(k in question_lower for k in ['max', 'maximum', 'highest']) else 'min'
        column_map = {
            'Usage Time (mins)': ['usage time', 'time', 'login time', 'logout time', 'duration', 'minutes'],
            'Count of Survey Attempts': ['survey attempts', 'survey', 'attempts', 'tries']
        }
        target_column = None
        for column, synonyms in column_map.items():
            if any(synonym in question_lower for synonym in synonyms) or column.lower() in question_lower:
                target_column = column
                break
        if target_column and target_column in df.columns:
            filtered_df = filter_dataframe_from_question(df, question_lower)
            if filtered_df.empty:
                return "I couldn’t find any data matching your filters. Maybe try a different region or specialty?"
            stat_value = filtered_df[target_column].max() if operation == 'max' else filtered_df[target_column].min()
            npi_list = filtered_df[filtered_df[target_column] == stat_value]['NPI'].tolist()
            if not npi_list:
                return f"No NPIs found for the {operation} {target_column.lower()}."
            unit = " minutes" if target_column == 'Usage Time (mins)' else ""
            response = f"The {operation} {target_column.lower()} is {stat_value:.2f}{unit}, and here’s who had it:\n\n"
            for npi in npi_list:
                details = get_npi_details(df, npi)
                response += f"{details}\n\n"
            return response.strip()
    
    if 'npi' in question_lower and not any(digit in question for digit in '0123456789'):
        if any(keyword in question_lower for keyword in ['average', 'mean', 'median', 'total', 'sum']):
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in question_lower or any(synonym in question_lower for synonym in get_column_synonyms(col)):
                    filtered_df = filter_dataframe_from_question(df, question_lower)
                    if isinstance(filtered_df, pd.DataFrame):
                        try:
                            if 'usage time' in question_lower or 'time' in question_lower:
                                result = filtered_df['Usage Time (mins)'].mean()
                                return f"On average, usage time is {result:.2f} minutes across the filtered data."
                            elif 'survey' in question_lower or 'attempts' in question_lower:
                                result = filtered_df['Count of Survey Attempts'].mean()
                                return f"The average number of survey attempts is {result:.2f}."
                        except Exception as e:
                            pass
        return "Could you share a specific 10-digit NPI or clarify what stats you’re looking for?"
    
    if any(ref in question_lower for ref in ['previous answer', 'last question', 'you just said', 'earlier response', 'previous response']):
        if not chat_history:
            return "I don’t have any previous chats to refer to yet. What’s your question?"
        conversation_context = "Here’s what we talked about recently:\n\n"
        for i, exchange in enumerate(chat_history[-3:]):
            conversation_context += f"Q{len(chat_history)-3+i+1}: {exchange['question']}\n"
            conversation_context += f"A{len(chat_history)-3+i+1}: {exchange['answer']}\n\n"
    
    if 'average' in question_lower or 'mean' in question_lower:
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in question_lower or any(synonym in question_lower for synonym in get_column_synonyms(col)):
                filtered_df = filter_dataframe_from_question(df, question_lower)
                if isinstance(filtered_df, pd.DataFrame):
                    try:
                        if 'usage time' in question_lower or 'time' in question_lower:
                            result = filtered_df['Usage Time (mins)'].mean()
                            return f"On average, usage time is {result:.2f} minutes."
                        elif 'survey' in question_lower or 'attempts' in question_lower:
                            result = filtered_df['Count of Survey Attempts'].mean()
                            return f"The average number of survey attempts is {result:.2f}."
                    except Exception as e:
                        pass
    
    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            context = f"""
            You’re a friendly data analyst helping with medical professional data. The dataset has:
            - NPI: Unique ID for medical professionals
            - State: US state code
            - Usage Time (mins): Time spent using the system
            - Region: Geographic region (Northeast, Midwest, etc.)
            - Speciality: Medical specialty
            - Count of Survey Attempts: Number of survey attempts
            - Login Date/Time, Logout Date/Time
            
            Dataset summary:
            - Total records: {summary['total_records']}
            - Date range: {summary['date_range']['min']} to {summary['date_range']['max']}
            - Most frequent login hour: {summary['frequent_login_hour']}:00
            - Specialties: {', '.join(summary['specialties'][:5]) if len(summary['specialties']) > 5 else ', '.join(summary['specialties'])}... {f"(and {len(summary['specialties'])-5} more)" if len(summary['specialties']) > 5 else ""}
            - Regions: {', '.join(summary['regions'])}
            - States: {', '.join(summary['states'][:10]) if len(summary['states']) > 10 else ', '.join(summary['states'])}... {f"(and {len(summary['states'])-10} more)" if len(summary['states']) > 10 else ""}
            - Usage Time: Avg = {summary['usage_time']['mean']:.2f} mins, Median = {summary['usage_time']['median']:.2f}, Min = {summary['usage_time']['min']}, Max = {summary['usage_time']['max']}
            - Survey Attempts: Avg = {summary['survey_attempts']['mean']:.2f}, Median = {summary['survey_attempts']['median']:.2f}, Min = {summary['survey_attempts']['min']}, Max = {summary['survey_attempts']['max']}
            
            {conversation_context}
            
            Current question: {question}
            
            Answer in a friendly, conversational way based on the data. Be precise with stats and include numbers. If more analysis is needed, let me know. Keep the context of previous questions in mind.
            """
            response = model.generate_content([context])
            response_text = response.text
            if "specific analysis" in response_text.lower() or "more data" in response_text.lower():
                top_specialties = df['Speciality'].value_counts().head(5).to_dict()
                region_usage = df.groupby('Region')['Usage Time (mins)'].mean().to_dict()
                additional_info = f"""
                Here’s some extra insight:
                - Top specialties by count: {top_specialties}
                - Average usage time by region: {region_usage}
                """
                response_text += "\n\n" + additional_info
            return response_text
        except Exception as e:
            return local_fallback_processing(question, df, summary, chat_history)
    else:
        return local_fallback_processing(question, df, summary, chat_history)

def local_fallback_processing(question, df, summary, chat_history):
    question_lower = question.lower()
    if 'how many' in question_lower:
        if 'specialties' in question_lower or 'specialities' in question_lower:
            return f"There are {len(summary['specialties'])} specialties in the dataset."
        elif 'regions' in question_lower:
            return f"There are {len(summary['regions'])} regions in the dataset."
        elif 'states' in question_lower:
            return f"There are {len(summary['states'])} states in the dataset."
        elif 'records' in question_lower or 'entries' in question_lower or 'rows' in question_lower:
            return f"The dataset has {summary['total_records']} records."
    
    if 'frequent login time' in question_lower:
        df['Login Hour'] = df['Login DateTime'].dt.hour
        frequent_hour = df['Login Hour'].mode()[0]
        return f"The most frequent login hour is {frequent_hour}:00."
    if 'total count of npi' in question_lower or 'how many npi' in question_lower:
        return f"The total number of NPIs in the dataset is {summary['total_records']}."
    
    if 'average' in question_lower or 'mean' in question_lower:
        if 'usage time' in question_lower:
            return f"On average, providers spend {summary['usage_time']['mean']:.2f} minutes using the system."
        elif 'survey attempts' in question_lower:
            return f"The average number of survey attempts is {summary['survey_attempts']['mean']:.2f}."
    
    if 'median' in question_lower:
        if 'usage time' in question_lower:
            return f"The median usage time is {summary['usage_time']['median']:.2f} minutes."
        elif 'survey attempts' in question_lower:
            return f"The median number of survey attempts is {summary['survey_attempts']['median']:.2f}."
    
    if 'minimum' in question_lower or 'min' in question_lower:
        if 'usage time' in question_lower:
            return f"The shortest usage time is {summary['usage_time']['min']} minutes."
        elif 'survey attempts' in question_lower:
            return f"The fewest survey attempts is {summary['survey_attempts']['min']}."
    
    if 'maximum' in question_lower or 'max' in question_lower:
        if 'usage time' in question_lower:
            return f"The longest usage time is {summary['usage_time']['max']} minutes."
        elif 'survey attempts' in question_lower:
            return f"The most survey attempts is {summary['survey_attempts']['max']}."
    
    if 'top' in question_lower:
        if 'specialties' in question_lower or 'specialities' in question_lower:
            top_n = 5
            for num in re.findall(r'\d+', question):
                top_n = int(num)
                break
            top_specs = df['Speciality'].value_counts().head(top_n)
            result = "Here are the top specialties by number of providers:\n"
            for specialty, count in top_specs.items():
                result += f"- {specialty}: {count} providers\n"
            return result
        if 'regions' in question_lower:
            top_regions = df['Region'].value_counts().head(5)
            result = "Here are the top regions by number of providers:\n"
            for region, count in top_regions.items():
                result += f"- {region}: {count} providers\n"
            return result
    
    return "I’m not quite sure what you’re asking. Could you rephrase or provide more details?"

def display_message(is_user, message):
    if is_user:
        with st.chat_message("user"):
            st.markdown(message)
    else:
        with st.chat_message("assistant"):
            st.markdown(message)

# Main Streamlit UI
def main():
    tab1, tab2 = st.tabs(["📊 Data Management", "💬 Chat Interface"])

    with tab1:
        st.markdown("### Manage Your Data")
        with st.container():
            st.markdown("#### Upload Dataset", unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="card upload-section">', unsafe_allow_html=True)
                uploaded_file = st.file_uploader("Drag and drop or click to upload a CSV file", type=['csv'])
                if uploaded_file is not None:
                    with st.spinner("Processing your data..."):
                        try:
                            df = pd.read_csv(uploaded_file)
                            processed_df, error_msg = process_uploaded_data(df)
                            if processed_df is not None:
                                st.session_state.df = processed_df
                                st.session_state.summary = generate_data_summary(processed_df)
                                st.success(f"Dataset loaded! Found {len(processed_df)} records.")
                            else:
                                st.error(error_msg)
                        except Exception as e:
                            st.error(f"Oops, something went wrong: {e}")
                
                sample_data = {
                    'NPI': [1234567890, 2345678901, 3456789012, 4567890123, 5678901234],
                    'State': ['CA', 'NY', 'TX', 'FL', 'IL'],
                    'Region': ['West', 'Northeast', 'South', 'South', 'Midwest'],
                    'Speciality': ['Cardiology', 'Neurology', 'Pediatrics', 'Oncology', 'Family Medicine'],
                    'Usage Time (mins)': [45, 60, 30, 75, 25],
                    'Count of Survey Attempts': [3, 5, 2, 4, 1],
                    'Login Date': ['2025-01-15', '2025-01-16', '2025-01-17', '2025-01-18', '2025-01-19'],
                    'Login Time': ['09:00:00', '10:30:00', '08:15:00', '13:45:00', '11:20:00'],
                    'Logout Date': ['2025-01-15', '2025-01-16', '2025-01-17', '2025-01-18', '2025-01-19'],
                    'Logout Time': ['09:45:00', '11:30:00', '08:45:00', '15:00:00', '11:45:00']
                }
                sample_df = pd.DataFrame(sample_data)
                
                col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])
                with col2:
                    if st.button("Load Sample Data", use_container_width=True, key="load_sample_btn"):
                        with st.spinner("Loading sample data..."):
                            try:
                                processed_df, error_msg = process_uploaded_data(sample_df)
                                if processed_df is not None:
                                    st.session_state.df = processed_df
                                    st.session_state.summary = generate_data_summary(processed_df)
                                    st.success(f"Sample data loaded! Found {len(processed_df)} records.")
                                else:
                                    st.error(error_msg)
                            except Exception as e:
                                st.error(f"Error loading sample data: {e}")
                with col3:
                    if st.button("View Sample Data", use_container_width=True, key="view_sample_btn"):
                        st.markdown('<div class="sample-data-table">', unsafe_allow_html=True)
                        st.table(sample_df)
                        st.markdown('</div>', unsafe_allow_html=True)
                with col4:
                    csv = sample_df.to_csv(index=False)
                    st.download_button(
                        label="Download Sample Data",
                        data=csv,
                        file_name="sample_data.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_sample_btn"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.df is not None:
            st.markdown("#### Quick NPI Lookup", unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                col1, col2 = st.columns([4, 1])
                with col1:
                    npi_input = st.text_input("Enter a 10-digit NPI", key="npi_lookup")
                with col2:
                    if st.button("Search", use_container_width=True):
                        if npi_input and npi_input.isdigit() and len(npi_input) == 10:
                            result = get_npi_details(st.session_state.df, npi_input)
                            st.markdown(f'<div class="npi-result">{result}</div>', unsafe_allow_html=True)
                        else:
                            st.error("Please enter a valid 10-digit NPI.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("#### Data Insights", unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                summary = st.session_state.summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", summary['total_records'])
                with col2:
                    st.metric("Avg Usage Time", f"{summary['usage_time']['mean']:.1f} mins")
                with col3:
                    st.metric("Avg Survey Attempts", f"{summary['survey_attempts']['mean']:.1f}")
                
                st.markdown(f"**Date Range**: {summary['date_range']['min']} to {summary['date_range']['max']}")
                
                st.markdown("##### Analyze Data", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    stat_column = st.selectbox("Select Column", 
                                              ['Usage Time (mins)', 'Count of Survey Attempts'],
                                              key="stat_column")
                with col2:
                    stat_operation = st.selectbox("Select Operation", 
                                                 ['mean', 'median', 'min', 'max', 'std'],
                                                 key="stat_op")
                
                filter_option = st.selectbox("Filter By", 
                                            ['None', 'Region', 'Speciality', 'State'],
                                            key="filter_option")
                
                if filter_option != 'None':
                    filter_value = st.selectbox(f"Select {filter_option}", 
                                               st.session_state.df[filter_option].unique(),
                                               key="filter_value")
                    filter_dict = {filter_option: filter_value}
                else:
                    filter_dict = None
                
                if st.button("Calculate", use_container_width=True):
                    result = analyze_data(st.session_state.df, stat_column, stat_operation, filter_dict)
                    if isinstance(result, (float, int)):
                        st.metric("Result", f"{result:.2f}")
                    else:
                        st.write("Result:", result)
                st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="chat-section">', unsafe_allow_html=True)
        st.markdown("### Chat with Your Data")
        if st.session_state.df is None:
            st.info("Please upload a dataset in the Data Management tab to start chatting.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                display_message(True, message["question"])
                display_message(False, message["answer"])
        st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            with st.form(key="question_form", clear_on_submit=True):
                user_question = st.text_area(
                    "Ask a Question",
                    placeholder="E.g., Which state has the most NPIs? or What’s the NPI with max usage time?",
                    key="question_input",
                    height=100
                )
                col1, col2 = st.columns([1, 1])
                with col1:
                    submit_button = st.form_submit_button("Send", use_container_width=True)
                with col2:
                    if st.form_submit_button("Clear Chat", use_container_width=True):
                        st.session_state.chat_history = []
                        st.rerun()
            if submit_button and user_question:
                display_message(True, user_question)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = process_question(
                            user_question, 
                            st.session_state.df, 
                            st.session_state.summary,
                            st.session_state.chat_history
                        )
                        st.session_state.chat_history.append({
                            "question": user_question,
                            "answer": response,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.markdown(response)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="conversation-history-section">', unsafe_allow_html=True)
        st.markdown("#### Conversation History")
        if st.session_state.chat_history:
            for i, exchange in enumerate(st.session_state.chat_history):
                with st.expander(f"Q{i+1}: {exchange['question'][:50]}..." if len(exchange['question']) > 50 else f"Q{i+1}: {exchange['question']}"):
                    st.markdown(f"**Question**: {exchange['question']}")
                    st.markdown(f"**Answer**: {exchange['answer']}")
        else:
            st.write("No questions asked yet. Start chatting above!")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if not st.session_state.chat_history:
            st.markdown("#### Try These Questions", unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                sample_questions = [
                    ("What are all the specialties?", "sample1"),
                    ("What is the average usage time?", "sample2"),
                    ("Which region has the highest survey attempts?", "sample3"),
                    ("Show me the NPI range in the dataset", "sample4")
                ]
                for idx, (col, (question, key)) in enumerate(zip([col1, col2, col3, col4], sample_questions)):
                    with col:
                        if st.button(question, key=key, use_container_width=True):
                            st.session_state.chat_history.append({
                                "question": question,
                                "answer": process_question(
                                    question, 
                                    st.session_state.df, 
                                    st.session_state.summary,
                                    []
                                ),
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()