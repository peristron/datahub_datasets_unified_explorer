#   brightspace_datasets_explorer_unified02092026.py, LKG 02092026
# =============================================================================
# unified brightspace dataset explorer
# combines the best of all three code-bases with simple/advanced modes
# run: streamlit run brightspace_datasets_explorer_unified02092026.py
# =============================================================================

import streamlit as st
st.cache_data.clear()   # temporary debug line
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import openai
import re
import logging
from typing import List, Dict, Optional
import math

# =============================================================================
# 1. app configuration & styling
# =============================================================================

st.set_page_config(
    page_title="Brightspace Datasets Expl0rer",
    layout="wide",
    page_icon="🔗",
    initial_sidebar_state="expanded"
)

#------------------------------
# configure structured logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# suppress insecure request warnings for d2l scrapers
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
#------------------------------
# known acronyms that .title() would corrupt (e.g., "JIT" -> "Jit")
PRESERVE_ACRONYMS = [
    'JIT', 'LTI', 'SIS', 'CPD', 'SCORM', 'PLOE',
    'SSO', 'API', 'IPSIS', 'UDF', 'LMS', 'SAML', 'LDAP',
]


def smart_title(text: str) -> str:
    """applies title case while preserving known acronyms."""
    if not text:
        return text
    result = text.title()
    for acronym in PRESERVE_ACRONYMS:
        titled_form = acronym.title()  # e.g., "JIT" -> "Jit", "SCORM" -> "Scorm"
        result = re.sub(r'\b' + re.escape(titled_form) + r'\b', acronym, result)
    return result
# apply professional ui css
st.markdown("""
<style>
    /* metric cards styling */
    div[data-testid="stMetric"] {
        background-color: #1E232B;
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    div[data-testid="stMetricLabel"] { color: #8B949E; }
    div[data-testid="stMetricValue"] { color: #58A6FF; font-size: 24px; }

    /* tabs styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #0E1117;
        border-radius: 4px;
        padding: 8px 16px;
        color: #C9D1D9;
        border: 1px solid transparent;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #238636;
        color: white;
        border-color: #30363D;
    }

    /* code blocks */
    .stCode { font-family: 'Fira Code', monospace; }

    /* sidebar expander styling */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        font-size: 1.1rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    /* hub badge styling */
    .hub-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4px 12px;
        border-radius: 12px;
        color: white;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. constants (urls, pricing registry)
# =============================================================================

DEFAULT_URLS = """
https://community.d2l.com/brightspace/kb/articles/4539-list-of-advanced-data-sets
https://community.d2l.com/brightspace/kb/articles/4752-accommodations-data-sets
https://community.d2l.com/brightspace/kb/articles/4712-activity-feed-data-sets
https://community.d2l.com/brightspace/kb/articles/4723-announcements-data-sets
https://community.d2l.com/brightspace/kb/articles/4767-assignments-data-sets
https://community.d2l.com/brightspace/kb/articles/4519-attendance-data-sets
https://community.d2l.com/brightspace/kb/articles/4520-awards-data-sets
https://community.d2l.com/brightspace/kb/articles/4521-calendar-data-sets
https://community.d2l.com/brightspace/kb/articles/4523-checklist-data-sets
https://community.d2l.com/brightspace/kb/articles/4754-competency-data-sets
https://community.d2l.com/brightspace/kb/articles/4713-content-data-sets
https://community.d2l.com/brightspace/kb/articles/22812-content-service-data-sets
https://community.d2l.com/brightspace/kb/articles/26020-continuous-professional-development-cpd-data-sets
https://community.d2l.com/brightspace/kb/articles/4725-course-copy-data-sets
https://community.d2l.com/brightspace/kb/articles/4524-course-publisher-data-sets
https://community.d2l.com/brightspace/kb/articles/26161-creator-data-sets
https://community.d2l.com/brightspace/kb/articles/4525-discussions-data-sets
https://community.d2l.com/brightspace/kb/articles/4526-exemptions-data-sets
https://community.d2l.com/brightspace/kb/articles/4527-grades-data-sets
https://community.d2l.com/brightspace/kb/articles/4528-intelligent-agents-data-sets
https://community.d2l.com/brightspace/kb/articles/5782-jit-provisioning-data-sets
https://community.d2l.com/brightspace/kb/articles/4714-local-authentication-data-sets
https://community.d2l.com/brightspace/kb/articles/4727-lti-data-sets
https://community.d2l.com/brightspace/kb/articles/4529-organizational-units-data-sets
https://community.d2l.com/brightspace/kb/articles/4796-outcomes-data-sets
https://community.d2l.com/brightspace/kb/articles/4530-portfolio-data-sets
https://community.d2l.com/brightspace/kb/articles/4531-questions-data-sets
https://community.d2l.com/brightspace/kb/articles/4532-quizzes-data-sets
https://community.d2l.com/brightspace/kb/articles/4533-release-conditions-data-sets
https://community.d2l.com/brightspace/kb/articles/33182-reoffer-course-data-sets
https://community.d2l.com/brightspace/kb/articles/4534-role-details-data-sets
https://community.d2l.com/brightspace/kb/articles/4535-rubrics-data-sets
https://community.d2l.com/brightspace/kb/articles/4536-scorm-data-sets
https://community.d2l.com/brightspace/kb/articles/4537-sessions-and-system-access-data-sets
https://community.d2l.com/brightspace/kb/articles/19147-sis-course-merge-data-sets
https://community.d2l.com/brightspace/kb/articles/33427-source-course-deploy-data-sets
https://community.d2l.com/brightspace/kb/articles/4538-surveys-data-sets
https://community.d2l.com/brightspace/kb/articles/4540-tools-data-sets
https://community.d2l.com/brightspace/kb/articles/4740-users-data-sets
https://community.d2l.com/brightspace/kb/articles/4541-virtual-classroom-data-sets
""".strip()

# define supported ai models and their costs (usd per 1m tokens)
PRICING_REGISTRY = {
    # xai models
    "grok-2-1212":        {"in": 2.00, "out": 10.00, "provider": "xAI"},
    "grok-2-vision-1212": {"in": 2.00, "out": 10.00, "provider": "xAI"},
    "grok-3":             {"in": 3.00, "out": 15.00, "provider": "xAI"},
    "grok-3-mini":        {"in": 0.30, "out": 0.50,  "provider": "xAI"},
    "grok-4-0709":        {"in": 3.00, "out": 15.00, "provider": "xAI"},

    # openai models
    "gpt-4o":       {"in": 2.50, "out": 10.00, "provider": "OpenAI"},
    "gpt-4o-mini":  {"in": 0.15, "out": 0.60,  "provider": "OpenAI"},
    "gpt-4.1":      {"in": 2.00, "out": 8.00,  "provider": "OpenAI"},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60,  "provider": "OpenAI"},
}

#------------------------------
# common d2l enumeration mappings (the "decoder ring")
ENUM_DEFINITIONS = {
    "GradeObjectTypeId": {
        1: "Numeric", 2: "Pass/Fail", 3: "Selectbox", 4: "Text",
        6: "Calculated", 7: "Formula"
    },
#------------------------------
    "OrgUnitTypeId": {
        1: "Organization", 2: "Course Offering", 3: "Course Template",
        4: "Department", 5: "Semester", 6: "Group", 7: "Section",
        8: "Program", 9: "Faculty"  # Common additional OrgUnit types in D2L
    },
    "SessionType": {
        1: "Instructor", 2: "Student", 3: "Admin", 4: "Impersonated"
    },
    "ActionType": {
        1: "Login", 2: "Logout", 3: "Time Out", 4: "Impersonated"
    },
    "InputDeviceType": {
        1: "PC", 2: "Mobile", 3: "Tablet"
    },
    "OutcomeType": {
        1: "General", 2: "Specific", 3: "Program"
    },
    "CompletionStatus": {
        0: "Unknown", 1: "Incomplete", 2: "Complete"
    },
    "ContentTypeId": {
        1: "Module", 2: "Topic", 3: "Link", 4: "File"
    },
    "QuizAttemptStatus": {
        0: "In Progress", 1: "Submitted", 2: "Graded"
    },
    "AssignmentSubmissionType": {
        1: "File", 2: "Text", 3: "On Paper", 4: "Observed in Person"
    },
    "EnrollmentRoleId": {
        110: "Learner", 111: "Instructor", 112: "TA",
        113: "Course Builder", 114: "Auditor"
    },
    "DiscussionPostType": {
        1: "Thread Starter", 2: "Reply"
    }
}

# SQL templates for common business metrics
RECIPE_REGISTRY = {
    "Learner Engagement": [
        {
            "title": "Course Access Frequency",
            "description": "Counts how many times each student accessed a specific course, including their last access date.",
            "datasets": ["Users", "Organizational Units", "Course Access"],
            "difficulty": "Intermediate",
            "sql_template": """
SELECT
    u.UserName,
    o.Name AS CourseName,
    COUNT(ca.CourseAccessId) AS TotalLogins,
    MAX(ca.LastAccessed) AS LastLoginDate
FROM CourseAccess ca
INNER JOIN Users u ON ca.UserId = u.UserId
INNER JOIN OrganizationalUnits o ON ca.OrgUnitId = o.OrgUnitId
GROUP BY u.UserName, o.Name
ORDER BY TotalLogins DESC
"""
        },
        {
            "title": "Inactive Students (At-Risk)",
            "description": "Identifies students who have not accessed the system in the last 30 days.",
            "datasets": ["Users", "System Access Log"],
            "difficulty": "Basic",
            "sql_template": """
SELECT
    u.UserName,
    u.FirstName,
    u.LastName,
    MAX(sal.Timestamp) AS LastSystemAccess
FROM Users u
LEFT JOIN SystemAccessLog sal ON u.UserId = sal.UserId
GROUP BY u.UserName, u.FirstName, u.LastName
HAVING MAX(sal.Timestamp) < DATEADD(day, -30, GETDATE()) -- Note: Syntax varies by DB
"""
        }
    ],
    "Assessments & Grades": [
        {
            "title": "Grade Distribution by Course",
            "description": "Calculates the average grade for each course offering to identify outliers.",
            "datasets": ["Grade Objects", "Grade Results", "Organizational Units"],
            "difficulty": "Intermediate",
            "sql_template": """
SELECT
    o.Name AS CourseName,
    go.Name AS AssignmentName,
    AVG(gr.PointsNumerator) AS AverageScore,
    COUNT(gr.UserId) AS SubmissionCount
FROM GradeResults gr
JOIN GradeObjects go ON gr.GradeObjectId = go.GradeObjectId
JOIN OrganizationalUnits o ON go.OrgUnitId = o.OrgUnitId
WHERE go.GradeObjectTypeId = 1 -- Numeric Grades only
GROUP BY o.Name, go.Name
"""
        },
        {
            "title": "Quiz Item Analysis",
            "description": "Analyzes which specific questions (InteractionIds) are most frequently answered incorrectly.",
            "datasets": ["Quiz Attempts", "Quiz User Answers", "Quiz Objects"],
            "difficulty": "Advanced",
            "sql_template": """
SELECT
    qo.Name AS QuizName,
    qua.QuestionId,
    COUNT(CASE WHEN qua.IsCorrect = 0 THEN 1 END) AS IncorrectCount,
    COUNT(qua.AttemptId) AS TotalAttempts,
    (COUNT(CASE WHEN qua.IsCorrect = 0 THEN 1 END) * 100.0 / COUNT(qua.AttemptId)) AS FailureRate
FROM QuizUserAnswers qua
JOIN QuizAttempts qa ON qua.AttemptId = qa.AttemptId
JOIN QuizObjects qo ON qa.QuizId = qo.QuizId
GROUP BY qo.Name, qua.QuestionId
ORDER BY FailureRate DESC
"""
        }
    ],
    "Data Cleaning & Deduplication": [
        {
            "title": "Get Latest Row Version",
            "description": "Many datasets (like Activity Feed) track edits using a 'Version' column. Use this pattern to filter for only the most recent version of each record.",
            "datasets": ["Activity Feed Post Objects", "Content Objects", "Wiki Pages"],
            "difficulty": "Advanced",
            "sql_template": """
WITH RankedRecords AS (
    SELECT
        *,
        -- Partition by the Primary Key, Order by Version DESC
        ROW_NUMBER() OVER (
            PARTITION BY ActivityId
            ORDER BY Version DESC
        ) as RowNum
    FROM ActivityFeedPostObjects
)
SELECT *
FROM RankedRecords
WHERE RowNum = 1 -- Keeps only the latest version
"""
        }
    ]
}

# =============================================================================
# 3. session state management
# =============================================================================

#------------------------------
def init_session_state():
    """initializes streamlit session state variables safely."""
    defaults = {
        'authenticated': False,
        'auth_error': False,
        'messages': [],
        'total_cost': 0.0,
        'total_tokens': 0,
        'experience_mode': 'simple',  # 'simple' or 'advanced'
        'selected_datasets': [],
        'scrape_msg': None,
        'show_url_editor': False,
        'custom_urls': None,
        #------------------------------
        'show_health_check': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# =============================================================================
# 4. authentication logic
# =============================================================================

#------------------------------
def get_secret(key_name: str) -> Optional[str]:
    """retrieves a secret, checking as-provided, lowercase, and uppercase variations."""
    return (
        st.secrets.get(key_name) or
        st.secrets.get(key_name.lower()) or
        st.secrets.get(key_name.upper())
    )


def perform_login():
    """verifies password against streamlit secrets or allows dev mode."""
    pwd_secret = get_secret("app_password")

    # dev mode: if no secret is configured, allow access
    if not pwd_secret:
        logger.warning("No password configured. Allowing open access.")
        st.session_state['authenticated'] = True
        return

    # production mode: check input
    if st.session_state.get("password_input") == pwd_secret:
        st.session_state['authenticated'] = True
        st.session_state['auth_error'] = False
    else:
        st.session_state['auth_error'] = True
        st.session_state['authenticated'] = False


#------------------------------
def logout():
    """clears authentication state."""
    st.session_state['authenticated'] = False
    st.session_state['auth_error'] = False
    st.session_state['password_input'] = ""
    st.session_state['messages'] = []


#------------------------------
def clear_all_selections():
    """clears all dataset selections."""
    st.session_state['selected_datasets'] = []
    # clear path finder results
    if 'path_finder_results' in st.session_state:
        del st.session_state['path_finder_results']
    # clear any selection-related keys
    for key in list(st.session_state.keys()):
        if key.startswith("sel_") or key == "global_search" or key == "dataset_multiselect":
            if isinstance(st.session_state.get(key), list):
                st.session_state[key] = []

# =============================================================================
# 5. data layer (scraper & storage)
# =============================================================================

#------------------------------
def clean_description(text: str) -> str:
    """
    Logic to convert raw documentation text into a concise summary.
    Removes boilerplate like 'The User data set describes...'
    """
    if not text:
        return ""

    # 1. Normalize whitespace (collapse multiple spaces, strip leading/trailing)
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Remove common D2L boilerplate patterns
    boilerplate_patterns = [
        r'^The .*? data set (describes|contains|provides|returns|includes) ',
        r'^This (data set|table|dataset) (describes|contains|provides|returns|includes) ',
        r'^Use this data set to ',
        r'^This is a ',
        r'^Contains ',
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 3. Remove any leading connector words left after stripping
    text = re.sub(r'^(the|a|an|all|each|every) ', '', text, flags=re.IGNORECASE)

    # 4. Capitalize first letter if needed
    if text:
        text = text[0].upper() + text[1:]

#------------------------------
    # 5. Remove "See also" or "Related" references before splitting
    text = re.sub(r'(?i)(see also|related (data|link)|more info|details at).*$', '', text)

    # 6. Limit to the first 2 complete sentences for brevity
    sentences = re.split(r'(?<=[.!?]) +', text)
    summary = ' '.join(sentences[:2]).rstrip('.,!?')  # Strip trailing punctuation if incomplete

    # 7. Ensure summary ends with proper punctuation (avoid trailing fragments)
    if summary and summary[-1] not in '.!?':
        # Find last sentence boundary and truncate
        last_boundary = max(summary.rfind('.'), summary.rfind('!'), summary.rfind('?'))
        if last_boundary > 0:
            summary = summary[:last_boundary + 1]

    return summary



def scrape_table(url: str, category_name: str) -> List[Dict]:
    """
    parses a d2l knowledge base page to extract dataset definitions AND context descriptions.
    returns a list of dictionaries representing columns.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    #------------
    IGNORE_HEADERS = [
        # boilerplate
        "returned fields", "available filters", "required permission", "required permissions",
        "about", "notes", "filters", "description", "version history", "table of contents",
        "referenced data sets", "interpreting the data",
        "about data hub", "set up data hub", "export data in data hub",
        "brightspace data sets", "advanced data sets", "performance+",
        "brightspace parent & guardian", "creator+", "accessibility",
        "platform requirements", "hosting", "user administration",
        "org administration", "system administration", "security administration",
        "release information", "documentation",

        # --- added fixes
        "required config variable", 
        # (removed: "program level outcomes evaluation (ploe)")
        "practices in learning objects and data sets",
        "entity relationship diagram",
        "note: calculating content completed",
        "generating the report"
    ]

    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        if response.status_code != 200:
            logger.warning(f"Status {response.status_code} for {url}")
            return []

#------------------------------
        soup = BeautifulSoup(response.text, 'html.parser')
        data = []
        current_dataset = category_name
        current_desc = ""

        elements = soup.find_all(['h2', 'h3', 'h4', 'table'])

        #----------------------------------------------------------------------------------------
        for element in elements:
            if element.name in ['h2', 'h3', 'h4']:
                text = element.text.strip()
                clean_text_lower = text.lower()

#------------------------------
                if any(x == clean_text_lower for x in IGNORE_HEADERS):
                    continue
                # substring check catches variants like "Returned Fields For Quizzes"
                # that the exact match above would miss
                if any(phrase in clean_text_lower for phrase in ["returned fields", "available filters"]):
                    continue
                if clean_text_lower.startswith("about "):  # NEW: Ignore subheaders like "About Time Tracking" to prevent overwrite
                    continue

#------------------------------
                if len(text) > 3:
                    current_dataset = smart_title(text)

                    next_sibling = element.find_next_sibling()
                    if next_sibling and next_sibling.name == 'p':
                        raw_text = next_sibling.text.strip()
                        current_desc = clean_description(raw_text)
                    else:
                        current_desc = ""

            elif element.name == 'table':
                rows = element.find_all('tr')
                if not rows:
                    continue

#------------------------------
                header_cells = element.find_all('th')
                if not header_cells and rows:
                    header_cells = rows[0].find_all('td')
                    data_rows = rows[1:]
                else:
                    data_rows = rows[1:]  # skip the header <tr> which contains <th> cells

                if not header_cells:
                    continue

                table_headers = [th.text.strip().lower().replace(' ', '_') for th in header_cells]

                valid_indicators = ['type', 'description', 'data_type', 'field', 'name', 'column']
                if not table_headers or not any(x in table_headers for x in valid_indicators):
                    continue

                for row in data_rows:
                    columns_ = row.find_all('td')
                    if not columns_ or len(columns_) < len(table_headers):
                        continue

                    entry = {}
                    for i, header in enumerate(table_headers):
                        if i < len(columns_):
                            entry[header] = columns_[i].text.strip()

#------------------------------
#------------------------------
                    header_map = {
                        'field': 'column_name',
                        'field_name': 'column_name',
                        'name': 'column_name',
                        'column': 'column_name',
                        'type': 'data_type',
                        'data_type': 'data_type',
                        'description': 'description',
                        'can_be_null?': 'is_nullable',
                        'version_added': 'version_history',
                        'version': 'version_history'
                    }

                    clean_entry = {header_map.get(k, k): v for k, v in entry.items()}

                    if 'column_name' in clean_entry and clean_entry['column_name']:
                        col = clean_entry['column_name']

                        # --- FINAL NORMALIZATION LOGIC ---

#------------------------------
                        # 1. Fix Capitalization of "ID" at end of word (OrgUnitID -> OrgUnitId)
                        # Require a preceding lowercase letter to avoid corrupting "GUID" -> "GUId"
                        col = re.sub(r'(?<=[a-z])I[dD]\b', 'Id', col)

                        # 2. Remove ALL spaces (Org Unit Id -> OrgUnitId)
                        # This aligns "Advanced Report" headers with "Standard Extract" headers
                        col = col.replace(' ', '')

                        # ---------------------------------

                        clean_entry['column_name'] = col
                        clean_entry['dataset_name'] = current_dataset
                        clean_entry['category'] = category_name
                        clean_entry['url'] = url
                        clean_entry['dataset_description'] = current_desc

                        if 'key' not in clean_entry:
                            clean_entry['key'] = ''

                        data.append(clean_entry)

        return data
    except Exception as e:
        logger.error(f"Failed to scrape {url}: {e}")
        return []


def scrape_and_save(urls: List[str]) -> pd.DataFrame:
    """
    orchestrates the scraping process using threadpoolexecutor.
    saves the result to 'dataset_metadata.csv'.
    """
    all_data = []
    progress_bar = st.progress(0, "Initializing Scraper...")

    # helper to clean urls and extract category
#------------------------------
    def extract_category(url):
        filename = os.path.basename(url).split('?')[0]

        if "advanced" in filename.lower():
            return "Advanced Data Sets"

        # strip leading digits AND any leading dashes/spaces left behind
        clean_name = re.sub(r'^[\d\s-]+', '', filename)
        return clean_name.replace('-data-sets', '').replace('-', ' ').strip().lower()

    with ThreadPoolExecutor(max_workers=10) as executor:
        args = [(url, extract_category(url)) for url in urls]
        future_to_url = {executor.submit(scrape_table, *arg): arg[0] for arg in args}

        for i, future in enumerate(future_to_url):
            try:
                result = future.result()
                all_data.extend(result)
            except Exception as e:
                logger.error(f"Thread error: {e}")

            progress_bar.progress((i + 1) / len(urls), f"Scraping {i + 1}/{len(urls)}...")

    progress_bar.empty()

    if not all_data:
        st.error("Scraper returned no data. Check URLs.")
        return pd.DataFrame()

    # create dataframe
    df = pd.DataFrame(all_data)
    df = df.fillna('')

    # clean up text - title case for readability
#------------------------------
#------------------------------
    df['dataset_name'] = df['dataset_name'].astype(str).apply(smart_title)
    df['category'] = df['category'].astype(str).apply(smart_title)

    # tag dataset type: reports (Advanced Data Sets) vs extracts (everything else)
    df['dataset_type'] = df['category'].apply(
        lambda c: 'report' if 'advanced' in str(c).lower() else 'extract'
    )

    # ensure expected columns exist
    expected_cols = ['category', 'dataset_name', 'dataset_description', 'column_name',
                     'data_type', 'description', 'key', 'url', 'version_history',
                     'is_nullable', 'dataset_type']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ''

    # logic flags for joins based on key column
    df['is_primary_key'] = df['key'].astype(str).str.contains(r'\bpk\b', case=False, regex=True)
    df['is_foreign_key'] = df['key'].astype(str).str.contains(r'\bfk\b', case=False, regex=True)

    # persist to csv
    df.to_csv('dataset_metadata.csv', index=False)
    logger.info(f"Scraping complete. Saved {len(df)} rows.")

    # Store fresh data in session state and clear cache
    st.session_state['current_df'] = df.copy()
    st.session_state['scrape_msg'] = f"Success: {df['dataset_name'].nunique()} datasets loaded"
    
    load_data.clear()   # Force reload on next run
    st.rerun()
    
    return df   # Keep return for backward compatibility if needed elsewhere

#------------------------------

@st.cache_data(ttl=60)  # Cache for 60 seconds to reduce disk I/O on every rerun
def load_data() -> pd.DataFrame:
    """Loads the latest metadata. Prefers fresh data from session_state, then falls back to disk."""
    # 1. Prefer fresh data from a recent scrape
    if 'current_df' in st.session_state:
        return st.session_state.current_df.copy()   # .copy() prevents accidental mutation

    # 2. Fall back to disk
    if os.path.exists('dataset_metadata.csv') and os.path.getsize('dataset_metadata.csv') > 100:
        try:
            df = pd.read_csv('dataset_metadata.csv', encoding='utf-8').fillna('')
            return df
        except Exception as e:
            logger.error(f"Failed to load metadata CSV: {e}")
    
    return pd.DataFrame()


import hashlib

@st.cache_data(ttl=300)  # 5-minute cache — good balance
def get_possible_joins(df_hash: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates join conditions.
    Nuses a much stronger, content-aware cache key
    """
    if df.empty:
        return pd.DataFrame()



    # Filter to extracts only — Advanced Data Sets (reports) are pre-joined
    # and should not participate in the relationship graph.
    # If 'dataset_type' column is missing (old CSV), treat everything as extract.
    if 'dataset_type' in df.columns:
        extract_df = df[df['dataset_type'] != 'report']
    else:
        extract_df = df

    if extract_df.empty:
        return pd.DataFrame()

    # 1. Identify definitive Primary Keys (Targets) from Scrape Data
    # We create a copy so we can inject implicit keys without altering the main df
    pks = extract_df[extract_df['is_primary_key'] == True].copy()
    
    # --- SAFETY NET: HARDCODED / IMPLICIT PRIMARY KEYS ---
    # This ensures that even if the scraper misses the "PK" flag in the docs (common for Hubs),
    # we force the application to recognize these tables as the "Owners" of these keys.
#------------------------------
    IMPLICIT_PKS = {
        'Users': ['UserId'],
        'Organizational Units': ['OrgUnitId'],
        'Role Details': ['RoleId'],
        'Semester': ['SemesterId'],
        'Department': ['DepartmentId'],
        'Course Offerings': ['CourseOfferingId'],
        'Quiz Objects': ['QuizId'],
        'Assignment Objects': ['AssignmentId'],
        'Content Objects': ['ContentObjectId'],
        'Discussion Forums': ['ForumId'],
        'Discussion Topics': ['TopicId']
    }

    # Inject implicit PKs if they exist in the dataframe but aren't currently flagged in 'pks'
    for ds_name, cols in IMPLICIT_PKS.items():
        for col in cols:
            # Check if this dataset+column exists in the loaded data
#------------------------------
            mask = (extract_df['dataset_name'] == ds_name) & (extract_df['column_name'] == col)
            if mask.any():
                # Check if it's already in our pks list
                already_exists = not pks[
                    (pks['dataset_name'] == ds_name) & (pks['column_name'] == col)
                ].empty
                
                if not already_exists:
                    # Add it to the PKs list manually
#------------------------------
                    row = extract_df[mask].iloc[0].to_dict()
                    row['is_primary_key'] = True
                    pks = pd.concat([pks, pd.DataFrame([row])], ignore_index=True)

    if pks.empty:
        return pd.DataFrame()

    # 2. Identify Potential Foreign Keys (Sources)
    pk_names = pks['column_name'].unique()
    
    # CRITICAL FIX: The Logic must allow a column to be a FK if:
    # A) It is explicitly marked is_foreign_key=True (even if is_primary_key is also True)
    # B) OR It is NOT a primary key (standard FK)
#------------------------------
    potential_fks = extract_df[
        (extract_df['column_name'].isin(pk_names)) & 
        (
            (extract_df['is_primary_key'] == False) | 
            (extract_df['is_foreign_key'] == True)
        )
    ].copy()
    
    # Remove rows where the dataset is the same as the PK dataset (prevent strict self-joins here)
    # We do this by ensuring we don't match a table to itself in the next step, 
    # but we can also filter here if needed. For now, the merge + filtering later handles it.

    # 3. Perform Exact Match Merge
    exact_joins = pd.merge(potential_fks, pks, on='column_name', suffixes=('_fk', '_pk'))

    # 4. Perform Synonym/Alias Match (The "Smart" Logic)
#------------------------------
    # alias_map: maps variant column names to (canonical_pk, hub_dataset)
    # the hub_dataset constraint prevents spurious joins to any table
    # that happens to have the canonical PK (e.g., UserId in composite keys)
    alias_map = {
        'CourseOfferingId': ('OrgUnitId', 'Organizational Units'),
        'SectionId':        ('OrgUnitId', 'Organizational Units'),
        'DepartmentId':     ('OrgUnitId', 'Organizational Units'),
        'SemesterId':       ('OrgUnitId', 'Organizational Units'),
        'ParentOrgUnitId':  ('OrgUnitId', 'Organizational Units'),
        'AuditorId':        ('UserId', 'Users'),
        'EvaluatorId':      ('UserId', 'Users'),
        'AssignedToUserId': ('UserId', 'Users'),
        'CreatedBy':        ('UserId', 'Users'),
        'ActionUserId':     ('UserId', 'Users'),
        'TargetUserId':     ('UserId', 'Users'),
        'LastModifiedBy':   ('UserId', 'Users'),
    }

    alias_joins = pd.DataFrame()
    for alias_col, (target_pk, hub_dataset) in alias_map.items():
#------------------------------
        alias_candidates = extract_df[
            (extract_df['column_name'] == alias_col) & 
            (
                (extract_df['is_primary_key'] == False) | 
                (extract_df['is_foreign_key'] == True)
            )
        ]
        
        # constrain to ONLY the canonical hub table for this PK
        target_pk_rows = pks[
            (pks['column_name'] == target_pk) &
            (pks['dataset_name'] == hub_dataset)
        ]
        
        if not alias_candidates.empty and not target_pk_rows.empty:
            temp_join = pd.merge(alias_candidates, target_pk_rows, how='cross', suffixes=('_fk', '_pk'))
            temp_join['column_name'] = temp_join['column_name_fk']
            alias_joins = pd.concat([alias_joins, temp_join])

    # 5. Combine and Clean
    all_joins = pd.concat([exact_joins, alias_joins], ignore_index=True)

    if all_joins.empty:
        return pd.DataFrame()

    # Exclude self-joins (joining a table to itself)
    joins = all_joins[all_joins['dataset_name_fk'] != all_joins['dataset_name_pk']]
    
    # Final cleanup to ensure distinct relationships
    joins = joins.drop_duplicates(subset=['dataset_name_fk', 'column_name', 'dataset_name_pk'])

    return joins


def get_joins(df: pd.DataFrame) -> pd.DataFrame:
    """wrapper to call cached join calculation with hash for cache key."""
    if df.empty:
        return pd.DataFrame()
    # create a simple hash for cache invalidation
    df_hash = f"{len(df)}_{df['dataset_name'].nunique()}"
    return get_possible_joins(df_hash, df)


def get_joins_for_selection(df: pd.DataFrame, selected_datasets: List[str]) -> pd.DataFrame:
    """
    Filter the global join table to only relationships that touch the selected datasets.
    This keeps all tools (graphs, tables, builder) consistent with get_joins(df).
    """
    if df.empty or not selected_datasets:
        return pd.DataFrame()

    joins = get_joins(df)
    if joins.empty:
        return pd.DataFrame()

    selected = set(selected_datasets)

    # Keep joins where either side is in the selected set
    filtered = joins[
        joins['dataset_name_fk'].isin(selected) |
        joins['dataset_name_pk'].isin(selected)
    ].copy()

    if filtered.empty:
        return pd.DataFrame()

    # Shape into the structure expected by downstream UI
    filtered = filtered.rename(columns={
        'dataset_name_fk': 'Source Dataset',
        'dataset_name_pk': 'Target Dataset',
        'category_pk': 'Target Category'
    })

    return (
        filtered[['Source Dataset', 'column_name', 'Target Dataset', 'Target Category']]
        .drop_duplicates()
        .reset_index(drop=True)
    )

# =============================================================================
# 6. analysis helpers
# =============================================================================

def get_dataset_connectivity(df: pd.DataFrame) -> pd.DataFrame:
    """calculates connectivity metrics for all datasets."""
    joins = get_joins(df)
    datasets = df['dataset_name'].unique()

    connectivity = []
    for ds in datasets:
        if joins.empty:
            outgoing = 0
            incoming = 0
        else:
            outgoing = len(joins[joins['dataset_name_fk'] == ds])
            incoming = len(joins[joins['dataset_name_pk'] == ds])

#------------------------------
        ds_subset = df[df['dataset_name'] == ds]
        category = ds_subset['category'].iloc[0] if not ds_subset.empty else ''

        connectivity.append({
            'dataset_name': ds,
            'outgoing_fks': outgoing,
            'incoming_fks': incoming,
            'total_connections': outgoing + incoming,
            'category': category
        })

#------------------------------
    return pd.DataFrame(connectivity).sort_values(
        ['total_connections', 'incoming_fks'],
        ascending=[False, False]
    )


#------------------------------
def get_hub_datasets(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """returns the most connected datasets (hubs)."""
    connectivity = get_dataset_connectivity(df)
    hubs = connectivity[connectivity['total_connections'] > 0]
    return hubs.head(top_n)

#------------------------------
def get_orphan_datasets(df: pd.DataFrame) -> List[str]:
    """returns datasets with zero connections, sorted alphabetically."""
    connectivity = get_dataset_connectivity(df)
    orphans = connectivity[connectivity['total_connections'] == 0]['dataset_name'].tolist()
    return sorted(orphans)


#------------
def find_all_paths(df: pd.DataFrame,
                   source_dataset: str,
                   target_dataset: str,
                   cutoff: int = 4,
                   limit: int = 5,
                   allowed_keys: Optional[List[str]] = None) -> List[List[str]]:
    """
    Finds paths using Shortest Path First strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Full metadata dataframe.
    source_dataset : str
        Starting dataset name.
    target_dataset : str
        Ending dataset name.
    cutoff : int
        Maximum number of hops (joins) allowed.
    limit : int
        Maximum number of paths to return.
    allowed_keys : Optional[List[str]]
        If provided, restricts edges to joins whose column_name is in this list
        (e.g., ['UserId', 'OrgUnitId']) to bias paths toward core dimensions.

    Returns
    -------
    List[List[str]]
        A list of paths, where each path is a list of dataset names.
    """
    joins = get_joins(df)

    if joins.empty:
        return []

    # Optional filter: restrict to certain join keys (e.g., UserId, OrgUnitId)
    if allowed_keys:
        allowed_set = set(allowed_keys)
        joins = joins[joins['column_name'].isin(allowed_set)]

        if joins.empty:
            return []

    # Build undirected graph from (possibly filtered) joins
    G = nx.Graph()
    for _, r in joins.iterrows():
        G.add_edge(r['dataset_name_fk'], r['dataset_name_pk'], key=r['column_name'])

    try:
        path_generator = nx.shortest_simple_paths(G, source_dataset, target_dataset)

        valid_paths: List[List[str]] = []
        for path in path_generator:
            # checks length constraint (nodes = hops + 1)
            if len(path) - 1 > cutoff:
                break

            valid_paths.append(path)

            # checks quantity constraint
            if len(valid_paths) >= limit:
                break

        return valid_paths
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def get_path_details(df: pd.DataFrame, path: List[str]) -> List[Dict]:
    """gets the join column details for each step in a path."""
    if not path or len(path) < 2:
        return []

    joins = get_joins(df)
    if joins.empty:
        return []

    details = []
    for i in range(len(path) - 1):
        src = path[i]
        tgt = path[i + 1]

        # find the join column
        match = joins[
            ((joins['dataset_name_fk'] == src) & (joins['dataset_name_pk'] == tgt)) |
            ((joins['dataset_name_fk'] == tgt) & (joins['dataset_name_pk'] == src))
        ]

        if not match.empty:
            details.append({
                'from': src,
                'to': tgt,
                'column': match.iloc[0]['column_name']
            })
        else:
            details.append({
                'from': src,
                'to': tgt,
                'column': '?'
            })

    return details


#------------------------------
def show_relationship_summary(df: pd.DataFrame, dataset_name: str, show_details: bool = True):
    """
    shows quick stats about a dataset's connectivity.
    optionally shows the actual connected dataset names.
    """
    joins = get_joins(df)

    if joins.empty:
        outgoing_joins = pd.DataFrame()
        incoming_joins = pd.DataFrame()
    else:
        # Outgoing: This dataset HAS the Foreign Key (it points TO others)
        outgoing_joins = joins[joins['dataset_name_fk'] == dataset_name]
        # Incoming: This dataset HAS the Primary Key (others point TO it)
        incoming_joins = joins[joins['dataset_name_pk'] == dataset_name]

    outgoing_count = len(outgoing_joins)
    incoming_count = len(incoming_joins)
    total = outgoing_count + incoming_count

    # metrics display
    st.metric("References (Outgoing)", outgoing_count,
              help=f"This dataset contains {outgoing_count} Foreign Keys pointing to other tables.")
    st.metric("Referenced By (Incoming)", incoming_count,
              help=f"{incoming_count} other tables have Foreign Keys pointing to this dataset.")
    st.metric("Total Connections", total)

    # optional detail expansion
    if show_details and total > 0:
        with st.expander("🔗 View Connected Datasets", expanded=False):
            if outgoing_count > 0:
                st.markdown("**References (points to):**")
                for _, row in outgoing_joins.iterrows():
                    target = row['dataset_name_pk']
                    key = row['column_name']
                    st.markdown(f"- `{target}` via `{key}`")

            if incoming_count > 0:
                if outgoing_count > 0:
                    st.markdown("---")
                st.markdown("**Referenced by (pointed to from):**")
                for _, row in incoming_joins.iterrows():
                    source = row['dataset_name_fk']
                    key = row['column_name']
                    st.markdown(f"- `{source}` via `{key}`")

# =============================================================================
# 7. visualization engine
# =============================================================================

#------------------------------
def get_category_colors(categories: List[str]) -> Dict[str, str]:
    """generates consistent colors for categories using deterministic hash."""
    import hashlib

    def stable_hash(s: str) -> int:
        return int(hashlib.md5(s.encode()).hexdigest(), 16)

    return {cat: f"hsl({(stable_hash(cat) * 137.5) % 360}, 70%, 50%)" for cat in categories}


#------------------------------
#------------------------------
def create_spring_graph(
    df: pd.DataFrame,
    selected_datasets: List[str],
    mode: str = 'focused',
    graph_font_size: int = 14,
    node_separation: float = 0.9,
    graph_height: int = 600,
    show_edge_labels: bool = True,
    hide_hubs: bool = False,
    edge_font_size: int = 12,
    edge_thickness: float = 1.5
) -> go.Figure:
    """
    creates a spring-layout graph visualization.
    mode: 'focused' shows only connections between selected datasets
    mode: 'discovery' shows all outgoing connections from selected datasets
    """
    if not selected_datasets:
        # (Empty state code remains same, omitted for brevity...)
        fig = go.Figure()
        fig.add_annotation(text="Select datasets to visualize", showarrow=False)
        return fig

    join_data = get_joins_for_selection(df, selected_datasets)
    G = nx.DiGraph()

    # Define High-Traffic Hubs to optionally hide
    # We only hide them if they are NOT in the primary selection list
    HUBS = ['Users', 'Organizational Units', 'Role Details', 'Semester', 'Department']
    
    def should_include(ds_name):
        if not hide_hubs:
            return True
        # Always include if the user specifically selected it
        if ds_name in selected_datasets:
            return True
        # Otherwise, exclude if it's a known hub
        return ds_name not in HUBS

    if mode == 'focused':
        # add only selected datasets
        for ds in selected_datasets:
            if should_include(ds):
                G.add_node(ds, type='focus')

        # add only edges between selected datasets
        if not join_data.empty:
            for _, row in join_data.iterrows():
                s = row['Source Dataset']
                t = row['Target Dataset']
                if s in selected_datasets and t in selected_datasets:
                    if should_include(s) and should_include(t):
                        G.add_edge(s, t, label=row['column_name'])
    else:
        # discovery mode
        for ds in selected_datasets:
            if should_include(ds):
                G.add_node(ds, type='focus')

        # add all outgoing connections
        if not join_data.empty:
            for _, row in join_data.iterrows():
                s = row['Source Dataset']
                t = row['Target Dataset']
                
                # Logic: If source is selected, look at target
                if s in selected_datasets:
                    if should_include(t): # Only add neighbor if it's not a hidden hub
                        if not G.has_node(t):
                            G.add_node(t, type='neighbor')
                        G.add_edge(s, t, label=row['column_name'])

    if G.number_of_nodes() == 0:
        # (Empty state code remains same...)
        fig = go.Figure()
        fig.add_annotation(text="No nodes to display (Hubs Hidden)", showarrow=False)
        return fig

    # (Layout calculation and Plotly tracing remains exactly the same below...)
    pos = nx.spring_layout(G, k=node_separation, iterations=50)
    
    # .
    # build edge traces
    edge_x = []
    edge_y = []
    annotations = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
#------------------------------
        if show_edge_labels:
            label_text = edge[2].get('label', '')
            
            # VISUAL IMPROVEMENT: Clutter Reduction
            # In 'Discovery' mode, common keys like 'UserId' appear dozens of times, obscuring the graph
            # suppresses these labels to improve readability, assuming the user knows 'Users' connects via 'UserId'
            is_common_key = label_text in ['UserId', 'OrgUnitId', 'RoleId', 'SemesterId', 'DepartmentId']
            
            # if in discovery mode AND it's a common key, skip drawing the text label
            if mode == 'discovery' and is_common_key:
                pass 
            elif label_text:
                annotations.append(dict(
                    x=(x0 + x1) / 2,
                    y=(y0 + y1) / 2,
                    text=label_text,
                    showarrow=False,
#------------------------------
                    font=dict(color="#58A6FF", size=edge_font_size, family="monospace"),
                    bgcolor="#1E232B",
                    borderpad=2,
                    opacity=0.9
                ))
#------------------------------
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=edge_thickness, color='#666'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )

    # build node traces with category colors
    categories = df['category'].unique().tolist()
    cat_colors = get_category_colors(categories)

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_hover = []
    node_size = []
    node_symbol = []
    node_line_color = []
    node_line_width = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        node_type = G.nodes[node].get('type', 'focus')
        subset = df[df['dataset_name'] == node]
        category = subset['category'].iloc[0] if not subset.empty else 'unknown'
        node_color.append(cat_colors.get(category, '#ccc'))
        node_hover.append(f"<b>{node}</b><br>Category: {category}<br>Type: {node_type.title()}")

#------------------------------
        # scale node size by number of edges (degree) in the current graph
        degree = G.degree(node)
        if node_type == 'focus':
            base_size = 30
            scaled_size = base_size + min(degree * 5, 40)  # 30–70 range
            node_size.append(scaled_size)
            node_symbol.append('square')
            node_text.append(f'<b>{node}</b>')
            node_line_color.append('white')
            node_line_width.append(3)
        else:
            base_size = 15
            scaled_size = base_size + min(degree * 4, 25)  # 15–40 range
            node_size.append(scaled_size)
            node_symbol.append('circle')
            node_text.append(node)
            node_line_color.append('gray')
            node_line_width.append(1)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_hover,
        text=node_text,
        textposition="top center",
        textfont=dict(size=graph_font_size, color='#fff'),
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            symbol=node_symbol,
            line=dict(color=node_line_color, width=node_line_width)
        ),
        showlegend=False
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            annotations=annotations,
            height=graph_height
        )
    )

    # build category legend from nodes in the graph
    legend_categories = {}
    for node in G.nodes():
        subset = df[df['dataset_name'] == node]
        if not subset.empty:
            cat = subset['category'].iloc[0]
            if cat not in legend_categories:
                legend_categories[cat] = cat_colors.get(cat, '#ccc')

    # add invisible traces to create legend entries
    for cat_name, color in sorted(legend_categories.items()):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color, symbol='square'),
            name=cat_name,
            showlegend=True
        ))

    fig.update_layout(
        showlegend=True,
        legend=dict(
            title=dict(text="Categories", font=dict(color='#8B949E', size=12)),
            font=dict(color='#C9D1D9', size=11),
            bgcolor='rgba(30, 35, 43, 0.85)',
            bordercolor='#30363D',
            borderwidth=1,
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01
        )
    )

#------------------------------
    return fig


@st.cache_data
def create_orbital_map(df_hash: str, df: pd.DataFrame,
                       target_node: str = None, filter_keys: tuple = None) -> go.Figure:
    """
    generates the 'solar system' map.
    Includes dynamic coloring for specific keys/columns.
    """
    if df.empty:
        return go.Figure()

    # prepare data
    categories = sorted(df['category'].unique())

    required_cols = ['dataset_name', 'category']
    # Prefer dataset_description (dataset-level prose), fall back to column description
    optional_cols = ['dataset_description', 'description']
    cols_to_use = required_cols + [c for c in optional_cols if c in df.columns]
    datasets = df[cols_to_use].drop_duplicates('dataset_name')

    # layout parameters
    pos = {}
    center_x = 0
    center_y = 0
    orbit_radius_cat = 20

    cat_step = 2 * math.pi / len(categories) if categories else 1

    # trace containers
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    node_line_width, node_line_color = [], []
    cat_x, cat_y, cat_text = [], [], []

    # Group edges by their KEY (Column Name) so we can color them differently
    edge_groups = {}
    active_neighbors = set()

    if target_node:
        # Helper to add edge to groups
        def add_edge(k, s, t):
            if k not in edge_groups:
                edge_groups[k] = []
            edge_groups[k].append((s, t))

        # LOGIC START
        if filter_keys:
            # Attribute/Weak Link Mode
            for key in filter_keys:
                matches = df[df['column_name'] == key]['dataset_name'].unique()
                for match in matches:
                    if match != target_node:
                        add_edge(key, target_node, match)
                        active_neighbors.add(match)
        else:
            # Standard PK/FK Mode
            joins = get_joins(df)
            if not joins.empty:
                out_ = joins[joins['dataset_name_fk'] == target_node]
                for _, r in out_.iterrows():
                    add_edge(r['column_name'], target_node, r['dataset_name_pk'])
                    active_neighbors.add(r['dataset_name_pk'])

                in_ = joins[joins['dataset_name_pk'] == target_node]
                for _, r in in_.iterrows():
                    add_edge(r['column_name'], r['dataset_name_fk'], target_node)
                    active_neighbors.add(r['dataset_name_fk'])
        # LOGIC END

    # build nodes
    for i, cat in enumerate(categories):
        angle = i * cat_step
        cx = center_x + orbit_radius_cat * math.cos(angle)
        cy = center_y + orbit_radius_cat * math.sin(angle)
        pos[cat] = (cx, cy)

        # add category node
        node_x.append(cx)
        node_y.append(cy)
        node_text.append(f"Category: {cat}")

        is_dim = (target_node is not None)
        node_color.append('rgba(255, 215, 0, 0.2)' if is_dim else 'rgba(255, 215, 0, 1)')
        node_size.append(35)
        node_line_width.append(0)
        node_line_color.append('rgba(0,0,0,0)')

        cat_x.append(cx)
        cat_y.append(cy + 3)
        cat_text.append(cat)

        # dataset positions
        cat_ds = datasets[datasets['category'] == cat]
        ds_count = len(cat_ds)

        if ds_count > 0:
            min_radius = 3
            radius_per_node = 0.5
            ds_radius = min_radius + (ds_count * radius_per_node / (2 * math.pi))
            ds_step = 2 * math.pi / ds_count

            for j, (_, row) in enumerate(cat_ds.iterrows()):
                ds_name = row['dataset_name']
                ds_angle = j * ds_step
                dx = cx + ds_radius * math.cos(ds_angle)
                dy = cy + ds_radius * math.sin(ds_angle)
                pos[ds_name] = (dx, dy)

                node_x.append(dx)
                node_y.append(dy)

                if target_node:
                    if ds_name == target_node:
                        node_color.append('#00FF00')
                        node_size.append(50)
                        node_line_width.append(5)
                        node_line_color.append('white')
                    elif ds_name in active_neighbors:
                        node_color.append('#FFFFFF')  # White nodes for connected items
                        node_size.append(15)
                        node_line_width.append(1)
                        node_line_color.append('white')
                    else:
                        node_color.append('rgba(50,50,50,0.3)')
                        node_size.append(8)
                        node_line_width.append(0)
                        node_line_color.append('rgba(0,0,0,0)')
                else:
                    node_color.append('#00CCFF')
                    node_size.append(10)
                    node_line_width.append(1)
                    node_line_color.append('rgba(255,255,255,0.3)')

                desc_source = row.get('dataset_description') or row.get('description') or ''
                desc_short = str(desc_source)[:80]
                hover_text = f"<b>{ds_name}</b><br>{desc_short}..." if desc_short else f"<b>{ds_name}</b>"
                node_text.append(hover_text)

    # build traces list (starting with nodes/cats)
    traces = []

    # 1. Generate Colorful Lines for each Key Group
    palette = [
        "#00FF00", "#FF00FF", "#00FFFF", "#FFA500", "#FF4500",
        "#ADFF2F", "#FF69B4", "#1E90FF", "#FFFF00", "#00CED1"
    ]

    for idx, (key, pairs) in enumerate(edge_groups.items()):
        ex, ey = [], []
        for s, t in pairs:
            if s in pos and t in pos:
                x0, y0 = pos[s]
                x1, y1 = pos[t]
                ex.extend([x0, x1, None])
                ey.extend([y0, y1, None])

        color = palette[idx % len(palette)]

        traces.append(go.Scatter(
            x=ex, y=ey,
            mode='lines',
            name=key,
            line=dict(width=2, color=color),
            hoverinfo='name'
        ))

    # 2. Add Nodes and Labels
    traces.append(go.Scatter(
        x=node_x, y=node_y, mode='markers',
        hoverinfo='text', hovertext=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=node_line_width, color=node_line_color)
        ),
        showlegend=False
    ))

    traces.append(go.Scatter(
        x=cat_x, y=cat_y, mode='text',
        text=cat_text,
        textfont=dict(color='gold', size=10),
        hoverinfo='none',
        showlegend=False
    ))

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            showlegend=True,
            legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.5)"),
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=700
        )
    )
    return fig


def get_orbital_map(df: pd.DataFrame, target_node: str = None,
                    filter_keys: List[str] = None) -> go.Figure:
    """wrapper to call cached orbital map with proper cache key."""
    df_hash = f"{len(df)}_{df['dataset_name'].nunique()}"
    safe_keys = tuple(sorted(filter_keys)) if filter_keys else None
    return create_orbital_map(df_hash, df, target_node, safe_keys)


#------------------------------
def create_relationship_matrix(df: pd.DataFrame, filter_connected_only: bool = True) -> go.Figure:
    """
    creates a heatmap showing which datasets connect to which.
    optionally filters to only show datasets with at least one connection.
    """
    joins = get_joins(df)

    if joins.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No relationships detected in schema",
            showarrow=False,
            font=dict(size=16, color='gray')
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e'
        )
        return fig

    # determine which datasets to include
    if filter_connected_only:
        connected_datasets = set(joins['dataset_name_fk']).union(set(joins['dataset_name_pk']))
        datasets = sorted(connected_datasets)
    else:
        datasets = sorted(df['dataset_name'].unique())

    # create adjacency matrix
    matrix = pd.DataFrame(0, index=datasets, columns=datasets)

    # track join keys for hover text
    join_keys = pd.DataFrame("", index=datasets, columns=datasets)

    for _, r in joins.iterrows():
        src = r['dataset_name_fk']
        tgt = r['dataset_name_pk']
        key = r['column_name']
        if src in matrix.index and tgt in matrix.columns:
            matrix.loc[src, tgt] += 1
            existing = join_keys.loc[src, tgt]
            if existing:
                join_keys.loc[src, tgt] = f"{existing}, {key}"
            else:
                join_keys.loc[src, tgt] = key

    # build custom hover text
    hover_text = []
    for src in datasets:
        row_hover = []
        for tgt in datasets:
            count = matrix.loc[src, tgt]
            keys = join_keys.loc[src, tgt]
            if count > 0:
                row_hover.append(f"<b>{src}</b> → <b>{tgt}</b><br>Joins: {count}<br>Keys: {keys}")
            else:
                row_hover.append(f"{src} → {tgt}<br>No direct relationship")
        hover_text.append(row_hover)

    # create heatmap with custom hover
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=datasets,
        y=datasets,
        hoverinfo='text',
        text=hover_text,
        colorscale=[
            [0.0, '#1e1e1e'],
            [0.01, '#1e3a5f'],
            [0.5, '#3182bd'],
            [1.0, '#08519c']
        ],
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Joins",
                side="right",
                font=dict(color='white')
            ),
            tickfont=dict(color='white')
        )
    ))

    # style the layout
    fig.update_layout(
        height=max(500, len(datasets) * 20),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        xaxis=dict(
            title=dict(
                text="Target Dataset (has Primary Key)",
                font=dict(color='#8B949E')
            ),
            tickangle=45,
            tickfont=dict(color='#C9D1D9', size=10),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(
                text="Source Dataset (has Foreign Key)",
                font=dict(color='#8B949E')
            ),
            tickfont=dict(color='#C9D1D9', size=10),
            showgrid=False,
            autorange='reversed'
        ),
        margin=dict(l=150, r=50, t=50, b=150)
    )

    return fig

# =============================================================================
# 8. sql builder engine
# =============================================================================

#------------
#------------------------------
def generate_sql_for_path(path: List[str],
                          df: pd.DataFrame,
                          dialect: str = "T-SQL") -> str:
    """
    Generate a LEFT JOIN query that follows a specific dataset path.
    Now uses the shared resolver for consistency.
    """
    path = list(dict.fromkeys(path))  # preserve order, remove duplicates

    if len(path) < 2:
        return "-- need at least 2 tables in the path to generate a JOIN."

    # Dialect configuration
    if dialect == "T-SQL":
        q_start, q_end = "[", "]"
        limit_syntax = "TOP 100"
        limit_suffix = ""
    else:
        q_start, q_end = '"', '"'
        limit_syntax = ""
        limit_suffix = "LIMIT 100"

    def quote(name: str) -> str:
        return f"{q_start}{name}{q_end}"

    # Use the shared resolver
    join_steps = resolve_joins_for_selection(path, df)

    base_table = path[0]
    aliases = {ds: f"t{i+1}" for i, ds in enumerate(path)}

    sql_lines = [
        f"SELECT {limit_syntax}" if limit_syntax else "SELECT",
        f"    {aliases[base_table]}.*",
        f"FROM {quote(base_table)} {aliases[base_table]}"
    ]

    for step in join_steps:
        left = step['left']
        right = step['right']
        conditions = step['conditions']

        if conditions:
            on_clause = " AND ".join(conditions)
            sql_lines.append(
                f"LEFT JOIN {quote(right)} {aliases[right]} ON {on_clause}"
            )
        else:
            sql_lines.append(
                f"CROSS JOIN {quote(right)} {aliases[right]} "
                f"-- ⚠️ no direct relationship found"
            )

    if limit_suffix:
        sql_lines.append(limit_suffix)

    return "\n".join(sql_lines)

#------------
#------------------------------
def generate_pandas_for_path(path: List[str], df: pd.DataFrame) -> str:
    """
    Generate pandas code for a specific path using the shared resolver.
    """
    path = list(dict.fromkeys(path))

    if len(path) < 2:
        return "# need at least 2 tables in the path to generate a JOIN."

    def clean_var(name: str) -> str:
        clean = name.lower()
        clean = re.sub(r'[^a-z0-9_]', '_', clean)
        clean = re.sub(r'_+', '_', clean)
        clean = clean.strip('_')
        return f"df_{clean}"

    # Use the shared resolver
    join_steps = resolve_joins_for_selection(path, df)

    lines = ["import pandas as pd", "", "# 1. Load Dataframes"]

    for ds in path:
        var = clean_var(ds)
        lines.append(f"{var} = pd.read_csv('{ds}.csv')")

    lines.append("")
    lines.append("# 2. Perform Merges")

    base_var = clean_var(path[0])
    lines.append(f"final_df = {base_var}")

    for step in join_steps:
        left = step['left']
        right = step['right']
        conditions = step['conditions']

        right_var = clean_var(right)

        lines.append("")
        lines.append(f"# Joining {right} to {left}")

        if conditions:
            # Take first condition's key for simplicity (pandas merge)
            key = conditions[0].split('=')[1].strip().split('.')[-1]
            lines.append(f"final_df = pd.merge(final_df, {right_var}, on='{key}', how='left')")
        else:
            lines.append(f"final_df = final_df.merge({right_var}, how='cross')  # no direct key")

    lines.append("")
    lines.append("# 3. Preview Result")
    lines.append("print(final_df.head())")

    return "\n".join(lines)

# =============================================================================
# shared Join Resolver (single source of truth)
# =============================================================================

def resolve_joins_for_selection(
    selected_datasets: List[str], 
    df: pd.DataFrame
) -> List[Dict]:
    """
    Returns an ordered list of join steps for the selected datasets.
    Each step = {'left': str, 'right': str, 'conditions': List[str]}
    This eliminates duplication across generate_sql, generate_pandas, etc.
    """
    if len(selected_datasets) < 2:
        return []

    selected_datasets = list(dict.fromkeys(selected_datasets))  # preserve order

    # Build full graph
    G_full = nx.Graph()
    joins = get_joins(df)
    if not joins.empty:
        for _, r in joins.iterrows():
            src, tgt, key = r['dataset_name_fk'], r['dataset_name_pk'], r['column_name']
            if G_full.has_edge(src, tgt):
                if key not in G_full[src][tgt].get('keys', []):
                    G_full[src][tgt]['keys'].append(key)
            else:
                G_full.add_edge(src, tgt, keys=[key])

    # Universal keys + alias mapping (centralized)
    UNIVERSAL_KEYS = ['UserId', 'OrgUnitId', 'SectionId', 'SemesterId', 'DepartmentId', 'SessionId']
    ALIAS_MAP = {
        'SubmitterId': 'UserId', 'GradedByUserId': 'UserId', 'AssignedToUserId': 'UserId',
        'EvaluatorId': 'UserId', 'AuditorId': 'UserId', 'LastModifiedBy': 'UserId',
        'CourseOfferingId': 'OrgUnitId', 'SectionId': 'OrgUnitId', 'ParentOrgUnitId': 'OrgUnitId'
    }

    steps = []
    joined_tables = {selected_datasets[0]}

    for current_table in selected_datasets[1:]:
        found = False

        curr_cols_raw = set(df[df['dataset_name'] == current_table]['column_name'])
        curr_cols_resolved = set(curr_cols_raw)
        col_lookup_curr = {c: c for c in curr_cols_raw}
        for c in curr_cols_raw:
            if c in ALIAS_MAP:
                canonical = ALIAS_MAP[c]
                curr_cols_resolved.add(canonical)
                if canonical not in col_lookup_curr:
                    col_lookup_curr[canonical] = c

        for existing_table in joined_tables:
            exist_cols_raw = set(df[df['dataset_name'] == existing_table]['column_name'])
            exist_cols_resolved = set(exist_cols_raw)
            col_lookup_exist = {c: c for c in exist_cols_raw}
            for c in exist_cols_raw:
                if c in ALIAS_MAP:
                    canonical = ALIAS_MAP[c]
                    exist_cols_resolved.add(canonical)
                    if canonical not in col_lookup_exist:
                        col_lookup_exist[canonical] = c

            # Graph keys
            graph_keys = G_full.get_edge_data(existing_table, current_table, default={}).get('keys', [])

            # Shared universal keys
            shared_universal = [k for k in UNIVERSAL_KEYS 
                               if k in curr_cols_resolved and k in exist_cols_resolved]

            final_keys = sorted(list(set(graph_keys + shared_universal)))

            if final_keys:
                conditions = []
                for k in final_keys:
                    left_col = col_lookup_exist.get(k, k)
                    right_col = col_lookup_curr.get(k, k)
                    conditions.append(f"{existing_table}.{left_col} = {current_table}.{right_col}")

                steps.append({
                    'left': existing_table,
                    'right': current_table,
                    'conditions': conditions
                })
                joined_tables.add(current_table)
                found = True
                break

        if not found:
            steps.append({
                'left': selected_datasets[0],
                'right': current_table,
                'conditions': []  # cross join
            })
            joined_tables.add(current_table)

    return steps

#------------------------------
#------------------------------
def generate_sql(selected_datasets: List[str], df: pd.DataFrame,
                 dialect: str = "T-SQL") -> str:
    """
    Generates a deterministic SQL JOIN query using the shared resolver.
    Much cleaner and eliminates duplication.
    """
    if len(selected_datasets) < 2:
        return "-- please select at least 2 datasets to generate a join."

    # Configuration based on dialect
    if dialect == "T-SQL":
        q_start, q_end = "[", "]"
        limit_syntax = "TOP 100"
        limit_suffix = ""
    else:  # Snowflake / PostgreSQL
        q_start, q_end = '"', '"'
        limit_syntax = ""
        limit_suffix = "LIMIT 100"

    def quote(name: str) -> str:
        return f"{q_start}{name}{q_end}"

    # Use the new shared resolver
    join_steps = resolve_joins_for_selection(selected_datasets, df)

    base_table = selected_datasets[0]
    aliases = {ds: f"t{i+1}" for i, ds in enumerate(selected_datasets)}

    sql_lines = [
        f"SELECT {limit_syntax}" if limit_syntax else "SELECT",
        f"    {aliases[base_table]}.*",
        f"FROM {quote(base_table)} {aliases[base_table]}"
    ]

    for step in join_steps:
        left = step['left']
        right = step['right']
        conditions = step['conditions']

        if conditions:
            on_clause = " AND ".join(conditions)
            sql_lines.append(
                f"LEFT JOIN {quote(right)} {aliases[right]} ON {on_clause}"
            )
        else:
            sql_lines.append(
                f"CROSS JOIN {quote(right)} {aliases[right]} "
                f"-- ⚠️ no direct relationship found"
            )

    if limit_suffix:
        sql_lines.append(limit_suffix)

    return "\n".join(sql_lines)
                     
#------------------------------

def generate_pandas(selected_datasets: List[str], df: pd.DataFrame) -> str:
    """
    Generates Python/Pandas code using the shared join resolver.
    Much cleaner and consistent with generate_sql().
    """
    if len(selected_datasets) < 2:
        return "# please select at least 2 datasets to generate code."

    # Clean variable names
    def clean_var(name: str) -> str:
        clean = name.lower()
        clean = re.sub(r'[^a-z0-9_]', '_', clean)
        clean = re.sub(r'_+', '_', clean)
        clean = clean.strip('_')
        return f"df_{clean}"

    # Use the shared resolver
    join_steps = resolve_joins_for_selection(selected_datasets, df)

    lines = ["import pandas as pd", "", "# 1. Load Dataframes"]

    for ds in selected_datasets:
        var = clean_var(ds)
        lines.append(f"{var} = pd.read_csv('{ds}.csv')")

    lines.append("")
    lines.append("# 2. Perform Merges")

    base_ds = selected_datasets[0]
    base_var = clean_var(base_ds)

    lines.append(f"# Starting with {base_ds}")
    lines.append(f"final_df = {base_var}")

    for step in join_steps:
        left = step['left']
        right = step['right']
        conditions = step['conditions']

        left_var = clean_var(left)
        right_var = clean_var(right)

        lines.append("")
        lines.append(f"# Joining {right} to {left}")

        if conditions:
            # Use the first condition for the merge key (pandas merge supports one key or list)
            # For simplicity we take the first condition's right side as the key
            key = conditions[0].split('=')[1].strip().split('.')[-1]
            lines.append(f"final_df = pd.merge(final_df, {right_var}, on='{key}', how='left')")
        else:
            lines.append(f"final_df = final_df.merge({right_var}, how='cross')  # no direct key found")

    lines.append("")
    lines.append("# 3. Preview Result")
    lines.append("print(final_df.head())")

    return "\n".join(lines)

def render_sidebar(df: pd.DataFrame) -> tuple:
    """renders the sidebar navigation and returns (view, selected_datasets)."""
    with st.sidebar:
        st.title("🔗 Datahub Datasets Explorer")

        # experience mode toggle
        st.session_state['experience_mode'] = st.radio(
            "Experience Mode",
            ["simple", "advanced"],
            format_func=lambda x: "🟢 Quick Explorer" if x == "simple" else "🔷 Power User",
            horizontal=True,
            help="Quick Explorer: Streamlined interface. Power User: All features and controls."
        )

        is_advanced = st.session_state['experience_mode'] == 'advanced'

        st.divider()

        # navigation based on mode
        if is_advanced:
            options = [
                "📊 Dashboard",
                "🗺️ Relationship Map",
                "📋 Schema Browser",
                "📚 KPI Recipes",
                "⚡ SQL Builder",
                "🔀 SQL Translator",
                "🔧 UDF Flattener",
                "✨ Schema Diff",
                "🌐 3D Explorer",
                "📋 Dataset ID Reference",
                "🤖 AI Assistant"
            ]
            captions = [
                "Overview, Search & Context",
                "Visualize Connections (PK/FK)",
                "Compare Tables Side-by-Side",
                "Pre-packaged SQL Solutions",
                "Generate JOIN Code",
                "Convert T-SQL ↔ Postgres",
                "Pivot Custom Fields (EAV)",
                "Compare against backups",
                "Full Schema in 3D (mostly for fun)",
                "SchemaID + PluginID Reference (experimental)",
                "Ask questions about data"
            ]

            view = st.radio(
                "Navigation",
                options,
                captions=captions,
                label_visibility="collapsed"
            )
        else:
            options = ["📊 Dashboard", "🗺️ Relationship Map", "🤖 AI Assistant"]
            captions = ["Overview & Search", "Visualize Connections", "Ask questions"]

            view = st.radio(
                "Navigation",
                options,
                captions=captions,
                label_visibility="collapsed"
            )

        st.divider()

        # data status and scraper - ROBUST VERSION
        if not df.empty:
            try:
                mod_time = os.path.getmtime('dataset_metadata.csv')
                last_updated = pd.Timestamp(mod_time, unit='s').strftime('%Y-%m-%d')
            except Exception:
                last_updated = "Unknown"

            # Safe column handling (protects against old CSVs)
            if 'dataset_name' in df.columns:
                dataset_count = df['dataset_name'].nunique()
            elif 'dataset' in df.columns:           # legacy name
                dataset_count = df['dataset'].nunique()
                df = df.rename(columns={'dataset': 'dataset_name'})
            else:
                dataset_count = 0

            st.success(f"✅ **{dataset_count}** Datasets Loaded")
            st.caption(f"📅 Schema updated: {last_updated}")
            st.caption(f"🔢 Total Columns: {len(df):,}")
        else:
            st.warning("⚠️ No schema data loaded yet")

        # Data Management / Backup
        with st.expander("⚙️ Data Management", expanded=df.empty):
            current_urls = st.session_state.get('custom_urls') or DEFAULT_URLS
            url_count = len([u for u in current_urls.strip().split('\n') if u.strip().startswith('http')])
            
            st.caption(f"Currently configured: **{url_count}** URLs")
            
            if st.button("✏️ Edit URLs (Full View)", use_container_width=True):
                st.session_state['show_url_editor'] = True
                st.rerun()

            if st.button("🔄 Scrape & Update", type="primary", use_container_width=True):
                urls_text = st.session_state.get('custom_urls') or DEFAULT_URLS
                urls = [u.strip() for u in urls_text.split('\n') if u.strip().startswith('http')]
                if urls:
                    with st.spinner(f"Scraping {len(urls)} pages..."):
                        new_df = scrape_and_save(urls)
                        if not new_df.empty:
                            st.session_state['scrape_msg'] = f"Success: {new_df['dataset_name'].nunique()} datasets loaded"
                            st.session_state['current_df'] = new_df
                            load_data.clear()
                            st.rerun()
                else:
                    st.error("No valid URLs configured.")

            if not df.empty:
                timestamp = pd.Timestamp.now().strftime('%Y-%m-%d')
                csv = df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="💾 Download Metadata Backup (CSV)",
                    data=csv,
                    file_name=f"brightspace_metadata_backup_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            if st.button("🩺 Health Check", use_container_width=True):
                st.session_state['show_health_check'] = True
                st.rerun()

        # dataset selection (when applicable)
        selected_datasets: List[str] = []
        if not df.empty and view in ["🗺️ Relationship Map", "⚡ SQL Builder"]:
            st.divider()
            st.subheader("Dataset Selection")

            if is_advanced:
                select_mode = st.radio(
                    "Method:", ["Templates", "By Category", "List All"],
                    horizontal=True,
                    label_visibility="collapsed"
                )
            else:
                select_mode = "Templates"

            if select_mode == "Templates":
                templates = {
                    "User Progress": ["Users", "User Enrollments", "Content User Progress", "Course Access"],
                    "Grades & Feedback": ["Users", "Grade Objects", "Grade Results", "Rubric Assessment Results"],
                    "Discussions": ["Discussion Forums", "Discussion Topics", "Discussion Posts"],
                    "Quizzes": ["Quiz Objects", "Quiz Attempts", "Quiz User Answers"],
                    "Assignments": ["Assignment Objects", "Assignment Submissions", "Assignment Feedback"]
                }

                # pre-validate which templates have all datasets available
                available_datasets = set(df['dataset_name'].unique())
                template_status = {}
                for name, datasets in templates.items():
                    available = [ds for ds in datasets if ds in available_datasets]
                    missing = [ds for ds in datasets if ds not in available_datasets]
                    template_status[name] = {
                        'available': available,
                        'missing': missing,
                        'complete': len(missing) == 0
                    }

                # build display labels showing availability
                template_options = ["Custom Selection..."]
                for name in templates.keys():
                    status = template_status[name]
                    if status['complete']:
                        template_options.append(f"✅ {name}")
                    elif len(status['available']) > 0:
                        template_options.append(f"⚠️ {name} (partial)")
                    else:
                        template_options.append(f"❌ {name} (unavailable)")

                chosen_option = st.selectbox(
                    "Select a Scenario:",
                    template_options
                )

                if chosen_option != "Custom Selection...":
                    chosen_template = chosen_option.split(" ", 1)[1].replace(" (partial)", "").replace(" (unavailable)", "")
                    status = template_status[chosen_template]

                    if status['missing']:
                        st.warning(f"⚠️ Missing datasets: {', '.join(status['missing'])}")
                        st.caption("These datasets may not have been scraped or may have different names.")

                    if status['available']:
                        st.session_state['selected_datasets'] = status['available']
                        selected_datasets = status['available']
                        st.success(f"Loaded {len(selected_datasets)}/{len(templates[chosen_template])} datasets for {chosen_template}")
                    else:
                        st.error("No datasets from this template are available.")
                        selected_datasets = []
                else:
                    all_ds = sorted(df['dataset_name'].unique())
                    selected_datasets = st.multiselect(
                        "Select Datasets:",
                        all_ds,
                        default=st.session_state.get('selected_datasets', [])
                    )

            elif select_mode == "By Category":
                all_cats = sorted(df['category'].unique())
                selected_cats = st.multiselect("Filter Categories:", all_cats, default=[])
                if selected_cats:
                    for cat in selected_cats:
                        cat_ds = sorted(df[df['category'] == cat]['dataset_name'].unique())
                        s = st.multiselect(f"📦 {cat}", cat_ds, key=f"sel_{cat}")
                        selected_datasets.extend(s)
            else:
                all_ds = sorted(df['dataset_name'].unique())
                selected_datasets = st.multiselect(
                    "Select Datasets:", all_ds, key="dataset_multiselect"
                )

            if selected_datasets:
                st.button("🗑️ Clear Selection", on_click=clear_all_selections)

        # authentication
        st.divider()

        # decoy input for password manager protection
        st.markdown(
            """
            <div style="height:0px; overflow:hidden; opacity:0; position:absolute; z-index:-1;">
                <input type="text" name="decoy_username" autocomplete="off" tabindex="-1">
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.session_state['authenticated']:
            st.success("🔓 AI Unlocked")
            if st.button("Logout"):
                logout()
                st.rerun()
        else:
            with st.expander("🔐 AI Login", expanded=True):
                with st.form("login_form"):
                    st.text_input(
                        "Password",
                        type="password",
                        key="password_input",
                        help="Enter password to unlock AI features."
                    )
                    submitted = st.form_submit_button("Unlock")

                if submitted:
                    perform_login()
                    if st.session_state.get('authenticated'):
                        st.rerun()

                if st.session_state['auth_error']:
                    st.error("Incorrect password.")

        # cross-links (advanced mode only)
        if is_advanced:
            st.divider()
            st.markdown("### 🔗 Related Tools")

            st.link_button(
                "🧠 Signal Foundry",
                "https://signalfoundry.streamlit.app/",
                help=(
                    "An advanced NLP engine for unstructured data. "
                    "Use this to analyze Discussion Posts, Survey Comments, and Assignment Feedback."
                )
            )

            if st.button("📋 Dataset ID Reference", use_container_width=True):
                render_dataset_id_reference()

            c_t1, c_t2 = st.columns(2)
            with c_t1:
                st.link_button(
                    "🔎 CSV Query Tool",
                    "https://csvexpl0rer.streamlit.app/",
                    help="Run SQL queries on CSV files."
                )
            with c_t2:
                st.link_button(
                    "✂️ CSV Splitter",
                    "https://csvsplittertool.streamlit.app/",
                    help="Split large CSVs into smaller chunks."
                )

    # How to use section
    with st.expander("ℹ️ How to use this application", expanded=False):
        st.markdown("""
**Welcome to the Brightspace Dataset Explorer**  
This tool attempts to be a sort of...Rosetta Stone for the D2L Data Hub — helping you navigate schemas, understand relationships, and build queries across an ever-growing landscape of datasets.

---

**🟢 Quick Explorer Mode** (default)

| Feature | What It Does |
|---|---|
| **🔍 Intelligent Search** | Find where columns (e.g., `OrgUnitId`) live, with context descriptions and relationship summaries |
| **🗺️ Relationship Map** | Visualize PK/FK connections between datasets. Use **Focused** mode for selected datasets or **Discovery** mode to explore outward |
| **🤖 AI Assistant** | Ask plain-language questions about the data model (requires login) |

---

**🔷 Power User Mode** (toggle in sidebar)

| Feature | What It Does |
|---|---|
| **📋 Schema Browser** | Compare multiple datasets side-by-side. The **Shared Column Analysis** automatically highlights potential join keys across your selection |
| **🛤️ Path Finder** | Find all valid join paths between any two datasets, with configurable hop depth. Generate SQL or Pandas code for each path |
| **⚡ SQL Builder** | Select datasets to auto-generate `LEFT JOIN` queries (T-SQL, Snowflake, PostgreSQL) or Python/Pandas merge code |
| **📚 KPI Recipes** | Pre-packaged SQL for common questions (engagement, grades, quiz analysis) with dataset availability indicators |
| **🔀 SQL Translator** | AI-powered dialect conversion (e.g., T-SQL → PostgreSQL, SQL → Pandas) |
| **🔧 UDF Flattener** | Generate PIVOT SQL to transform vertical EAV data into horizontal columns |
| **✨ Schema Diff** | Upload a backup CSV to detect added/removed datasets, columns, and metadata changes |
| **🩺 Health Check** | Validate scraped data against live D2L documentation to detect schema drift |

---

**💡 Common Workflows**

**Re-create a Report from Raw Tables:**
1. Open **Schema Browser** → select an Advanced Dataset (e.g., *All Grades*) and a raw table (e.g., *Grade Results*)
2. Review the **Shared Column Analysis** to identify matching columns
3. Use **SQL Builder** to generate the join query

**Find How Two Datasets Connect:**
1. Open **Path Finder** on the Dashboard (Power User mode)
2. Select source and target datasets
3. Review the shortest paths and generate SQL for any path

**Track Schema Changes Over Time:**
1. Download a **Metadata Backup** from Data Management
2. After re-scraping later, upload the backup to **Schema Diff**
3. Review added/removed datasets and columns

**Validate Data Freshness:**
1. Open **Data Management** → click **🩺 Health Check**
2. Run offline checks instantly, or sample live pages for drift detection
""")

    return view, selected_datasets


def render_dataset_id_reference(df: pd.DataFrame):
    """Hybrid Dataset ID Reference: Hard-coded stable IDs + auto-detect new datasets from scraper."""
    st.header("📋 Brightspace Dataset ID Reference")
    st.caption("SchemaID + Full / Differential PluginIDs (stable across environments)")

    # ── FULLY POPULATED FROM YOUR CODE-BLOCK 1 (Sorted Alphabetically) ──
    hardcoded = [
        {"Dataset Name": "Accommodations Profile Log", "SchemaID": "e1da7ff3-8578-4659-bb34-bb901d3a032c", "Full PluginID": "729711ba-ca1d-11eb-b8bc-0242ac130003", "Differential PluginID": "d0d3c00a-ca1a-11eb-b8bc-0242ac130003"},
        {"Dataset Name": "Activity Exemptions Log", "SchemaID": "6ed33466-03ce-4702-9402-d8089ccaf5cc", "Full PluginID": "ef65e37a-7ae4-4389-9de0-c0d1ab7a9596", "Differential PluginID": "53198ba5-d30f-476b-bbb1-bbf9cdf1ed4c"},
        {"Dataset Name": "Activity Feed Comment Log", "SchemaID": "2adc9fe9-bfb9-46b1-9302-af45d35fa293", "Full PluginID": "937ae575-58fd-46fe-9c38-728c6f29ec85", "Differential PluginID": "ad92a587-c296-4f73-b35d-b8a3785ead7c"},
        {"Dataset Name": "Activity Feed Comment Objects", "SchemaID": "acc45d5d-c5df-416d-a9fe-a6b1a43ec2c5", "Full PluginID": "4bb03b07-7e16-49cd-ac0e-0200ae3c85e2", "Differential PluginID": "b2756271-f0d8-4b95-a41a-9660ed70ac14"},
        {"Dataset Name": "Activity Feed Post Log", "SchemaID": "2f22beed-81c7-4d99-9935-70eb30610084", "Full PluginID": "8fd22e52-5a44-4c30-9f96-282ea0c04a8c", "Differential PluginID": "0dfe6caa-a8d4-4480-98be-00d74f62e7b9"},
        {"Dataset Name": "Activity Feed Post Objects", "SchemaID": "4320d948-b06a-4fa2-b4ad-723ed9b002aa", "Full PluginID": "553ee539-6ecc-4096-b845-acb1e8efd9eb", "Differential PluginID": "15679ea1-ef36-4839-a57e-8e70da5a98ac"},
        {"Dataset Name": "Announcements", "SchemaID": "5b76e84f-189e-4c64-bcef-960c33b02a18", "Full PluginID": "d9f4035c-cb0f-45af-9646-08e46b341b1f", "Differential PluginID": "a698893e-b831-403c-9da1-b23170220d3e"},
        {"Dataset Name": "Assignment Special Access", "SchemaID": "f4664070-bd96-4d85-9f17-0da6a91dddf8", "Full PluginID": "fc349213-2f2f-4602-bf92-448bdec6b858", "Differential PluginID": "05277094-834b-4688-b5b9-f1f47b42386e"},
        {"Dataset Name": "Assignment Submission Details", "SchemaID": "8e3916fd-3678-4440-b0ba-9a9242889b75", "Full PluginID": "b12a4203-3169-4dbb-9e6b-e979fc1620a9", "Differential PluginID": "ec1de75c-4384-4943-8690-07012f1c378c"},
        {"Dataset Name": "Assignment Submissions", "SchemaID": "a5686dc9-78fc-4495-ada4-50db954bebea", "Full PluginID": "041dde83-3a29-4a37-97de-9ee615318111", "Differential PluginID": "7c7094f9-6268-49a8-aea3-a952369a849d"},
        {"Dataset Name": "Assignment Summary", "SchemaID": "16a36efe-0a07-4381-9570-6d8b391ac317", "Full PluginID": "d9923de9-de6a-41ea-a63e-e8fd771b7b93", "Differential PluginID": "fe38c3e3-33bc-41a0-843a-274614583925"},
        {"Dataset Name": "Attendance Registers", "SchemaID": "9b1086cd-e527-4dc5-8687-98484c1553a8", "Full PluginID": "86ca50e9-77ae-43a7-b646-8ce7794161a0", "Differential PluginID": "41b95b72-7ad9-4c49-acad-b7ffe103ef45"},
        {"Dataset Name": "Attendance Schemes", "SchemaID": "74c92e7e-d5c5-4616-b7a2-df2871e411f0", "Full PluginID": "eacabdd4-af98-4a90-816b-5763a4950e7e", "Differential PluginID": "28dc1e1b-36bd-44b2-bfda-e6f76af3cfae"},
        {"Dataset Name": "Attendance Sessions", "SchemaID": "879a19b1-1b72-42b0-b0c5-1304331a6443", "Full PluginID": "78035701-db72-463b-9dc8-a6f3eed2041e", "Differential PluginID": "d191c862-ae01-43a0-acfa-b670f1128ce3"},
        {"Dataset Name": "Attendance User Sessions", "SchemaID": "9e167b3b-7a0b-470c-ad58-8a19f9c79e22", "Full PluginID": "ad84c484-000b-48b4-85d2-ba5f781e9c18", "Differential PluginID": "82cc7e73-4bb0-49fb-90d0-66b8da08c8d0"},
        {"Dataset Name": "Audio Video Processed", "SchemaID": "3282ebf6-d8e2-477d-9ce0-99feae4c9778", "Full PluginID": "39bbb587-62e4-4e77-8455-7556f74bc6fd", "Differential PluginID": "78a9687e-6407-43a2-91d0-60bd450c0b94"},
        {"Dataset Name": "Auditor Relationships Log", "SchemaID": "a83525de-2ac8-4dd7-9e29-f6cd527e0984", "Full PluginID": "8c426cdc-0545-42a7-a292-83f297cf7427", "Differential PluginID": "1e0c21d7-7d3d-47c2-9fa2-0b00f3720371"},
        {"Dataset Name": "Award Objects", "SchemaID": "56fa41c5-142f-4b99-885a-a738d0a09d54", "Full PluginID": "429f9046-49e5-4b29-8818-295ec3814593", "Differential PluginID": "99b375ed-9247-465b-9955-0313b800a07b"},
        {"Dataset Name": "Awards Issued", "SchemaID": "dcef8789-1e6b-4d98-ac76-588583b3ba30", "Full PluginID": "6d6cd2d8-c714-41fd-9465-f797dfd69c76", "Differential PluginID": "627c8792-65b2-4484-8398-ac21b91fb07d"},
        {"Dataset Name": "Calendar Events", "SchemaID": "c68bce39-87ba-4a44-84c7-a2e8a3dab424", "Full PluginID": "2fddff98-4a27-4d5f-83c8-1de977dd5a4e", "Differential PluginID": "d5a7d50d-2a82-41a1-9ccb-e806b6bfe865"},
        {"Dataset Name": "Checklist Category Details", "SchemaID": "e0541d86-16f0-492a-8fc7-1495f5bc69a4", "Full PluginID": "d984b4b7-8bd2-456a-b082-6708b5454d23", "Differential PluginID": "b87cfe45-ed94-4264-8c12-2f5a3f1f4011"},
        {"Dataset Name": "Checklist Completions", "SchemaID": "decc7bbf-716c-44c1-9754-4356405fb9aa", "Full PluginID": "b45ebe15-e737-4794-8204-11c27c469bbc", "Differential PluginID": "c3f8eca0-d886-45f4-b18b-5c68e5ed2c93"},
        {"Dataset Name": "Checklist Item Details", "SchemaID": "49907f5d-4de2-48dd-838c-5b143a5565c5", "Full PluginID": "490ab141-85b5-40b0-995b-e07daea3ca23", "Differential PluginID": "de613ceb-0d60-46f4-ad57-366fddf10cd2"},
        {"Dataset Name": "Checklist Objects", "SchemaID": "d4a354f1-edfd-4443-9a86-95e219935a78", "Full PluginID": "96a69e76-1045-4e06-8517-5783a844b50e", "Differential PluginID": "8ce4c7b2-7313-4610-a98a-e8b6e11c70c0"},
        {"Dataset Name": "Competency Activities", "SchemaID": "b544c470-29a8-449e-9a77-7825388fbba4", "Full PluginID": "a9887522-31cb-429c-90d1-5f2f31bf10b8", "Differential PluginID": "01d3b529-5929-4f54-8348-d9e92fdb726a"},
        {"Dataset Name": "Competency Activity Log", "SchemaID": "d33bfddd-5151-46f7-966e-0afa6023f300", "Full PluginID": "dd43a7db-801d-4fb3-b8f9-f94844e012f6", "Differential PluginID": "ee58980a-4884-4dee-861b-ca685b00d77b"},
        {"Dataset Name": "Competency Activity Results", "SchemaID": "6cecdaab-bff9-484f-a0d9-61c558707fad", "Full PluginID": "7d3bda26-1c97-4f31-a8fa-1df8fbd83dc0", "Differential PluginID": "91593014-7d61-4a04-adaf-54c86061ae43"},
        {"Dataset Name": "Competency Log", "SchemaID": "b7bb72a8-2ae0-43a3-83e8-2d1a0f9165ca", "Full PluginID": "1f2ca72d-f39d-4ef6-95f6-ad5e991b8b8e", "Differential PluginID": "279cd3bf-70db-4f4d-be40-8183b54c8ee4"},
        {"Dataset Name": "Competency Objects", "SchemaID": "5992b233-bf20-4991-b6e4-ea15f7fa6330", "Full PluginID": "47e0822b-832a-4cf7-bcf6-4d6481dd97c1", "Differential PluginID": "e6360178-a6b9-492b-8f52-5a55f8361f18"},
        {"Dataset Name": "Competency Structure", "SchemaID": "4a579623-dac9-4f38-a8a0-991ed6577f86", "Full PluginID": "ad3f4d61-3314-447a-a521-d1b73fa9308d", "Differential PluginID": "21c458ed-10f0-452d-abd3-99cb6b454234"},
        {"Dataset Name": "Content Files Properties Log", "SchemaID": "c73385ca-9ac4-439a-a625-bb473f450b49", "Full PluginID": "7ddc7dcd-6da0-4119-ae03-a6f6d631d739", "Differential PluginID": "becb2eba-fecb-49f9-ada8-f06dd59ddc15"},
        {"Dataset Name": "Content Objects", "SchemaID": "ae3fe47d-0fee-43cb-94b3-7e9e15c4e14e", "Full PluginID": "7e16311c-d302-45da-afd9-98af90706ccb", "Differential PluginID": "ad64a823-099c-4f5d-ad56-0806daddc286"},
        {"Dataset Name": "Content User Completion", "SchemaID": "b7f3452e-9cae-4172-ad03-f242f61d7c61", "Full PluginID": "1c50d2a2-990b-4897-a8c2-89a7a3202514", "Differential PluginID": "39c598e7-ac6e-475b-a42a-219c60cb6f9f"},
        {"Dataset Name": "Content User Progress", "SchemaID": "2572cc01-77f8-481c-b2a6-907971ec2b83", "Full PluginID": "428ad0cb-6203-486d-be85-adb01c79578b", "Differential PluginID": "28905553-dab1-405a-9ecc-8e33ee467e7e"},
        {"Dataset Name": "Course Access", "SchemaID": "2386cc16-2058-4495-a3ee-2148e7dddf0f", "Full PluginID": "e260902a-582c-48c9-8dd8-80aa7dfa6b76", "Differential PluginID": "01078704-43a5-45b4-a0ae-8fccdd89c6fd"},
        {"Dataset Name": "Course Access Log", "SchemaID": "b41cdecc-9ef9-444b-ade1-e67f8bc55473", "Full PluginID": "ed57eba8-f6e0-4d2d-8030-dbb6645e5b7b", "Differential PluginID": "ef0d4243-6d14-4a7f-b0cb-43c42b881f26"},
        {"Dataset Name": "Course Awards", "SchemaID": "a89d5e34-b4d7-45ad-bbfa-7cc327aac819", "Full PluginID": "ebb6cb39-1d1c-4e97-8974-f658414d2272", "Differential PluginID": "e82ae523-ff2c-43e2-a274-14d2a852e86d"},
        {"Dataset Name": "Course Copy Logs", "SchemaID": "d3c500d0-c384-48b3-9770-70023a6b2ca7", "Full PluginID": "1bca1fba-4edf-4f88-9f9f-daee9ca3a0c5", "Differential PluginID": "4bb85466-3b1c-4a39-ae27-281d6c57ebc5"},
        {"Dataset Name": "Course Publisher Launches", "SchemaID": "4be23ec6-9cf7-43b5-bf38-a82251d96e5b", "Full PluginID": "08f2847c-c80e-4dd0-9414-a1be8005f5ae", "Differential PluginID": "33761584-26d6-4053-a5f8-0a744a9f2fb9"},
        {"Dataset Name": "Course Publisher Recipients", "SchemaID": "6fd5da94-5270-4359-a6b2-8d0896527c1f", "Full PluginID": "e8b4670e-fda5-4c93-a8db-689e6881230c", "Differential PluginID": "1eaa086d-9073-4857-afc9-12dde5e8a93a"},
        {"Dataset Name": "Creator+ Practices Adoption", "SchemaID": "45f8a83e-78e0-4444-8d3f-d242e35c7158", "Full PluginID": "ef4e1555-0136-4aae-9003-d9c8ca3e074e", "Differential PluginID": "44af38cf-2e49-4789-b2cc-ad50d1c0c32b"},
        {"Dataset Name": "Creator+ Practices Engagement", "SchemaID": "ebc1b345-76b7-4daa-b07d-f2a892e32c7f", "Full PluginID": "d8894e7f-06e6-4dc8-8810-585bfe61fdd5", "Differential PluginID": "d6ca0aed-16c7-4d9c-ada9-51332e4f99a5"},
        {"Dataset Name": "Discussion Forums", "SchemaID": "fd57b574-f156-48ef-bc53-670e2b3d0f58", "Full PluginID": "8851ce21-6049-4004-9990-78c372bbd3b7", "Differential PluginID": "1fb4c71f-f9ee-4a71-9448-cf2df83d76f6"},
        {"Dataset Name": "Discussion Posts", "SchemaID": "f7c47f8b-35f2-466c-95e9-04c315ec07ff", "Full PluginID": "bce64f34-acee-415e-aceb-e3a38ddf476f", "Differential PluginID": "8d3ee4fc-2bc1-4708-abbe-450651c7ad24"},
        {"Dataset Name": "Discussion Posts Read Status", "SchemaID": "b4c8636c-b717-4422-97e3-aa40c9722e70", "Full PluginID": "ac51124b-6038-4b04-a186-92eb4cef40b0", "Differential PluginID": "dfad9cf9-dc5f-482e-87c6-5c69bdfca161"},
        {"Dataset Name": "Discussion Topic User Scores", "SchemaID": "dd162f3c-65f1-40f1-b81d-9a15c36c1cf2", "Full PluginID": "1c4add93-4905-4b24-b50d-a14fd10c971a", "Differential PluginID": "7a6fec74-d0e8-4c96-a5e6-f17fa999edba"},
        {"Dataset Name": "Discussion Topics", "SchemaID": "92068fe7-3976-426c-8406-fa655977ae04", "Full PluginID": "0646bbe1-79af-48ef-89d9-91f677419259", "Differential PluginID": "454c1b3f-be89-49c9-a9e7-5d6d6365fb6a"},
        {"Dataset Name": "Enrollments and Withdrawals", "SchemaID": "05dc704d-2a2c-4bc6-8cf6-ee6b998dc2db", "Full PluginID": "88cfcc22-ce8b-4dab-8d42-2b9da92f29cf", "Differential PluginID": "b6660b04-aabe-4603-b415-c9520d7931fe"},
        {"Dataset Name": "Grade Objects", "SchemaID": "dacc3bad-81ed-4cec-975f-88598c660f02", "Full PluginID": "793668a8-2c58-4e5e-b263-412d28d5703f", "Differential PluginID": "e0856750-abf2-4f1b-9c3a-ad82a0cfedc1"},
        {"Dataset Name": "Grade Objects Log", "SchemaID": "d725ab09-9369-4919-a8dc-7e8bd8c1acff", "Full PluginID": "1fa8ff9c-8702-46fc-a863-18ca6c2cc4d1", "Differential PluginID": "c8ccb9d4-8e5b-47d6-baea-b6d96a4877a5"},
        {"Dataset Name": "Grade Results", "SchemaID": "4a8f154b-9a55-4782-af80-9360d56ff3c9", "Full PluginID": "9d8a96b4-8145-416d-bd18-11402bc58f8d", "Differential PluginID": "3c44270c-9224-4e61-abee-696ba1b4882f"},
        {"Dataset Name": "Grade Scheme Ranges", "SchemaID": "dfc437cd-f9a5-4926-be0a-c94c4ea09556", "Full PluginID": "93caee13-773a-4e7f-9e49-5cb430ef7072", "Differential PluginID": "bed50bd2-9f0b-49bd-9b39-3317e4b82d62"},
        {"Dataset Name": "Grade Schemes", "SchemaID": "bcd6ca2c-51b8-452f-a689-aacaf0e6726c", "Full PluginID": "74308e1e-b0c0-437c-b3df-3a19e3b6f305", "Differential PluginID": "8d1b1c9b-0ad6-419c-bed2-3a720fad7d81"},
        {"Dataset Name": "Gradebook Settings", "SchemaID": "4196a13f-4431-47d7-b2e0-a97197b5d18c", "Full PluginID": "c3672b39-846e-49bf-8cf4-69015a8f15c1", "Differential PluginID": "8532e304-e08e-469f-a503-172b1b93345b"},
        {"Dataset Name": "Intelligent Agent Objects", "SchemaID": "d336eec3-b1eb-41ef-8245-ff20ca98fe14", "Full PluginID": "b069488a-aff7-42f2-828c-14bb0f71f3f4", "Differential PluginID": "063dced8-0e14-43f6-a1e9-a6261b316e50"},
        {"Dataset Name": "Intelligent Agent Run Log", "SchemaID": "748e30e9-7a04-449e-a557-d1bde27f6c0a", "Full PluginID": "c6045d32-c269-47ac-9d39-ce6be77c2015", "Differential PluginID": "23a9bff4-5c9a-446b-bd4d-46a924369997"},
        {"Dataset Name": "Intelligent Agent Run Users", "SchemaID": "a4f709a6-34ea-490d-bad1-22268db8c6ac", "Full PluginID": "ba6315cf-4c0d-42b4-827f-b9f29e1e34f5", "Differential PluginID": "1dc1fc77-8906-4685-ad0c-66a841cb5142"},
        {"Dataset Name": "JIT Provisioned Users Log", "SchemaID": "72e99089-3a7c-46d8-be69-b3e4dd0c13d6", "Full PluginID": "99de58c7-df43-4b83-a295-dcddcd943407", "Differential PluginID": "2261e717-de0c-4753-bde4-3a2540bc1bb1"},
        {"Dataset Name": "LTI Advantage Deployment Audit", "SchemaID": "56ab3878-f6ec-41a0-af94-3358fc85729a", "Full PluginID": "581ecbef-ca17-48c0-853b-c275c7aaf4f1", "Differential PluginID": "194176cf-17e8-4a81-8436-c5704ef2a20a"},
        {"Dataset Name": "LTI Advantage Registration Audit", "SchemaID": "4f75c912-cb31-46a1-bf5c-61f983228f0c", "Full PluginID": "38e40e20-d76f-49e6-a466-53e44ef4c191", "Differential PluginID": "545a5c59-eff6-4701-8050-fff99762e549"},
        {"Dataset Name": "LTI Launches", "SchemaID": "4f47d05d-951c-4cc8-8627-edc1e4eff481", "Full PluginID": "f233257d-c0f9-431f-89ca-3d55f633eb67", "Differential PluginID": "d23a1754-c088-455c-a653-8bf6b601b093"},
        {"Dataset Name": "LTI Link Migration Audit", "SchemaID": "804fff95-a6c7-48f2-9c57-6de4ed040d1e", "Full PluginID": "95951262-4878-4da9-86f0-188e5d1489b3", "Differential PluginID": "c36c12fc-2b76-4646-9214-e9ac2d223faa"},
        {"Dataset Name": "LTI Links", "SchemaID": "7a0d551f-0bbb-4544-a077-e1c427a0e5f3", "Full PluginID": "e00227de-5563-4c89-80f7-c6847eafe6f9", "Differential PluginID": "5d85d8e7-5141-42bc-8855-c25c913feed0"},
        {"Dataset Name": "Local Authentication Security Log", "SchemaID": "66dc53a4-4f62-4235-9727-af59f670f0b5", "Full PluginID": "b8ada4c7-d5d7-4377-bf8a-81718978ba01", "Differential PluginID": "01227024-8c3f-4c63-9498-83d8c4eb85b5"},
        {"Dataset Name": "Media Consumption Log", "SchemaID": "ce26f4a9-098b-40be-8e75-f5f9b8571972", "Full PluginID": "18652288-c12a-48e2-9b22-0556f4c5a2aa", "Differential PluginID": "4c910031-488b-443f-92d2-e94c1e6b21f3"},
        {"Dataset Name": "Media Objects", "SchemaID": "75c16f2d-a40f-42db-b8a4-314608320ccf", "Full PluginID": "377a2e14-09ba-407d-9836-726c7592a79ab", "Differential PluginID": "c657a528-877d-40d9-b56a-3d3c5cf64ae1"},
        {"Dataset Name": "Organizational Unit Ancestors", "SchemaID": "c0b0740f-896e-4afa-bfd9-81d8e43006d9", "Full PluginID": "61726e1b-bf42-4cab-910d-e5a226dec4f0", "Differential PluginID": "42846d2d-cce6-4215-ab21-4228a952a0db"},
        {"Dataset Name": "Organizational Unit Descendants", "SchemaID": "1168c2bc-c734-4727-b53c-062824124e74", "Full PluginID": "2e20f325-6fef-4065-9b5d-1400304611db", "Differential PluginID": "56d9e64a-0076-4fe7-8fd8-2f68feeb6161"},
        {"Dataset Name": "Organizational Unit Parents", "SchemaID": "4ed08e9f-d294-478c-912e-6b0ba4282e4a", "Full PluginID": "cb7caa4a-c35f-48d0-a9ae-59eefea299df", "Differential PluginID": "54be8f9d-b6ec-48e7-a18e-ab17c7fa8d42"},
        {"Dataset Name": "Organizational Unit Recent Access", "SchemaID": "e87a0eed-992d-475a-8e82-3598f939d11e", "Full PluginID": "41783af1-7030-4453-b342-89d44bbd8c5b", "Differential PluginID": "9b93ed8b-1b12-45a8-8277-adcbfeabe48a"},
        {"Dataset Name": "Organizational Units", "SchemaID": "53d5273c-1dc0-412b-beb3-417298bd0c6d", "Full PluginID": "07a9e561-e22f-4e82-8dd6-7bfb14c91776", "Differential PluginID": "867fb940-2b80-49da-9c8b-277c99686fc3"},
        {"Dataset Name": "Outcome Details", "SchemaID": "ed2401a6-e596-4ae9-b9c0-029a2b433efc", "Full PluginID": "4c4eaad1-8c54-4188-aaa0-bcdf4a346349", "Differential PluginID": "9c012fc0-b161-42fe-9c9e-e27594991490"},
        {"Dataset Name": "Outcome Registry Owners", "SchemaID": "cf6ef813-481d-45ac-8e4a-5df1e13863af", "Full PluginID": "a78aec93-5652-4b47-9a4e-865465b19ee6", "Differential PluginID": "77c494d0-08e2-4fab-8228-65e9d01b5aa2"},
        {"Dataset Name": "Outcomes Aligned to Tool Objects", "SchemaID": "ca9ba3aa-6207-41dc-a4cc-f75bf2b0ddda", "Full PluginID": "8b30bf00-35bb-4a54-86f1-c5fb11685907", "Differential PluginID": "57dfaca2-3e64-46a6-b71a-657e3a3e2fcf"},
        {"Dataset Name": "Outcomes Assessed Checkpoints", "SchemaID": "645fc68b-04a5-4d42-8282-be71e54b2068", "Full PluginID": "928e8a7a-b07a-400a-8ecd-f00af6ace96c", "Differential PluginID": "c1900ac5-b2be-4c6a-9dd6-0a3ec19ea3b1"},
        {"Dataset Name": "Outcomes Course Specific Scales", "SchemaID": "b4d68640-c259-481b-927c-90162f25fda1", "Full PluginID": "7d6998c0-4fc5-4582-b9a4-3e93e73c4c24", "Differential PluginID": "df5a8a7a-1fbc-4fad-bfc7-e928ec8073d8"},
        {"Dataset Name": "Outcomes Demonstrations", "SchemaID": "2e5f74b3-be2e-4cd5-afe8-d6fb8c31c9c3", "Full PluginID": "b07cae00-72c4-43e5-b083-4da09e425e17", "Differential PluginID": "837d26ec-25c3-4ace-8afd-5084ca98aca7"},
        {"Dataset Name": "Outcomes In Registries", "SchemaID": "99376909-7e3c-44cf-a723-76e50243a54d", "Full PluginID": "eb28c5ce-d299-4ba0-9956-fe94b6e4fd30", "Differential PluginID": "c6521348-a7a9-49e8-8b27-2e845b29ea18"},
        {"Dataset Name": "Outcomes Program Details", "SchemaID": "b120d95d-6baf-4687-8502-c03ec8b109d1", "Full PluginID": "90699954-a040-47b2-86ec-b603a7b73f67", "Differential PluginID": "e05c4bab-effd-467a-b241-4694ff4abc74"},
        {"Dataset Name": "Outcomes Rubric Alignments", "SchemaID": "a38958d0-f56b-49fd-90fe-c7e712c6e3f5", "Full PluginID": "aed60e0d-1f4a-4c2c-b129-57be73fbadf8", "Differential PluginID": "545d5972-757f-49d2-be04-10e6df21a667"},
        {"Dataset Name": "Outcomes Scale Definition", "SchemaID": "b3037586-4d6f-4069-a138-3d3069af9657", "Full PluginID": "b5504ee3-fb43-4bf2-83cc-23ecd0e0077b", "Differential PluginID": "df4f3513-fe31-4564-8f5c-453269ab1613"},
        {"Dataset Name": "Outcomes Scale Level Definition", "SchemaID": "2cfd2cc4-c003-4033-bc1d-ee5b25d9dbf9", "Full PluginID": "c31e05cc-b1a4-4b74-ad9b-b4a9baa1eed5", "Differential PluginID": "36088317-eefc-491b-80cc-94ae8d3cbaa4"},
        {"Dataset Name": "Outcomes Set Course", "SchemaID": "dc669c5d-30a0-489e-9ee5-8ef1e0307376", "Full PluginID": "95912ea7-e30f-4b81-8f0e-ac232c651fbd", "Differential PluginID": "184f7a53-0a64-490a-9a9a-f808cc0ed1b3"},
        {"Dataset Name": "Outcomes Set Org", "SchemaID": "011231c5-579b-4db1-94c2-45f1d21555c0", "Full PluginID": "93aeb3f1-e2b8-411c-9ba6-e07ffbfe4e63", "Differential PluginID": "28feb868-943c-4391-9714-461ab9cd57d1"},
        {"Dataset Name": "Portfolio Categories", "SchemaID": "6ef67fee-bc34-4a67-9ac5-75be0953b24b", "Full PluginID": "45da28c2-81f9-4b31-ad7c-edbc1838ae63", "Differential PluginID": "ee6afb72-7781-478f-9996-321611490c58"},
        {"Dataset Name": "Portfolio Evidence Categories", "SchemaID": "b3ada66d-100e-4cab-85ca-d6e103cfd673", "Full PluginID": "5b0d3990-7459-4a82-abe1-ee791d4d1bab", "Differential PluginID": "fa10ab40-f3c1-4774-a5e5-fae391b2901f"},
        {"Dataset Name": "Portfolio Evidence Log", "SchemaID": "a5246ef3-66b2-4396-ba90-a145095ad5d2", "Full PluginID": "3d2f0520-0ac8-4e10-82e0-52c93de34586", "Differential PluginID": "790dfc62-1bae-4c99-aec4-403eae17c16c"},
        {"Dataset Name": "Portfolio Evidence Objects", "SchemaID": "f7736415-b09e-4227-beed-1e559be1c40f", "Full PluginID": "07c902c8-15e0-4124-b0f5-b83f2a87fb24", "Differential PluginID": "6ed27c39-16ab-4211-a279-baad7e41b06e"},
        {"Dataset Name": "PreRequisite Conditions Met", "SchemaID": "15cf3c9c-fc92-46d4-bc0b-25610e376e8a", "Full PluginID": "1e44c2f7-a5ac-4d06-9a60-0ad55c941c59", "Differential PluginID": "6c2f8bc3-a4b1-48d3-bd3b-41bd66bdc862"},
        {"Dataset Name": "Question Answer Options", "SchemaID": "b3362aa3-f74d-4424-bf38-32c57a4d75a4", "Full PluginID": "005ebd00-00be-4830-8ff4-cb3853312585", "Differential PluginID": "bbeca5be-12b4-4bc8-8280-65ff4042d3fc"},
        {"Dataset Name": "Question Answers", "SchemaID": "0900c10e-1862-4b21-95d1-43eee5e51e1b", "Full PluginID": "faa8bab8-25f9-4921-ade5-becaedc526e8", "Differential PluginID": "a2b2c14e-3ffd-431c-b64f-53b577c781d9"},
        {"Dataset Name": "Question Library", "SchemaID": "c9edec37-1322-44ed-a922-f68d11472f6e", "Full PluginID": "5c0f2c70-4737-44ee-8780-be67bfa43594", "Differential PluginID": "708469f2-92ef-43f3-bdd3-dfef560e3432"},
        {"Dataset Name": "Question Relationships", "SchemaID": "7a1a456f-3bd9-403e-bf04-a384036da3ae", "Full PluginID": "1a0f6b1d-513c-474c-b7ce-0ee1fbea8d02", "Differential PluginID": "2127ae26-4ec6-4cfa-9dc7-fb741e3fbcfd"},
        {"Dataset Name": "Quiz Attempts", "SchemaID": "a7d6e843-bf8d-4965-9274-95028f3c4d86", "Full PluginID": "f1623581-c5d7-4562-93fe-6ad16010c96b", "Differential PluginID": "d8c9b542-0f2d-4d7e-9774-c07bebe2eff6"},
        {"Dataset Name": "Quiz Attempts Log", "SchemaID": "a8e9249d-3412-4dff-bfd4-758c66fd2f55", "Full PluginID": "d1c3127a-b8a4-48ff-924d-eb5a6ac6a344", "Differential PluginID": "5a0f30dc-4294-4d78-9721-ad6fe0867915"},
        {"Dataset Name": "Quiz Objects", "SchemaID": "f6fc270a-20ec-4fe0-9e90-c461fb2c53b2", "Full PluginID": "eef7ca81-86bb-430c-96ee-382b83f5c0f9", "Differential PluginID": "ca67a7b2-5b01-44a6-be86-ca95e6ca33e3"},
        {"Dataset Name": "Quiz Survey Sections", "SchemaID": "560e2992-1633-4770-905f-b2445fe4c0c2", "Full PluginID": "b2cfad85-ab8b-4e0a-a3e3-67495aa2875b", "Differential PluginID": "45c79f1b-3d6f-4f47-a6ec-dc8ac480becc"},
        {"Dataset Name": "Quiz User Answer Responses", "SchemaID": "08d61f34-f5c0-4896-a345-e3941535b4d4", "Full PluginID": "24d9051c-509a-4ea3-81bc-735f36bf94f0", "Differential PluginID": "436b1311-fe12-4939-a678-b2c108110b29"},
        {"Dataset Name": "Quiz User Answers", "SchemaID": "241c5dfb-4807-4f54-a7c4-78fbd7fb2671", "Full PluginID": "93d6063b-61d4-4629-a6af-b4fad71f8c55", "Differential PluginID": "79d6974c-76f7-4bb9-94c3-1fb30c47d0b3"},
        {"Dataset Name": "Reoffer Course", "SchemaID": "5d73f0ef-55a0-45d0-892b-1c5120e333a7", "Full PluginID": "65514d61-c74c-43b7-a386-58a11f9695ff", "Differential PluginID": "ebcce1cf-a3db-4786-b638-6337c957f7bb"},
        {"Dataset Name": "Release Condition Objects", "SchemaID": "93deef8a-db75-463d-8448-bc7924e241fa", "Full PluginID": "156ff1c7-fbaa-4c61-af19-23682018702d", "Differential PluginID": "2ff23328-8f79-46ed-9d95-fab0107b25a8"},
        {"Dataset Name": "Release Condition Results", "SchemaID": "ed253ac1-c0ec-44f1-89b6-e15cfccaacff", "Full PluginID": "86a481b1-de9d-4e5c-bf06-da03a18c09a5", "Differential PluginID": "2181b243-ac22-4a5c-8c13-a4f1fe808dbd"},
        {"Dataset Name": "Role Details", "SchemaID": "d70f64e0-ad63-4140-aac5-e337560b8371", "Full PluginID": "bd61f20b-be91-4b93-b449-46361e2c323f", "Differential PluginID": "e49a4837-72d7-4175-80d9-bd4dc10bdd08"},
        {"Dataset Name": "Rubric Assessment", "SchemaID": "d197b592-0c59-438c-8186-f42af6fddd35", "Full PluginID": "cd7fa762-841e-48c5-abd7-6379b84963bf", "Differential PluginID": "9e8d3ed5-f0f0-4834-a9e5-b64fe61beada"},
        {"Dataset Name": "Rubric Assessment Criteria", "SchemaID": "7c01ead6-8011-4d04-8b9a-7289b731a9fc", "Full PluginID": "612e3196-52ad-42bd-b460-8b850f7a7be1", "Differential PluginID": "d22a84ac-412e-428a-a094-08d39360e68a"},
        {"Dataset Name": "Rubric Criteria Levels", "SchemaID": "a69d02ee-82d8-4246-8573-16c4723cc86c", "Full PluginID": "f2fe26f9-fd27-4e1f-bb14-09f339963519", "Differential PluginID": "5e4e0fa0-b4b7-4077-81b6-511c8b64cacd"},
        {"Dataset Name": "Rubric Object Criteria", "SchemaID": "fac02315-302f-41b8-8b07-10c00f7a8d1d", "Full PluginID": "df537dc9-8358-4c28-9ab9-ddb8d364a9fc", "Differential PluginID": "17bec595-147d-4483-aee5-b52d7b8bd69a"},
        {"Dataset Name": "Rubric Object Levels", "SchemaID": "a1abc89e-5458-4127-8756-c8df9d859992", "Full PluginID": "bbe237cd-5afa-4ad1-b936-4d3404b9a6ca", "Differential PluginID": "26038f5a-539d-4933-86cb-fcb1bbf24cb7"},
        {"Dataset Name": "Rubric Objects", "SchemaID": "c473942c-74e4-4ef2-83a6-638434a7db26", "Full PluginID": "841308a2-e761-498e-a4cc-0c3619791c19", "Differential PluginID": "11b43b24-560b-4ed4-bcba-5ae46d39b949"},
        {"Dataset Name": "Rubrics Edit", "SchemaID": "d0294b4f-7c1e-43e7-8114-5f391c8f9626", "Full PluginID": "b64c7afb-ba9f-48da-bc95-ec54adb0fe6d", "Differential PluginID": "9c143916-3878-4ce6-8745-4b2243e4cbfc"},
        {"Dataset Name": "SCORM Activities", "SchemaID": "7e93eb03-3df3-4135-8e4c-080372b80149", "Full PluginID": "396e9578-04e3-49b2-bf89-c17757e501de", "Differential PluginID": "ac416eb4-49c8-4599-b6a5-c5a640340a5b"},
        {"Dataset Name": "SCORM Activity Attempts", "SchemaID": "435ee960-871f-484f-8e66-44886dea08f8", "Full PluginID": "d18ed567-e0a3-4fb7-912f-84d294620830", "Differential PluginID": "7e6de3f4-23ec-4c8d-a8ac-22a1ddf9d795"},
        {"Dataset Name": "SCORM Interaction Attempts", "SchemaID": "4e723260-b2e7-41d0-a165-a975f1a5aabc", "Full PluginID": "74d0cf71-0ca6-485c-80e2-fe054b3d8e8c", "Differential PluginID": "538b3b7b-f0f3-419d-9b0c-ee856bb217bc"},
        {"Dataset Name": "SCORM Interaction Objectives", "SchemaID": "fd6d617f-f3bf-498a-a45e-c8db9d1c3615", "Full PluginID": "fdec3d31-39f6-4615-9118-8c8dbe81ccbd", "Differential PluginID": "67795928-7917-4dec-b325-e8f308c178c2"},
        {"Dataset Name": "SCORM Interactions", "SchemaID": "11a4bf10-7d8f-486a-a028-6cffd542b1e7", "Full PluginID": "30760ee3-2db6-4303-8e00-c4d155d95a25", "Differential PluginID": "0cb43c48-5e89-4031-8937-670f0f621332"},
        {"Dataset Name": "SCORM Objective Attempts", "SchemaID": "7430daa7-2b6a-4291-9409-25b77fceff74", "Full PluginID": "f8128ba4-5fa2-4768-ac3e-117f6390c6e0", "Differential PluginID": "2d3e86f5-22d6-47e9-85ea-93be2b992e98"},
        {"Dataset Name": "SCORM Objectives", "SchemaID": "6e59993c-1708-4e36-bdae-2c440027823b", "Full PluginID": "51b028b4-d5ae-4784-b03e-18556b50e590", "Differential PluginID": "2ce9fd3b-cc9f-4d9f-8efc-1adfeb193c00"},
        {"Dataset Name": "SCORM Objects", "SchemaID": "d611f5e9-749e-4db0-b564-0549efe89d57", "Full PluginID": "ff5813ca-87f7-4def-983d-a2a88c42dbb4", "Differential PluginID": "e19f7212-d517-43d9-9078-ee97b63fdd1d"},
        {"Dataset Name": "SCORM Visits", "SchemaID": "ff935065-2431-4382-8524-ac40af9831d8", "Full PluginID": "30d67ca7-8c8b-4dde-93ef-becd2ac2e223", "Differential PluginID": "44556e27-057d-451f-9227-2a6e1b30a19c"},
        {"Dataset Name": "SIS Course Merge Log", "SchemaID": "b9611500-29d6-471f-95ff-014a599e0e74", "Full PluginID": "270a2afa-1f09-439c-b0df-f83b166b0dbd", "Differential PluginID": "12da5758-164c-4e00-af5a-e850c4403528"},
        {"Dataset Name": "Session History", "SchemaID": "02c0eb11-c1ae-4fe7-9e41-d9c3ffdc8567", "Full PluginID": "847d2e44-4fc1-4060-a6ce-80b6c6f95f7d", "Differential PluginID": "ff786a77-918c-4300-a9d2-a4e538c7e357"},
        {"Dataset Name": "Source Course Deploy History", "SchemaID": "c7e7e7c6-8d4a-4e3a-9b2e-1f2a3b4c5d6e", "Full PluginID": "a1b2c3d4-e5f6-47a8-9b0c-1d2e3f4a5b6c", "Differential PluginID": "c1d2e3f4-a5b6-47a8-9b0c-1d2e3f4a5b6d"},
        {"Dataset Name": "Survey Attempts", "SchemaID": "c597fc3e-b7ee-4d30-af3f-a7c825cc59f3", "Full PluginID": "10c06cbe-7171-4d6f-bd06-8330a93e5d2e", "Differential PluginID": "bd785b7d-5177-4529-8df2-48bf6affdaf1"},
        {"Dataset Name": "Survey Objects", "SchemaID": "ed15df0c-49e7-4ace-aeb3-22a415975a3f", "Full PluginID": "6bb3c6c2-7a61-44df-a081-d8762d93a3b5", "Differential PluginID": "3388f90c-2060-4583-a3df-50e9734b67d6"},
        {"Dataset Name": "Survey User Answer Responses", "SchemaID": "54d695fe-1c34-4fae-b1d5-a5923ca933a2", "Full PluginID": "20923295-981b-4d3c-8ab8-aa149abfdb45", "Differential PluginID": "0629be42-9f33-40c2-ad31-03f252b3689f"},
        {"Dataset Name": "Survey User Answers", "SchemaID": "75939752-2645-4fd7-89ff-663881ccf1af", "Full PluginID": "810d9b4e-6f05-4f02-b32d-c86d78999db1", "Differential PluginID": "fade45b2-ea09-4f65-8c2e-cc3a8a6615f5"},
        {"Dataset Name": "System Access Log", "SchemaID": "c3336250-30d1-45bc-809c-7b68b983f305", "Full PluginID": "5813e618-49ec-4e5c-90e7-1fb4fe4b59c6", "Differential PluginID": "a5bd3c98-3582-4232-8ddd-653d5ff7f074"},
        {"Dataset Name": "Tools", "SchemaID": "81d4bd50-9db6-495d-93ed-a3dbb6597a61", "Full PluginID": "c437b117-16b3-46b8-bae9-ac64948c8882", "Differential PluginID": "f931c869-db64-4c3c-a564-a3c54fa6598d"},
        {"Dataset Name": "TurnItIn Submissions", "SchemaID": "ac48b6d1-6655-48e4-907b-61c0fdbb37d1", "Full PluginID": "e4b3d080-b4f8-4d6c-abf3-98bf887829bc", "Differential PluginID": "341c1278-f32b-43c7-b29e-db4881a463f4"},
        {"Dataset Name": "User Attribute Definitions", "SchemaID": "0d1f2f4b-1b61-4373-9e33-a99a8d8dbfd9", "Full PluginID": "6fc12759-a014-4aec-8cdb-50ae6ff18530", "Differential PluginID": "1e11923c-267c-400b-bfb5-690ce15b2ff9"},
        {"Dataset Name": "User Attribute Values", "SchemaID": "cc69b71f-5186-46c0-85ba-cb5e02e4fabc", "Full PluginID": "fdf8ec1c-cec8-4c2d-bdde-e1b212aeeaa6", "Differential PluginID": "d301362c-ec67-4713-9bd6-23a84a58a24d"},
        {"Dataset Name": "User Enrollments", "SchemaID": "5ced736e-4c4c-4b01-96c8-1c7d404ac5c2", "Full PluginID": "533f84c8-b2ad-4688-94dc-c839952e9c4f", "Differential PluginID": "a78735f2-7210-4a57-aac1-e0f6bd714349"},
        {"Dataset Name": "User Logins", "SchemaID": "1c2ab1cc-c483-4657-906a-fed026be8750", "Full PluginID": "20794201-b8fe-4010-9197-9f4997f91531", "Differential PluginID": "49ac9b6f-8cbc-4a98-a95c-6ce0d89bca57"},
        {"Dataset Name": "Users", "SchemaID": "b21a6414-38f8-4da8-9a65-8b5586f9fe3b", "Full PluginID": "1d6d722e-b572-456f-97c1-d526570daa6b", "Differential PluginID": "e8339b7a-2d32-414e-9136-2adf3215a09c"}
    ]

    ref_df = pd.DataFrame(hardcoded)

    # ── DUPLICATE PREVENTION LOGIC ──
    # Create a set of existing names for fast, accurate looking up
    existing_names = set(ref_df['Dataset Name'])

    if not df.empty and 'dataset_name' in df.columns:
        new_list = []
        # Iterate through unique raw names in the scraper dataframe
        for name in df['dataset_name'].unique():
            # Format the name FIRST (snake_case -> Title Case)
            nice_name = ' '.join(word.capitalize() for word in name.replace('_', ' ').split())

            # Check if the FORMATTED name exists in the hardcoded list
            if nice_name not in existing_names:
                new_list.append({
                    "Dataset Name": nice_name,
                    "SchemaID": "(New – ID not mapped yet)",
                    "Full PluginID": "(New – ID not mapped yet)",
                    "Differential PluginID": "(New – ID not mapped yet)"
                })
                # Add to set to prevent duplicates if the loop encounters the same name again
                existing_names.add(nice_name)

        if new_list:
            new_df = pd.DataFrame(new_list)
            ref_df = pd.concat([ref_df, new_df], ignore_index=True)

    ref_df = ref_df.sort_values("Dataset Name").reset_index(drop=True)

    st.dataframe(
        ref_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Dataset Name": st.column_config.TextColumn("Dataset Name", width="medium"),
            "SchemaID": st.column_config.TextColumn("SchemaID"),
            "Full PluginID": st.column_config.TextColumn("Full PluginID"),
            "Differential PluginID": st.column_config.TextColumn("Differential PluginID"),
        }
    )

    st.success(f"✅ Showing {len(ref_df)} datasets (hard-coded + any newly discovered)")
    st.info("All available IDs are populated; any gaps will be addressed at a later date.")


def render_relationship_map(df: pd.DataFrame, selected_datasets: List[str]):
    """renders the relationship visualization with multiple graph types."""
    st.header("🗺️ Relationship Map")

    is_advanced = st.session_state['experience_mode'] == 'advanced'

    # graph type selection
    if is_advanced:
        graph_type = st.radio(
            "Visualization Style",
            ["Spring Layout (Network)", "Orbital Map (Galaxy)", "Relationship Matrix (Heatmap)"],
            horizontal=True
        )
    else:
        graph_type = st.radio(
            "Visualization Style",
            ["Spring Layout (Network)", "Orbital Map (Galaxy)"],
            horizontal=True
        )

    if graph_type == "Spring Layout (Network)":
        col_mode, col_controls = st.columns([2, 1])

        with col_mode:
            graph_mode = st.radio(
                "Graph Mode:",
                ["Focused (Between Selected)", "Discovery (From Selected)"],
                horizontal=True,
                help=(
                    "**Focused:** Shows only connections between your selected datasets. "
                    "**Discovery:** Shows all datasets your selection connects to."
                )
            )

        with st.expander("🛠️ Graph Settings", expanded=False):
            col_c1, col_c2, col_c3, col_c4 = st.columns(4)
#------------------------------
            with col_c1:
                graph_height = st.slider("Graph Height", 400, 1200, 600)
                edge_font_size = st.slider(
                    "Edge Label Size", 6, 20, 16,
                    help="Font size for join key labels on connections."
                )
                edge_thickness = st.slider(
                    "Edge Thickness", 0.5, 5.0, 1.5, step=0.5,
                    help="Line thickness for connections between datasets."
                )
#------------------------------
#------------------------------
            with col_c2:
                show_edge_labels = st.checkbox("Show Join Labels", True)
                validate_joins = st.checkbox("Validate Joins", False, help="Check selection for connectivity issues.")
                
                # NEW CHECKBOX
                hide_hubs = st.checkbox(
                    "Hide Common Hubs", 
                    False, 
                    help="Hides 'Users' and 'Organizational Units' to reduce clutter, unless selected."
                )

            # Joins Validation Logic
            if validate_joins and selected_datasets:
                joins = get_joins_for_selection(df, selected_datasets)
                if joins.empty:
                    st.warning("⚠️ No joins detected in selection. Consider adding bridges like 'Users' or 'Organizational Units'.")
                else:
                    isolated = [ds for ds in selected_datasets if ds not in joins['Source Dataset'].unique() and ds not in joins['Target Dataset'].unique()]
                    if isolated:
                        st.warning(f"⚠️ Isolated datasets: {', '.join(isolated)}. They lack connections—review relationships.")

            if is_advanced:
                with col_c3:
                    graph_font_size = st.slider("Font Size", 8, 24, 14)
                with col_c4:
                    node_separation = st.slider("Node Separation", 0.1, 2.5, 0.9)
            else:
                graph_font_size = 14
                node_separation = 0.9

        if not selected_datasets:
            st.info("👈 Select a Template or Datasets from the sidebar to visualize their relationships.")
        else:
            mode = 'focused' if 'Focused' in graph_mode else 'discovery'

            # bridge finder logic
            if len(selected_datasets) > 1 and mode == 'focused':
                current_joins = get_joins_for_selection(df, selected_datasets)

                if current_joins.empty:
                    st.warning("⚠️ These datasets don't connect directly. You might be missing a 'bridge' table.")

                    if st.button("🕵️ Find Missing Link"):
                        with st.spinner("Searching for a bridge table..."):
                            all_ds = df['dataset_name'].unique()
                            candidates = []
                            for candidate in all_ds:
                                if candidate in selected_datasets:
                                    continue
                                temp_group = list(selected_datasets) + [candidate]
                                temp_joins = get_joins_for_selection(df, temp_group)
                                # if it connects to at least 2 of our original selection
                                if len(temp_joins['Source Dataset'].unique()) >= 2:
                                    candidates.append(candidate)

                            if candidates:
                                st.success(f"Try adding: {', '.join(candidates[:3])}")
                            else:
                                st.error(
                                    "No direct bridge found. These datasets might be unrelated."
                                )

            if mode == 'focused':
                st.caption("Showing direct PK-FK connections between selected datasets only.")
            else:
                st.caption(
                    "Showing all datasets that your selection connects to via foreign keys."
                )

            config = {
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'brightspace_entity_diagram',
                    'height': 1200,
                    'width': 1600,
                    'scale': 2
                }
            }

            # CALL UPDATED FUNCTION
#------------------------------
            fig = create_spring_graph(
                df, selected_datasets, mode,
                graph_font_size, node_separation, graph_height, show_edge_labels,
                hide_hubs=hide_hubs,
                edge_font_size=edge_font_size,
                edge_thickness=edge_thickness
            )
            st.plotly_chart(fig, use_container_width=True, config=config)

            # Export Diagram
            with st.expander("📤 Export Diagram (Visio / LucidChart / PNG)"):
                c_dot, c_png = st.columns([2, 1])

                with c_png:
                    st.info(
                        "📷 **To get a PNG Image:**\n"
                        "Hover over the graph above and click the Camera icon (📸) "
                        "in the top-right corner. It is configured for high-res export."
                    )

                with c_dot:
                    st.markdown("#### GraphViz / DOT Export")
                    st.caption(
                        "Copy this code into **LucidChart** (Import -> GraphViz), "
                        "**Visio**, or **WebGraphViz** to create editable diagrams."
                    )

                    dot_lines = [
                        "digraph BrightspaceData {",
                        "  rankdir=LR;",
                        "  node [shape=box, style=filled, color=lightblue];"
                    ]

                    export_joins = get_joins_for_selection(df, selected_datasets)

#------------------------------
                    for ds in selected_datasets:
                        safe_ds = ds.replace('"', '\\"')
                        dot_lines.append(f'  "{safe_ds}" [label="{safe_ds}"];')

                    if not export_joins.empty:
                        for _, row in export_joins.iterrows():
                            s = row['Source Dataset'].replace('"', '\\"')
                            t = row['Target Dataset'].replace('"', '\\"')
                            k = row['column_name'].replace('"', '\\"')

                            if mode == 'focused':
                                if s in selected_datasets and t in selected_datasets:
                                    dot_lines.append(
                                        f'  "{s}" -> "{t}" [label="{k}", fontsize=10];'
                                    )
                            else:
                                if s in selected_datasets:
                                    dot_lines.append(
                                        f'  "{s}" -> "{t}" [label="{k}", fontsize=10];'
                                    )

                    dot_lines.append("}")
                    dot_string = "\n".join(dot_lines)

                    st.text_area("DOT Code", dot_string, height=150)
                    st.download_button("📥 Download .gv File", dot_string, "diagram.gv")
#------------------------------
                with c_png:
                    st.markdown("#### Mermaid.js Export")
                    st.caption("Copy into GitHub, Notion, or Obsidian.")
                    
                    mermaid_lines = ["graph LR"]
                    
                    # Add styles
                    mermaid_lines.append("    classDef focus fill:#0969da,stroke:#fff,stroke-width:2px,color:#fff;")
                    mermaid_lines.append("    classDef neighbor fill:#f6f8fa,stroke:#d0d7de,stroke-width:1px,color:#24292f;")

                    # Add nodes
                    for ds in selected_datasets:
                        safe_id = re.sub(r'[^a-zA-Z0-9]', '_', ds)
                        mermaid_lines.append(f"    {safe_id}[{ds}]:::focus")

                    # Add edges
                    if not export_joins.empty:
                        for _, row in export_joins.iterrows():
                            s = row['Source Dataset']
                            t = row['Target Dataset']
                            k = row['column_name']
                            
                            s_id = re.sub(r'[^a-zA-Z0-9]', '_', s)
                            t_id = re.sub(r'[^a-zA-Z0-9]', '_', t)
                            
                            # Filter based on mode (Focused vs Discovery) - reusing logic from DOT generation
                            include_edge = False
                            if mode == 'focused':
                                if s in selected_datasets and t in selected_datasets:
                                    include_edge = True
                            else:
                                if s in selected_datasets:
                                    include_edge = True
                                    # Ensure target node is defined if it wasn't a focus node
                                    if t not in selected_datasets:
                                        mermaid_lines.append(f"    {t_id}[{t}]:::neighbor")

                            if include_edge:
                                mermaid_lines.append(f"    {s_id} -- {k} --> {t_id}")

                    mermaid_string = "\n".join(mermaid_lines)
                    st.text_area("Mermaid Code", mermaid_string, height=150)
            # Integrated SQL generation
            if mode == 'focused' and len(selected_datasets) > 1:
                with st.expander("⚡ Get SQL for this View", expanded=False):
                    col_dial, col_cap = st.columns([2, 3])
                    with col_dial:
                        dialect = st.radio(
                            "SQL Dialect:",
                            ["T-SQL", "Snowflake", "PostgreSQL"],
                            horizontal=True,
                            label_visibility="collapsed"
                        )
                    with col_cap:
                        st.caption(f"Generating syntax for **{dialect}**.")

                    sql_code = generate_sql(selected_datasets, df, dialect)

                    st.code(sql_code, language="sql")

                    col_copy, col_goto = st.columns([1, 4])
                    with col_copy:
                        st.download_button(
                            label="📥 Download .sql",
                            data=sql_code,
                            file_name=f"graph_query_{dialect.lower()}.sql",
                            mime="application/sql"
                        )

            # Relationships table
            join_data = get_joins_for_selection(df, selected_datasets)

            if mode == 'focused' and not join_data.empty:
                join_data = join_data[join_data['Target Dataset'].isin(selected_datasets)]

            if not join_data.empty:
                with st.expander("📋 View Relationships Table", expanded=True):
                    sources = set(join_data['Source Dataset'])
                    targets = set(join_data['Target Dataset'])
                    parents = [
                        ds for ds in selected_datasets
                        if ds in targets and ds not in sources
                    ]

                    if parents:
                        st.info(
                            "ℹ️ **Note:** **"
                            + ", ".join(parents)
                            + "** appear in the 'To Table' column because they are "
                            "**Parent Tables** (they hold the Primary Key)."
                        )

                    st.dataframe(
                        join_data,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Source Dataset": "From Table (Child)",
                            "column_name": "Join Key",
                            "Target Dataset": "To Table (Parent)",
                            "Target Category": "Parent Category"
                        }
                    )
                    # Export Joins as CSV
                    joins_csv = join_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Export Joins CSV",
                        data=joins_csv,
                        file_name="brightspace_joins.csv",
                        mime="text/csv",
                        help="Download the relationships table as CSV."
                    )
            elif mode == 'focused' and len(selected_datasets) > 1:
                with st.expander("📋 View Relationships Table"):
                    st.info("No direct joins found between these specific datasets.")

#------------------------------
    elif graph_type == "Orbital Map (Galaxy)":
        st.caption("Categories are shown as golden suns, datasets orbit around their category.")

        with st.expander("🛠️ Graph Settings", expanded=False):
            col_o1, col_o2, col_o3 = st.columns(3)
            with col_o1:
                orbital_height = st.slider("Graph Height", 400, 1200, 700, key="orbital_height")
                orbital_line_width = st.slider(
                    "Connection Line Width", 0.5, 6.0, 2.0, step=0.5,
                    key="orbital_line_width",
                    help="Thickness of connection lines between datasets."
                )
            with col_o2:
                orbital_cat_font = st.slider(
                    "Category Label Size", 6, 24, 10, key="orbital_cat_font",
                    help="Font size for category names."
                )
                orbital_node_scale = st.slider(
                    "Node Size Scale", 0.5, 3.0, 1.0, step=0.25,
                    key="orbital_node_scale",
                    help="Multiplier for all node sizes."
                )
            with col_o3:
                orbital_legend_font = st.slider(
                    "Legend Font Size", 8, 20, 12, key="orbital_legend_font",
                    help="Font size for key names in the legend."
                )

        all_ds = sorted(df['dataset_name'].unique())

        target = st.selectbox(
            "🎯 Target Dataset (click to highlight connections)",
            ["None"] + all_ds
        )
        target_val = None if target == "None" else target

        active_keys_filter = None

        if target_val:
            target_cols = set(df[df['dataset_name'] == target_val]['column_name'])
            other_cols = set(df[df['dataset_name'] != target_val]['column_name'])
            shared_attributes = target_cols.intersection(other_cols)

            if shared_attributes:
                available_keys = sorted(list(shared_attributes))

                st.info(
                    "ℹ️ Filter now allows selecting **Shared Attributes** "
                    "(e.g. Username) in addition to strict Keys."
                )
                active_keys_filter = st.multiselect(
                    "Filter Connections by Column Name:",
                    available_keys,
                    placeholder=(
                        "Select columns (e.g. Username, OrgUnitId) "
                        "to find shared connections..."
                    ),
                    help=(
                        "Selecting a column here will highlight ANY dataset that shares this column name."
                    )
                )

        col_map, col_details = st.columns([3, 1])

#------------------------------
        with col_map:
            fig = get_orbital_map(df, target_val, active_keys_filter)

            # apply visual overrides post-cache
            # 1. Scale node sizes
            for trace in fig.data:
                if hasattr(trace, 'marker') and trace.marker is not None:
                    if trace.marker.size is not None:
                        sizes = trace.marker.size
                        if isinstance(sizes, (list, tuple)):
                            trace.marker.size = [s * orbital_node_scale for s in sizes]
                        elif isinstance(sizes, (int, float)):
                            trace.marker.size = sizes * orbital_node_scale

                # 2. Scale connection line widths
                if hasattr(trace, 'line') and trace.line is not None:
                    if trace.line.width is not None:
                        trace.line.width = orbital_line_width

                # 3. Scale category label font
                if hasattr(trace, 'textfont') and trace.textfont is not None:
                    if trace.textfont.color == 'gold':
                        trace.textfont.size = orbital_cat_font

            # 4. Scale legend font and graph height
            fig.update_layout(
                height=orbital_height,
                legend=dict(
                    font=dict(size=orbital_legend_font)
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            if active_keys_filter:
                with st.expander("💾 View & Download Connection Data", expanded=True):
                    conn_data = []

                    for key in active_keys_filter:
                        matches = df[df['column_name'] == key]['dataset_name'].unique()
                        for match in matches:
                            if match != target_val:
                                conn_data.append({
                                    "Target Dataset": target_val,
                                    "Shared Column": key,
                                    "Connected Dataset": match,
                                    "Connected Category": df[df['dataset_name'] == match]['category'].iloc[0]
                                })

                    if conn_data:
                        conn_df = pd.DataFrame(conn_data)
                        st.dataframe(conn_df, hide_index=True, use_container_width=True)

                        csv = conn_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Connections as CSV",
                            data=csv,
                            file_name=f"{target_val}_connections.csv",
                            mime="text/csv"
                        )

        with col_details:
            if target_val:
                st.markdown(f"### {target_val}")
                show_relationship_summary(df, target_val)

                ds_data = df[df['dataset_name'] == target_val]
                if not ds_data.empty:
                    st.caption(f"Category: {ds_data.iloc[0]['category']}")

                    if 'url' in ds_data.columns and ds_data.iloc[0]['url']:
                        st.link_button("📄 Documentation", ds_data.iloc[0]['url'])

                    with st.expander("Schema", expanded=True):
                        display_cols = ['column_name', 'data_type', 'key']
                        available_cols = [c for c in display_cols if c in ds_data.columns]
                        st.dataframe(ds_data[available_cols], hide_index=True, use_container_width=True)
            else:
                st.info("Select a target dataset to see its details and connections.")

    elif graph_type == "Relationship Matrix (Heatmap)":
        st.caption("Heatmap showing which datasets reference which via foreign keys.")

        fig = create_relationship_matrix(df)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "💡 Tip: Hover over cells to see the exact connection count. "
            "Darker colors = more connections."
        )


def render_schema_browser(df: pd.DataFrame):
    """renders the schema browser and search functionality."""
    st.header("📋 Schema Browser")

    col_search, col_browse = st.columns([1, 2])

    with col_search:
        st.subheader("🔍 Column Search")
        
        search = st.text_input(
            "Find Column",
            placeholder="e.g. OrgUnitId, UserId, LastAccessed...",
            key="schema_browser_column_search",
            help="Press Enter after typing to search across all datasets"
        )

        if search:
            escaped_search = re.escape(search)
            hits = df[
                df['column_name'].str.contains(escaped_search, case=False, na=False, regex=True) |
                df['description'].str.contains(escaped_search, case=False, na=False, regex=True)
            ].copy()

            if not hits.empty:
                st.success(f"Found **{len(hits)}** matching columns across **{hits['dataset_name'].nunique()}** datasets")

                for ds_name in sorted(hits['dataset_name'].unique()):
                    ds_hits = hits[hits['dataset_name'] == ds_name]
                    with st.expander(f"📦 {ds_name} ({len(ds_hits)} matches)", expanded=len(ds_hits) <= 8):
                        display_cols = ['column_name', 'data_type', 'description', 'key']
                        available_cols = [c for c in display_cols if c in ds_hits.columns]
                        st.dataframe(
                            ds_hits[available_cols],
                            hide_index=True,
                            use_container_width=True,
                            height=min(300, len(ds_hits) * 35)
                        )
            else:
                st.warning("No matches found.")
        else:
            st.caption("Type a column name or keyword, in whole or partially, and press **Enter** to search")

    with col_browse:
        st.subheader("📂 Browse by Dataset")

        all_ds = sorted(df['dataset_name'].unique())

        selected_ds_list = st.multiselect(
            "Select Datasets",
            options=all_ds,
            placeholder="Choose one or more datasets to inspect..."
        )

        if selected_ds_list:
            # shared column analysis when 2+ datasets selected
            if len(selected_ds_list) >= 2:
                with st.expander("🔗 Shared Column Analysis", expanded=True):
                    st.caption(
                        "Columns that appear in multiple selected datasets — "
                        "potential join keys or shared dimensions."
                    )

                    # build column-to-dataset mapping
                    col_to_datasets = {}
                    for ds in selected_ds_list:
                        ds_cols = df[df['dataset_name'] == ds]['column_name'].unique()
                        for col in ds_cols:
                            if col not in col_to_datasets:
                                col_to_datasets[col] = []
                            col_to_datasets[col].append(ds)

                    # filter to columns appearing in 2+ selected datasets
                    shared = {
                        col: datasets
                        for col, datasets in col_to_datasets.items()
                        if len(datasets) >= 2
                    }

                    if shared:
                        shared_rows = []
                        for col, datasets in sorted(shared.items()):
                            shared_rows.append({
                                "Column": col,
                                "Shared By": ", ".join(sorted(datasets)),
                                "Count": len(datasets),
                                "Is Key": "🔑" if col in ENUM_DEFINITIONS or col.endswith("Id") else ""
                            })

                        shared_df = pd.DataFrame(shared_rows).sort_values(
                            ["Count", "Column"], ascending=[False, True]
                        )

                        st.dataframe(
                            shared_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Count": st.column_config.ProgressColumn(
                                    "Overlap",
                                    help="Number of selected datasets containing this column",
                                    format="%d",
                                    min_value=0,
                                    max_value=len(selected_ds_list),
                                )
                            }
                        )

                        st.caption(
                            f"**{len(shared)}** shared columns found across "
                            f"**{len(selected_ds_list)}** selected datasets."
                        )
                    else:
                        st.info("No shared columns found between the selected datasets.")

                st.divider()

            for i, selected_ds in enumerate(selected_ds_list):
                if i > 0:
                    st.divider()

                st.markdown(f"### 📦 {selected_ds}")

                subset = df[df['dataset_name'] == selected_ds]

                # Contextual Description
                if 'dataset_description' in subset.columns:
                    desc_text = subset['dataset_description'].iloc[0]
                    if desc_text:
                        st.info(f"**Context:** {desc_text}", icon="💡")
                    else:
                        st.caption("No context description available.")

                col_info, col_stats = st.columns([2, 1])

                with col_info:
                    if not subset.empty and 'category' in subset.columns:
                        st.caption(f"Category: **{subset.iloc[0]['category']}**")
                    if 'url' in subset.columns and subset.iloc[0]['url']:
                        st.link_button("📄 View Documentation", subset.iloc[0]['url'])

                with col_stats:
                    show_relationship_summary(df, selected_ds)

                # Enum decoder ring
                ds_columns = subset['column_name'].tolist()
                found_enums = {
                    col: ENUM_DEFINITIONS[col]
                    for col in ds_columns if col in ENUM_DEFINITIONS
                }

                if found_enums:
                    with st.expander(f"💡 Column Value Decoders ({selected_ds})", expanded=True):
                        st.caption(
                            "This dataset contains columns with coded integer values. "
                            "Here's what they mean:"
                        )

                        if len(found_enums) > 1:
                            tabs = st.tabs(list(found_enums.keys()))
                            for idx, (col_name, mapping) in enumerate(found_enums.items()):
                                with tabs[idx]:
                                    enum_df = pd.DataFrame(
                                        list(mapping.items()),
                                        columns=["Value (ID)", "Meaning"]
                                    )
                                    st.dataframe(
                                        enum_df,
                                        hide_index=True,
                                        use_container_width=True
                                    )
                        else:
                            col_name = list(found_enums.keys())[0]
                            mapping = found_enums[col_name]
                            st.markdown(f"**{col_name}**")
                            enum_df = pd.DataFrame(
                                list(mapping.items()),
                                columns=["Value (ID)", "Meaning"]
                            )
                            st.dataframe(
                                enum_df,
                                hide_index=True,
                                use_container_width=True
                            )

                # Schema table
                st.markdown("#### Schema")
                display_cols = ['column_name', 'data_type', 'description', 'key']
                available_cols = [c for c in display_cols if c in subset.columns]

                st.dataframe(
                    subset[available_cols],
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                # Quick Export Button for this Dataset's Schema
                schema_csv = subset[available_cols].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Schema CSV",
                    data=schema_csv,
                    file_name=f"{selected_ds.lower().replace(' ', '_')}_schema.csv",
                    mime="text/csv",
                    help="Download this dataset's columns as CSV for offline use."
                )
                
                st.divider()
                
                # Mock Data Generator Logic
                with st.expander("🎲 Generate Mock Data (CSV)", expanded=False):
                    st.caption("Generate dummy rows for testing ETL scripts or reports.")
                    
                    c_rows, c_seed = st.columns(2)
                    with c_rows:
                        num_rows = st.number_input("Rows to Generate", min_value=1, max_value=1000, value=10, key=f"mock_n_{selected_ds}")
                    
                    if st.button("🚀 Generate Data", key=f"btn_mock_{selected_ds}"):
                        mock_data = []
                        import random
                        from datetime import datetime, timedelta

                        # Simple generator map
                        def get_mock_value(dtype, col_name):
                            dtype = dtype.lower()
                            col_lower = col_name.lower()
                            
                            # ID / Key logic
                            if 'id' in col_lower:
                                return random.randint(100, 99999)
                            
                            # Boolean
                            if 'bit' in dtype or 'bool' in dtype:
                                return random.choice([0, 1])
                            
                            # Numeric
                            if 'int' in dtype or 'num' in dtype or 'decimal' in dtype or 'float' in dtype:
                                return random.randint(0, 100)
                            
                            # Dates
                            if 'date' in dtype or 'time' in dtype:
                                start_date = datetime.now() - timedelta(days=365)
                                random_days = random.randint(0, 365)
                                return (start_date + timedelta(days=random_days)).strftime('%Y-%m-%d %H:%M:%S')
                            
                            # Strings - Context aware
                            if 'name' in col_lower:
                                return random.choice(['Introduction to Data', 'Advanced Analysis', 'Biology 101', 'History of Art'])
                            if 'user' in col_lower:
                                return random.choice(['jsmith', 'adoe', 'bwayne', 'ckent'])
                            if 'code' in col_lower:
                                return f"CODE-{random.randint(100,999)}"
                            if 'guid' in dtype or 'uuid' in dtype:
                                import uuid
                                return str(uuid.uuid4())
                            
                            # Default String
                            return "Lorem Ipsum"

                        # Generate Rows
                        cols_to_gen = subset['column_name'].tolist()
                        types_to_gen = subset['data_type'].tolist()
                        
                        for _ in range(num_rows):
                            row = {}
                            for col, dtype in zip(cols_to_gen, types_to_gen):
                                row[col] = get_mock_value(dtype, col)
                            mock_data.append(row)
                        
                        mock_df = pd.DataFrame(mock_data)
                        
                        st.dataframe(mock_df.head(), use_container_width=True, hide_index=True)
                        
                        # CSV Download
                        mock_csv = mock_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"📥 Download {num_rows} Mock Rows",
                            data=mock_csv,
                            file_name=f"MOCK_{selected_ds.replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                
                col_pk, col_fk = st.columns(2)

                with col_pk:
                    if 'is_primary_key' in subset.columns:
                        pks = subset[subset['is_primary_key']]['column_name'].tolist()
                        if pks:
                            st.markdown(f"🔑 **Primary Keys:** {', '.join(pks)}")

                with col_fk:
                    if 'is_foreign_key' in subset.columns:
                        fks = subset[subset['is_foreign_key']]['column_name'].tolist()
                        if fks:
                            st.markdown(f"🔗 **Foreign Keys:** {', '.join(fks)}")

                # DDL Export
                with st.expander("🏗️ Export as DDL (CREATE TABLE)", expanded=False):
                    ddl_dialect = st.radio(
                        "Dialect:",
                        ["T-SQL", "Snowflake", "PostgreSQL"],
                        horizontal=True,
                        key=f"ddl_dialect_{selected_ds}"
                    )
                    
                    ddl_code = generate_ddl(df, selected_ds, ddl_dialect)
                    st.code(ddl_code, language="sql")
                    
                    col_ddl_dl, col_ddl_info = st.columns([1, 2])
                    with col_ddl_dl:
                        st.download_button(
                            label="📥 Download DDL",
                            data=ddl_code,
                            file_name=f"{selected_ds.replace(' ', '_').lower()}_ddl_{ddl_dialect.lower()}.sql",
                            mime="application/sql",
                            key=f"ddl_download_{selected_ds}"
                        )
                    with col_ddl_info:
                        st.caption(
                            "⚠️ Review data types before use. D2L types are mapped to common SQL equivalents "
                            "but may need adjustment for your specific database."
                        )


def render_sql_builder(df: pd.DataFrame, selected_datasets: List[str]):
    """renders the sql builder interface with python/pandas support."""
#------------------------------
    st.header("⚡ Query Builder")

    st.info(
        "The builder analyzes PK/FK relationships between your selected datasets and generates "
        "deterministic JOIN queries. It supports composite keys, alias resolution "
        "(e.g., `SubmitterId` → `UserId`), and sibling joins through shared dimensions like `OrgUnitId`.\n\n"
        "Select **2 or more datasets** from the sidebar or use Quick Select below.",
        icon="💡"
    )

    if not selected_datasets:
        st.subheader("Quick Select")
        all_ds = sorted(df['dataset_name'].unique())
        quick_select = st.multiselect(
            "Select datasets here:", all_ds, key="sql_quick_select"
        )

        if quick_select:
            selected_datasets = quick_select
#------------------------------
        # Suggest Related Datasets Button
        if st.button("🤖 Suggest Related Datasets", help="AI-powered recommendations based on joins"):
            if not selected_datasets:
                st.warning("Select at least one dataset first.")
            else:
                with st.spinner("Analyzing connections..."):
                    joins = get_joins_for_selection(df, selected_datasets)
                    if not joins.empty:
                        related = set(joins['Target Dataset']) - set(selected_datasets)
                        suggestions = sorted(list(related))[:3]  # Top 3 by alpha for simplicity
                        if suggestions:
                            st.success(f"Try adding: {', '.join(suggestions)}")
                            # Auto-add to selection (optional; comment out if unwanted)
                            st.session_state['selected_datasets'] = list(set(selected_datasets + suggestions))
                            st.rerun()
                        else:
                            st.info("No direct connections found for suggestions.")
                    else:
                        st.info("No joins detected for these datasets.")
#------------------------------
    if selected_datasets:
        if len(selected_datasets) < 2:
            st.warning("Select at least 2 datasets to generate a JOIN.")
        else:
            st.markdown(f"**Selected:** {', '.join(selected_datasets)}")

            col_lang, col_opts, _ = st.columns([1, 1, 2])

            with col_lang:
                output_format = st.radio(
                    "Output Format",
                    ["SQL", "Python (Pandas)"],
                    horizontal=True,
                    help="Choose between generating a SQL query or Python code for CSV analysis."
                )

            if output_format == "SQL":
                with col_opts:
                    dialect = st.selectbox(
                        "Target Database Dialect",
                        ["T-SQL", "Snowflake", "PostgreSQL"],
                        help="Adjusts syntax for quotes ([], \"\") and limits (TOP vs LIMIT)."
                    )
                generated_code = generate_sql(selected_datasets, df, dialect)
                lang_label = "sql"
                file_ext = "sql"
                mime_type = "application/sql"
                download_label = f"Download {dialect} Query"
            else:
                with col_opts:
                    st.caption(
                        "Generates `pd.read_csv` and `pd.merge` code for local analysis."
                    )

                generated_code = generate_pandas(selected_datasets, df)
                lang_label = "python"
                file_ext = "py"
                mime_type = "text/x-python"
                download_label = "Download Python Script"

            col_code, col_schema = st.columns([2, 1])

            with col_code:
                st.markdown(f"#### Generated {output_format}")
                st.code(generated_code, language=lang_label)

                st.download_button(
                    label=f"📥 {download_label}",
                    data=generated_code,
                    file_name=f"brightspace_extract.{file_ext}",
                    mime=mime_type
                )

            with col_schema:
                st.markdown("#### Field Reference")
                for ds in selected_datasets:
                    with st.expander(f"📦 {ds}", expanded=False):
                        subset = df[df['dataset_name'] == ds]
                        display_cols = ['column_name', 'data_type', 'key']
                        available_cols = [c for c in display_cols if c in subset.columns]
                        st.dataframe(
                            subset[available_cols],
                            hide_index=True,
                            use_container_width=True,
                            height=200
                        )

            with st.expander("🗺️ Join Visualization"):
                fig = create_spring_graph(
                    df, selected_datasets, 'focused', 12, 1.0, 400, True
                )
                st.plotly_chart(fig, use_container_width=True)


def render_sql_translator():
    """renders the sql dialect translation tool."""
#------------------------------
    st.header("🔀 SQL Dialect Translator")
    st.markdown(
        "Convert queries between dialects (e.g., T-SQL to PostgreSQL) or to Python/Pandas."
    )

    st.info(
        "Use this when migrating queries between database platforms (e.g., SQL Server to Snowflake) "
        "or when converting SQL to Python for local CSV analysis.\n\n"
        "The AI engine handles function mappings (e.g., `GETDATE()` → `CURRENT_TIMESTAMP`), "
        "syntax differences (`TOP` → `LIMIT`), and type casting variations. "
        "Review the output before running in production — edge cases may require manual adjustment.",
        icon="💡"
    )

    if not st.session_state['authenticated']:
        st.warning(
            "🔒 Login required. This feature uses the AI engine to ensure accurate syntax translation."
        )
        return

    c1, c2 = st.columns(2)
    with c1:
        source_lang = st.selectbox(
            "Source Dialect",
            ["Auto-Detect", "T-SQL (SQL Server)", "MySQL", "Oracle", "PostgreSQL", "Snowflake"]
        )
    with c2:
        target_lang = st.selectbox(
            "Target Dialect",
            ["PostgreSQL", "Snowflake", "T-SQL (SQL Server)", "MySQL", "Python (Pandas)"]
        )

    input_query = st.text_area(
        "Paste Source Query",
        height=200,
        placeholder="SELECT TOP 10 * FROM Users WHERE CAST(Created AS DATE) = GETDATE()..."
    )

    if st.button("✨ Translate Code", type="primary"):
        if not input_query:
            st.error("Please enter a query to translate.")
            return

        system_msg = f"""You are an expert SQL Code Translator.
Task: Convert the following {source_lang} query into optimized {target_lang}.

Rules:
1. Preserve the original logic exactly.
2. Convert specific functions (e.g., GETDATE() -> NOW(), TOP -> LIMIT).
3. If converting to Python/Pandas, assume 'df' is the dataframe.
4. Output ONLY the code block. No conversational filler.
5. Add short comments explaining complex changes if necessary.
"""

        model = "gpt-4o-mini"
        provider = "OpenAI"

#------------------------------
        openai_key = get_secret("openai_api_key")
        xai_key = get_secret("xai_api_key")

        if openai_key:
            secret_key = openai_key
            base_url = None
            model_name = "gpt-4o-mini"
        elif xai_key:
            secret_key = xai_key
            base_url = "https://api.x.ai/v1"
            model_name = "grok-3-mini"
        else:
            st.error("No API Key found. Please login.")
            return

        try:
            with st.spinner(f"Translating to {target_lang}..."):
                client = openai.OpenAI(api_key=secret_key, base_url=base_url)

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": input_query}
                    ]
                )

                translated_code = response.choices[0].message.content

                st.subheader(f"Output ({target_lang})")
                st.code(
                    translated_code,
                    language="sql" if "Python" not in target_lang else "python"
                )
        except Exception as e:
            st.error(f"Translation failed: {str(e)}")


def render_ai_assistant(df: pd.DataFrame, selected_datasets: List[str]):
    """renders the ai chat interface."""
    st.header("🤖 AI Data Architect Assistant")

    if not st.session_state['authenticated']:
        st.warning(
            "🔒 Login required to use AI features. Please enter password in the sidebar."
        )

        st.info("""
**What the AI Assistant can do:**
- Explain dataset relationships and join strategies
- Suggest optimal query patterns
- Answer questions about the Brightspace data model
- Help design complex SQL queries
""")
        return

    col_settings, col_chat = st.columns([1, 3])

    with col_settings:
        st.markdown("#### ⚙️ Settings")

        model_options = list(PRICING_REGISTRY.keys())
        selected_model = st.selectbox("Model", model_options, index=3)

        model_info = PRICING_REGISTRY[selected_model]
        provider = model_info['provider']

        st.caption(f"Provider: **{provider}**")
        st.caption(f"Cost: ${model_info['in']:.2f}/${model_info['out']:.2f} per 1M tokens")

        key_name = "openai_api_key" if provider == "OpenAI" else "xai_api_key"
        secret_key = get_secret(key_name)
        if secret_key:
            st.success(f"✅ {provider} Key Loaded")
            api_key = secret_key
        else:
            api_key = st.text_input(f"{provider} API Key", type="password")

        use_full_context = st.checkbox(
            "Include Full Schema", value=False,
            help=(
                "Send entire database schema to AI. "
                "Higher cost but more comprehensive answers."
            )
        )

        with st.expander("💰 Session Cost", expanded=True):
            st.metric("Tokens", f"{st.session_state['total_tokens']:,}")
            st.metric("Cost", f"${st.session_state['total_cost']:.4f}")
            if st.button("Reset"):
                st.session_state['total_cost'] = 0.0
                st.session_state['total_tokens'] = 0
                st.rerun()

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    with col_chat:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if prompt := st.chat_input("Ask about the data model..."):
            if not api_key:
                st.error("Please provide an API key.")
                st.stop()

            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                if use_full_context:
                    schema_text = []
                    for ds_name, group in df.groupby('dataset_name'):
                        url = (
                            group['url'].iloc[0]
                            if 'url' in group.columns and pd.notna(group['url'].iloc[0])
                            else ""
                        )
                        cols = []
                        for _, row in group.iterrows():
                            c = row['column_name']
                            if row.get('is_primary_key'):
                                c += " (PK)"
                            elif row.get('is_foreign_key'):
                                c += " (FK)"
                            cols.append(c)
                        schema_text.append(
                            f"TABLE: {ds_name}\nURL: {url}\nCOLS: {', '.join(cols)}"
                        )

                    context = "\n\n".join(schema_text)
                    scope_msg = "FULL DATABASE SCHEMA"
                else:
                    relationships_context = ""

                    if selected_datasets:
                        context_df = df[df['dataset_name'].isin(selected_datasets)]
                        scope_msg = f"SELECTED DATASETS: {', '.join(selected_datasets)}"

                        known_joins = get_joins_for_selection(df, selected_datasets)

                        if not known_joins.empty:
                            relationships_context = (
                                "\n\nVERIFIED RELATIONSHIPS (Use these strictly for JOIN conditions):\n"
                            )
                            for _, row in known_joins.iterrows():
                                relationships_context += (
                                    f"- {row['Source Dataset']} joins to {row['Target Dataset']} "
                                    f"ON column '{row['column_name']}'\n"
                                )
                    else:
                        context_df = df.head(100)
                        scope_msg = "SAMPLE DATA (first 100 rows)"

                    cols_to_use = ['dataset_name', 'column_name', 'data_type', 'description', 'key']
                    available_cols = [c for c in cols_to_use if c in context_df.columns]

                    context = context_df[available_cols].to_csv(index=False) + relationships_context

                system_msg = (
                    "You are an expert SQL Data Architect specializing in Brightspace (D2L) data sets.\n\n"
                    f"Context: {scope_msg}\n"
                    "INSTRUCTIONS:\n"
                    "1. Provide clear, actionable answers about the data model\n"
                    "2. When suggesting JOINs, use proper syntax and explain the relationship\n"
                    "3. If dataset URLs are available, reference them for documentation\n"
                    "4. Be concise but thorough\n"
                    "SCHEMA DATA:\n"
                    f"{context[:60000]}"
                )

                base_url = "https://api.x.ai/v1" if provider == "xAI" else None
                client = openai.OpenAI(api_key=api_key, base_url=base_url)

                with st.spinner(f"Consulting {selected_model}..."):
                    response = client.chat.completions.create(
                        model=selected_model,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": prompt}
                        ]
                    )

                    reply = response.choices[0].message.content

#------------------------------
                    if hasattr(response, 'usage') and response.usage:
                        in_tok = response.usage.prompt_tokens or 0
                        out_tok = response.usage.completion_tokens or 0
                        cost = (
                            in_tok * model_info['in'] / 1_000_000 +
                            out_tok * model_info['out'] / 1_000_000
                        )
                        st.session_state['total_tokens'] += (in_tok + out_tok)
                        st.session_state['total_cost'] += cost

                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()

            except Exception as e:
                st.error(f"AI Error: {str(e)}")


#------------------------------
def convert_sql_dialect(sql: str, target_dialect: str) -> str:
    """
    converts t-sql syntax to target dialect.
    handles common function and syntax differences.
    """
    if target_dialect == "T-SQL":
        # ensure TOP clause for T-SQL
        if "SELECT TOP" not in sql and "SELECT" in sql:
            sql = sql.replace("SELECT", "SELECT TOP 100", 1)
        return sql

    # for snowflake and postgresql, remove TOP and add LIMIT
    sql = re.sub(r'SELECT\s+TOP\s+(\d+)', r'SELECT', sql, flags=re.IGNORECASE)
    if "LIMIT" not in sql.upper():
        sql = sql.rstrip().rstrip(';') + "\nLIMIT 100"

    # common conversions for both snowflake and postgresql
    sql = sql.replace("GETDATE()", "CURRENT_TIMESTAMP")
    sql = sql.replace("GETUTCDATE()", "CURRENT_TIMESTAMP")
    sql = sql.replace("ISNULL(", "COALESCE(")

    # dateadd conversion: DATEADD(day, -30, GETDATE()) -> CURRENT_TIMESTAMP - INTERVAL '30 days'
    dateadd_pattern = r"DATEADD\s*\(\s*(\w+)\s*,\s*(-?\d+)\s*,\s*([^)]+)\)"

    def dateadd_replacement(match):
        unit = match.group(1).lower()
        value = int(match.group(2))
        base_expr = match.group(3).strip()

        # normalize the base expression
        if base_expr.upper() in ("GETDATE()", "CURRENT_TIMESTAMP"):
            base_expr = "CURRENT_TIMESTAMP"

        # handle negative values
        if value < 0:
            operator = "-"
            value = abs(value)
        else:
            operator = "+"

        # pluralize unit if needed
        unit_map = {
            "day": "days",
            "month": "months",
            "year": "years",
            "hour": "hours",
            "minute": "minutes",
            "second": "seconds",
            "week": "weeks"
        }
        unit_plural = unit_map.get(unit, f"{unit}s")

        return f"({base_expr} {operator} INTERVAL '{value} {unit_plural}')"

    sql = re.sub(dateadd_pattern, dateadd_replacement, sql, flags=re.IGNORECASE)

    # datediff conversion: DATEDIFF(day, start, end) -> (end::date - start::date) for postgres
    # or DATEDIFF('day', start, end) for snowflake
    if target_dialect == "PostgreSQL":
        datediff_pattern = r"DATEDIFF\s*\(\s*(\w+)\s*,\s*([^,]+)\s*,\s*([^)]+)\)"

        def datediff_replacement(match):
            unit = match.group(1).lower()
            start_expr = match.group(2).strip()
            end_expr = match.group(3).strip()

            if unit == "day":
                return f"(({end_expr})::date - ({start_expr})::date)"
            else:
                return f"EXTRACT({unit.upper()} FROM ({end_expr}) - ({start_expr}))"

        sql = re.sub(datediff_pattern, datediff_replacement, sql, flags=re.IGNORECASE)

        # convert CONVERT(type, expr) to expr::type
        convert_pattern = r"CONVERT\s*\(\s*(\w+)\s*,\s*([^)]+)\)"
        sql = re.sub(convert_pattern, r"(\2)::\1", sql, flags=re.IGNORECASE)

    elif target_dialect == "Snowflake":
        # snowflake uses quoted unit in datediff
        datediff_pattern = r"DATEDIFF\s*\(\s*(\w+)\s*,"
        sql = re.sub(datediff_pattern, r"DATEDIFF('\1',", sql, flags=re.IGNORECASE)

        # convert CONVERT to TRY_CAST for snowflake
        convert_pattern = r"CONVERT\s*\(\s*(\w+)\s*,\s*([^)]+)\)"
        sql = re.sub(convert_pattern, r"TRY_CAST(\2 AS \1)", sql, flags=re.IGNORECASE)

    return sql

#------------------------------
def generate_ddl(df: pd.DataFrame, dataset_name: str, dialect: str = "T-SQL") -> str:
    """
    generates a CREATE TABLE statement for a dataset based on its schema.
    maps d2l data types to sql types as best as possible.
    """
    subset = df[df['dataset_name'] == dataset_name]
    if subset.empty:
        return f"-- No schema found for {dataset_name}"

    # dialect-specific quoting
    if dialect == "T-SQL":
        q_start, q_end = "[", "]"
    else:
        q_start, q_end = '"', '"'

    def quote(name: str) -> str:
        return f"{q_start}{name}{q_end}"

    # clean table name for SQL
    table_name = dataset_name.replace(' ', '_').replace('-', '_')

    # map D2L types to SQL types
    type_map = {
        # common mappings
        'int': 'INT',
        'bigint': 'BIGINT',
        'smallint': 'SMALLINT',
        'bit': 'BIT' if dialect == "T-SQL" else 'BOOLEAN',
        'float': 'FLOAT',
        'decimal': 'DECIMAL(18,4)',
        'numeric': 'DECIMAL(18,4)',
        'datetime': 'DATETIME' if dialect == "T-SQL" else 'TIMESTAMP',
        'datetime2': 'DATETIME2' if dialect == "T-SQL" else 'TIMESTAMP',
        'date': 'DATE',
        'time': 'TIME',
        'nvarchar': 'NVARCHAR(MAX)' if dialect == "T-SQL" else 'TEXT',
        'varchar': 'VARCHAR(MAX)' if dialect == "T-SQL" else 'TEXT',
        'text': 'TEXT',
        'ntext': 'NTEXT' if dialect == "T-SQL" else 'TEXT',
        'uniqueidentifier': 'UNIQUEIDENTIFIER' if dialect == "T-SQL" else 'UUID',
        'guid': 'UNIQUEIDENTIFIER' if dialect == "T-SQL" else 'UUID',
        'boolean': 'BIT' if dialect == "T-SQL" else 'BOOLEAN',
        'bool': 'BIT' if dialect == "T-SQL" else 'BOOLEAN',
    }

    def map_type(d2l_type: str) -> str:
        if not d2l_type:
            return 'VARCHAR(255)' if dialect == "T-SQL" else 'TEXT'
        
        d2l_lower = d2l_type.lower().strip()
        
        # check for exact match first
        if d2l_lower in type_map:
            return type_map[d2l_lower]
        
        # check for partial matches
        for key, val in type_map.items():
            if key in d2l_lower:
                return val
        
        # default fallback
        return 'VARCHAR(255)' if dialect == "T-SQL" else 'TEXT'

    # build column definitions
    columns = []
    pk_columns = []
    
    for _, row in subset.iterrows():
        col_name = row['column_name']
        data_type = row.get('data_type', '')
        is_pk = row.get('is_primary_key', False)
        is_nullable = row.get('is_nullable', '')
        
        sql_type = map_type(data_type)
        
        # determine nullability
        if is_pk:
            null_clause = "NOT NULL"
            pk_columns.append(col_name)
        elif 'no' in str(is_nullable).lower() or 'false' in str(is_nullable).lower():
            null_clause = "NOT NULL"
        else:
            null_clause = "NULL"
        
        columns.append(f"    {quote(col_name)} {sql_type} {null_clause}")

    # build the statement
    lines = [f"CREATE TABLE {quote(table_name)} ("]
    lines.append(",\n".join(columns))
    
    # add primary key constraint if applicable
    if pk_columns:
        pk_cols = ", ".join([quote(c) for c in pk_columns])
        lines.append(f",\n    CONSTRAINT PK_{table_name} PRIMARY KEY ({pk_cols})")
    
    lines.append(");")
    
    # add header comment
    header = [
        f"-- DDL for {dataset_name}",
        f"-- Dialect: {dialect}",
        f"-- Generated from Brightspace Dataset Explorer",
        f"-- Columns: {len(subset)}",
        ""
    ]
    
    return "\n".join(header) + "\n".join(lines)

def render_kpi_recipes(df: pd.DataFrame):
    """renders the cookbook of sql recipes."""
#------------------------------
    st.header("📚 KPI Recipes")
    st.markdown("Pre-packaged SQL queries for common educational analysis questions.")

    st.info(
        "These are ready-to-use SQL templates for common reporting questions. "
        "Each recipe targets specific datasets — look for ✅/❌ indicators to confirm "
        "availability in your schema.\n\n"
        "**How to use:** Select a category below, choose your SQL dialect, then copy or download the query. "
        "Adapt the `WHERE` clauses and column selections to match your institution's needs.",
        icon="💡"
    )

    all_cats = list(RECIPE_REGISTRY.keys())
    selected_cat = st.radio(
        "Category", all_cats, horizontal=True, label_visibility="collapsed"
    )

    recipes = RECIPE_REGISTRY[selected_cat]

    st.divider()

    for recipe in recipes:
        with st.container():
            c1, c2 = st.columns([3, 1])
            with c1:
                st.subheader(recipe["title"])
                st.write(recipe["description"])

#------------------------------
                available_datasets = set(df['dataset_name'].unique())
                tags = []
                for d in recipe["datasets"]:
                    if d in available_datasets:
                        tags.append(f"✅ {d}")
                    else:
                        tags.append(f"❌ {d}")
                tags.append(f"⚡ {recipe['difficulty']}")
                st.caption(" • ".join(tags))

            with c2:
                dialect = st.selectbox(
                    "Dialect",
                    ["T-SQL", "Snowflake", "PostgreSQL"],
                    key=f"rec_{recipe['title']}",
                    label_visibility="collapsed"
                )

            sql = convert_sql_dialect(recipe["sql_template"].strip(), dialect)

            with st.expander("👨‍🍳 View SQL Recipe", expanded=False):
                st.code(sql, language="sql")

                col_dl, col_note = st.columns([1, 2])
                with col_dl:
                    st.download_button(
                        label="📥 Download SQL",
                        data=sql,
                        file_name=f"recipe_{recipe['title'].lower().replace(' ', '_')}_{dialect.lower()}.sql",
                        mime="application/sql"
                    )
                with col_note:
                    if dialect != "T-SQL":
                        st.caption(f"⚠️ Converted from T-SQL to {dialect}. Verify syntax before use.")

            st.divider()

#------------------------------
def render_schema_diff(df: pd.DataFrame):
    """renders the schema diff tool to compare current schema against a backup."""
    st.header("✨ Schema Diff")
    st.markdown("Compare the current schema against a previous backup to identify changes.")

    uploaded_file = st.file_uploader(
        "Upload Backup CSV",
        type="csv",
        help="Upload a previously downloaded metadata backup CSV to compare against the current schema."
    )

    if not uploaded_file:
        st.info("👆 Upload a backup CSV file to begin comparison.")

        with st.expander("ℹ️ How to use Schema Diff", expanded=True):
            st.markdown("""
**Purpose:** Track how the Brightspace Datahub schema evolves over time.

**Steps:**
1. Download a backup using **💾 Download Metadata Backup** in the sidebar
2. Later, after re-scraping or updating, upload that backup here
3. Review which datasets/columns were added, removed, or modified

**What We Detect:**
- 📦 **Dataset-level:** New or removed datasets
- 📋 **Column-level:** New or removed columns within datasets
- ✏️ **Value-level:** Changes to column metadata (data_type, description, key)

**Use Cases:**
- Identify new datasets added by D2L in product updates
- Track columns that have been deprecated or renamed
- Spot documentation updates or type changes
- Audit changes before updating ETL pipelines
""")
        return

    try:
        backup_df = pd.read_csv(uploaded_file).fillna('')
    except Exception as e:
        st.error(f"Failed to parse uploaded CSV: {e}")
        return

    # Validate backup has expected columns
    required_cols = ['dataset_name', 'column_name']
    missing_cols = [c for c in required_cols if c not in backup_df.columns]
    if missing_cols:
        st.error(f"Backup CSV is missing required columns: {', '.join(missing_cols)}")
        return

    st.success(f"✅ Loaded backup with **{backup_df['dataset_name'].nunique()}** datasets and **{len(backup_df)}** columns")

    st.divider()

    # Dataset-level comparison
    current_datasets = set(df['dataset_name'].unique())
    backup_datasets = set(backup_df['dataset_name'].unique())

    added_datasets = current_datasets - backup_datasets
    removed_datasets = backup_datasets - current_datasets
    common_datasets = current_datasets & backup_datasets

    # Prepare column-level and value-level change tracking
    datasets_with_column_changes = []
    datasets_with_value_changes = []
    all_value_changes = []

    # Fields to compare for value-level changes
    compare_fields = ['data_type', 'description', 'key']
    available_compare_fields = [f for f in compare_fields if f in df.columns and f in backup_df.columns]

    for ds in sorted(common_datasets):
        current_cols = set(df[df['dataset_name'] == ds]['column_name'])
        backup_cols = set(backup_df[backup_df['dataset_name'] == ds]['column_name'])

        added_cols = current_cols - backup_cols
        removed_cols = backup_cols - current_cols
        common_cols = current_cols & backup_cols

        if added_cols or removed_cols:
            datasets_with_column_changes.append({
                'dataset': ds,
                'added': added_cols,
                'removed': removed_cols
            })

        # Value-level comparison for common columns
        if available_compare_fields and common_cols:
            current_subset = df[df['dataset_name'] == ds].set_index('column_name')
            backup_subset = backup_df[backup_df['dataset_name'] == ds].set_index('column_name')

            ds_value_changes = []

            for col in sorted(common_cols):
                if col in current_subset.index and col in backup_subset.index:
                    current_row = current_subset.loc[col]
                    backup_row = backup_subset.loc[col]

                    # Handle case where index returns multiple rows (duplicates)
                    if isinstance(current_row, pd.DataFrame):
                        current_row = current_row.iloc[0]
                    if isinstance(backup_row, pd.DataFrame):
                        backup_row = backup_row.iloc[0]

                    for field in available_compare_fields:
                        current_val = str(current_row.get(field, '')).strip()
                        backup_val = str(backup_row.get(field, '')).strip()

                        if current_val != backup_val:
                            change_record = {
                                'dataset': ds,
                                'column': col,
                                'field': field,
                                'old_value': backup_val if backup_val else '(empty)',
                                'new_value': current_val if current_val else '(empty)'
                            }
                            ds_value_changes.append(change_record)
                            all_value_changes.append(change_record)

            if ds_value_changes:
                datasets_with_value_changes.append({
                    'dataset': ds,
                    'changes': ds_value_changes
                })

    # Summary metrics
    col_summary, col_details = st.columns([1, 2])

    with col_summary:
        st.subheader("📊 Summary")
        st.metric("Datasets in Current", len(current_datasets))
        st.metric("Datasets in Backup", len(backup_datasets))
        st.metric("Datasets Added", len(added_datasets), delta=f"+{len(added_datasets)}" if added_datasets else None)
        st.metric("Datasets Removed", len(removed_datasets), delta=f"-{len(removed_datasets)}" if removed_datasets else None, delta_color="inverse")
        st.metric("Columns Changed", len(datasets_with_column_changes), help="Datasets with added or removed columns")
        st.metric("Values Modified", len(all_value_changes), help="Individual field value changes detected")

    with col_details:
        st.subheader("🔍 Dataset Changes")

        if added_datasets:
            with st.expander(f"➕ Added Datasets ({len(added_datasets)})", expanded=True):
                for ds in sorted(added_datasets):
                    col_count = len(df[df['dataset_name'] == ds])
                    category = df[df['dataset_name'] == ds]['category'].iloc[0] if not df[df['dataset_name'] == ds].empty else "Unknown"
                    st.markdown(f"- **{ds}** ({category}) — {col_count} columns")

        if removed_datasets:
            with st.expander(f"➖ Removed Datasets ({len(removed_datasets)})", expanded=True):
                for ds in sorted(removed_datasets):
                    col_count = len(backup_df[backup_df['dataset_name'] == ds])
                    category = backup_df[backup_df['dataset_name'] == ds]['category'].iloc[0] if not backup_df[backup_df['dataset_name'] == ds].empty else "Unknown"
                    st.markdown(f"- **{ds}** ({category}) — {col_count} columns")

        if not added_datasets and not removed_datasets:
            st.success("No datasets were added or removed.")

    st.divider()

    # Column-level changes (added/removed columns)
    st.subheader("📋 Column-Level Changes (Added/Removed)")

    if datasets_with_column_changes:
        st.warning(f"**{len(datasets_with_column_changes)}** dataset(s) have column additions or removals.")

        for change in datasets_with_column_changes:
            with st.expander(f"📦 {change['dataset']} (+{len(change['added'])} / -{len(change['removed'])})"):
                c1, c2 = st.columns(2)

                with c1:
                    if change['added']:
                        st.markdown("**Added Columns:**")
                        for col in sorted(change['added']):
                            st.markdown(f"- `{col}` ➕")
                    else:
                        st.caption("No columns added.")

                with c2:
                    if change['removed']:
                        st.markdown("**Removed Columns:**")
                        for col in sorted(change['removed']):
                            st.markdown(f"- `{col}` ➖")
                    else:
                        st.caption("No columns removed.")
    else:
        st.success("No columns were added or removed in common datasets.")

    st.divider()

    # Value-level changes (metadata modifications)
    st.subheader("✏️ Value-Level Changes (Metadata Modifications)")

    if not available_compare_fields:
        st.warning("Cannot compare values — backup CSV is missing comparison fields (data_type, description, key).")
    elif datasets_with_value_changes:
        st.warning(f"**{len(all_value_changes)}** value change(s) detected across **{len(datasets_with_value_changes)}** dataset(s).")

        # Group by field type for filtering
        field_counts = {}
        for change in all_value_changes:
            field_counts[change['field']] = field_counts.get(change['field'], 0) + 1

        st.caption(f"By field: {', '.join([f'{k}: {v}' for k, v in field_counts.items()])}")

        # Filter option
        filter_field = st.selectbox(
            "Filter by field type:",
            ["All Fields"] + available_compare_fields,
            help="Focus on specific types of changes."
        )

        for ds_change in datasets_with_value_changes:
            ds = ds_change['dataset']
            changes = ds_change['changes']

            if filter_field != "All Fields":
                changes = [c for c in changes if c['field'] == filter_field]

            if not changes:
                continue

            with st.expander(f"📦 {ds} ({len(changes)} change(s))"):
                for change in changes:
                    col_name = change['column']
                    field = change['field']
                    old_val = change['old_value']
                    new_val = change['new_value']

                    # Truncate long values for display
                    old_display = old_val[:100] + "..." if len(old_val) > 100 else old_val
                    new_display = new_val[:100] + "..." if len(new_val) > 100 else new_val

                    st.markdown(f"**`{col_name}`** — *{field}* changed:")
                    st.markdown(f"  - Old: `{old_display}`")
                    st.markdown(f"  - New: `{new_display}`")
                    st.markdown("---")
    else:
        st.success("No value-level changes detected in common columns.")

    # Export diff report
    st.divider()
    st.subheader("📥 Export Diff Report")

    diff_data = []

    for ds in sorted(added_datasets):
        cat = df[df['dataset_name'] == ds]['category'].iloc[0] if not df[df['dataset_name'] == ds].empty else ""
        diff_data.append({
            "Change Type": "Dataset Added",
            "Dataset": ds,
            "Column": "",
            "Field": "",
            "Old Value": "",
            "New Value": "",
            "Category": cat
        })

    for ds in sorted(removed_datasets):
        cat = backup_df[backup_df['dataset_name'] == ds]['category'].iloc[0] if not backup_df[backup_df['dataset_name'] == ds].empty else ""
        diff_data.append({
            "Change Type": "Dataset Removed",
            "Dataset": ds,
            "Column": "",
            "Field": "",
            "Old Value": "",
            "New Value": "",
            "Category": cat
        })

    for change in datasets_with_column_changes:
        ds = change['dataset']
        for col in sorted(change['added']):
            diff_data.append({
                "Change Type": "Column Added",
                "Dataset": ds,
                "Column": col,
                "Field": "",
                "Old Value": "",
                "New Value": "",
                "Category": ""
            })
        for col in sorted(change['removed']):
            diff_data.append({
                "Change Type": "Column Removed",
                "Dataset": ds,
                "Column": col,
                "Field": "",
                "Old Value": "",
                "New Value": "",
                "Category": ""
            })

    for change in all_value_changes:
        diff_data.append({
            "Change Type": "Value Modified",
            "Dataset": change['dataset'],
            "Column": change['column'],
            "Field": change['field'],
            "Old Value": change['old_value'],
            "New Value": change['new_value'],
            "Category": ""
        })

    if diff_data:
        diff_df = pd.DataFrame(diff_data)
        st.dataframe(diff_df, use_container_width=True, hide_index=True)

        csv = diff_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Diff Report (CSV)",
            data=csv,
            file_name="schema_diff_report.csv",
            mime="text/csv"
        )
    else:
        st.success("🎉 No differences detected! The schemas are identical.")

#------------------------------
def render_url_editor():
    """renders the full-width URL editor view."""
    st.header("✏️ URL Configuration")
    st.markdown("Edit the list of D2L Knowledge Base URLs to scrape for dataset metadata.")

    col_edit, col_help = st.columns([3, 1])

    with col_help:
        st.markdown("#### Tips")
        st.markdown("""
- **One URL per line**
- URLs must start with `http`
- Remove the top 2 URLs to exclude Advanced Data Sets
- Add new KB article URLs as D2L releases them
""")
        
        st.markdown("#### Quick Actions")
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.session_state['custom_urls'] = DEFAULT_URLS
            st.rerun()

    with col_edit:
        current_urls = st.session_state.get('custom_urls') or DEFAULT_URLS
        
        # Full-width text area with more height
        edited_urls = st.text_area(
            "URLs to Scrape (1 per line; remove the top 2 to exclude ADSs metadata)",
            value=current_urls,
            height=500,
            help="Each line should contain one complete URL to a D2L Knowledge Base article."
        )

        # Validation feedback
        lines = edited_urls.strip().split('\n')
        valid_urls = [u.strip() for u in lines if u.strip().startswith('http')]
        invalid_lines = [u.strip() for u in lines if u.strip() and not u.strip().startswith('http')]

        col_status, col_actions = st.columns([2, 1])

        with col_status:
            st.success(f"✅ **{len(valid_urls)}** valid URLs detected")
            if invalid_lines:
                st.warning(f"⚠️ **{len(invalid_lines)}** invalid line(s) will be ignored")
                with st.expander("View invalid lines"):
                    for line in invalid_lines[:10]:
                        st.code(line[:80] + "..." if len(line) > 80 else line)

        with col_actions:
            col_save, col_cancel = st.columns(2)
            
            with col_save:
                if st.button("💾 Save & Close", type="primary", use_container_width=True):
                    st.session_state['custom_urls'] = edited_urls
                    st.session_state['show_url_editor'] = False
                    st.success("URLs saved!")
                    st.rerun()

            with col_cancel:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state['show_url_editor'] = False
                    st.rerun()

        # Preview section
        with st.expander("👁️ Preview Valid URLs", expanded=False):
            for i, url in enumerate(valid_urls, 1):
                st.markdown(f"{i}. `{url}`")


# =========================================
# 10. UDF Flattener (EAV → wide)
# =============================================================================
#------------------------------
def render_health_check(df: pd.DataFrame):
    """validates stored schema against live D2L documentation."""
    st.header("🩺 Scrape Health Check")
    st.markdown("Validate the integrity of your scraped data against live D2L documentation.")

    # =============================================
    # TIER 1: Offline Checks (instant, no network)
    # =============================================
    st.subheader("⚡ Offline Checks")

    issues = []
    passes = []

    # 1. CSV Staleness
    try:
        mod_time = os.path.getmtime('dataset_metadata.csv')
        days_old = (pd.Timestamp.now() - pd.Timestamp(mod_time, unit='s')).days
        if days_old > 30:
            issues.append(f"⚠️ **Stale Data:** CSV is **{days_old}** days old. Consider re-scraping.")
        elif days_old > 7:
            passes.append(f"🟡 CSV is {days_old} days old.")
        else:
            passes.append(f"✅ CSV is fresh ({days_old} day(s) old).")
    except Exception:
        issues.append("❌ Cannot determine CSV age.")

    # 2. Datasets with zero columns
    ds_col_counts = df.groupby('dataset_name')['column_name'].count()
    zero_col_ds = ds_col_counts[ds_col_counts == 0].index.tolist()
    if zero_col_ds:
        issues.append(
            f"❌ **Empty Datasets:** {len(zero_col_ds)} dataset(s) have zero columns: "
            f"{', '.join(zero_col_ds[:5])}"
        )
    else:
        passes.append(f"✅ All {df['dataset_name'].nunique()} datasets have at least 1 column.")

    # 3. Duplicate dataset names (case-insensitive)
    from collections import Counter
    name_counts = Counter(df['dataset_name'].str.lower())
    dupes = [name for name, count in name_counts.items() if count > 1]
    if dupes:
        issues.append(f"⚠️ **Possible Duplicate Datasets:** {', '.join(dupes[:5])}")
    else:
        passes.append("✅ No duplicate dataset names detected.")

    # 4. Categories with only 1 dataset
    cat_counts = df.groupby('category')['dataset_name'].nunique()
    singleton_cats = cat_counts[cat_counts == 1].index.tolist()
    if singleton_cats:
        issues.append(
            f"🟡 **Singleton Categories:** {len(singleton_cats)} category(ies) have only 1 dataset: "
            f"{', '.join(singleton_cats[:5])}"
        )
    else:
        passes.append("✅ All categories have multiple datasets.")

    # 5. Suspect column names (possible header-as-data ingestion)
    suspect_names = ['field', 'type', 'description', 'name', 'column', 'data type']
    suspects = df[df['column_name'].str.lower().isin(suspect_names)]
    if not suspects.empty:
        ds_list = suspects['dataset_name'].unique()[:5]
        issues.append(
            f"⚠️ **Suspect Column Names:** Found columns named like table headers in: "
            f"{', '.join(ds_list)}"
        )
    else:
        passes.append("✅ No suspect header-as-data columns detected")

    # 6. Datasets missing descriptions
    desc_col = 'dataset_description'
    if desc_col in df.columns:
        ds_descs = df.groupby('dataset_name')[desc_col].first()
        missing_desc = ds_descs[ds_descs.astype(str).str.strip() == ''].index.tolist()
        total_ds = df['dataset_name'].nunique()
        has_desc = total_ds - len(missing_desc)
        if len(missing_desc) > total_ds * 0.5:
            issues.append(
                f"🟡 **Low Description Coverage:** Only {has_desc}/{total_ds} datasets "
                f"have context descriptions."
            )
        else:
            passes.append(f"✅ Description coverage: {has_desc}/{total_ds} datasets")

    # Display offline results
    col_pass, col_issue = st.columns(2)
    with col_pass:
        for p in passes:
            st.markdown(p)
    with col_issue:
        for i in issues:
            st.markdown(i)

    if not issues:
        st.success("All offline checks passed")

    st.divider()

    # =============================================
    # tier 2: online validation (hitting the D2L/community pages)
    # =============================================
    st.subheader("🌐 Online Validation")
    st.caption("Re-scrapes a sample of pages and compares against stored data to detect drift")

    urls_text = st.session_state.get('custom_urls') or DEFAULT_URLS
    all_urls = [u.strip() for u in urls_text.split('\n') if u.strip().startswith('http')]

    col_config, col_action = st.columns([2, 1])

    with col_config:
        sample_size = st.slider(
            "Pages to Sample",
            min_value=1,
            max_value=min(len(all_urls), 10),
            value=min(5, len(all_urls)),
            help="Number of random URLs to re-scrape for comparison."
        )

    with col_action:
        st.write("")  # spacer
        run_check = st.button("🔍 Run Live Check", type="primary", use_container_width=True)

    if run_check:
        import random

        sampled_urls = random.sample(all_urls, sample_size)
        results = []
        progress = st.progress(0, "Starting validation...")

        for i, url in enumerate(sampled_urls):
            progress.progress(
                (i + 1) / len(sampled_urls),
                f"Checking {i + 1}/{len(sampled_urls)}..."
            )

            # Extract category using same logic as scrape_and_save
            filename = os.path.basename(url).split('?')[0]
            if "advanced" in filename.lower():
                category = "Advanced Data Sets"
            else:
                clean_name = re.sub(r'^[\d\s-]+', '', filename)
                category = clean_name.replace('-data-sets', '').replace('-', ' ').strip().lower()

            try:
                live_data = scrape_table(url, category)
                if not live_data:
                    results.append({
                        'url': url,
                        'status': '❌ Empty',
                        'detail': 'Live scrape returned no data',
                        'added_ds': set(),
                        'removed_ds': set(),
                        'col_diffs': []
                    })
                    continue

                live_df = pd.DataFrame(live_data)
                live_df['dataset_name'] = live_df['dataset_name'].astype(str).apply(smart_title)

                # Get stored data for this URL
                stored_for_url = df[df['url'] == url]

                stored_ds = set(stored_for_url['dataset_name'].unique())
                live_ds = set(live_df['dataset_name'].unique())

                added_ds = live_ds - stored_ds
                removed_ds = stored_ds - live_ds
                common_ds = stored_ds & live_ds

                # Column-level comparison per common dataset
                col_diffs = []
                for ds in sorted(common_ds):
                    stored_cols = set(
                        stored_for_url[stored_for_url['dataset_name'] == ds]['column_name']
                    )
                    live_cols = set(
                        live_df[live_df['dataset_name'] == ds]['column_name']
                    ) if 'column_name' in live_df.columns else set()

                    new_cols = live_cols - stored_cols
                    gone_cols = stored_cols - live_cols

                    if new_cols or gone_cols:
                        col_diffs.append({
                            'dataset': ds,
                            'added': new_cols,
                            'removed': gone_cols
                        })

                # Determine overall status
                if not added_ds and not removed_ds and not col_diffs:
                    status = '✅ Match'
                    detail = 'No drift detected'
                else:
                    parts = []
                    if added_ds:
                        parts.append(f"+{len(added_ds)} new dataset(s)")
                    if removed_ds:
                        parts.append(f"-{len(removed_ds)} missing dataset(s)")
                    if col_diffs:
                        parts.append(f"{len(col_diffs)} dataset(s) with column changes")
                    status = '⚠️ Drift'
                    detail = '; '.join(parts)

                results.append({
                    'url': url,
                    'status': status,
                    'detail': detail,
                    'added_ds': added_ds,
                    'removed_ds': removed_ds,
                    'col_diffs': col_diffs
                })

            except Exception as e:
                results.append({
                    'url': url,
                    'status': '❌ Error',
                    'detail': str(e)[:100],
                    'added_ds': set(),
                    'removed_ds': set(),
                    'col_diffs': []
                })

        progress.empty()

        # Summary metrics
        st.markdown("### 📋 Results")

        match_count = sum(1 for r in results if '✅' in r['status'])
        drift_count = sum(1 for r in results if '⚠️' in r['status'])
        error_count = sum(1 for r in results if '❌' in r['status'])

        col_m, col_d, col_e = st.columns(3)
        col_m.metric("✅ Match", match_count)
        col_d.metric("⚠️ Drift", drift_count)
        col_e.metric("❌ Error", error_count)

        # Per-URL detail
        for result in results:
            page_name = os.path.basename(result['url']).split('?')[0][:50]
            is_problem = '⚠️' in result['status'] or '❌' in result['status']

            with st.expander(
                f"{result['status']} {page_name}",
                expanded=is_problem
            ):
                st.caption(f"URL: {result['url']}")
                st.markdown(f"**Status:** {result['detail']}")

                if result.get('added_ds'):
                    st.markdown("**New Datasets (on live page, not in stored CSV):**")
                    for ds in sorted(result['added_ds']):
                        st.markdown(f"- `{ds}` ➕")

                if result.get('removed_ds'):
                    st.markdown("**Missing Datasets (in stored CSV, not on live page):**")
                    for ds in sorted(result['removed_ds']):
                        st.markdown(f"- `{ds}` ➖")

                if result.get('col_diffs'):
                    st.markdown("**Column Changes:**")
                    for diff in result['col_diffs']:
                        st.markdown(f"**{diff['dataset']}:**")
                        if diff['added']:
                            st.markdown(
                                f"  - Added: `{'`, `'.join(sorted(diff['added']))}`"
                            )
                        if diff['removed']:
                            st.markdown(
                                f"  - Removed: `{'`, `'.join(sorted(diff['removed']))}`"
                            )

        # Recommendation
        st.divider()
        if drift_count > 0:
            st.warning(
                f"⚠️ **Drift detected on {drift_count}/{len(results)} page(s).** "
                f"Consider running **🔄 Scrape & Update** to refresh the schema."
            )
        elif error_count > 0:
            st.error(
                f"❌ **{error_count} page(s) had errors.** "
                f"Check if these URLs are still valid in the URL editor."
            )
        else:
            st.success("🎉 All sampled pages match the stored schema. No action needed.")

    # Back button
    st.divider()
    if st.button("← Back to App", use_container_width=True):
        st.session_state['show_health_check'] = False
        st.rerun()

#------------------------------
#------------------------------
#------------------------------
@st.cache_data
def compute_3d_layout(df_hash: str, df: pd.DataFrame,
                      selected_categories: tuple = None,
                      show_all: bool = False,
                      selected_datasets: tuple = None,
                      ds_mode: str = 'focus') -> dict:
    """computes cached 3D spring layout for the dataset relationship graph."""
    joins = get_joins(df)

    G = nx.Graph()

    # determine which datasets are eligible based on category filter
    if selected_categories:
        valid_ds = set(df[df['category'].isin(selected_categories)]['dataset_name'].unique())
    else:
        valid_ds = set(df['dataset_name'].unique())

#------------------------------
    # further narrow to specific datasets if selected
    if selected_datasets:
        focus_set = set(selected_datasets)

        if ds_mode == 'focus':
            # strict: only show selected datasets and edges between them
            valid_ds = focus_set & valid_ds
        else:
            # discovery: include direct neighbors for context
            neighbor_ds = set()
            if not joins.empty:
                for _, r in joins.iterrows():
                    src = r['dataset_name_fk']
                    tgt = r['dataset_name_pk']
                    if src in focus_set and tgt in valid_ds:
                        neighbor_ds.add(tgt)
                    if tgt in focus_set and src in valid_ds:
                        neighbor_ds.add(src)

            valid_ds = (focus_set | neighbor_ds) & valid_ds

    # add edges from joins (filtered by category)
    if not joins.empty:
        for _, r in joins.iterrows():
            src = r['dataset_name_fk']
            tgt = r['dataset_name_pk']
            key = r['column_name']

            if src in valid_ds and tgt in valid_ds:
                if G.has_edge(src, tgt):
                    G[src][tgt]['keys'].append(key)
                else:
                    G.add_edge(src, tgt, keys=[key])

    # optionally add disconnected nodes (orphans, reports)
    if show_all:
        for ds in valid_ds:
            if not G.has_node(ds):
                G.add_node(ds)

    if G.number_of_nodes() == 0:
        return {'positions': {}, 'edges': [], 'node_info': {}}

    # compute 3D spring layout (seed for determinism)
    pos = nx.spring_layout(G, dim=3, k=2.0, iterations=50, seed=42)

    # build node info
    node_info = {}
    for node in G.nodes():
        subset = df[df['dataset_name'] == node]
        cat = subset['category'].iloc[0] if not subset.empty else 'unknown'
        ds_type = 'extract'
        if 'dataset_type' in subset.columns and not subset.empty:
            ds_type = subset['dataset_type'].iloc[0]
        node_info[node] = {
            'category': cat,
            'degree': G.degree(node),
            'dataset_type': ds_type
        }

    # build edge list
    edge_data = []
    for u, v, data in G.edges(data=True):
        edge_data.append({
            'source': u,
            'target': v,
            'keys': data.get('keys', [])
        })

    return {
        'positions': {k: [float(c) for c in v] for k, v in pos.items()},
        'edges': edge_data,
        'node_info': node_info
    }


def render_3d_explorer(df: pd.DataFrame):
    """renders the interactive 3D schema visualization."""
    st.header("🌐 3D Schema Explorer")
    st.caption(
        "Interactive 3D visualization of all dataset relationships. "
        "Drag to rotate, scroll to zoom, right-drag to pan."
    )

    # controls
    with st.expander("🛠️ Settings", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            graph_height = st.slider(
                "Graph Height", 500, 1200, 800, key="3d_height"
            )
            node_scale = st.slider(
                "Node Scale", 0.5, 3.0, 1.0, step=0.25, key="3d_node_scale",
                help="Multiplier for all node sizes."
            )

        with col2:
            edge_width = st.slider(
                "Edge Width", 0.5, 5.0, 1.5, step=0.5, key="3d_edge_width",
                help="Thickness of connection lines."
            )
            edge_opacity = st.slider(
                "Edge Opacity", 0.1, 1.0, 0.3, step=0.1, key="3d_edge_opacity",
                help="Transparency of connection lines."
            )

#------------------------------
        with col3:
            show_all = st.checkbox(
                "Show Disconnected Datasets", False, key="3d_show_all",
                help="Include orphan datasets and reports that have no join relationships."
            )
            label_font_size = st.slider(
                "Label Font Size", 6, 24, 10, key="3d_label_font",
                help="Font size for dataset name labels in Focus mode."
            )

#------------------------------
    # filters
    col_filter_cat, col_filter_ds = st.columns(2)

    with col_filter_cat:
        all_cats = sorted(df['category'].unique())
        selected_cats = st.multiselect(
            "Filter by Category",
            all_cats,
            default=all_cats,
            key="3d_categories",
            help="Limit the graph to specific categories. Remove categories to reduce clutter."
        )

    # narrow dataset list based on selected categories
    if selected_cats:
        available_ds = sorted(
            df[df['category'].isin(selected_cats)]['dataset_name'].unique()
        )
    else:
        available_ds = sorted(df['dataset_name'].unique())

#------------------------------
    with col_filter_ds:
        selected_ds = st.multiselect(
            "Filter by Dataset",
            available_ds,
            default=[],
            key="3d_datasets",
            placeholder="All datasets shown (select to focus)...",
            help="Select specific datasets to focus on. Leave empty to show all within selected categories."
        )

    # dataset mode toggle (only shown when datasets are selected)
    ds_mode = 'focus'
    if selected_ds:
        ds_mode = st.radio(
            "Dataset View Mode",
            ["focus", "discovery"],
            format_func=lambda x: "🎯 Focus (selected only)" if x == "focus" else "🔭 Discovery (with neighbors)",
            horizontal=True,
            key="3d_ds_mode",
            help=(
                "**Focus:** Shows only connections between your selected datasets. "
                "**Discovery:** Also shows datasets your selection connects to."
            )
        )

    if not selected_cats:
        st.warning("Select at least one category to display.")
        return

    # compute layout (cached)
    df_hash = f"{len(df)}_{df['dataset_name'].nunique()}"
    safe_cats = tuple(sorted(selected_cats))
    safe_ds = tuple(sorted(selected_ds)) if selected_ds else None

    layout_data = compute_3d_layout(df_hash, df, safe_cats, show_all, safe_ds, ds_mode)

    positions = layout_data['positions']
    edges = layout_data['edges']
    node_info = layout_data['node_info']

    if not positions:
        st.warning("No datasets to display with current filters.")
        return

    # build traces
    categories_in_graph = list(set(info['category'] for info in node_info.values()))
    cat_colors = get_category_colors(categories_in_graph)

    traces = []

    # 1. edge lines
    edge_x, edge_y, edge_z = [], [], []

    for edge in edges:
        src = edge['source']
        tgt = edge['target']

        if src in positions and tgt in positions:
            x0, y0, z0 = positions[src]
            x1, y1, z1 = positions[tgt]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

    if edge_x:
        traces.append(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(
                width=edge_width,
                color=f'rgba(150,150,150,{edge_opacity})'
            ),
            hoverinfo='none',
            showlegend=False
        ))

    # 2. edge midpoint hover markers (show join keys on hover)
    if edges:
        mid_x, mid_y, mid_z = [], [], []
        mid_hover = []

        for edge in edges:
            src = edge['source']
            tgt = edge['target']
            keys = edge['keys']

            if src in positions and tgt in positions:
                x0, y0, z0 = positions[src]
                x1, y1, z1 = positions[tgt]
                mid_x.append((x0 + x1) / 2)
                mid_y.append((y0 + y1) / 2)
                mid_z.append((z0 + z1) / 2)
                mid_hover.append(
                    f"<b>{src}</b> ↔ <b>{tgt}</b><br>"
                    f"Keys: {', '.join(keys)}"
                )

        if mid_x:
            traces.append(go.Scatter3d(
                x=mid_x, y=mid_y, z=mid_z,
                mode='markers',
                marker=dict(
                    size=2,
                    color='rgba(88,166,255,0.5)',
                    symbol='diamond'
                ),
                text=mid_hover,
                hoverinfo='text',
                showlegend=False
            ))

    # 3. node traces (one per category for legend)
    cat_nodes = {}
    for node, info in node_info.items():
        cat = info['category']
        if cat not in cat_nodes:
            cat_nodes[cat] = []
        cat_nodes[cat].append(node)

#------------------------------
    # determine if we're in detailed focus mode (small dataset count)
    is_detailed_focus = selected_ds and ds_mode == 'focus' and len(positions) <= 15

    if is_detailed_focus:
        # DETAILED MODE: one trace per dataset for granular legend

        # build edge color map — assign a color to each join key
        all_keys_in_view = set()
        for edge in edges:
            for k in edge['keys']:
                all_keys_in_view.add(k)
        sorted_keys = sorted(all_keys_in_view)

        key_palette = [
            "#58A6FF", "#FF6B6B", "#69DB7C", "#FFD43B", "#CC5DE8",
            "#20C997", "#FF922B", "#339AF0", "#F06595", "#A9E34B"
        ]
        key_colors = {k: key_palette[i % len(key_palette)] for i, k in enumerate(sorted_keys)}

        # redraw edges with per-key coloring
        traces.clear()

        for key_name in sorted_keys:
            ex, ey, ez = [], [], []

            for edge in edges:
                if key_name in edge['keys']:
                    src = edge['source']
                    tgt = edge['target']
                    if src in positions and tgt in positions:
                        x0, y0, z0 = positions[src]
                        x1, y1, z1 = positions[tgt]
                        ex.extend([x0, x1, None])
                        ey.extend([y0, y1, None])
                        ez.extend([z0, z1, None])

            if ex:
                traces.append(go.Scatter3d(
                    x=ex, y=ey, z=ez,
                    mode='lines',
                    name=f"🔑 {key_name}",
                    line=dict(
                        width=edge_width + 1,
                        color=key_colors[key_name]
                    ),
                    hoverinfo='name',
                    showlegend=True
                ))

        # draw one trace per dataset
        for node, info in sorted(node_info.items()):
            x, y, z = positions[node]
            cat = info['category']
            degree = info['degree']
            ds_type = info.get('dataset_type', 'extract')
            color = cat_colors.get(cat, '#ccc')

            base_size = 8
            scaled = base_size + min(degree * 3, 20)

            # build column list for hover
            node_cols = df[df['dataset_name'] == node]['column_name'].tolist()
            col_preview = ', '.join(node_cols[:10])
            if len(node_cols) > 10:
                col_preview += f"... (+{len(node_cols) - 10} more)"

            type_label = "📊 Report" if ds_type == 'report' else "📦 Extract"

            hover_text = (
                f"<b>{node}</b><br>"
                f"Category: {cat}<br>"
                f"Type: {type_label}<br>"
                f"Connections: {degree}<br>"
                f"Columns ({len(node_cols)}): {col_preview}"
            )

#------------------------------
            traces.append(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers+text',
                name=f"📦 {node}",
                text=[node],
                textposition='top center',
                textfont=dict(size=label_font_size, color='white'),
                marker=dict(
                    size=scaled * node_scale,
                    color=color,
                    line=dict(width=2, color='white'),
                    opacity=0.95,
                    symbol='diamond'
                ),
                hoverinfo='text',
                hovertext=[hover_text],
                showlegend=True
            ))

    else:
        # STANDARD MODE: one trace per category (unchanged)
        for cat in sorted(cat_nodes.keys()):
            nodes = cat_nodes[cat]
            color = cat_colors.get(cat, '#ccc')

            nx_list, ny_list, nz_list = [], [], []
            sizes = []
            hover_texts = []

            for node in nodes:
                x, y, z = positions[node]
                nx_list.append(x)
                ny_list.append(y)
                nz_list.append(z)

                info = node_info[node]
                degree = info['degree']
                ds_type = info.get('dataset_type', 'extract')

                base_size = 4
                scaled = base_size + min(degree * 2, 16)
                sizes.append(scaled * node_scale)

                type_label = "📊 Report" if ds_type == 'report' else "📦 Extract"
                hover_texts.append(
                    f"<b>{node}</b><br>"
                    f"Category: {cat}<br>"
                    f"Type: {type_label}<br>"
                    f"Connections: {degree}"
                )

            traces.append(go.Scatter3d(
                x=nx_list, y=ny_list, z=nz_list,
                mode='markers',
                name=cat,
                marker=dict(
                    size=sizes,
                    color=color,
                    line=dict(width=0.5, color='rgba(255,255,255,0.3)'),
                    opacity=0.9
                ),
                text=hover_texts,
                hoverinfo='text',
                showlegend=True
            ))

    # build figure
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            showlegend=True,
            legend=dict(
                title=dict(text="Categories", font=dict(color='#8B949E', size=12)),
                font=dict(color='#C9D1D9', size=11),
                bgcolor='rgba(30, 35, 43, 0.85)',
                bordercolor='#30363D',
                borderwidth=1
            ),
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='#0E1117'
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='#0E1117',
            height=graph_height
        )
    )

    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'brightspace_3d_schema',
            'height': 1200,
            'width': 1600,
            'scale': 2
        }
    }

    st.plotly_chart(fig, use_container_width=True, config=config)

    # summary stats
    st.divider()

    total_nodes = len(positions)
    total_edges = len(edges)
    connected_nodes = sum(1 for info in node_info.values() if info['degree'] > 0)
    isolated_nodes = total_nodes - connected_nodes

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Datasets Shown", total_nodes)
    col_s2.metric("Connections", total_edges)
    col_s3.metric("Connected", connected_nodes)
    col_s4.metric("Isolated", isolated_nodes)

    # dataset details table
    with st.expander("📋 Dataset Details", expanded=False):
        detail_data = []
        for node, info in sorted(node_info.items()):
            detail_data.append({
                "Dataset": node,
                "Category": info['category'],
                "Type": info['dataset_type'],
                "Connections": info['degree']
            })

        detail_df = pd.DataFrame(detail_data).sort_values(
            "Connections", ascending=False
        )

        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Connections": st.column_config.ProgressColumn(
                    "Connections",
                    format="%d",
                    min_value=0,
                    max_value=int(detail_df['Connections'].max()) if not detail_df.empty else 1
                )
            }
        )


#------------------------------
def render_udf_flattener(df: pd.DataFrame):
    """renders the EAV pivot tool for user defined fields."""
#------------------------------
    st.header("🔧 UDF Flattener")

    st.markdown(
        "Transform 'vertical' custom data lists into standard 'horizontal' tables. "
        "This is commonly needed for **User Defined Fields (UDFs)** — custom attributes "
        "that your institution adds to users, courses, or org units."
    )

    st.info(
        "**What is this for?**\n\n"
        "Many Brightspace instances store custom data (e.g., Pronouns, Department, Start Date) "
        "in an [Entity-Attribute-Value (EAV)](https://en.wikipedia.org/wiki/Entity%E2%80%93attribute%E2%80%93value_model) pattern, "
        "where each custom field is a **row** rather than a **column**. "
        "This makes the data hard to query directly.\n\n"
        "This tool generates the SQL `PIVOT` logic to convert those rows into proper columns, "
        "so a query like `SELECT UserId, Department, Pronouns FROM ...` becomes possible.\n\n"
        "**You'll need:** Your specific Field IDs or Field Names from your database — "
        "see the guide below for how to find them.",
        icon="💡"
    )

    with st.expander("ℹ️ How to use & Where to find Field IDs", expanded=True):
        c_concept, c_action = st.columns([1, 1])

        with c_concept:
            st.markdown("**1. The Concept (Pivoting)**")
            st.code(
                "# BEFORE (Vertical EAV)\n"
                "UserId | FieldId | Value\n"
                "101    | 4       | \"Marketing\"\n"
                "101    | 9       | \"He/Him\"\n"
                "# AFTER (Flattened)\n"
                "UserId | Dept_Marketing | Pronouns_HeHim\n"
                "101    | \"Marketing\"    | \"He/Him\"",
                language="text"
            )

        with c_action:
            st.markdown("**2. Finding your Field IDs**")
            st.caption(
                "Since this app cannot see your data, you must look up your specific "
                "Field IDs in your database."
            )
            st.markdown("Run this SQL in your environment:")
            st.code(
                "SELECT FieldId, Name\n"
                "FROM UserDefinedFields\n"
                "-- Look for IDs like 4, 9, 12...",
                language="sql"
            )

    st.divider()

    # 1. Table selection
    st.subheader("1. Configuration")

    col_main, col_eav, col_dialect = st.columns(3)
    all_ds = sorted(df['dataset_name'].unique())

    def_main = "Users" if "Users" in all_ds else (all_ds[0] if all_ds else "")
    def_eav = "UserUserDefinedFields" if "UserUserDefinedFields" in all_ds else (
        all_ds[1] if len(all_ds) > 1 else (all_ds[0] if all_ds else "")
    )

    with col_main:
        main_table = st.selectbox(
            "Main Entity Table (The Rows)",
            all_ds,
            index=all_ds.index(def_main) if def_main in all_ds else 0
        )
    with col_eav:
        eav_table = st.selectbox(
            "Attribute Table (The Data)",
            all_ds,
            index=all_ds.index(def_eav) if def_eav in all_ds else 0
        )
    with col_dialect:
        dialect = st.selectbox(
            "SQL Dialect",
            ["T-SQL", "Snowflake", "PostgreSQL"],
            help="Adjusts identifier quoting and syntax."
        )

    # Warn if same table selected
    if main_table == eav_table:
        st.warning("⚠️ Main table and Attribute table are the same. This is unusual for EAV patterns.")

    # 2. Column mapping
    st.subheader("2. Column Mapping")

    main_cols = df[df['dataset_name'] == main_table]['column_name'].tolist()
    eav_cols = df[df['dataset_name'] == eav_table]['column_name'].tolist()
    common = list(set(main_cols) & set(eav_cols)) or ["UserId"]

    c1, c2, c3 = st.columns(3)

    with c1:
        default_join_key = 'UserId' if 'UserId' in common else common[0]
        join_key = st.selectbox(
            "Join Key (PK)", common,
            index=common.index(default_join_key),
            help="The ID connecting both tables."
        )
    with c2:
        if 'FieldId' in eav_cols:
            piv_idx = eav_cols.index('FieldId')
        elif 'Name' in eav_cols:
            piv_idx = eav_cols.index('Name')
        else:
            piv_idx = 0
        pivot_col = st.selectbox(
            "Attribute Name Column", eav_cols,
            index=piv_idx,
            help="The column containing the field identifiers (e.g. 'FieldId' or 'Name')."
        )
    with c3:
        val_idx = eav_cols.index('Value') if 'Value' in eav_cols else 0
        val_col = st.selectbox(
            "Value Column", eav_cols,
            index=val_idx,
            help="The column containing the actual data."
        )

    # 3. Field definition
    st.subheader("3. Define Fields")

    col_input, col_fields = st.columns([1, 2])

    with col_input:
        input_type = st.radio(
            "Key Type", ["IDs (Integers)", "Names (Strings)"],
            help="Are we pivoting on '1, 2, 3' or 'Gender, Dept'?"
        )

    with col_fields:
        if input_type == "IDs (Integers)":
            placeholder = "e.g. 1, 4, 9, 12"
            help_text = "Enter the Field IDs you found using the SQL tip above."
        else:
            placeholder = "e.g. Pronouns, Department, Start Date"
            help_text = (
                "Enter the exact Names of the fields you want to turn into columns. "
                "These must match the data exactly."
            )

        raw_fields = st.text_area(
            "Fields to Flatten (comma separated)",
            placeholder=placeholder,
            help=help_text
        )

    # 4. Generator
    if st.button("Generate Pivot SQL", type="primary"):
        if not raw_fields:
            st.error("Please enter at least one field to flatten.")
        else:
            fields = [f.strip() for f in raw_fields.split(',') if f.strip()]

            # Validate integer input if IDs selected
            if input_type == "IDs (Integers)":
                invalid_fields = [f for f in fields if not f.isdigit()]
                if invalid_fields:
                    st.error(f"Invalid Field IDs (expected integers): {', '.join(invalid_fields)}")
                    return

            # Configure dialect-specific quoting
            if dialect == "T-SQL":
                q_start, q_end = "[", "]"
            else:
                q_start, q_end = '"', '"'

            def quote(name: str) -> str:
                return f"{q_start}{name}{q_end}"

            # Build the SQL
            lines = [
                f"-- UDF Flattener: Pivot {eav_table} into {main_table}",
                f"-- Dialect: {dialect}",
                f"-- Fields: {', '.join(fields)}",
                "",
                "SELECT"
            ]
            lines.append(f"    m.{quote(join_key)},")

            for i, f in enumerate(fields):
                comma = "," if i < len(fields) - 1 else ""

                if input_type == "IDs (Integers)":
                    match_logic = f"{quote(pivot_col)} = {f}"
                    alias = f"Field_{f}"
                else:
                    safe_f = f.replace("'", "''")
                    match_logic = f"{quote(pivot_col)} = '{safe_f}'"
                    alias = f.replace(' ', '_').replace("'", "")

                lines.append(
                    f"    MAX(CASE WHEN e.{match_logic} THEN e.{quote(val_col)} END) AS {quote(alias)}{comma}"
                )

            lines.append(f"FROM {quote(main_table)} m")
            lines.append(f"LEFT JOIN {quote(eav_table)} e ON m.{quote(join_key)} = e.{quote(join_key)}")
            lines.append(f"GROUP BY m.{quote(join_key)}")

            sql_output = "\n".join(lines)

            st.code(sql_output, language="sql")

            col_dl, col_info = st.columns([1, 2])
            with col_dl:
                st.download_button(
                    label="📥 Download SQL",
                    data=sql_output,
                    file_name=f"udf_pivot_{dialect.lower()}.sql",
                    mime="application/sql"
                )
            with col_info:
                st.caption(f"Generated {dialect} syntax. Copy this SQL to query your database.")

# =============================================================================
# 11. main orchestrator
# =============================================================================

def render_dashboard(df: pd.DataFrame):
    """Main dashboard view with metrics, search, hubs, orphans, and Path Finder."""
    st.header("📊 Dashboard")

    is_advanced = st.session_state.get('experience_mode', 'basic') == 'advanced'

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    total_datasets = df['dataset_name'].nunique() if 'dataset_name' in df.columns else 0
    total_columns = len(df)
    total_categories = df['category'].nunique() if 'category' in df.columns else 0

    joins = get_joins(df)
    total_relationships = len(joins) if not joins.empty else 0

    col1.metric("Total Datasets", total_datasets)
    col2.metric("Total Columns", f"{total_columns:,}")
    col3.metric("Categories", total_categories)
    col4.metric(
        "Unique Joins",
        total_relationships,
        help="Total count of unique directional links (A → B) detected across the entire schema."
    )

    st.divider()

    # ---------------------------------------------------------
    # 🔍 Intelligent Search
    # ---------------------------------------------------------
    st.subheader("🔍 Intelligent Search")

    all_datasets = sorted(df['dataset_name'].unique()) if 'dataset_name' in df.columns else []
    all_columns = sorted(df['column_name'].unique()) if 'column_name' in df.columns else []

    search_index = [f"📦 {ds}" for ds in all_datasets] + [f"🔑 {col}" for col in all_columns]

    col_search, col_stats = st.columns([3, 1])

    with col_search:
        # Key added to prevent StreamlitDuplicateElementKey error
        search_selection = st.selectbox(
            "Search for a Dataset or Column",
            options=search_index,
            index=None,
            placeholder="Type to search (e.g. 'Users', 'OrgUnitId')...",
            label_visibility="collapsed",
            key="dashboard_main_search_box"
        )

    if search_selection:
        st.divider()

        search_type = "dataset" if "📦" in search_selection else "column"
        parts = search_selection.split(" ", 1)
        term = parts[1] if len(parts) > 1 else search_selection

        if search_type == "dataset":
            st.markdown(f"### Results for Dataset: **{term}**")

            ds_data = df[df['dataset_name'] == term]
            if not ds_data.empty:
                meta = ds_data.iloc[0]

                with st.container():
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        st.caption(f"Category: **{meta.get('category', 'Unknown')}**")
                        if meta.get('url'):
                            st.markdown(f"📄 [**Official Documentation**]({meta['url']})")

                    with c2:
                        show_relationship_summary(df, term)

                with st.expander("📋 View Schema", expanded=True):
                    display_cols = ['column_name', 'data_type', 'description', 'key']
                    available_cols = [c for c in display_cols if c in ds_data.columns]
                    st.dataframe(ds_data[available_cols], hide_index=True, use_container_width=True)

        else:
            st.markdown(f"### Datasets containing column: `{term}`")
            hits = df[df['column_name'] == term]['dataset_name'].unique() if 'column_name' in df.columns else []

            if len(hits) > 0:
                st.info(f"Found **{len(hits)}** datasets containing `{term}`")

                for ds_name in sorted(hits):
                    ds_rows = df[df['dataset_name'] == ds_name]
                    if ds_rows.empty: continue
                    ds_meta = ds_rows.iloc[0]
                    category = ds_meta.get('category', 'Unknown')

                    with st.expander(f"📦 {ds_name}  ({category})"):
                        c_info, c_rel = st.columns([2, 1])
                        with c_info:
                            desc = ds_meta.get('dataset_description', '')
                            if desc: st.caption(f"💡 {desc}")
                            if ds_meta.get('url'): st.markdown(f"[View Documentation]({ds_meta['url']})")

                            col_row = df[(df['dataset_name'] == ds_name) & (df['column_name'] == term)]
                            st.caption("Column Details:")
                            st.dataframe(col_row[['data_type', 'description', 'key']], hide_index=True, use_container_width=True)

                        with c_rel:
                            show_relationship_summary(df, ds_name)
            else:
                st.warning("No matches found.")

    else:
        # ---------------------------------------------------------
        # Default Dashboard View (Hubs + Orphans)
        # ---------------------------------------------------------
        st.divider()
        col_hubs, col_orphans = st.columns(2)

        with col_hubs:
            st.subheader("🌟 Most Connected Datasets ('Hubs')")
            with st.expander("ℹ️ Why are these numbers so high?", expanded=False):
                st.caption("""
**High Outgoing (Refers To):** This dataset contains \"Super Keys\" like `OrgUnitId` or `UserId`
which allows it to join to dozens of other structural tables.

**High Incoming (Referenced By):** This is a central \"Dimension\" table (like `Users`)
that almost every other table links to.
""")

            hubs = get_hub_datasets(df, top_n=10)
            if not hubs.empty and hubs['total_connections'].sum() > 0:
                st.dataframe(
                    hubs[['dataset_name', 'category', 'outgoing_fks', 'incoming_fks']],
                    column_config={
                        "dataset_name": "Dataset",
                        "category": "Category",
                        "outgoing_fks": st.column_config.ProgressColumn("Refers To (Outgoing)", format="%d"),
                        "incoming_fks": st.column_config.ProgressColumn("Referenced By (Incoming)", format="%d")
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No relationship data available yet.")

        with col_orphans:
            st.subheader("🏝️ Orphan Datasets")
            orphans = get_orphan_datasets(df)
            if orphans:
                st.warning(f"{len(orphans)} datasets have no detected relationships")
                st.caption("These tables usually lack standard keys like `OrgUnitId` or `UserId`.")
                orphan_details = (
                    df[df['dataset_name'].isin(orphans)]
                    [['dataset_name', 'category', 'dataset_description']]
                    .drop_duplicates('dataset_name')
                    .sort_values('dataset_name')
                    .rename(columns={'dataset_description': 'description'})
                )
                st.dataframe(
                    orphan_details,
                    column_config={
                        "dataset_name": "Dataset",
                        "category": "Category",
                        "description": st.column_config.TextColumn("Description", width="medium")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.success("All datasets have at least one connection")

        # ---------------------------------------------------------
        # Advanced Features (Category Chart & Path Finder)
        # ---------------------------------------------------------
        if is_advanced:
            st.divider()
            st.subheader("📁 Category Breakdown")

            cat_stats = df.groupby('category').agg({
                'dataset_name': 'nunique',
                'column_name': 'count'
            }).reset_index()
            cat_stats.columns = ['Category', 'Datasets', 'Columns']
            cat_stats = cat_stats.sort_values('Datasets', ascending=False)

            col_chart, col_table = st.columns([2, 1])
            with col_chart:
                fig = px.bar(
                    cat_stats, x='Category', y='Datasets',
                    color='Columns',
                    title="Datasets per Category",
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_table:
                st.dataframe(cat_stats, use_container_width=True, hide_index=True)

            # ---------------------------------------------------------
            # 🛤️ Path Finder (Restored Functionality)
            # ---------------------------------------------------------
            st.divider()
            st.subheader("🛤️ Path Finder")
            st.caption("Find all valid join paths between two datasets.")

            st.info(
                "A **path** is a chain of JOINs connecting two datasets that don't directly share a key. "
                "Each **hop** is one JOIN through an intermediate table. Shorter paths generally produce "
                "cleaner queries with fewer potential data issues.\n\n"
                "**Example:** `Quiz Attempts` → `Users` → `User Enrollments` is a 2-hop path, "
                "joining through the `Users` table via `UserId`.",
                icon="💡"
            )

            # Reusing all_datasets computed earlier
            col_from, col_to = st.columns(2)
            with col_from:
                source_ds = st.selectbox("From Dataset", all_datasets, index=None, placeholder="Select source...", key="path_source")
            with col_to:
                target_ds = st.selectbox("To Dataset", all_datasets, index=None, placeholder="Select target...", key="path_target")

            # Row 2: Configuration & Action
            col_hops, col_limit, col_find = st.columns([1, 1, 2])
            
            with col_hops:
                max_hops = st.number_input(
                    "Max Hops",
                    min_value=1,
                    max_value=6,
                    value=4,
                    help="Max number of joins allowed (depth)."
                )
            with col_limit:
                top_k = st.number_input(
                    "Results to Show",
                    min_value=1,
                    max_value=50,
                    value=5,
                    help="Number of paths to display."
                )
                use_core_keys_only = st.checkbox(
                    "Use core keys only (UserId / OrgUnitId)",
                    value=False,
                    help=(
                        "When enabled, restricts paths to joins on UserId and OrgUnitId. "
                        "This biases results toward dimension-style paths through Users / Org Units."
                    )
                )
            with col_find:
                st.write("")  # Spacer for alignment
                st.write("")
                find_path = st.button(
                    "Find Paths",
                    type="primary",
                    use_container_width=True
                )
            
            # Logic: Calculate Paths
            if find_path and source_ds and target_ds:
                if source_ds == target_ds:
                    st.warning("Please select two different datasets.")
                else:
                    # Decide which join keys are allowed for this search
                    allowed_keys = ['UserId', 'OrgUnitId'] if use_core_keys_only else None

                    with st.spinner("Calculating network paths..."):
                        # Assumes find_all_paths exists in your file
                        paths = find_all_paths(
                            df,
                            source_ds,
                            target_ds,
                            cutoff=max_hops,
                            limit=top_k,
                            allowed_keys=allowed_keys
                        )
                    
                    # Store in session state to persist across reruns
                    st.session_state['path_finder_results'] = {
                        'paths': paths,
                        'source_ds': source_ds,
                        'target_ds': target_ds,
                        'max_hops': max_hops,
                        'use_core_keys_only': use_core_keys_only
                    }
            
            # Logic: Display Results
            if 'path_finder_results' in st.session_state:
                results = st.session_state['path_finder_results']
                paths = results['paths']
                
                if paths:
                    count = len(paths)
                    st.success(f"Found top {count} shortest path(s) within {results['max_hops']} hops.")
                    
                    for i, path in enumerate(paths):
                        hops = len(path) - 1
                        label = f"Option {i+1}: {hops} Join(s)"
                        if i == 0:
                            label += " (Shortest)"
                        
                        with st.expander(label, expanded=(i == 0)):
                            # Breadcrumb visual
                            st.markdown(" → ".join([f"**{p}**" for p in path]))
                            
                            # Detailed breakdown
                            # Assumes get_path_details exists
                            path_details = get_path_details(df, path)
                            st.markdown("---")
                            for step in path_details:
                                st.markdown(
                                    f"- `{step['from']}` joins to `{step['to']}` on column `{step['column']}`"
                                )

                            # Generate SQL for this specific path
                            st.markdown("#### Generate Query for This Path")
                            col_sql_dialect, col_sql_btn = st.columns([2, 1])
                            with col_sql_dialect:
                                path_sql_dialect = st.selectbox(
                                    "Dialect",
                                    ["T-SQL", "Snowflake", "PostgreSQL"],
                                    key=f"path_sql_dialect_{i}"
                                )
                            with col_sql_btn:
                                st.write("")  # spacer
                                gen_sql_for_path = st.button(
                                    "Generate SQL",
                                    key=f"gen_sql_for_path_{i}",
                                    use_container_width=True
                                )

                            if gen_sql_for_path:
                                # Assumes generate_sql_for_path exists
                                sql_from_path = generate_sql_for_path(
                                    path, df, dialect=path_sql_dialect
                                )
                                st.code(sql_from_path, language="sql")
                            
                            # Generate Pandas Code
                            st.markdown("#### Generate Pandas Code for This Path")
                            gen_pandas_for_path = st.button(
                                "Generate Pandas",
                                key=f"gen_pandas_for_path_{i}",
                                use_container_width=True
                            )

                            if gen_pandas_for_path:
                                # Assumes generate_pandas_for_path exists
                                pandas_from_path = generate_pandas_for_path(path, df)
                                st.code(pandas_from_path, language="python")
                else:
                    st.error(
                        f"No path found within {results['max_hops']} hops. "
                        f"{'Try disabling the core key filter or increasing Max Hops.' if results['use_core_keys_only'] else 'These datasets may be unrelated or require a deeper search.'}"
                    )
                
                # Add reset button
                if st.button("Reset Search"):
                    if 'path_finder_results' in st.session_state:
                        del st.session_state['path_finder_results']
                    st.rerun()

#------------------------------
def main():
    """main entry point that orchestrates the application."""

    # ──────────────────────────────────────────────────────────────
    # Development Banner (shows on every page until dismissed)
    # ──────────────────────────────────────────────────────────────
    if 'dev_banner_dismissed' not in st.session_state:
        st.session_state.dev_banner_dismissed = False

    if not st.session_state.dev_banner_dismissed:
        col_msg, col_btn = st.columns([9, 1])
        with col_msg:
            st.info(
                "🚧 **This application is under active development.** "
                "We're making major changes soon. Some features may be unstable or change without notice (just check back in a little while if you see an issue).",
                icon="⚠️"
            )
        with col_btn:
            if st.button("✕", help="Dismiss this message", key="dismiss_dev_banner"):
                st.session_state.dev_banner_dismissed = True
                st.rerun()

    # Show scrape success message if it exists
    if st.session_state.get('scrape_msg'):
        st.success(st.session_state['scrape_msg'])
        st.session_state['scrape_msg'] = None

    # Single source of truth for loading data
    df = load_data()

    # Prefer fresh data from session_state after a successful scrape
    # (this makes hot-reloads and immediate UI updates much smoother)
    if 'current_df' in st.session_state:
        df = st.session_state.current_df

    # Render sidebar and get navigation state
    view, selected_datasets = render_sidebar(df)

    # URL Editor takes priority
    if st.session_state.get('show_url_editor'):
        render_url_editor()
        return

    # Health Check takes priority
    if st.session_state.get('show_health_check'):
        render_health_check(df)
        return

    # Improved empty state (much friendlier first-time experience)
    if df.empty:
        st.title("🔗 Brightspace Dataset Explorer")
        st.info("👋 Welcome! No schema data has been loaded yet.")

        st.markdown("""
        ### First-time setup
        To populate this application with data, please use the **sidebar controls** on the left:

        1. Locate the **⚙️ Data Management** section in the sidebar.
        2. Click the red **🔄 Scrape & Update** button.

        This will pull the latest dataset definitions from D2L documentation.
        """)

        return

    # Route to the selected view
    # (The duplicate block below this one has been removed)
    if view == "📊 Dashboard":
        render_dashboard(df)
    elif view == "🗺️ Relationship Map":
        render_relationship_map(df, selected_datasets)
    elif view == "📋 Schema Browser":
        render_schema_browser(df)
    elif view == "📚 KPI Recipes":
        render_kpi_recipes(df)
    elif view == "⚡ SQL Builder":
        render_sql_builder(df, selected_datasets)
    elif view == "🔀 SQL Translator":
        render_sql_translator()
    elif view == "🔧 UDF Flattener":
        render_udf_flattener(df)
    elif view == "✨ Schema Diff":
        render_schema_diff(df)
    elif view == "🌐 3D Explorer":
        render_3d_explorer(df)
    elif view == "🤖 AI Assistant":
        render_ai_assistant(df, selected_datasets)
    elif view == "📋 Dataset ID Reference":
        render_dataset_id_reference(df)

if __name__ == "__main__":
    main()
