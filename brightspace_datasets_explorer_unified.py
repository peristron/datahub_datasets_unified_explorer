#   brightspace_datasets_explorer_unified01312026.py, LKG 01312026
# =============================================================================
# unified brightspace dataset explorer
# combines the best of all three code-bases with simple/advanced modes
# run: streamlit run unified_dataset_explorer.py
# =============================================================================

import streamlit as st
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

# configure structured logging
logging.basicConfig(
    filename='app.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# suppress insecure request warnings for d2l scrapers
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

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
https://community.d2l.com/brightspace/kb/articles/4745-competency-reporting-with-ploe-advanced-data-set
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

# common d2l enumeration mappings (the "decoder ring")
ENUM_DEFINITIONS = {
    "GradeObjectTypeId": {
        1: "Numeric", 2: "Pass/Fail", 3: "Selectbox", 4: "Text",
        6: "Calculated", 7: "Formula"
    },
    "OrgUnitTypeId": {
        1: "Organization", 2: "Course Offering", 3: "Course Template",
        4: "Department", 5: "Semester", 6: "Group", 7: "Section"
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
        'scrape_msg': None
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


def clear_all_selections():
    """clears all dataset selections."""
    st.session_state['selected_datasets'] = []
    # clear any selection-related keys
    for key in list(st.session_state.keys()):
        if key.startswith("sel_") or key == "global_search" or key == "dataset_multiselect":
            if isinstance(st.session_state.get(key), list):
                st.session_state[key] = []

# =============================================================================
# 5. data layer (scraper & storage)
# =============================================================================

def clean_description(text: str) -> str:
    """
    Logic to convert raw documentation text into a concise summary.
    Removes boilerplate like 'The User data set describes...'
    """
    if not text:
        return ""

    # 1. Remove common D2L boilerplate
    text = re.sub(r'^The .*? data set (describes|contains|provides) ', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^This (data set|table) (describes|contains|provides) ', '', text, flags=re.IGNORECASE)

    # 2. Capitalize first letter if needed
    text = text[0].upper() + text[1:] if text else text

    # 3. Limit to the first 2 sentences for brevity
    sentences = re.split(r'(?<=[.!?]) +', text)
    summary = ' '.join(sentences[:2])

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

        soup = BeautifulSoup(response.content, 'html.parser')
        data = []
        current_dataset = category_name
        current_desc = ""

        elements = soup.find_all(['h2', 'h3', 'h4', 'table'])

        #----------------------------------------------------------------------------------------
        for element in elements:
            if element.name in ['h2', 'h3', 'h4']:
                text = element.text.strip()
                clean_text_lower = text.lower()

                if any(x == clean_text_lower for x in IGNORE_HEADERS):
                    continue
                if "returned fields" in clean_text_lower or "available filters" in clean_text_lower:
                    continue
                if clean_text_lower.startswith("about "):  # NEW: Ignore subheaders like "About Time Tracking" to prevent overwrite
                    continue

                if len(text) > 3:
                    current_dataset = text.title()

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

                header_cells = element.find_all('th')
                if not header_cells and rows:
                    header_cells = rows[0].find_all('td')
                    data_rows = rows[1:]
                else:
                    data_rows = rows

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

                    header_map = {
                        'field': 'column_name',
                        'field_name': 'column_name',
                        'name': 'column_name',
                        'type': 'data_type',
                        'data_type': 'data_type',
                        'description': 'description',
                        'can_be_null?': 'is_nullable'
                    }

                    clean_entry = {header_map.get(k, k): v for k, v in entry.items()}

                    if 'column_name' in clean_entry and clean_entry['column_name']:
                        col = clean_entry['column_name']

                        # --- FINAL NORMALIZATION LOGIC ---

                        # 1. Fix Capitalization of "ID" at end of word (OrgUnitID -> OrgUnitId)
                        col = re.sub(r'I[dD]\b', 'Id', col)

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
    def extract_category(url):
        filename = os.path.basename(url).split('?')[0]

        # NEW LOGIC: Force 'Advanced Data Sets' category for the new URLs
        if "advanced" in filename.lower():
            return "Advanced Data Sets"

        clean_name = re.sub(r'^\d+\s*', '', filename)
        return clean_name.replace('-data-sets', '').replace('-', ' ').lower()

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
    df['dataset_name'] = df['dataset_name'].astype(str).str.title()
    df['category'] = df['category'].astype(str).str.title()

    # ensure expected columns exist
    expected_cols = ['category', 'dataset_name', 'dataset_description', 'column_name',
                     'data_type', 'description', 'key', 'url']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ''

    # logic flags for joins based on key column
    df['is_primary_key'] = df['key'].astype(str).str.contains(r'\bpk\b', case=False, regex=True)
    df['is_foreign_key'] = df['key'].astype(str).str.contains(r'\bfk\b', case=False, regex=True)

    # persist to csv
    df.to_csv('dataset_metadata.csv', index=False)
    logger.info(f"Scraping complete. Saved {len(df)} rows.")
    return df


@st.cache_data
def load_data() -> pd.DataFrame:
    """loads the csv from disk if it exists and is valid."""
    if os.path.exists('dataset_metadata.csv') and os.path.getsize('dataset_metadata.csv') > 10:
        return pd.read_csv('dataset_metadata.csv').fillna('')
    return pd.DataFrame()


@st.cache_data
def get_possible_joins(df_hash: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates join conditions.
    Improves on strict PK/FK matching by inferring FKs if a column name matches a known PK.
    Also handles D2L specific synonyms (e.g. CourseOfferingId -> OrgUnitId).
    """
    if df.empty:
        return pd.DataFrame()

    # ensure required columns exist
    if 'is_primary_key' not in df.columns:
        return pd.DataFrame()

    # 1. Identify definitive Primary Keys
    pks = df[df['is_primary_key'] == True]
    if pks.empty:
        return pd.DataFrame()

    # 2. Identify potential foreign keys (Exact Match)
    pk_names = pks['column_name'].unique()

    potential_fks = df[
        (df['column_name'].isin(pk_names)) &
        (df['is_primary_key'] == False)
    ]

    # 3. Perform Exact Match Merge
    exact_joins = pd.DataFrame()
    if not potential_fks.empty:
        exact_joins = pd.merge(potential_fks, pks, on='column_name', suffixes=('_fk', '_pk'))

    # 4. Perform Synonym/Alias Match (The "Smart" Logic)
#------------
    alias_map = {
        'CourseOfferingId': 'OrgUnitId',
        'SectionId': 'OrgUnitId',
        'DepartmentId': 'OrgUnitId',
        'SemesterId': 'OrgUnitId',
        'ParentOrgUnitId': 'OrgUnitId',
        'AuditorId': 'UserId',
        'EvaluatorId': 'UserId',
        'AssignedToUserId': 'UserId',
        'CreatedBy': 'UserId',
        'ActionUserId': 'UserId',
        'TargetUserId': 'UserId'
    }

    # Get columns that match our alias list
    aliased_fks = df[
        (df['column_name'].isin(alias_map.keys())) &
        (df['is_primary_key'] == False)
    ].copy()

    alias_joins = pd.DataFrame()
    if not aliased_fks.empty:
        # Create a temporary column to bridge the join
        aliased_fks['target_pk_name'] = aliased_fks['column_name'].map(alias_map)

        # Use same suffixes as exact_joins so we get dataset_name_fk/pk, category_fk/pk, etc.
        alias_joins = pd.merge(
            aliased_fks,
            pks,
            left_on='target_pk_name',
            right_on='column_name',
            suffixes=('_fk', '_pk')
        )

        # Use the FK-side column name as the join label (e.g., CourseOfferingId)
        alias_joins['column_name'] = alias_joins['column_name_fk']

        # Clean up temp column
        alias_joins = alias_joins.drop(columns=['target_pk_name'])

    # 5. Combine and Clean
    all_joins = pd.concat([exact_joins, alias_joins], ignore_index=True)

    if all_joins.empty:
        return pd.DataFrame()

    # Exclude self-joins (joining a table to itself)
    if 'dataset_name_fk' in all_joins.columns and 'dataset_name_pk' in all_joins.columns:
        joins = all_joins[all_joins['dataset_name_fk'] != all_joins['dataset_name_pk']]

        # Ensure distinct relationships
        joins = joins.drop_duplicates(subset=['dataset_name_fk', 'column_name', 'dataset_name_pk'])
        return joins

    return pd.DataFrame()


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

        connectivity.append({
            'dataset_name': ds,
            'outgoing_fks': outgoing,
            'incoming_fks': incoming,
            'total_connections': outgoing + incoming,
            'category': df[df['dataset_name'] == ds]['category'].iloc[0]
            if len(df[df['dataset_name'] == ds]) > 0 else ''
        })

    return pd.DataFrame(connectivity).sort_values('total_connections', ascending=False)


def get_hub_datasets(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """returns the most connected datasets (hubs)."""
    connectivity = get_dataset_connectivity(df)
    return connectivity.head(top_n)


def get_orphan_datasets(df: pd.DataFrame) -> List[str]:
    """returns datasets with zero connections."""
    connectivity = get_dataset_connectivity(df)
    orphans = connectivity[connectivity['total_connections'] == 0]['dataset_name'].tolist()
    return orphans


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


def show_relationship_summary(df: pd.DataFrame, dataset_name: str):
    """shows quick stats about a dataset's connectivity."""
    joins = get_joins(df)

    if joins.empty:
        outgoing = 0
        incoming = 0
    else:
        # Outgoing: This dataset HAS the Foreign Key (it points TO others)
        outgoing = len(joins[joins['dataset_name_fk'] == dataset_name])
        # Incoming: This dataset HAS the Primary Key (others point TO it)
        incoming = len(joins[joins['dataset_name_pk'] == dataset_name])

    # vertical stacking for readability
    st.metric("References (Outgoing)", outgoing,
              help=f"This dataset contains {outgoing} Foreign Keys pointing to other tables.")
    st.metric("Referenced By (Incoming)", incoming,
              help=f"{incoming} other tables have Foreign Keys pointing to this dataset.")
    st.metric("Total Connections", outgoing + incoming)

# =============================================================================
# 7. visualization engine
# =============================================================================

def get_category_colors(categories: List[str]) -> Dict[str, str]:
    """generates consistent colors for categories using hsl hash."""
    return {cat: f"hsl({(hash(cat)*137.5) % 360}, 70%, 50%)" for cat in categories}


def create_spring_graph(
    df: pd.DataFrame,
    selected_datasets: List[str],
    mode: str = 'focused',
    graph_font_size: int = 14,
    node_separation: float = 0.9,
    graph_height: int = 600,
    show_edge_labels: bool = True
) -> go.Figure:
    """
    creates a spring-layout graph visualization.
    mode: 'focused' shows only connections between selected datasets
    mode: 'discovery' shows all outgoing connections from selected datasets
    """
    if not selected_datasets:
        fig = go.Figure()
        fig.add_annotation(
            text="Select datasets to visualize", showarrow=False,
            font=dict(size=16, color='gray')
        )
        fig.update_layout(
            height=graph_height,
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    join_data = get_joins_for_selection(df, selected_datasets)
    G = nx.DiGraph()

    if mode == 'focused':
        # add only selected datasets
        for ds in selected_datasets:
            G.add_node(ds, type='focus')

        # add only edges between selected datasets
        if not join_data.empty:
            for _, row in join_data.iterrows():
                s = row['Source Dataset']
                t = row['Target Dataset']
                if s in selected_datasets and t in selected_datasets:
                    G.add_edge(s, t, label=row['column_name'])
    else:
        # discovery mode - add selected datasets first
        for ds in selected_datasets:
            G.add_node(ds, type='focus')

        # add all outgoing connections
        if not join_data.empty:
            for _, row in join_data.iterrows():
                s = row['Source Dataset']
                t = row['Target Dataset']
                if s in selected_datasets:
                    if not G.has_node(t):
                        G.add_node(t, type='neighbor')
                    G.add_edge(s, t, label=row['column_name'])

    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No nodes to display", showarrow=False,
            font=dict(size=16, color='gray')
        )
        fig.update_layout(
            height=graph_height,
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    # calculate positions
    pos = nx.spring_layout(G, k=node_separation, iterations=50)

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
            if label_text:
                annotations.append(dict(
                    x=(x0 + x1) / 2,
                    y=(y0 + y1) / 2,
                    text=label_text,
                    showarrow=False,
                    font=dict(color="#58A6FF", size=max(10, graph_font_size - 1), family="monospace"),
                    bgcolor="#1E232B",
                    borderpad=2,
                    opacity=0.9
                ))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#666'),
        hoverinfo='none',
        mode='lines'
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

        if node_type == 'focus':
            node_size.append(40)
            node_symbol.append('square')
            node_text.append(f'<b>{node}</b>')
            node_line_color.append('white')
            node_line_width.append(3)
        else:
            node_size.append(20)
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
        )
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


def create_relationship_matrix(df: pd.DataFrame) -> go.Figure:
    """creates a heatmap showing which datasets connect to which."""
    joins = get_joins(df)
    datasets = sorted(df['dataset_name'].unique())

    # create adjacency matrix
    matrix = pd.DataFrame(0, index=datasets, columns=datasets)

    if not joins.empty:
        for _, r in joins.iterrows():
            src = r['dataset_name_fk']
            tgt = r['dataset_name_pk']
            if src in matrix.index and tgt in matrix.columns:
                matrix.loc[src, tgt] += 1

    fig = px.imshow(
        matrix,
        labels=dict(x="Target (PK)", y="Source (FK)", color="Connections"),
        color_continuous_scale="Blues"
    )
    fig.update_layout(height=600)
    return fig

# =============================================================================
# 8. sql builder engine
# =============================================================================

#------------
def generate_sql_for_path(path: List[str],
                          df: pd.DataFrame,
                          dialect: str = "T-SQL") -> str:
    """
    Generate a LEFT JOIN query that follows a specific dataset path
    (e.g., ['Users', 'Course Access', 'Grade Results']).

    Uses the same join graph as generate_sql, but respects the exact
    dataset order of the provided path.
    """
    if len(path) < 2:
        return "-- need at least 2 tables in the path to generate a JOIN."

    # configuration based on dialect
    if dialect == "T-SQL":
        q_start, q_end = "[", "]"
        limit_syntax = "TOP 100"   # Goes after SELECT
        limit_suffix = ""          # Goes at end
    else:  # Snowflake and PostgreSQL
        q_start, q_end = '"', '"'
        limit_syntax = ""          # Goes after SELECT
        limit_suffix = "LIMIT 100" # Goes at end

    def quote(name: str) -> str:
        return f"{q_start}{name}{q_end}"

    # Build connection graph from global joins
    G_full = nx.Graph()
    joins = get_joins(df)
    if not joins.empty:
        for _, r in joins.iterrows():
            G_full.add_edge(r['dataset_name_fk'], r['dataset_name_pk'], key=r['column_name'])

    base_table = path[0]
    aliases = {ds: f"t{i+1}" for i, ds in enumerate(path)}

    # SELECT clause
    select_part = f"SELECT {limit_syntax}" if limit_syntax else "SELECT"
    sql_lines = [select_part, f"    {aliases[base_table]}.*"]

    # FROM clause
    sql_lines.append(f"FROM {quote(base_table)} {aliases[base_table]}")

    # Walk the path sequentially
    for i in range(1, len(path)):
        current_table = path[i]
        prev_table = path[i - 1]

        if G_full.has_edge(current_table, prev_table):
            key = G_full[current_table][prev_table]['key']
            join_line = (
                f"LEFT JOIN {quote(current_table)} {aliases[current_table]} "
                f"ON {aliases[prev_table]}.{quote(key)} = {aliases[current_table]}.{quote(key)}"
            )
            sql_lines.append(join_line)
        else:
            # Fallback if metadata has no edge recorded between these two
            sql_lines.append(
                f"CROSS JOIN {quote(current_table)} {aliases[current_table]} "
                f"-- ⚠️ no direct relationship found between {prev_table} and {current_table}"
            )

    if limit_suffix:
        sql_lines.append(limit_suffix)

    return "\n".join(sql_lines)

#------------
def generate_pandas_for_path(path: List[str], df: pd.DataFrame) -> str:
    """
    Generate pandas code to follow a specific dataset path
    (e.g., ['Users', 'Course Access', 'Grade Results']).

    Uses the same join graph as generate_pandas, but respects the
    exact dataset order of the provided path.
    """
    if len(path) < 2:
        return "# need at least 2 tables in the path to generate a JOIN."

    # Helper to clean names for python variables (User Logins -> df_user_logins)
    def clean_var(name: str) -> str:
        return f"df_{name.lower().replace(' ', '_')}"

    # Build connection graph from global joins
    G_full = nx.Graph()
    joins = get_joins(df)
    if not joins.empty:
        for _, r in joins.iterrows():
            G_full.add_edge(r['dataset_name_fk'], r['dataset_name_pk'], key=r['column_name'])

    lines: List[str] = ["import pandas as pd", "", "# 1. Load Dataframes"]

    # Load steps for each table in the path
    for ds in path:
        var = clean_var(ds)
        lines.append(f"{var} = pd.read_csv('{ds}.csv')")

    lines.append("")
    lines.append("# 2. Perform Merges")

    base_ds = path[0]
    base_var = clean_var(base_ds)

    lines.append(f"# Starting with {base_ds}")
    lines.append(f"final_df = {base_var}")

    # Walk the path sequentially, joining each table to its immediate predecessor
    for i in range(1, len(path)):
        current_ds = path[i]
        prev_ds = path[i - 1]
        current_var = clean_var(current_ds)

        if G_full.has_edge(current_ds, prev_ds):
            key = G_full[current_ds][prev_ds]['key']

            lines.append("")
            lines.append(f"# Joining {current_ds} to {prev_ds} on {key}")
            lines.append("final_df = pd.merge(")
            lines.append("    final_df,")
            lines.append(f"    {current_var},")
            lines.append(f"    on='{key}',")
            lines.append("    how='left'")
            lines.append(")")
        else:
            lines.append("")
            lines.append(
                f"# ⚠️ No direct key found between {prev_ds} and {current_ds} in metadata. "
                f"Performing cross join (use with caution)."
            )
            lines.append("final_df = final_df.merge(")
            lines.append(f"    {current_var},")
            lines.append("    how='cross'")
            lines.append(")")

    lines.append("")
    lines.append("# 3. Preview Result")
    lines.append("print(final_df.head())")

    return "\n".join(lines)

def generate_sql(selected_datasets: List[str], df: pd.DataFrame,
                 dialect: str = "T-SQL") -> str:
    """
    generates a deterministic sql join query with dialect-specific syntax.
    supported dialects: 'T-SQL', 'Snowflake', 'PostgreSQL'
    """
    if len(selected_datasets) < 2:
        return "-- please select at least 2 datasets to generate a join."

    # configuration based on dialect
    if dialect == "T-SQL":
        q_start, q_end = "[", "]"
        limit_syntax = "TOP 100"   # Goes after SELECT
        limit_suffix = ""          # Goes at end
    else:  # snowflake and postgreSQL
        q_start, q_end = '"', '"'
        limit_syntax = ""          # Goes after SELECT
        limit_suffix = "LIMIT 100" # Goes at end

    # helper to quote identifiers
    def quote(name):
        return f"{q_start}{name}{q_end}"

    # build the full connection graph
    G_full = nx.Graph()
    joins = get_joins(df)

    if not joins.empty:
        for _, r in joins.iterrows():
            G_full.add_edge(r['dataset_name_fk'], r['dataset_name_pk'], key=r['column_name'])

    # initialize query
    base_table = selected_datasets[0]
    aliases = {ds: f"t{i+1}" for i, ds in enumerate(selected_datasets)}

    # SELECT clause
    select_part = f"SELECT {limit_syntax}" if limit_syntax else "SELECT"
    sql_lines = [f"{select_part}", f"    {aliases[base_table]}.*"]

    # FROM clause
    sql_lines.append(f"FROM {quote(base_table)} {aliases[base_table]}")

    joined_tables = {base_table}
    remaining_tables = selected_datasets[1:]

    # join strategy
    for current_table in remaining_tables:
        found_connection = False

        for existing_table in joined_tables:
            if G_full.has_edge(current_table, existing_table):
                key = G_full[current_table][existing_table]['key']

                join_line = (
                    f"LEFT JOIN {quote(current_table)} {aliases[current_table]} "
                    f"ON {aliases[existing_table]}.{quote(key)} = {aliases[current_table]}.{quote(key)}"
                )
                sql_lines.append(join_line)

                joined_tables.add(current_table)
                found_connection = True
                break

        if not found_connection:
            sql_lines.append(
                f"CROSS JOIN {quote(current_table)} {aliases[current_table]} "
                f"-- ⚠️ no direct relationship found in metadata"
            )
            joined_tables.add(current_table)

    # added LIMIT for Postgres/Snowflake
    if limit_suffix:
        sql_lines.append(limit_suffix)

    return "\n".join(sql_lines)


def generate_pandas(selected_datasets: List[str], df: pd.DataFrame) -> str:
    """
    generates python pandas code to load and merge the selected datasets.
    """
    if len(selected_datasets) < 2:
        return "# please select at least 2 datasets to generate code."

    # helper to clean names for python variables (User Logins -> df_user_logins)
    def clean_var(name):
        return f"df_{name.lower().replace(' ', '_')}"

    # build connection graph
    G_full = nx.Graph()
    joins = get_joins(df)
    if not joins.empty:
        for _, r in joins.iterrows():
            G_full.add_edge(r['dataset_name_fk'], r['dataset_name_pk'], key=r['column_name'])

    lines = ["import pandas as pd", "", "# 1. Load Dataframes"]

    # load steps
    for ds in selected_datasets:
        var = clean_var(ds)
        lines.append(f"{var} = pd.read_csv('{ds}.csv')")

    lines.append("")
    lines.append("# 2. Perform Merges")

    # connection logic
    base_ds = selected_datasets[0]
    base_var = clean_var(base_ds)

    lines.append(f"# Starting with {base_ds}")
    lines.append(f"final_df = {base_var}")

    joined_tables = {base_ds}
    remaining_tables = selected_datasets[1:]

    for current_ds in remaining_tables:
        current_var = clean_var(current_ds)
        found_connection = False

        for existing_ds in joined_tables:
            if G_full.has_edge(current_ds, existing_ds):
                key = G_full[current_ds][existing_ds]['key']

                lines.append("")
                lines.append(f"# Joining {current_ds} to {existing_ds} on {key}")
                lines.append("final_df = pd.merge(")
                lines.append("    final_df,")
                lines.append(f"    {current_var},")
                lines.append(f"    on='{key}',")
                lines.append("    how='left'")
                lines.append(")")

                joined_tables.add(current_ds)
                found_connection = True
                break

        if not found_connection:
            lines.append("")
            lines.append(f"# ⚠️ No direct key found for {current_ds}. Performing cross join (careful!)")
            lines.append(f"final_df = final_df.merge({current_var}, how='cross')")
            joined_tables.add(current_ds)

    lines.append("")
    lines.append("# 3. Preview Result")
    lines.append("print(final_df.head())")

    return "\n".join(lines)

# =============================================================================
# 9. view controllers (modular ui)
# =============================================================================

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

        # data status and scraper
        if not df.empty:
            try:
                mod_time = os.path.getmtime('dataset_metadata.csv')
                last_updated = pd.Timestamp(mod_time, unit='s').strftime('%Y-%m-%d')
            except Exception:
                last_updated = "Unknown"

            st.success(f"✅ **{df['dataset_name'].nunique()}** Datasets Loaded")
            st.caption(f"📅 Schema updated: {last_updated}")
            st.caption(f"🔢 Total Columns: {len(df):,}")
        else:
            st.error("❌ No data loaded")

        # Data Management / Backup
        with st.expander("⚙️ Data Management", expanded=df.empty):
            st.caption("Remove the top 2 URLs below if you don't want to see the ADSs.")
            pasted_text = st.text_area("URLs to Scrape", height=100, value=DEFAULT_URLS)

            if st.button(
                "🔄 Scrape & Update All URLs",
                type="primary",
                use_container_width=True,
                help="Scrape the URLs listed above, add any new datasets found, and refresh the schema."
            ):
                urls = [u.strip() for u in pasted_text.split('\n') if u.strip().startswith('http')]
                if urls:
                    with st.spinner(f"Scraping {len(urls)} pages..."):
                        new_df = scrape_and_save(urls)
                        if not new_df.empty:
                            st.session_state['scrape_msg'] = (
                                f"Success: {new_df['dataset_name'].nunique()} datasets loaded"
                            )
                            load_data.clear()
                            st.rerun()
                else:
                    st.error("No valid URLs found")

            if not df.empty:
                timestamp = pd.Timestamp.now().strftime('%Y-%m-%d')
                csv = df.to_csv(index=False).encode('utf-8')

                st.write("")

                st.download_button(
                    label="💾 Download Metadata Backup (CSV)",
                    data=csv,
                    file_name=f"brightspace_metadata_backup_{timestamp}.csv",
                    mime="text/csv",
                    help="Save a backup of the current schema state. Useful for comparisons or offline analysis.",
                    use_container_width=True
                )

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

                chosen_template = st.selectbox(
                    "Select a Scenario:",
                    ["Custom Selection..."] + list(templates.keys())
                )

                if chosen_template != "Custom Selection...":
                    st.session_state['selected_datasets'] = templates[chosen_template]
                    selected_datasets = st.session_state['selected_datasets']
                    st.success(f"Loaded {len(selected_datasets)} datasets for {chosen_template}")
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
            with st.expander("🔐 AI Login", expanded=False):
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

    return view, selected_datasets


def render_dashboard(df: pd.DataFrame):
    """renders the main dashboard with overview statistics and intelligent search."""
    st.header("📊 Datahub Datasets Overview")

    # How to use section
    with st.expander("ℹ️ How to use this application", expanded=False):
        st.markdown("""
**Welcome to the Brightspace Dataset Explorer**  
This tool attempts to be a sort of...Rosetta Stone for the D2L Data Hub, helping you navigate schemas and build queries, across an ever-growing landscape of available datasets.

1. **🔍 Search & Context:** Find where columns (e.g., `OrgUnitId`) live and read **summaries** of what each dataset actually does.
2. **📋 Compare Schemas:** Use the **Schema Browser** to select multiple datasets and inspect their structures side-by-side.
3. **🔄 Map Analogs (Reports vs. Extracts):**
   * *Goal:* Re-create an Advanced Dataset (Report) using Brightspace Datasets (Raw Extracts).
   * *Action:* Use **Schema Browser** to open the Advanced Dataset (e.g., *All Grades*) next to raw tables (e.g., *Grades*).
   * *Result:* Identify which raw columns match the report columns to build your own custom version.
4. **⚡ Build Queries:** Select datasets in the **SQL Builder** to auto-generate the correct `LEFT JOIN` syntax.
5. **🤖 Ask AI:** Unlock the **AI Assistant** to ask plain-language questions about the data model.

**💡 Pro Tip:** Toggle **"Power User"** mode in the sidebar to reveal advanced tools like the *UDF Flattener* and *KPI Recipes*.
""")

    is_advanced = st.session_state['experience_mode'] == 'advanced'

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    total_datasets = df['dataset_name'].nunique()
    total_columns = len(df)
    total_categories = df['category'].nunique()

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

    # Intelligent search
    st.subheader("🔍 Intelligent Search")

    all_datasets = sorted(df['dataset_name'].unique())
    all_columns = sorted(df['column_name'].unique())

    search_index = [f"📦 {ds}" for ds in all_datasets] + [f"🔑 {col}" for col in all_columns]

    col_search, col_stats = st.columns([3, 1])

    with col_search:
        search_selection = st.selectbox(
            "Search for a Dataset or Column",
            options=search_index,
            index=None,
            placeholder="Type to search (e.g. 'Users', 'OrgUnitId')...",
            label_visibility="collapsed"
        )

    if search_selection:
        st.divider()

        search_type = "dataset" if "📦" in search_selection else "column"
        term = search_selection.split(" ", 1)[1]

        if search_type == "dataset":
            # Single Dataset View
            st.markdown(f"### Results for Dataset: **{term}**")

            ds_data = df[df['dataset_name'] == term]
            if not ds_data.empty:
                meta = ds_data.iloc[0]

                with st.container():
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        st.caption(f"Category: **{meta['category']}**")
                        if meta['url']:
                            st.markdown(f"📄 [**Official Documentation**]({meta['url']})")
                        else:
                            st.caption("No documentation link available.")

                    with c2:
                        show_relationship_summary(df, term)

                with st.expander("📋 View Schema", expanded=True):
                    display_cols = ['column_name', 'data_type', 'description', 'key']
                    available_cols = [c for c in display_cols if c in ds_data.columns]
                    st.dataframe(ds_data[available_cols], hide_index=True, use_container_width=True)

        else:
            # Column View (List of Datasets)
            st.markdown(f"### Datasets containing column: `{term}`")

            hits = df[df['column_name'] == term]['dataset_name'].unique()

            if len(hits) > 0:
                st.info(f"Found **{len(hits)}** datasets containing `{term}`")

                for ds_name in sorted(hits):
                    ds_meta = df[df['dataset_name'] == ds_name].iloc[0]
                    category = ds_meta['category']

                    with st.expander(f"📦 {ds_name}  ({category})"):
                        c_info, c_rel = st.columns([2, 1])

                        with c_info:
                            if ds_meta['url']:
                                st.markdown(f"[View Documentation]({ds_meta['url']})")

                            col_row = df[
                                (df['dataset_name'] == ds_name) &
                                (df['column_name'] == term)
                            ]
                            st.caption("Column Details:")
                            st.dataframe(
                                col_row[['data_type', 'description', 'key']],
                                hide_index=True,
                                use_container_width=True
                            )

                        with c_rel:
                            show_relationship_summary(df, ds_name)
            else:
                st.warning(
                    "Odd, this column is in the index but no datasets were found. Try reloading."
                )

    else:
        # Default Dashboard View
        st.divider()
        col_hubs, col_orphans = st.columns(2)

        with col_hubs:
            st.subheader("🌟 Most Connected Datasets ('Hubs')")

            with st.expander("ℹ️  Why are these numbers so high?", expanded=False):
                st.caption("""
**High Outgoing (Refers To):** This dataset contains \"Super Keys\" like `OrgUnitId` or `UserId`
which allows it to join to dozens of other structural tables (e.g., a Log table joining to every Org Unit type).

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
                        "outgoing_fks": st.column_config.ProgressColumn(
                            "Refers To (Outgoing)",
                            help="Number of tables this dataset points TO (contains FKs)",
                            format="%d",
                            min_value=0,
                            max_value=int(hubs['outgoing_fks'].max()),
                        ),
                        "incoming_fks": st.column_config.ProgressColumn(
                            "Referenced By (Incoming)",
                            help="Number of tables pointing TO this dataset (contains PKs)",
                            format="%d",
                            min_value=0,
                            max_value=int(hubs['incoming_fks'].max()),
                        )
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
                    .rename(columns={'dataset_description': 'description'})
                )

                st.dataframe(
                    orphan_details,
                    column_config={
                        "dataset_name": "Dataset",
                        "category": "Category",
                        "description": st.column_config.TextColumn(
                            "Description", width="medium"
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.success("All datasets have at least one connection!")

        # Category chart (advanced only)
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

    # Path finder (advanced only)
#-----------------------------------------------------------------------------------------------------------				


#------------------------------------------------------------------------------------------------------------------------------------------------------------

    if is_advanced:
        st.divider()
        st.subheader("🛤️ Path Finder")
        st.caption("Find all valid join paths between two datasets.")

        all_ds = sorted(df['dataset_name'].unique())

        col_from, col_to = st.columns(2)
        with col_from:
            source_ds = st.selectbox("From Dataset", [""] + all_ds, key="path_source")
        with col_to:
            target_ds = st.selectbox("To Dataset", [""] + all_ds, key="path_target")
#---------------------------------------------------------
        #------------
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
        
        #------------
        if find_path and source_ds and target_ds:
            if source_ds == target_ds:
                st.warning("Please select two different datasets.")
            else:
                # Decide which join keys are allowed for this search
                allowed_keys = ['UserId', 'OrgUnitId'] if use_core_keys_only else None

                with st.spinner("Calculating network paths..."):
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
        
        # Display results if available in session state
        if 'path_finder_results' in st.session_state:
            results = st.session_state['path_finder_results']
            paths = results['paths']
            #use_core_keys_only = results['use_core_keys_only']  # Retrieve from stored state
            
            if paths:
                count = len(paths)
                st.success(f"Found top {count} shortest path(s) within {results['max_hops']} hops.")
                
                for i, path in enumerate(paths):
                    hops = len(path) - 1
                    label = f"Option {i+1}: {hops} Join(s)"
                    if i == 0:
                        label += " (Shortest)"
                    
                    with st.expander(label, expanded=(i == 0)):
                        # breadcrumb visual
                        st.markdown(" → ".join([f"**{p}**" for p in path]))
                        
                        # detailed breakdown
                        path_details = get_path_details(df, path)
                        st.markdown("---")
                        for step in path_details:
                            st.markdown(
                                f"- `{step['from']}` joins to `{step['to']}` on column `{step['column']}`"
                            )

                        # --- NEW: Generate SQL for this specific path ---
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
                            sql_from_path = generate_sql_for_path(
                                path, df, dialect=path_sql_dialect
                            )
                            st.code(sql_from_path, language="sql")
                                                    #------------
                        # generate pandas code for this path as well
                        st.markdown("#### Generate Pandas Code for This Path")
                        gen_pandas_for_path = st.button(
                            "Generate Pandas",
                            key=f"gen_pandas_for_path_{i}",
                            use_container_width=True
                        )

                        if gen_pandas_for_path:
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
            with col_c1:
                graph_height = st.slider("Graph Height", 400, 1200, 600)
            with col_c2:
                show_edge_labels = st.checkbox("Show Join Labels", True)

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

            fig = create_spring_graph(
                df, selected_datasets, mode,
                graph_font_size, node_separation, graph_height, show_edge_labels
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

                    for ds in selected_datasets:
                        dot_lines.append(f'  "{ds}" [label="{ds}"];')

                    if not export_joins.empty:
                        for _, row in export_joins.iterrows():
                            s = row['Source Dataset']
                            t = row['Target Dataset']
                            k = row['column_name']

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
            elif mode == 'focused' and len(selected_datasets) > 1:
                with st.expander("📋 View Relationships Table"):
                    st.info("No direct joins found between these specific datasets.")

    elif graph_type == "Orbital Map (Galaxy)":
        st.caption("Categories are shown as golden suns, datasets orbit around their category.")

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

        with col_map:
            fig = get_orbital_map(df, target_val, active_keys_filter)
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
        search = st.text_input("Find Column", placeholder="e.g. OrgUnitId, UserId...")

        if search:
            hits = df[df['column_name'].str.contains(search, case=False, na=False)]
            if not hits.empty:
                st.success(f"Found in **{hits['dataset_name'].nunique()}** datasets")

                for ds_name in sorted(hits['dataset_name'].unique()):
                    ds_hits = hits[hits['dataset_name'] == ds_name]
                    with st.expander(f"📦 {ds_name} ({len(ds_hits)} matches)"):
                        display_cols = ['column_name', 'data_type', 'description', 'key']
                        available_cols = [c for c in display_cols if c in ds_hits.columns]
                        st.dataframe(
                            ds_hits[available_cols],
                            hide_index=True,
                            use_container_width=True
                        )
            else:
                st.warning("No matches found.")

    with col_browse:
        st.subheader("📂 Browse by Dataset")

        all_ds = sorted(df['dataset_name'].unique())

        selected_ds_list = st.multiselect(
            "Select Datasets",
            options=all_ds,
            placeholder="Choose one or more datasets to inspect..."
        )

        if selected_ds_list:
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


def render_sql_builder(df: pd.DataFrame, selected_datasets: List[str]):
    """renders the sql builder interface with python/pandas support."""
    st.header("⚡ Query Builder")

    if not selected_datasets:
        st.info("👈 Select 2 or more datasets from the sidebar to generate code.")

        st.subheader("Quick Select")
        all_ds = sorted(df['dataset_name'].unique())
        quick_select = st.multiselect(
            "Select datasets here:", all_ds, key="sql_quick_select"
        )

        if quick_select:
            selected_datasets = quick_select

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
    st.header("🔀 SQL Dialect Translator")
    st.markdown(
        "Convert queries between dialects (e.g., T-SQL to PostgreSQL) or to Python/Pandas."
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

        secret_key = get_secret("openai_api_key") or get_secret("xai_api_key")
        if not secret_key:
            st.error("No API Key found. Please login.")
            return

        try:
            with st.spinner(f"Translating to {target_lang}..."):
                base_url = "https://api.x.ai/v1" if "xai" in str(secret_key).lower() else None
                client = openai.OpenAI(api_key=secret_key, base_url=base_url)

                model_name = model if "xai" not in str(secret_key).lower() else "grok-3-mini"

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

                    if hasattr(response, 'usage') and response.usage:
                        in_tok = response.usage.prompt_tokens
                        out_tok = response.usage.completion_tokens
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


def render_kpi_recipes(df: pd.DataFrame):
    """renders the cookbook of sql recipes."""
    st.header("📚 KPI Recipes")
    st.markdown("Pre-packaged SQL queries for common educational analysis questions.")

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

                tags = [f"📊 {d}" for d in recipe["datasets"]]
                tags.append(f"⚡ {recipe['difficulty']}")
                st.caption(" • ".join(tags))

            with c2:
                dialect = st.selectbox(
                    "Dialect",
                    ["T-SQL", "Snowflake", "PostgreSQL"],
                    key=f"rec_{recipe['title']}",
                    label_visibility="collapsed"
                )

            sql = recipe["sql_template"].strip()

            if dialect == "T-SQL":
                if "SELECT TOP" not in sql and "SELECT" in sql:
                    sql = sql.replace("SELECT", "SELECT TOP 100", 1)
            elif dialect in ["Snowflake", "PostgreSQL"]:
                sql = sql.replace("SELECT TOP 100", "SELECT")
                if "LIMIT" not in sql:
                    sql += "\nLIMIT 100"
                if dialect == "PostgreSQL":
                    sql = sql.replace("GETDATE()", "NOW()").replace("DATEADD", "AGE")

            with st.expander("👨‍🍳 View SQL Recipe", expanded=False):
                st.code(sql, language="sql")
                st.download_button(
                    label="📥 Download SQL",
                    data=sql,
                    file_name=f"recipe_{recipe['title'].lower().replace(' ', '_')}.sql",
                    mime="application/sql"
                )

            st.divider()

# =============================================================================
# 10. UDF Flattener (EAV → wide)
# =============================================================================

def render_udf_flattener(df: pd.DataFrame):
    """renders the EAV pivot tool for user defined fields."""
    st.header("🔧 UDF Flattener")

    st.markdown("Transform 'vertical' custom data lists into standard 'horizontal' tables.")

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

    col_main, col_eav = st.columns(2)
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

            lines = ["SELECT"]
            lines.append(f"    m.{join_key},")

            for i, f in enumerate(fields):
                comma = "," if i < len(fields) - 1 else ""

                if input_type == "IDs (Integers)":
                    match_logic = f"{pivot_col} = {f}"
                    alias = f"Field_{f}"
                else:
                    safe_f = f.replace("'", "''")
                    match_logic = f"{pivot_col} = '{safe_f}'"
                    alias = f.replace(' ', '_').replace("'", "")

                lines.append(
                    f"    MAX(CASE WHEN e.{match_logic} THEN e.{val_col} END) AS {alias}{comma}"
                )

            lines.append(f"FROM {main_table} m")
            lines.append(f"LEFT JOIN {eav_table} e ON m.{join_key} = e.{join_key}")
            lines.append(f"GROUP BY m.{join_key}")

            st.code("\n".join(lines), language="sql")
            st.caption("Copy this SQL to query your database.")

# =============================================================================
# 11. main orchestrator
# =============================================================================

def main():
    """main entry point that orchestrates the application."""

    # show scrape success message if exists
    if st.session_state.get('scrape_msg'):
        st.success(st.session_state['scrape_msg'])
        st.session_state['scrape_msg'] = None

    # load data
    df = load_data()

    # render sidebar and get navigation state
    view, selected_datasets = render_sidebar(df)

    # handle empty data state
    if df.empty:
        st.title("🔗 Brightspace Dataset Explorer")
        st.warning("No data loaded. Please use the sidebar to scrape the Knowledge Base articles.")

        st.markdown("""
### Getting Started
1. Open the **Data Management** section in the sidebar  
2. Click **Scrape & Update All URLs** to load dataset information  
3. Once loaded, explore relationships, search schemas, and use AI assistance
""")
        return

    # route to appropriate view
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
        st.header("✨ Schema Diff")
        st.info("Upload a backup CSV to compare against the current schema.")
        uploaded_file = st.file_uploader("Upload Backup CSV", type="csv")
        if uploaded_file:
            st.caption("Diff logic not yet implemented in this version.")
    elif view == "🤖 AI Assistant":
        render_ai_assistant(df, selected_datasets)


if __name__ == "__main__":
    main()
