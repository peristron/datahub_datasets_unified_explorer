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
    page_title="Brightspace Dataset Explorer",
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
    "grok-2-1212":             {"in": 2.00, "out": 10.00, "provider": "xAI"},
    "grok-2-vision-1212":      {"in": 2.00, "out": 10.00, "provider": "xAI"},
    "grok-3":                  {"in": 3.00, "out": 15.00, "provider": "xAI"},
    "grok-3-mini":             {"in": 0.30, "out": 0.50,  "provider": "xAI"},
    "grok-4-0709":             {"in": 3.00, "out": 15.00, "provider": "xAI"},
    
    # openai models
    "gpt-4o":                  {"in": 2.50, "out": 10.00, "provider": "OpenAI"},
    "gpt-4o-mini":             {"in": 0.15, "out": 0.60,  "provider": "OpenAI"},
    "gpt-4.1":                 {"in": 2.00, "out": 8.00,  "provider": "OpenAI"},
    "gpt-4.1-mini":            {"in": 0.40, "out": 1.60,  "provider": "OpenAI"},
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

def get_secret(key_name: str) -> Optional[str]:
    """retrieves a secret, checking both lowercase and uppercase variations."""
    return st.secrets.get(key_name) or st.secrets.get(key_name.upper())


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


def logout():
    """clears authentication state."""
    st.session_state['authenticated'] = False
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

def scrape_table(url: str, category_name: str) -> List[Dict]:
    """
    parses a d2l knowledge base page to extract dataset definitions.
    returns a list of dictionaries representing columns.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        if response.status_code != 200:
            logger.warning(f"Status {response.status_code} for {url}")
            return []
            
        soup = BeautifulSoup(response.content, 'html.parser')
        data = []
        current_dataset = category_name
        
        # logic: headers (h2/h3) denote the dataset name, following table is schema
        elements = soup.find_all(['h2', 'h3', 'table'])
        for element in elements:
            if element.name in ['h2', 'h3']:
                text = element.text.strip()
                if len(text) > 3: 
                    current_dataset = text.lower()
                    
            elif element.name == 'table':
                # normalize headers
                table_headers = [th.text.strip().lower().replace(' ', '_') for th in element.find_all('th')]
                
                # validation: ensure this is a metadata table
                if not table_headers or not any(x in table_headers for x in ['type', 'description', 'data_type']):
                    continue
                
                # extract rows
                for row in element.find_all('tr'):
                    columns_ = row.find_all('td')
                    if len(columns_) < len(table_headers): 
                        continue
                    
                    entry = {}
                    for i, header in enumerate(table_headers):
                        if i < len(columns_): 
                            entry[header] = columns_[i].text.strip()
                    
                    # normalize keys
                    header_map = {'field': 'column_name', 'name': 'column_name', 'type': 'data_type'}
                    clean_entry = {header_map.get(k, k): v for k, v in entry.items()}
                    
                    if 'column_name' in clean_entry and clean_entry['column_name']:
                        clean_entry['dataset_name'] = current_dataset
                        clean_entry['category'] = category_name
                        clean_entry['url'] = url
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
            
            progress_bar.progress((i + 1) / len(urls), f"Scraping {i+1}/{len(urls)}...")
            
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
    expected_cols = ['category', 'dataset_name', 'column_name', 'data_type', 'description', 'key', 'url']
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

    # 2. identify potential foreign keys
    # logic: any column that shares a name with a known PK is a potential FK, 
    # even if not explicitly marked as 'FK' in the documentation 
    # filtering out the PK rows themselves to avoid self-matching the PK definition
    
    pk_names = pks['column_name'].unique()
    
    # Get all columns that match a PK name but aren't the PK row itself
    potential_fks = df[
        (df['column_name'].isin(pk_names)) & 
        (df['is_primary_key'] == False)
    ]
    
    if potential_fks.empty:
        return pd.DataFrame()
    
    # merge to find connections (potential_fks -> pks)
    merged = pd.merge(potential_fks, pks, on='column_name', suffixes=('_fk', '_pk'))
    
    # clean up
    # exclude self-joins (joining a table to itself)
    joins = merged[merged['dataset_name_fk'] != merged['dataset_name_pk']]
    
    # ensure distinct relationships
    joins = joins.drop_duplicates(subset=['dataset_name_fk', 'column_name', 'dataset_name_pk'])
    
    return joins


def get_joins(df: pd.DataFrame) -> pd.DataFrame:
    """wrapper to call cached join calculation with hash for cache key."""
    if df.empty:
        return pd.DataFrame()
    # create a simple hash for cache invalidation
    df_hash = str(len(df)) + "_" + str(df['dataset_name'].nunique())
    return get_possible_joins(df_hash, df)


@st.cache_data
def find_pk_fk_joins_for_selection(df_hash: str, df: pd.DataFrame, selected_tuple: tuple) -> pd.DataFrame:
    """
    finds pk-fk joins for selected datasets.
    uses tuple for selected datasets to make it hashable for caching.
    """
    selected_datasets = list(selected_tuple)
    if df.empty or not selected_datasets:
        return pd.DataFrame()
        
    pks = df[df['is_primary_key'] == True]
    fks = df[(df['is_foreign_key'] == True) & (df['dataset_name'].isin(selected_datasets))]
    
    if pks.empty or fks.empty:
        return pd.DataFrame()
    
    merged = pd.merge(fks, pks, on='column_name', suffixes=('_fk', '_pk'))
    joins = merged[merged['dataset_name_fk'] != merged['dataset_name_pk']]
    
    if joins.empty:
        return pd.DataFrame()
        
    result = joins[['dataset_name_fk', 'column_name', 'dataset_name_pk', 'category_pk']].copy()
    result.columns = ['Source Dataset', 'Join Column', 'Target Dataset', 'Target Category']
    return result.drop_duplicates().reset_index(drop=True)


def get_joins_for_selection(df: pd.DataFrame, selected_datasets: List[str]) -> pd.DataFrame:
    """wrapper to call cached join finder with proper cache keys."""
    if df.empty or not selected_datasets:
        return pd.DataFrame()
    df_hash = str(len(df)) + "_" + str(df['dataset_name'].nunique())
    return find_pk_fk_joins_for_selection(df_hash, df, tuple(selected_datasets))

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
            'category': df[df['dataset_name'] == ds]['category'].iloc[0] if len(df[df['dataset_name'] == ds]) > 0 else ''
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


def find_join_path(df: pd.DataFrame, source_dataset: str, target_dataset: str) -> Optional[List[str]]:
    """finds the shortest path of joins between two datasets."""
    joins = get_joins(df)
    
    if joins.empty:
        return None
    
    G = nx.Graph()
    for _, r in joins.iterrows():
        G.add_edge(r['dataset_name_fk'], r['dataset_name_pk'], key=r['column_name'])
    
    try:
        path = nx.shortest_path(G, source_dataset, target_dataset)
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


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
    
    # layout improvements
    # switched to vertical stacking
    # since this function is rendered in narrow side columns, splitting it into columns 
    # causes text truncation. Vertical stacking guarantees the labels are readable
    
    st.metric("References (Outgoing)", outgoing, help=f"This dataset contains {outgoing} Foreign Keys pointing to other tables.")
    st.metric("Referenced By (Incoming)", incoming, help=f"{incoming} other tables have Foreign Keys pointing to this dataset.")
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
        fig.add_annotation(text="Select datasets to visualize", showarrow=False, font=dict(size=16, color='gray'))
        fig.update_layout(height=graph_height, plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                         xaxis=dict(visible=False), yaxis=dict(visible=False))
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
                    G.add_edge(s, t, label=row['Join Column'])
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
                    G.add_edge(s, t, label=row['Join Column'])
    
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No nodes to display", showarrow=False, font=dict(size=16, color='gray'))
        fig.update_layout(height=graph_height, plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                         xaxis=dict(visible=False), yaxis=dict(visible=False))
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
        if show_edge_labels:
            annotations.append(dict(
                x=(x0 + x1) / 2, 
                y=(y0 + y1) / 2, 
                text=edge[2].get('label', ''), 
                showarrow=False, 
                # styling improvement
                # larger font size (max(10, ...))
                # background box (bgcolor) to hide line behind text
                # mtches UI color (#58A6FF)
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
        category = df[df['dataset_name'] == node]['category'].iloc[0] if not df[df['dataset_name'] == node].empty else 'unknown'
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
        if show_edge_labels:
            annotations.append(dict(
                x=(x0 + x1) / 2, 
                y=(y0 + y1) / 2, 
                text=edge[2].get('label', ''), 
                showarrow=False, 
                font=dict(color="cyan", size=max(8, graph_font_size - 4))
            ))
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, 
        line=dict(width=1.5, color='#888'), 
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
        category = df[df['dataset_name'] == node]['category'].iloc[0] if not df[df['dataset_name'] == node].empty else 'unknown'
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
def create_orbital_map(df_hash: str, df: pd.DataFrame, target_node: str = None) -> go.Figure:
    """
    generates the 'solar system' map with deterministic geometry.
    categories are suns. datasets are planets.
    """
    if df.empty:
        return go.Figure()
    
    # prepare data
    categories = sorted(df['category'].unique())
    
    required_cols = ['dataset_name', 'category']
    optional_cols = ['description']
    cols_to_use = required_cols + [c for c in optional_cols if c in df.columns]
    datasets = df[cols_to_use].drop_duplicates('dataset_name')
    
    # layout parameters
    pos = {}
    center_x = 0
    center_y = 0
    orbit_radius_cat = 20
    
    cat_step = 2 * math.pi / len(categories) if categories else 1
    
    # trace containers
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    node_line_width = []
    node_line_color = []
    cat_x = []
    cat_y = []
    cat_text = []
    
    # determine highlights
    active_edges = []
    active_neighbors = set()
    
    if target_node:
        joins = get_joins(df)
        
        if not joins.empty:
            # find outgoing neighbors
            out_ = joins[joins['dataset_name_fk'] == target_node]
            for _, r in out_.iterrows():
                active_edges.append((target_node, r['dataset_name_pk'], r['column_name']))
                active_neighbors.add(r['dataset_name_pk'])
                
            # find incoming neighbors
            in_ = joins[joins['dataset_name_pk'] == target_node]
            for _, r in in_.iterrows():
                active_edges.append((r['dataset_name_fk'], target_node, r['column_name']))
                active_neighbors.add(r['dataset_name_fk'])
    
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
                        node_color.append('#00CCFF')
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
                
                desc_short = str(row.get('description', ''))[:80]
                if desc_short:
                    desc_short += "..."
                    hover_text = f"<b>{ds_name}</b><br>{desc_short}"
                else:
                    hover_text = f"<b>{ds_name}</b>"
                node_text.append(hover_text)
    
    # build edges
    edge_x = []
    edge_y = []
    label_x = []
    label_y = []
    label_text = []
    
    for s, t, k in active_edges:
        if s in pos and t in pos:
            x0, y0 = pos[s]
            x1, y1 = pos[t]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            label_x.append((x0 + x1) / 2)
            label_y.append((y0 + y1) / 2)
            label_text.append(k)
    
    # create traces
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines', 
        line=dict(width=2, color='#00FF00'), 
        hoverinfo='none'
    )
    
    label_trace = go.Scatter(
        x=label_x, y=label_y, mode='text', text=label_text,
        textfont=dict(color='#00FF00', size=11, family="monospace"),
        hoverinfo='none'
    )
    
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', 
        hoverinfo='text', hovertext=node_text,
        marker=dict(
            color=node_color, 
            size=node_size, 
            line=dict(width=node_line_width, color=node_line_color)
        )
    )
    
    cat_label_trace = go.Scatter(
        x=cat_x, y=cat_y, mode='text', text=cat_text,
        textfont=dict(color='gold', size=10), 
        hoverinfo='none'
    )
    
    fig = go.Figure(
        data=[edge_trace, label_trace, node_trace, cat_label_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=700
        )
    )
    return fig


def get_orbital_map(df: pd.DataFrame, target_node: str = None) -> go.Figure:
    """wrapper to call cached orbital map with proper cache key."""
    df_hash = str(len(df)) + "_" + str(df['dataset_name'].nunique())
    return create_orbital_map(df_hash, df, target_node)


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

def generate_sql(selected_datasets: List[str], df: pd.DataFrame) -> str:
    """
    generates a deterministic sql join query based on the graph relationships.
    uses a 'greedy' approach: connect each new table to the existing joined cluster.
    """
    if len(selected_datasets) < 2:
        return "-- please select at least 2 datasets to generate a join."
    
    # build the full connection graph
    G_full = nx.Graph()
    joins = get_joins(df)
    
    if not joins.empty:
        for _, r in joins.iterrows():
            G_full.add_edge(r['dataset_name_fk'], r['dataset_name_pk'], key=r['column_name'])
    
    # initialize query
    base_table = selected_datasets[0]
    aliases = {ds: f"t{i+1}" for i, ds in enumerate(selected_datasets)}
    
    sql_lines = [f"SELECT TOP 100", f"    {aliases[base_table]}.*"]
    sql_lines.append(f"FROM {base_table} {aliases[base_table]}")
    
    joined_tables = {base_table}
    remaining_tables = selected_datasets[1:]
    
    for current_table in remaining_tables:
        found_connection = False
        
        for existing_table in joined_tables:
            if G_full.has_edge(current_table, existing_table):
                key = G_full[current_table][existing_table]['key']
                sql_lines.append(
                    f"LEFT JOIN {current_table} {aliases[current_table]} "
                    f"ON {aliases[existing_table]}.{key} = {aliases[current_table]}.{key}"
                )
                joined_tables.add(current_table)
                found_connection = True
                break
        
        if not found_connection:
            sql_lines.append(
                f"CROSS JOIN {current_table} {aliases[current_table]} "
                f"-- ⚠️ no direct relationship found in metadata"
            )
            joined_tables.add(current_table)
            
    return "\n".join(sql_lines)

# =============================================================================
# 9. view controllers (modular ui)
# =============================================================================

def render_sidebar(df: pd.DataFrame) -> tuple:
    """renders the sidebar navigation and returns (view, selected_datasets)."""
    with st.sidebar:
        st.title("🔗 Dataset Explorer")
        
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
            view = st.radio(
                "Navigation", 
                ["📊 Dashboard", "🗺️ Relationship Map", "📋 Schema Browser", "⚡ SQL Builder", "🤖 AI Assistant"],
                label_visibility="collapsed"
            )
        else:
            view = st.radio(
                "Navigation", 
                ["📊 Dashboard", "🗺️ Relationship Map", "🤖 AI Assistant"],
                label_visibility="collapsed"
            )
        
        st.divider()
        
        # data status and scraper
        if not df.empty:
            st.caption(f"✅ Loaded {df['dataset_name'].nunique()} datasets ({len(df):,} columns)")
        else:
            st.error("❌ No data loaded")
        
        with st.expander("⚙️ Data Management", expanded=df.empty):
            pasted_text = st.text_area("URLs to Scrape", height=100, value=DEFAULT_URLS)
            if st.button("🔄 Scrape All URLs", type="primary"):
                urls = [u.strip() for u in pasted_text.split('\n') if u.strip().startswith('http')]
                if urls:
                    with st.spinner(f"Scraping {len(urls)} pages..."):
                        new_df = scrape_and_save(urls)
                        if not new_df.empty:
                            st.session_state['scrape_msg'] = f"Success: {new_df['dataset_name'].nunique()} datasets loaded"
                            load_data.clear()
                            st.rerun()
                else:
                    st.error("No valid URLs found")
        
        # dataset selection (when applicable)
        selected_datasets = []
        if not df.empty and view in ["🗺️ Relationship Map", "⚡ SQL Builder"]:
            st.divider()
            st.subheader("Dataset Selection")
            
            if is_advanced:
                select_mode = st.radio("Method:", ["List All", "By Category"], horizontal=True, label_visibility="collapsed")
            else:
                select_mode = "List All"
            
            if select_mode == "By Category":
                all_cats = sorted(df['category'].unique())
                selected_cats = st.multiselect("Filter Categories:", all_cats, default=[])
                if selected_cats:
                    for cat in selected_cats:
                        cat_ds = sorted(df[df['category'] == cat]['dataset_name'].unique())
                        s = st.multiselect(f"📦 {cat}", cat_ds, key=f"sel_{cat}")
                        selected_datasets.extend(s)
            else:
                all_ds = sorted(df['dataset_name'].unique())
                selected_datasets = st.multiselect("Select Datasets:", all_ds, key="dataset_multiselect")
            
            if selected_datasets:
                st.button("🗑️ Clear Selection", on_click=clear_all_selections)
        
        # authentication
        st.divider()
        if st.session_state['authenticated']:
            st.success("🔓 AI Unlocked")
            if st.button("Logout"):
                logout()
                st.rerun()
        else:
            with st.expander("🔐 AI Login", expanded=False):
                st.text_input(
                    "Password", 
                    type="password", 
                    key="password_input", 
                    on_change=perform_login,
                    help="Enter password to unlock AI features."
                )
                if st.session_state['auth_error']:
                    st.error("Incorrect password.")
        
        # cross-links (advanced mode only)
        if is_advanced:
            st.divider()
            st.markdown("### 🔗 Related Tools")
            st.link_button("🔎 CSV Query Tool", "https://csvexpl0rer.streamlit.app/")
            st.link_button("✂️ CSV Splitter", "https://csvsplittertool.streamlit.app/")
    
    return view, selected_datasets


def render_dashboard(df: pd.DataFrame):
    """renders the main dashboard with overview statistics and intelligent search."""
    st.header("📊 Datahub Datasets Overview")
    
    is_advanced = st.session_state['experience_mode'] == 'advanced'
    
    # -top metrics -
    col1, col2, col3, col4 = st.columns(4)
    
    total_datasets = df['dataset_name'].nunique()
    total_columns = len(df)
    total_categories = df['category'].nunique()
    
    joins = get_joins(df)
    total_relationships = len(joins) if not joins.empty else 0
    
    col1.metric("Total Datasets", total_datasets)
    col2.metric("Total Columns", f"{total_columns:,}")
    col3.metric("Categories", total_categories)
    # improvement: renamed to "Unique Joins" and added tooltip to explain the count
    col4.metric("Unique Joins", total_relationships, help="Total count of unique directional links (A → B) detected across the entire schema.")
    
    st.divider()
    
    # 2 intelligent search ---
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

    # --- 3. Search Results Logic ---
    if search_selection:
        st.divider()
        
        search_type = "dataset" if "📦" in search_selection else "column"
        term = search_selection.split(" ", 1)[1]
        
        if search_type == "dataset":
            # --- Single Dataset View ---
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
            # --- Column View (List of Datasets) ---
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
                            
                            col_row = df[(df['dataset_name'] == ds_name) & (df['column_name'] == term)]
                            st.caption("Column Details:")
                            st.dataframe(col_row[['data_type', 'description', 'key']], hide_index=True, use_container_width=True)
                            
                        with c_rel:
                            show_relationship_summary(df, ds_name)
            else:
                st.warning("Odd, this column is in the index but no datasets were found. Try reloading.")
                
    else:
        # --- 4. Default Dashboard View ---
        st.divider()
        col_hubs, col_orphans = st.columns(2)
        
        with col_hubs:
            st.subheader("🌟 Most Connected Datasets ('Hubs')")
            
            # Context helper
            with st.expander("ℹ️  Why are these numbers so high?", expanded=False):
                st.caption("""
                **High Outgoing (Refers To):** This dataset contains "Super Keys" like `OrgUnitId` or `UserId` 
                which allows it to join to dozens of other structural tables (e.g., a Log table joining to every Org Unit type).
                
                **High Incoming (Referenced By):** This is a central "Dimension" table (like `Users`) 
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
                
                # filtering main df to get details for these orphans
                # dropping duplicates to get one row per dataset, not one per column
                orphan_details = df[df['dataset_name'].isin(orphans)][['dataset_name', 'category', 'description']].drop_duplicates('dataset_name')
                
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
                st.success("All datasets have at least one connection!")
        
        # --- 5. Category Chart (Advanced Only) ---
        if is_advanced:
            st.divider()
            st.subheader("📁 Category Breakdown")
            
            cat_stats = df.groupby('category').agg({
                'dataset_name': 'nunique',
                'column_name': 'count'
            }).reset_index()
            cat_stats.columns = ['Category', 'Datasets', 'Columns']
            cat_stats = cat_stats.sort_values('Datasets', ascending=False)
            
            # FIXED: Defined columns before using them
            col_chart, col_table = st.columns([2, 1])
            
            with col_chart:
                fig = px.bar(cat_stats, x='Category', y='Datasets', color='Columns',
                            title="Datasets per Category", color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
            
            with col_table:
                st.dataframe(cat_stats, use_container_width=True, hide_index=True)
    
    # --- 6. Path Finder (Advanced Only) ---
    if is_advanced:
        st.divider()
        st.subheader("🛤️ Path Finder")
        st.caption("Find the shortest join path between two datasets")
        
        col_from, col_to, col_find = st.columns([2, 2, 1])
        
        all_ds = sorted(df['dataset_name'].unique())
        
        with col_from:
            source_ds = st.selectbox("From Dataset", [""] + all_ds, key="path_source")
        with col_to:
            target_ds = st.selectbox("To Dataset", [""] + all_ds, key="path_target")
        with col_find:
            st.write("")
            st.write("")
            find_path = st.button("Find Path", type="primary")
        
        if find_path and source_ds and target_ds and source_ds != target_ds:
            path = find_join_path(df, source_ds, target_ds)
            if path:
                st.success(f"Found path with {len(path) - 1} join(s)")
                
                path_details = get_path_details(df, path)
                
                path_text = []
                for i, step in enumerate(path_details):
                    path_text.append(f"**{step['from']}** → `{step['column']}` → **{step['to']}**")
                
                st.markdown(" → ".join([f"**{p}**" for p in path]))
                
                with st.expander("View Join Details"):
                    for step in path_details:
                        st.markdown(f"- `{step['from']}` joins to `{step['to']}` on column `{step['column']}`")
            else:
                st.error("No path found between these datasets. They may not be connected.")

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
        # graph mode selection
        col_mode, col_controls = st.columns([2, 1])
        
        with col_mode:
            graph_mode = st.radio(
                "Graph Mode:",
                ["Focused (Between Selected)", "Discovery (From Selected)"],
                horizontal=True,
                help="**Focused:** Shows only connections between your selected datasets. **Discovery:** Shows all datasets your selection connects to."
            )
        
        # advanced controls
        if is_advanced:
            with st.expander("🛠️ Graph Controls", expanded=False):
                col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                with col_c1:
                    graph_font_size = st.slider("Font Size", 8, 24, 14)
                with col_c2:
                    node_separation = st.slider("Node Separation", 0.1, 2.5, 0.9)
                with col_c3:
                    graph_height = st.slider("Graph Height", 400, 1200, 600)
                with col_c4:
                    show_edge_labels = st.checkbox("Show Join Labels", True)
        else:
            graph_font_size = 14
            node_separation = 0.9
            graph_height = 600
            show_edge_labels = True
        
        if not selected_datasets:
            st.info("👈 Select datasets from the sidebar to visualize their relationships.")
        else:
            mode = 'focused' if 'Focused' in graph_mode else 'discovery'
            
            if mode == 'focused':
                st.caption("Showing direct PK-FK connections between selected datasets only.")
            else:
                st.caption("Showing all datasets that your selection connects to via foreign keys.")
            
            fig = create_spring_graph(
                df, selected_datasets, mode,
                graph_font_size, node_separation, graph_height, show_edge_labels
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # relationships table
            join_data = get_joins_for_selection(df, selected_datasets)
            if not join_data.empty:
                with st.expander("📋 View Relationships Table"):
                    st.dataframe(join_data, use_container_width=True, hide_index=True)
    
    elif graph_type == "Orbital Map (Galaxy)":
        st.caption("Categories are shown as golden suns, datasets orbit around their category.")
        
        all_ds = sorted(df['dataset_name'].unique())
        target = st.selectbox("🎯 Target Dataset (click to highlight connections)", ["None"] + all_ds)
        target_val = None if target == "None" else target
        
        col_map, col_details = st.columns([3, 1])
        
        with col_map:
            fig = get_orbital_map(df, target_val)
            st.plotly_chart(fig, use_container_width=True)
        
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
        
        st.info("💡 Tip: Hover over cells to see the exact connection count. Darker colors = more connections.")


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
                
                # group by dataset
                for ds_name in sorted(hits['dataset_name'].unique()):
                    ds_hits = hits[hits['dataset_name'] == ds_name]
                    with st.expander(f"📦 {ds_name} ({len(ds_hits)} matches)"):
                        display_cols = ['column_name', 'data_type', 'description', 'key']
                        available_cols = [c for c in display_cols if c in ds_hits.columns]
                        st.dataframe(ds_hits[available_cols], hide_index=True, use_container_width=True)
            else:
                st.warning("No matches found.")
    
    with col_browse:
        st.subheader("📂 Browse by Dataset")
        
        all_ds = sorted(df['dataset_name'].unique())
        selected_ds = st.selectbox("Select a Dataset", [""] + all_ds)
        
        if selected_ds:
            subset = df[df['dataset_name'] == selected_ds]
            
            # dataset info
            col_info, col_stats = st.columns([2, 1])
            
            with col_info:
                if not subset.empty and 'category' in subset.columns:
                    st.caption(f"Category: **{subset.iloc[0]['category']}**")
                if 'url' in subset.columns and subset.iloc[0]['url']:
                    st.link_button("📄 View Documentation", subset.iloc[0]['url'])
            
            with col_stats:
                show_relationship_summary(df, selected_ds)
            
            # schema table
            st.markdown("#### Schema")
            display_cols = ['column_name', 'data_type', 'description', 'key']
            available_cols = [c for c in display_cols if c in subset.columns]
            st.dataframe(
                subset[available_cols], 
                use_container_width=True, 
                hide_index=True,
                height=400
            )
            
            # pk/fk breakdown
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
    """renders the sql builder interface."""
    st.header("⚡ SQL Builder")
    
    if not selected_datasets:
        st.info("👈 Select 2 or more datasets from the sidebar to generate SQL joins.")
        
        # quick select interface
        st.subheader("Quick Select")
        all_ds = sorted(df['dataset_name'].unique())
        quick_select = st.multiselect("Select datasets here:", all_ds, key="sql_quick_select")
        
        if quick_select:
            selected_datasets = quick_select
    
    if selected_datasets:
        if len(selected_datasets) < 2:
            st.warning("Select at least 2 datasets to generate a JOIN query.")
        else:
            # show selected datasets
            st.markdown(f"**Selected:** {', '.join(selected_datasets)}")
            
            # generate sql
            sql_code = generate_sql(selected_datasets, df)
            
            col_sql, col_schema = st.columns([2, 1])
            
            with col_sql:
                st.markdown("#### Generated SQL")
                st.code(sql_code, language="sql")
                
                # copy button (streamlit's code block has built-in copy)
                st.caption("💡 Click the copy icon in the code block to copy the SQL.")
            
            with col_schema:
                st.markdown("#### Field Reference")
                
                for ds in selected_datasets:
                    with st.expander(f"📦 {ds}", expanded=False):
                        subset = df[df['dataset_name'] == ds]
                        display_cols = ['column_name', 'data_type', 'key']
                        available_cols = [c for c in display_cols if c in subset.columns]
                        st.dataframe(subset[available_cols], hide_index=True, use_container_width=True, height=200)
            
            # show join visualization
            with st.expander("🗺️ Join Visualization"):
                fig = create_spring_graph(df, selected_datasets, 'focused', 12, 1.0, 400, True)
                st.plotly_chart(fig, use_container_width=True)


def render_ai_assistant(df: pd.DataFrame, selected_datasets: List[str]):
    """renders the ai chat interface."""
    st.header("🤖 AI Data Architect Assistant")
    
    if not st.session_state['authenticated']:
        st.warning("🔒 Login required to use AI features. Please enter password in the sidebar.")
        
        st.info("""
        **What the AI Assistant can do:**
        - Explain dataset relationships and join strategies
        - Suggest optimal query patterns
        - Answer questions about the Brightspace data model
        - Help design complex SQL queries
        """)
        return
    
    # ai settings
    col_settings, col_chat = st.columns([1, 3])
    
    with col_settings:
        st.markdown("#### ⚙️ Settings")
        
        # model selection
        model_options = list(PRICING_REGISTRY.keys())
        selected_model = st.selectbox("Model", model_options, index=3)  # default to grok-3-mini
        
        model_info = PRICING_REGISTRY[selected_model]
        provider = model_info['provider']
        
        st.caption(f"Provider: **{provider}**")
        st.caption(f"Cost: ${model_info['in']:.2f}/${model_info['out']:.2f} per 1M tokens")
        
        # api key
        key_name = "openai_api_key" if provider == "OpenAI" else "xai_api_key"
        secret_key = get_secret(key_name)
        
        if secret_key:
            st.success(f"✅ {provider} Key Loaded")
            api_key = secret_key
        else:
            api_key = st.text_input(f"{provider} API Key", type="password")
        
        # context options
        use_full_context = st.checkbox("Include Full Schema", value=False, 
                                       help="Send entire database schema to AI. Higher cost but more comprehensive answers.")
        
        # cost tracker
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
        # display chat history
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        
        # chat input
        if prompt := st.chat_input("Ask about the data model..."):
            if not api_key:
                st.error("Please provide an API key.")
                st.stop()
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            try:
                # build context
                
                if use_full_context:
                    schema_text = []
                    for ds_name, group in df.groupby('dataset_name'):
                        url = group['url'].iloc[0] if 'url' in group.columns and pd.notna(group['url'].iloc[0]) else ""
                        cols = []
                        for _, row in group.iterrows():
                            c = row['column_name']
                            if row.get('is_primary_key'):
                                c += " (PK)"
                            elif row.get('is_foreign_key'):
                                c += " (FK)"
                            cols.append(c)
                        schema_text.append(f"TABLE: {ds_name}\nURL: {url}\nCOLS: {', '.join(cols)}")
                    
                    context = "\n\n".join(schema_text)
                    scope_msg = "FULL DATABASE SCHEMA"
                else:
                    relationships_context = ""
                    
                    if selected_datasets:
                        context_df = df[df['dataset_name'].isin(selected_datasets)]
                        scope_msg = f"SELECTED DATASETS: {', '.join(selected_datasets)}"
                        
                        # improved logic
                        # to explicitly fetch the relationships we calculated earlier and feed them to the AI
                        # prevents the AI from guessing/hallucinating joins
                        known_joins = get_joins_for_selection(df, selected_datasets)
                        
                        if not known_joins.empty:
                            relationships_context = "\n\nVERIFIED RELATIONSHIPS (Use these strictly for JOIN conditions):\n"
                            for _, row in known_joins.iterrows():
                                relationships_context += f"- {row['Source Dataset']} joins to {row['Target Dataset']} ON column '{row['Join Column']}'\n"
                    else:
                        context_df = df.head(100)
                        scope_msg = "SAMPLE DATA (first 100 rows)"
                    
                    cols_to_use = ['dataset_name', 'column_name', 'data_type', 'description', 'key']
                    available_cols = [c for c in cols_to_use if c in context_df.columns]
                    
                    # appending the explicit relationships to the CSV data
                    context = context_df[available_cols].to_csv(index=False) + relationships_context
                
                system_msg = f"""You are an expert SQL Data Architect specializing in Brightspace (D2L) data sets.
                
Context: {scope_msg}

INSTRUCTIONS:
1. Provide clear, actionable answers about the data model
2. When suggesting JOINs, use proper syntax and explain the relationship
3. If dataset URLs are available, reference them for documentation
4. Be concise but thorough

SCHEMA DATA:
{context[:60000]}"""
                
                # api call
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
                    
                    # track cost
                    if hasattr(response, 'usage') and response.usage:
                        in_tok = response.usage.prompt_tokens
                        out_tok = response.usage.completion_tokens
                        cost = (in_tok * model_info['in'] / 1_000_000) + (out_tok * model_info['out'] / 1_000_000)
                        st.session_state['total_tokens'] += (in_tok + out_tok)
                        st.session_state['total_cost'] += cost
                
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()
                
            except Exception as e:
                st.error(f"AI Error: {str(e)}")

# =============================================================================
# 10. main orchestrator
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
        2. Click **Scrape All URLs** to load dataset information
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
    elif view == "⚡ SQL Builder":
        render_sql_builder(df, selected_datasets)
    elif view == "🤖 AI Assistant":
        render_ai_assistant(df, selected_datasets)


if __name__ == "__main__":
    main()
