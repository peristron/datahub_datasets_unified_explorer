# filename: brightspace_unified.py
# Run command: streamlit run brightspace_unified.py

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
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# =============================================================================
# 1. APP CONFIGURATION & STYLING
# =============================================================================

st.set_page_config(
    page_title="Brightspace Data Universe",
    layout="wide",
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

# Logging
logging.basicConfig(filename='app.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# CSS: Merging the "Pro" look from Codebase 3
st.markdown("""
<style>
    /* Global Clean Up */
    .main .block-container { padding-top: 2rem; }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #1E232B;
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    div[data-testid="stMetricLabel"] { color: #8B949E; }
    div[data-testid="stMetricValue"] { color: #58A6FF; font-size: 24px; }
    
    /* Tabs */
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
    
    /* Expander Headers */
    [data-testid="stExpander"] summary p { font-weight: 600; font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. CONSTANTS & AI CONFIG
# =============================================================================

PRICING_REGISTRY = {
    "gpt-4o":                  {"in": 2.50, "out": 10.00, "provider": "OpenAI"},
    "gpt-4o-mini":             {"in": 0.15, "out": 0.60,  "provider": "OpenAI"},
    "grok-2-1212":             {"in": 2.00, "out": 10.00, "provider": "xAI"},
    "grok-3":                  {"in": 3.00, "out": 15.00, "provider": "xAI"},
    "grok-3-mini":             {"in": 0.30, "out": 0.50,  "provider": "xAI"},
}

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

# =============================================================================
# 3. SESSION STATE & AUTH
# =============================================================================

if 'authenticated' not in st.session_state: st.session_state['authenticated'] = False
if 'auth_error' not in st.session_state: st.session_state['auth_error'] = False
if 'messages' not in st.session_state: st.session_state['messages'] = []
if 'total_cost' not in st.session_state: st.session_state['total_cost'] = 0.0
if 'total_tokens' not in st.session_state: st.session_state['total_tokens'] = 0

def get_secret(key_name: str) -> Optional[str]:
    return st.secrets.get(key_name) or st.secrets.get(key_name.upper())

def perform_login():
    pwd_secret = get_secret("app_password")
    if not pwd_secret: # Dev mode / Open access
        st.session_state['authenticated'] = True
        return
    if st.session_state.get("password_input") == pwd_secret:
        st.session_state['authenticated'] = True
        st.session_state['auth_error'] = False
    else:
        st.session_state['auth_error'] = True
        st.session_state['authenticated'] = False

def logout():
    st.session_state['authenticated'] = False
    st.session_state['messages'] = []

# =============================================================================
# 4. SCRAPING & DATA PROCESSING (The Foundation)
# =============================================================================

def scrape_table(url: str, category_name: str) -> List[Dict]:
    # 1. UPGRADE: Use browser-like headers to bypass 403/WAF blocks
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://community.d2l.com/',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        # 2. UPGRADE: Use a session and strict error logging
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=20, verify=False)
        
        if response.status_code != 200:
            logger.error(f"Failed to load {url} - Status Code: {response.status_code}")
            return []
            
        soup = BeautifulSoup(response.content, 'html.parser')
        data = []
        current_dataset = category_name
        
        # 3. UPGRADE: Normalize text to catch datasets hidden in article text
        elements = soup.find_all(['h2', 'h3', 'table'])
        
        for element in elements:
            if element.name in ['h2', 'h3']:
                text = element.get_text(strip=True)
                # Clean up generic titles to find the actual dataset name
                clean_text = re.sub(r'About the | data sets?', '', text, flags=re.IGNORECASE).strip()
                if len(clean_text) > 2 and "related" not in clean_text.lower():
                    current_dataset = clean_text
            
            elif element.name == 'table':
                # Parse headers more loosely to account for formatting inconsistencies
                rows = element.find_all('tr')
                if not rows: continue
                
                # Try to find the header row (sometimes it's <thead>, sometimes just first <tr>)
                header_cells = rows[0].find_all(['th', 'td'])
                table_headers = [cell.get_text(strip=True).lower().replace(' ', '_') for cell in header_cells]
                
                # Check if this looks like a schema table (must have name/field AND type/desc)
                valid_indicators = ['name', 'field', 'column', 'type', 'description', 'datatype']
                if not any(ind in h for h in table_headers for ind in valid_indicators):
                    continue

                for row in rows[1:]:
                    columns_ = row.find_all(['td', 'th'])
                    if len(columns_) < 2: continue # Skip empty rows
                    
                    entry = {}
                    # Map columns to known keys based on index
                    for i, col in enumerate(columns_):
                        if i < len(table_headers):
                            val = col.get_text(strip=True)
                            header = table_headers[i]
                            
                            # Normalize header names to our schema
                            if any(x in header for x in ['field', 'name', 'column']):
                                entry['column_name'] = val
                            elif any(x in header for x in ['type', 'datatype']):
                                entry['data_type'] = val
                            elif 'desc' in header:
                                entry['description'] = val
                            elif 'key' in header:
                                entry['key'] = val

                    if 'column_name' in entry and entry['column_name']:
                        entry['dataset_name'] = current_dataset
                        entry['category'] = category_name
                        entry['url'] = url
                        data.append(entry)
                        
        return data
    except Exception as e:
        logger.error(f"Scraping Error for {url}: {e}")
        return []

def scrape_and_save(urls: List[str]) -> pd.DataFrame:
    all_data = []
    progress = st.progress(0, "Initializing Scraper...")
    
    def extract_cat(url):
        return re.sub(r'^\d+\s*', '', os.path.basename(url).split('?')[0].replace('-data-sets', '').replace('-', ' ')).lower()
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        args = [(url, extract_cat(url)) for url in urls]
        future_to_url = {executor.submit(scrape_table, *arg): arg[0] for arg in args}
        for i, future in enumerate(future_to_url):
            try:
                all_data.extend(future.result())
            except Exception: pass
            progress.progress((i + 1) / len(urls), f"Scraping {i+1}/{len(urls)}...")
    progress.empty()
    
    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data).fillna('')
    df['dataset_name'] = df['dataset_name'].astype(str).str.title()
    df['category'] = df['category'].astype(str).str.title()
    if 'key' not in df.columns: df['key'] = ''
    df['is_primary_key'] = df['key'].astype(str).str.contains(r'\bpk\b', case=False, regex=True)
    df['is_foreign_key'] = df['key'].astype(str).str.contains(r'\bfk\b', case=False, regex=True)
    df.to_csv('dataset_metadata.csv', index=False)
    return df

@st.cache_data
def load_data() -> pd.DataFrame:
    if os.path.exists('dataset_metadata.csv'):
        return pd.read_csv('dataset_metadata.csv').fillna('')
    return pd.DataFrame()

@st.cache_data
def get_joins(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'is_primary_key' not in df.columns: return pd.DataFrame()
    pks = df[df['is_primary_key'] == True]
    fks = df[df['is_foreign_key'] == True]
    if pks.empty or fks.empty: return pd.DataFrame()
    merged = pd.merge(fks, pks, on='column_name', suffixes=('_fk', '_pk'))
    return merged[merged['dataset_name_fk'] != merged['dataset_name_pk']]

# =============================================================================
# 5. VISUALIZATION ENGINES
# =============================================================================

# Engine A: The "Macro" View (Solar System)
def create_orbital_map(df: pd.DataFrame, target_node: str = None) -> go.Figure:
    """Deterministic circular layout grouped by Category."""
    if df.empty: return go.Figure()
    
    categories = sorted(df['category'].unique())
    datasets = df[['dataset_name', 'category', 'description']].drop_duplicates('dataset_name')
    
    pos, node_x, node_y, node_text, node_color, node_size = {}, [], [], [], [], []
    cat_x, cat_y, cat_text = [], [], []
    center_x, center_y, orbit_radius = 0, 0, 20
    
    # 1. Place Categories (Suns)
    cat_step = 2 * math.pi / len(categories) if categories else 1
    for i, cat in enumerate(categories):
        angle = i * cat_step
        cx, cy = center_x + orbit_radius * math.cos(angle), center_y + orbit_radius * math.sin(angle)
        pos[cat] = (cx, cy)
        
        # Category Visuals
        cat_x.append(cx); cat_y.append(cy + 3); cat_text.append(cat)
        node_x.append(cx); node_y.append(cy)
        node_text.append(f"Category: {cat}")
        node_color.append('rgba(255, 215, 0, 0.2)' if target_node else 'rgba(255, 215, 0, 1)')
        node_size.append(30)

        # 2. Place Datasets (Planets)
        cat_ds = datasets[datasets['category'] == cat]
        if not cat_ds.empty:
            ds_radius = 4 + (len(cat_ds) * 0.1) # Grow radius slightly with density
            ds_step = 2 * math.pi / len(cat_ds)
            for j, (_, row) in enumerate(cat_ds.iterrows()):
                ds_name = row['dataset_name']
                dx, dy = cx + ds_radius * math.cos(j * ds_step), cy + ds_radius * math.sin(j * ds_step)
                pos[ds_name] = (dx, dy)
                node_x.append(dx); node_y.append(dy)
                
                # Visual Logic
                is_target = (ds_name == target_node)
                node_text.append(f"<b>{ds_name}</b>")
                if target_node:
                    col = '#00FF00' if is_target else ('#00CCFF' if is_target else 'rgba(50,50,50,0.3)')
                    sz = 40 if is_target else 10
                else:
                    col = '#00CCFF'; sz = 8
                node_color.append(col); node_size.append(sz)

    # 3. Edges (Only if target selected)
    edge_x, edge_y = [], []
    if target_node:
        joins = get_joins(df)
        rels = joins[(joins['dataset_name_fk'] == target_node) | (joins['dataset_name_pk'] == target_node)]
        for _, r in rels.iterrows():
            s, t = r['dataset_name_fk'], r['dataset_name_pk']
            if s in pos and t in pos:
                edge_x.extend([pos[s][0], pos[t][0], None])
                edge_y.extend([pos[s][1], pos[t][1], None])

    fig = go.Figure(layout=go.Layout(
        showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=700
    ))
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#00FF00'), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', hovertext=node_text, hoverinfo='text', marker=dict(color=node_color, size=node_size)))
    fig.add_trace(go.Scatter(x=cat_x, y=cat_y, mode='text', text=cat_text, textfont=dict(color='gold', size=10), hoverinfo='none'))
    return fig

# Engine B: The "Micro" View (Focused Network with Arrows & Categories)
def create_focused_graph(df: pd.DataFrame, selected_datasets: List[str]) -> go.Figure:
    """Force-directed graph with Arrows, Categories, and Physics."""
    if len(selected_datasets) < 1: return go.Figure()
    
    joins = get_joins(df)
    G = nx.DiGraph()
    
    # 1. Build Nodes & Edges
    for ds in selected_datasets: G.add_node(ds, type='focus')
    
    # Add relationships
    relevant_joins = joins[
        (joins['dataset_name_fk'].isin(selected_datasets)) & 
        (joins['dataset_name_pk'].isin(selected_datasets))
    ]
    
    # Discovery Mode: Add neighbors if we only have 1 node selected
    if len(selected_datasets) == 1 and not joins.empty:
        single = selected_datasets[0]
        neighbors = joins[(joins['dataset_name_fk'] == single) | (joins['dataset_name_pk'] == single)]
        for _, r in neighbors.iterrows():
            s, t = r['dataset_name_fk'], r['dataset_name_pk']
            if s == single: G.add_node(t, type='neighbor'); G.add_edge(s, t, label=r['column_name'])
            else: G.add_node(s, type='neighbor'); G.add_edge(s, t, label=r['column_name'])
    else:
        for _, r in relevant_joins.iterrows():
            G.add_edge(r['dataset_name_fk'], r['dataset_name_pk'], label=r['column_name'])

    if G.number_of_nodes() == 0: return go.Figure()

    # 2. Categories & Colors (The "Improvement")
    unique_cats = sorted(df[df['dataset_name'].isin(G.nodes())]['category'].unique())
    cat_colors = {cat: f"hsl({(i * 360 / len(unique_cats) + 20) % 360}, 70%, 50%)" for i, cat in enumerate(unique_cats)}

    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)
    
    # 3. Build Traces
    edge_traces = []
    annotations = []
    
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=1.5, color='#888'), hoverinfo='none', mode='lines'
        ))
        # Arrow Logic
        annotations.append(dict(
            ax=x0, ay=y0, axref='x', ayref='y', x=x1, y=y1, xref='x', yref='y',
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=1, arrowcolor='#888', opacity=0.8
        ))
        # Label Logic (Midpoint)
        edge_traces.append(go.Scatter(
            x=[(x0+x1)/2], y=[(y0+y1)/2], text=[data.get('label','')],
            mode='text', textfont=dict(color='cyan', size=10), hoverinfo='text',
            hovertext=f"{u} -> {v}<br>Key: {data.get('label','')}"
        ))

    node_x, node_y, node_text, node_bg, node_line = [], [], [], [], []
    for n in G.nodes(data=True):
        n_id, n_data = n[0], n[1]
        x, y = pos[n_id]
        node_x.append(x); node_y.append(y)
        
        # Lookup category
        cat = df[df['dataset_name'] == n_id]['category'].iloc[0] if not df[df['dataset_name'] == n_id].empty else 'Unknown'
        node_bg.append(cat_colors.get(cat, '#888'))
        node_text.append(f"<b>{n_id}</b><br>Category: {cat}")
        node_line.append('white' if n_data.get('type')=='focus' else '#444')

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=[n for n in G.nodes()], textposition="top center",
        textfont=dict(color='white', size=11), hoverinfo='text', hovertext=node_text,
        marker=dict(size=30, color=node_bg, line=dict(width=2, color=node_line))
    )

    fig = go.Figure(data=[*edge_traces, node_trace],
                    layout=go.Layout(
                        showlegend=False, hovermode='closest', annotations=annotations,
                        xaxis=dict(visible=False), yaxis=dict(visible=False),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        height=700, margin=dict(t=20, b=20, l=20, r=20)
                    ))
    return fig

# =============================================================================
# 6. SQL BUILDER
# =============================================================================

def generate_sql(selected_datasets: List[str], df: pd.DataFrame) -> str:
    if len(selected_datasets) < 2: return "-- Select 2+ tables to generate SQL"
    
    joins = get_joins(df)
    G = nx.Graph()
    if not joins.empty:
        for _, r in joins.iterrows():
            G.add_edge(r['dataset_name_fk'], r['dataset_name_pk'], key=r['column_name'])
            
    base = selected_datasets[0]
    aliases = {ds: f"t{i+1}" for i, ds in enumerate(selected_datasets)}
    sql = [f"SELECT TOP 100", f"    {aliases[base]}.*"]
    sql.append(f"FROM {base} {aliases[base]}")
    
    connected = {base}
    remaining = selected_datasets[1:]
    
    for curr in remaining:
        joined = False
        for existing in connected:
            if G.has_edge(curr, existing):
                key = G[curr][existing]['key']
                sql.append(f"LEFT JOIN {curr} {aliases[curr]} ON {aliases[existing]}.{key} = {aliases[curr]}.{key}")
                connected.add(curr)
                joined = True
                break
        if not joined:
            sql.append(f"CROSS JOIN {curr} {aliases[curr]} -- ⚠️ No direct FK found")
            connected.add(curr)
            
    return "\n".join(sql)

# =============================================================================
# 7. MAIN UI LAYOUTS
# =============================================================================

def render_visual_explorer(df: pd.DataFrame):
    """The 'Complex' Workflow: Maps and Graphs."""
    st.header("🌌 Visual Explorer")
    
    col_ctrl, col_view = st.columns([1, 4])
    
    with col_ctrl:
        st.caption("Visualization Controls")
        viz_mode = st.radio("View Mode", ["Macro (Orbit)", "Micro (Network)"], help="Orbit for big picture, Network for relationships.")
        
        all_ds = sorted(df['dataset_name'].unique())
        if viz_mode == "Macro (Orbit)":
            target = st.selectbox("Highlight Dataset", ["None"] + all_ds)
            target_val = None if target == "None" else target
            selected_ds = []
        else:
            selected_ds = st.multiselect("Select Focus Datasets", all_ds, help="Pick tables to map connections.")
            target_val = None
            
    with col_view:
        if viz_mode == "Macro (Orbit)":
            st.plotly_chart(create_orbital_map(df, target_val), use_container_width=True)
            if target_val:
                st.info(f"Showing connections for: {target_val}")
        else:
            if not selected_ds:
                st.info("👈 Select at least one dataset in the sidebar/controls to generate the network.")
            else:
                st.plotly_chart(create_focused_graph(df, selected_ds), use_container_width=True)
                
                # Contextual SQL for the visual selection
                if len(selected_ds) > 1:
                    with st.expander("⚡ Generated SQL for Selection"):
                        st.code(generate_sql(selected_ds, df), language="sql")

def render_data_catalog(df: pd.DataFrame):
    """The 'Simple' Workflow: Search and Lists."""
    st.header("📋 Data Catalog")
    
    col_search, col_filter = st.columns([3, 1])
    with col_search:
        search_term = st.text_input("Search Datasets or Columns", placeholder="e.g. 'Grade' or 'OrgUnitId'")
    with col_filter:
        cat_filter = st.multiselect("Filter Category", sorted(df['category'].unique()))

    # Filtering Logic
    filtered = df.copy()
    if cat_filter:
        filtered = filtered[filtered['category'].isin(cat_filter)]
    if search_term:
        filtered = filtered[
            filtered['dataset_name'].str.contains(search_term, case=False) | 
            filtered['column_name'].str.contains(search_term, case=False)
        ]

    unique_hits = filtered['dataset_name'].unique()
    st.caption(f"Found {len(unique_hits)} Datasets")

    # List View
    for ds in unique_hits[:20]: # Pagination limit for performance
        with st.expander(f"📦 **{ds}**", expanded=False):
            subset = df[df['dataset_name'] == ds]
            
            # 1. Quick Stats
            joins = get_joins(df)
            parents = joins[joins['dataset_name_fk'] == ds]['dataset_name_pk'].unique().tolist()
            children = joins[joins['dataset_name_pk'] == ds]['dataset_name_fk'].unique().tolist()
            
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Category:** {subset['category'].iloc[0]}")
            c2.markdown(f"**Parents (FKs to):** {len(parents)}")
            c3.markdown(f"**Children (Referenced by):** {len(children)}")
            
            # 2. Relationship details (The 'Inter-relationships' request)
            if parents or children:
                st.markdown("---")
                rc1, rc2 = st.columns(2)
                with rc1: 
                    if parents: st.markdown(f"**Feeds into (Foreign Keys):**\n" + ", ".join([f"`{p}`" for p in parents]))
                with rc2:
                    if children: st.markdown(f"**Referenced by:**\n" + ", ".join([f"`{c}`" for c in children]))
            
            # 3. Schema Table
            st.markdown("---")
            st.dataframe(subset[['column_name', 'data_type', 'description', 'key']], use_container_width=True, hide_index=True)
            
    if len(unique_hits) > 20:
        st.warning(f"Showing first 20 of {len(unique_hits)} results. Please refine search.")

def render_ai_architect(df: pd.DataFrame):
    """The AI Workflow."""
    st.header("🤖 AI Data Architect")
    
    if not st.session_state['authenticated']:
        st.warning("Please log in via the sidebar to use AI features.")
        return

    c_cfg, c_chat = st.columns([1, 3])
    
    with c_cfg:
        st.markdown("### Settings")
        model = st.selectbox("Model", list(PRICING_REGISTRY.keys()), index=0)
        provider = PRICING_REGISTRY[model]['provider']
        
        # Auth Key Logic
        key_name = "openai_api_key" if provider == "OpenAI" else "xai_api_key"
        api_key = get_secret(key_name)
        if not api_key: api_key = st.text_input(f"{provider} API Key", type="password")
        
        use_full = st.checkbox("Full Database Context", value=True, help="Sends all table names/columns to AI.")
        
        with st.expander("Cost Tracker"):
            st.metric("Session Cost", f"${st.session_state['total_cost']:.4f}")
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()

    with c_chat:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])
            
        if prompt := st.chat_input("Ask about connections, SQL, or definitions..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            if not api_key:
                st.error("API Key missing.")
            else:
                try:
                    # Context Construction
                    if use_full:
                        ctx = df.groupby('dataset_name').apply(lambda x: f"TABLE {x.name}: {','.join(x['column_name'])}").str.cat(sep="\n")
                        sys_msg = f"You are a D2L Brightspace SQL Expert. Full Schema:\n{ctx[:60000]}"
                    else:
                        sys_msg = "You are a D2L Brightspace SQL Expert. Answer based on general knowledge."

                    # API Call
                    base = "https://api.x.ai/v1" if provider == "xAI" else None
                    client = openai.OpenAI(api_key=api_key, base_url=base)
                    
                    with st.spinner("Thinking..."):
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}]
                        )
                        reply = resp.choices[0].message.content
                        
                        # Cost Calc
                        usage = resp.usage
                        cost = (usage.prompt_tokens * PRICING_REGISTRY[model]['in'] / 1e6) + \
                               (usage.completion_tokens * PRICING_REGISTRY[model]['out'] / 1e6)
                        st.session_state['total_cost'] += cost
                        
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        st.rerun()
                except Exception as e:
                    st.error(str(e))

# =============================================================================
# 8. APP ORCHESTRATOR
# =============================================================================

def main():
    df = load_data()
    
    # Sidebar
    with st.sidebar:
        st.title("🌌 Brightspace Universe")
        
        if df.empty:
            st.warning("No Data Loaded")
        else:
            st.success(f"Loaded {df['dataset_name'].nunique()} Datasets")
            
        with st.expander("⚙️ Admin / Scraper"):
            urls = st.text_area("URLs", value=DEFAULT_URLS, height=100)
            if st.button("Update Data"):
                scrape_and_save([u.strip() for u in urls.split('\n') if u.startswith('http')])
                st.rerun()
        
        st.divider()
        if st.session_state['authenticated']:
            st.write("Logged In 🔓")
            if st.button("Logout"): logout(); st.rerun()
        else:
            st.text_input("Password", type="password", key="password_input", on_change=perform_login)

    # Main Area
    if df.empty:
        st.info("👈 Please run the scraper in the sidebar to begin.")
        return

    # Tabs for Simple vs Complex Workflow
    tab_visual, tab_catalog, tab_ai = st.tabs(["🌌 Visual Explorer", "📋 Data Catalog", "🤖 AI Architect"])
    
    with tab_visual:
        render_visual_explorer(df)
    with tab_catalog:
        render_data_catalog(df)
    with tab_ai:
        render_ai_architect(df)

if __name__ == "__main__":
    main()
