Here is a comprehensive `README.md` tailored to the application's features and the specific "Power User" workflows we just polished.

***

# 🔗 Unified Brightspace Dataset Explorer

**A visual "GPS" for D2L Brightspace Data Hub datasets.**

The **Brightspace Dataset Explorer** is a Streamlit application designed to help Data Engineers, Educational Technologists, and Analysts navigate the complex web of 140+ D2L Data Hub datasets. It scrapes the official documentation, visualizes relationships, and automatically generates the SQL needed to join tables together.

## 🚀 Key Features

### 1. 🕷️ Dynamic Metadata Scraper
- Automatically scrapes the [D2L Community Knowledge Base](https://community.d2l.com/) to build a live schema of the Data Hub.
- Parses Primary Keys (PK) and Foreign Keys (FK) to infer relationships between tables.
- Categorizes datasets (e.g., *Users, Grades, Content, Outcomes*).

### 2. 🗺️ Interactive Relationship Maps
- **Network Graph:** Visualize how tables connect. Uses a physics-based spring layout to cluster related datasets.
- **Orbital Map:** A "Solar System" view where datasets orbit their parent Categories.
- **Focused Mode:** Filter the noise to see only the connections between specific datasets you care about.
- **Bridge Finder:** Automatically suggests intermediate tables to link two disconnected datasets.

### 3. ⚡ Instant SQL Generator
- Select any combination of datasets (e.g., *Users + Grades + OrgUnits*).
- The app calculates the shortest join path and generates production-ready **SQL JOIN** syntax.
- **Download .sql files** directly from the UI.

### 4. 🤖 AI Data Architect
- Integrated AI Assistant (OpenAI/xAI) locked behind a secure login.
- Context-aware: The AI knows the schema and relationships of the datasets you are currently viewing.
- Ask questions like *"How do I calculate time-in-content for a specific course?"*

### 5. 🔍 Schema Browser & Intelligent Search
- Search for any column (e.g., `OrgUnitId`, `OutcomeId`) to find every dataset it belongs to.
- Identify "Orphan" datasets that have no detected relationships.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/brightspace-explorer.git
   cd brightspace-explorer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Dependencies include: streamlit, pandas, networkx, plotly, beautifulsoup4, requests, openai)*

3. **Run the application:**
   ```bash
   streamlit run unified_dataset_explorer.py
   ```

---

## ⚙️ Configuration

To enable the AI Assistant and the Admin Login, you must configure your secrets. 

Create a file at `.streamlit/secrets.toml`:

```toml
# Password to unlock the AI Assistant tab
app_password = "your_secure_password"

# API Keys (Only one is strictly required if you want AI features)
openai_api_key = "sk-..."
xai_api_key = "..."
```

---

## 📖 User Guide

### The "Power User" Workflow
For the most efficient experience in finding connections:

1.  **Select Mode:** Switch to **"🔷 Power User"** in the sidebar.
2.  **Navigate:** Go to **"🗺️ Relationship Map"**.
3.  **Choose a Template:** In the sidebar, select a **Template** (e.g., *Grades & Feedback*) or manually select datasets.
4.  **Analyze:** The graph will show how these tables connect. 
    *   *Don't see a line?* Use the **"🕵️ Find Missing Link"** button to find the bridge table.
5.  **Export:** Expand the **"⚡ Get SQL for this View"** section below the graph and click **Download SQL**.

### Data Management
If the application shows "No data loaded":
1.  Open the **Data Management** expander in the sidebar.
2.  Click **"🔄 Scrape All URLs"**.
3.  The app will fetch the latest definitions from D2L and save them locally to `dataset_metadata.csv`.

---

## 🧠 Technical Details

*   **Graph Theory:** Uses `NetworkX` to calculate shortest paths and connectivity centrality (identifying "Hub" datasets).
*   **Visualization:** Uses `Plotly` for interactive, zoomable graphs.
*   **Browser Security:** Implements form isolation and "decoy inputs" to prevent browser password managers from interfering with dataset selection dropdowns.
*   **Caching:** Heavily leverages `@st.cache_data` to ensure instant graph rendering even with 1,600+ columns.

---

## 📄 License

[MIT License](LICENSE)

*Disclaimer: This tool is an unofficial utility and is not affiliated with or endorsed by D2L Corporation.*
