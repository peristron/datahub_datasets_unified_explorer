🔗 Unified Brightspace Dataset Explorer

The ultimate "GPS" and Workbench for D2L Brightspace Data Hub.

The Brightspace Dataset Explorer is a professional-grade Streamlit application designed to help Data Engineers, Analysts, and EdTech admins master the 140+ datasets in the D2L Data Hub. It goes beyond simple documentation by providing interactive visualization, semantic translation, and automated code generation.
✨ Key Features
1. 🗺️ Interactive Relationship Map

    Visualize the Network: See how tables connect via Primary and Foreign Keys.
    Bridge Finder: Selected two unrelated tables? The app automatically finds the "missing link" (intermediate table) needed to join them.
    Templates: One-click starter packs for common domains (Grades, Engagement, Quizzes).

2. ⚡ SQL Builder & Dialect Switching

    Auto-Joins: Select any combination of datasets; the app calculates the shortest path and writes the SQL.
    Multi-Dialect: Generates syntax for T-SQL (SQL Server), Snowflake, or PostgreSQL.
    Download: Export .sql files directly to your machine.

3. 📚 KPI Recipes (Cookbook)

    Business Logic: Don't just join tables—solve problems. Includes pre-written queries for:
        Learner Engagement (Last access, course activity)
        Assessments (Quiz item analysis, grade distribution)
    Data Cleaning: Special recipes for handling D2L's Row Versioning (deduplication).

4. 🕵️ Semantic "Decoder Ring"

    No More Magic Numbers: Automatically detects Enum columns (like GradeObjectTypeId or SessionType) and displays a cheat sheet of their values (e.g., 1 = Numeric, 2 = Pass/Fail).

5. ✨ Schema Version Diff

    Track Changes: Download a baseline of the metadata today. Upload it next month to see exactly which datasets or columns D2L added or removed.
    Safe Upgrades: Prevents your pipelines from breaking silently.

6. 🤖 AI Data Architect

    Context-Aware AI: A secure chat interface (OpenAI/xAI) that "knows" the specific schema relationships you are currently viewing.

🛠️ Installation

    Clone the repository:

Bash

git clone https://github.com/your-repo/brightspace-explorer.git
cd brightspace-explorer

Install dependencies:

Bash

pip install -r requirements.txt

(Requires: streamlit, pandas, networkx, plotly, beautifulsoup4, requests, openai)

Run the application:

Bash

    streamlit run unified_dataset_explorer.py

⚙️ Configuration

To enable the AI Assistant, configure your API keys in .streamlit/secrets.toml:

toml

# Admin Password for the AI Tab
app_password = "your_secure_password"

# API Keys (Only one required)
openai_api_key = "sk-..."
xai_api_key = "..."

📖 Recommended Workflow
For New Explorers:

    Go to Data Management in the sidebar and click "🔄 Scrape All URLs" to initialize the database.
    Switch to "🔷 Power User" mode.
    Navigate to "🗺️ Relationship Map".
    Select a Template (e.g., "Grades & Feedback").
    Visualise connections and click "⚡ Get SQL for this View" to generate the query.

For Maintenance (Monthly Updates):

    Go to "✨ Schema Diff".
    Upload your brightspace_metadata_backup.csv from the previous month.
    Review the "New Columns" and "New Datasets" report.

🧠 Technical Architecture

    Scraper: Threaded BeautifulSoup scraper that parses D2L Knowledge Base HTML tables.
    Graph Engine: NetworkX handles pathfinding (shortest path algorithms) and centrality metrics.
    Visualization: Plotly Spring Layouts for dynamic, physics-based network rendering.
    Security: Implements "Decoy Input" strategies to prevent browser password managers from interfering with UI dropdowns.

📄 License

MIT License

Disclaimer: This tool is an unofficial utility and is not affiliated with or endorsed by D2L Corporation.
