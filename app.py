"""
RAG-LLM Assessment Generator - Hasil Penelitian Dashboard
Dashboard untuk menampilkan hasil penelitian skripsi secara interaktif untuk sidang ujian.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Hasil Penelitian - RAG-LLM Assessment Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Professional CSS for Thesis Defense
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Base styling */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        color: #1e3a5f;
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        color: #64748b;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Section headers */
    h2, h3 {
        color: #1e3a5f !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
    }
    
    /* Metric cards - clean look */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    
    div[data-testid="stMetric"] > div {
        color: #1e293b !important;
    }
    
    div[data-testid="stMetric"] label,
    div[data-testid="stMetricLabel"] > div,
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #334155 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }
    
    div[data-testid="stMetricValue"] > div,
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1e40af !important;
    }
    
    div[data-testid="stMetricDelta"] > div,
    div[data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
        color: #059669 !important;
    }
    
    /* Tables - clean and readable */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        border-left-width: 4px;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #1e3a5f;
        background: #f8fafc;
        border-radius: 8px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #0f172a 100%);
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        color: white !important;
        font-weight: 500;
    }
    
    section[data-testid="stSidebar"] h1 {
        color: white !important;
    }
    
    /* Dividers */
    hr {
        margin: 2rem 0;
        border-color: #e2e8f0;
    }
    
    /* Success/Info boxes */
    .stSuccess {
        background: #ecfdf5;
        border-color: #10b981;
    }
    
    /* Charts */
    .stPlotlyChart, .stVegaLiteChart {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 2rem 0;
        font-size: 0.9rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }
    
    /* Summary cards */
    .summary-box {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .summary-value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .summary-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.25rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Determine base path
BASE_PATH = Path(__file__).parent / "hasil"

@st.cache_data
def load_evaluations():
    """Load expert evaluation data"""
    try:
        with open(BASE_PATH / "Data_Evaluasi_Expert.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading evaluations: {e}")
        return []

@st.cache_data
def load_assessments():
    """Load generated assessments"""
    try:
        with open(BASE_PATH / "Log_Hasil_Generate_Soal.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading assessments: {e}")
        return []

@st.cache_data
def load_retrieval_data():
    """Load and aggregate retrieval analysis data from raw results"""
    try:
        # Load raw results
        df = pd.read_csv(BASE_PATH / "Raw_Data_Retrieval.csv")
        
        # Calculate dynamic sigmoid scores first (0-100% scale for display)
        import numpy as np
        df["rerank_sigmoid"] = 1 / (1 + np.exp(-df["rerank_avg_score"]))
        df["rerank_top1_sigmoid"] = 1 / (1 + np.exp(-df["rerank_top1"]))
        
        # Aggregate by Subject
        agg_df = df.groupby("mata_kuliah").agg({
            "rerank_sigmoid": "mean",
            "rerank_top1_sigmoid": "mean", 
            "total_time_ms": "mean",
            "query": "count"
        }).reset_index()
        
        # Rename columns to match the UI expectations
        agg_df.columns = ["Mata Kuliah", "P(relevant)", "Top-1 Score (avg)", "Response Time (ms)", "Jumlah Query"]
        
        # Convert to percentage for display consistency (0-1 -> 0-100)
        agg_df["P(relevant)"] = agg_df["P(relevant)"] * 100
        agg_df["Top-1 Score (avg)"] = agg_df["Top-1 Score (avg)"] * 100
        
        return agg_df[["Mata Kuliah", "P(relevant)", "Response Time (ms)"]]
    except Exception as e:
        st.error(f"Error loading retrieval data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_rag_effectiveness():
    """Load RAG effectiveness summary from raw results"""
    try:
        # Load and process raw data
        df = pd.read_csv(BASE_PATH / "Raw_Data_Retrieval.csv")
        import numpy as np
        
        # Calculate dynamic sigmoid scores
        df["rerank_sigmoid"] = 1 / (1 + np.exp(-df["rerank_avg_score"]))
        df["rerank_top1_sigmoid"] = 1 / (1 + np.exp(-df["rerank_top1"]))
        
        # Calculate summary metrics
        total = len(df)
        avg_top_k = df["rerank_sigmoid"].mean()
        avg_top_1 = df["rerank_top1_sigmoid"].mean()
        success_70 = len(df[df["rerank_top1_sigmoid"] >= 0.7]) / total * 100
        success_50 = len(df[df["rerank_top1_sigmoid"] >= 0.5]) / total * 100
        avg_time = df["total_time_ms"].mean()
        
        # Create summary dataframe with new labels
        summary_data = {
            "Metrik": [
                "Avg P(relevant) - Top-K",
                "Avg P(relevant) - Top-1", 
                "Success Rate (P ≥ 70%)",
                "Success Rate (P ≥ 50%)",
                "Avg Response Time (ms)"
            ],
            "Nilai": [
                f"{avg_top_k:.4f} ({avg_top_k*100:.1f}%)",
                f"{avg_top_1:.4f} ({avg_top_1*100:.1f}%)",
                f"{success_70:.0f}%",
                f"{success_50:.0f}%",
                f"{avg_time:.2f} ms"
            ]
        }
        
        return pd.DataFrame(summary_data)
    except Exception as e:
        st.error(f"Error loading RAG effectiveness: {e}")
        return pd.DataFrame()

@st.cache_data
def load_sigmoid_analysis():
    """Load retrieval results and calculate sigmoid scores dynamically"""
    try:
        # Load raw results
        df = pd.read_csv(BASE_PATH / "Raw_Data_Retrieval.csv")
        
        # Calculate Sigmoid (1 / (1 + exp(-x))) for Rerank scores (Logits)
        import numpy as np
        
        # FAISS score is Cosine Similarity (already 0-1)
        df["faiss_sigmoid"] = df["faiss_avg_score"]
        
        # Rerank score is Logit -> Apply Sigmoid to get Probability (0-1)
        df["rerank_sigmoid"] = 1 / (1 + np.exp(-df["rerank_avg_score"]))
        df["rerank_top1_sigmoid"] = 1 / (1 + np.exp(-df["rerank_top1"]))
        
        return df
    except Exception as e:
        st.error(f"Error loading retrieval results: {e}")
        return pd.DataFrame()

def calculate_evaluation_stats(evaluations):
    """Calculate statistics from evaluations"""
    if not evaluations:
        return {}
    
    df = pd.DataFrame(evaluations)
    
    return {
        "total_evaluations": len(df),
        "unique_evaluators": df["evaluator_name"].nunique(),
        "unique_assessments": df["assessment_id"].nunique(),
        "avg_overall": df["overall"].mean(),
        "avg_relevance": df["relevance"].mean(),
        "avg_difficulty_match": df["difficulty_match"].mean(),
        "avg_structure": df["structure"].mean(),
        "avg_pedagogical": df["pedagogical_value"].mean(),
        "excellent_count": len(df[df["overall"] >= 4.25]),
        "good_count": len(df[(df["overall"] >= 3.5) & (df["overall"] < 4.25)]),
        "needs_improvement": len(df[df["overall"] < 3.5]),
    }

# ================== MAIN APP ==================

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Hasil Penelitian RAG-LLM Assessment Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Dashboard Interaktif untuk Sidang Ujian Skripsi</p>', unsafe_allow_html=True)
    
    # Load all data
    evaluations = load_evaluations()
    assessments = load_assessments()
    retrieval_data = load_retrieval_data()
    rag_effectiveness = load_rag_effectiveness()
    sigmoid_data = load_sigmoid_analysis()
    
    # Calculate stats
    eval_stats = calculate_evaluation_stats(evaluations)
    
    # Sidebar Navigation
    st.sidebar.title("📑 Navigasi")
    section = st.sidebar.radio(
        "Pilih Bagian:",
        ["🏠 Overview", "🔍 Efektivitas RAG", "📄 Hasil Generate Soal", "📋 Evaluasi Expert", "📈 Data Mentah"]
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📅 Penelitian 2025**")
    st.sidebar.markdown("Mahendra - Universitas Hasanuddin")
    
    # ==================== OVERVIEW ====================
    if section == "🏠 Overview":
        st.markdown("## 📊 Ringkasan Hasil Penelitian")
        
        # Key Metrics Row - 4 columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Evaluasi Expert",
                value=eval_stats.get("total_evaluations", 0),
                delta=f"{eval_stats.get('unique_evaluators', 0)} evaluator"
            )
        
        with col2:
            avg_score = eval_stats.get('avg_overall', 0)
            st.metric(
                label="Skor Rata-rata",
                value=f"{avg_score:.2f}/5.00",
                delta="Sangat Baik" if avg_score >= 4.21 else "Baik"
            )
        
        with col3:
            st.metric(
                label="Total Soal Dihasilkan",
                value=len(assessments),
                delta="5 Mata Kuliah"
            )
        
        with col4:
            st.metric(
                label="RAG Success Rate",
                value="74.1%",
                delta="100 Query"
            )
        
        st.markdown("---")
        
        # Two column layout for charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Distribusi Skor Evaluasi")
            if evaluations:
                df_eval = pd.DataFrame(evaluations)
                interpretation_counts = df_eval["interpretation"].value_counts()
                st.bar_chart(interpretation_counts)
                
                # Legend
                st.markdown(f"""
                | Kategori | Jumlah |
                |----------|--------|
                | 🌟 Sangat Baik (≥4.25) | {eval_stats.get('excellent_count', 0)} |
                | ✅ Baik (3.5-4.24) | {eval_stats.get('good_count', 0)} |
                | ⚠️ Perlu Perbaikan (<3.5) | {eval_stats.get('needs_improvement', 0)} |
                """)
        
        with col2:
            st.markdown("### 📈 Skor Per Aspek Evaluasi")
            
            aspect_df = pd.DataFrame({
                "Aspek": ["Relevansi Materi", "Kesesuaian Kesulitan", "Struktur Soal", "Nilai Pedagogis", "**Rata-rata Overall**"],
                "Skor": [
                    eval_stats.get("avg_relevance", 0),
                    eval_stats.get("avg_difficulty_match", 0),
                    eval_stats.get("avg_structure", 0),
                    eval_stats.get("avg_pedagogical", 0),
                    eval_stats.get("avg_overall", 0)
                ]
            })
            
            st.dataframe(
                aspect_df.style.format({"Skor": "{:.2f}"}).background_gradient(
                    cmap="Blues", subset=["Skor"], vmin=1, vmax=5
                ),
                use_container_width=True,
                hide_index=True
            )
            
            # Show image if available
            img_path = BASE_PATH / "grafik_skor_per_aspek.png"
            if img_path.exists():
                st.image(str(img_path), caption="Grafik Skor Expert Evaluation per Aspek")
        
        st.markdown("---")
        
        # RAG Retrieval Summary
        st.markdown("### 🔍 Efektivitas Retrieval RAG per Mata Kuliah")
        
        if not retrieval_data.empty:
            st.dataframe(
                retrieval_data.style.format({
                    "Relevance Score (avg)": "{:.2f}",
                    "Top-1 Score (avg)": "{:.2f}",
                    "Avg Response Time (ms)": "{:.0f}"
                }).background_gradient(
                    cmap="Greens", subset=["Relevance Score (avg)", "Top-1 Score (avg)"], vmin=0, vmax=5
                ),
                use_container_width=True,
                hide_index=True
            )
    
    # ==================== EVALUASI EXPERT ====================
    elif section == "📋 Evaluasi Expert":
        st.markdown("## 📋 Hasil Evaluasi Expert")
        
        st.info("**9 evaluator** melakukan evaluasi terhadap **26 sampel soal** yang dihasilkan sistem, menghasilkan **50 evaluasi** total.")
        
        if evaluations:
            df_eval = pd.DataFrame(evaluations)
            
            # Filters in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_matkul = st.selectbox(
                    "Filter Mata Kuliah:",
                    ["Semua"] + sorted(df_eval["mata_kuliah"].unique().tolist())
                )
            with col2:
                selected_evaluator = st.selectbox(
                    "Filter Evaluator:",
                    ["Semua"] + sorted(df_eval["evaluator_name"].unique().tolist())
                )
            with col3:
                selected_difficulty = st.selectbox(
                    "Filter Kesulitan:",
                    ["Semua", "Mudah", "Sedang", "Sulit"]
                )
            
            # Apply filters
            filtered_df = df_eval.copy()
            if selected_matkul != "Semua":
                filtered_df = filtered_df[filtered_df["mata_kuliah"] == selected_matkul]
            if selected_evaluator != "Semua":
                filtered_df = filtered_df[filtered_df["evaluator_name"] == selected_evaluator]
            if selected_difficulty != "Semua":
                filtered_df = filtered_df[filtered_df["difficulty"] == selected_difficulty]
            
            st.markdown(f"### Menampilkan {len(filtered_df)} evaluasi")
            
            # Display table
            display_cols = [
                "evaluator_name", "mata_kuliah", "topic", "difficulty",
                "relevance", "difficulty_match", "structure", "pedagogical_value",
                "overall", "interpretation"
            ]
            
            st.dataframe(
                filtered_df[display_cols].rename(columns={
                    "evaluator_name": "Evaluator",
                    "mata_kuliah": "Mata Kuliah",
                    "topic": "Topik",
                    "difficulty": "Kesulitan",
                    "relevance": "Relevansi",
                    "difficulty_match": "Kesesuaian",
                    "structure": "Struktur",
                    "pedagogical_value": "Pedagogis",
                    "overall": "Overall",
                    "interpretation": "Interpretasi"
                }).style.format({
                    "Relevansi": "{:.0f}",
                    "Kesesuaian": "{:.0f}",
                    "Struktur": "{:.0f}",
                    "Pedagogis": "{:.0f}",
                    "Overall": "{:.2f}"
                }).background_gradient(cmap="RdYlGn", subset=["Overall"], vmin=1, vmax=5),
                use_container_width=True,
                hide_index=True
            )
            
            # Comments section
            st.markdown("---")
            st.markdown("### 💬 Komentar Evaluator")
            comments = filtered_df[filtered_df["comments"].str.len() > 0][["evaluator_name", "topic", "comments"]]
            if not comments.empty:
                for _, row in comments.iterrows():
                    with st.expander(f"💬 {row['evaluator_name']} - {row['topic']}"):
                        st.write(row["comments"])
            else:
                st.info("Tidak ada komentar untuk filter yang dipilih.")
    
    # ==================== EFEKTIVITAS RAG ====================
    elif section == "🔍 Efektivitas RAG":
        st.markdown("## 🔍 Efektivitas Retrieval RAG")
        
        # Sigmoid Formula Explanation
        st.markdown("### 📐 Normalisasi Skor dengan Fungsi Sigmoid")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            CrossEncoder menghasilkan skor dalam bentuk **logit** (tanpa batas). 
            Untuk menginterpretasi sebagai probabilitas relevansi, digunakan fungsi sigmoid:
            """)
            
            st.latex(r"P(relevant) = \sigma(x) = \frac{1}{1 + e^{-x}}")
            
            st.markdown("""
            **Keterangan:**
            - **x** = skor CrossEncoder (raw logit)
            - **σ** = fungsi sigmoid  
            - **P(relevant)** = probabilitas relevansi (0-100%)
            """)
        
        with col2:
            st.markdown("**Tabel Konversi Referensi:**")
            conversion_table = pd.DataFrame({
                "Score": [-2.0, 0.0, 2.0, 4.0],
                "Sigmoid": [0.12, 0.50, 0.88, 0.98],
                "P(r)": ["12%", "50%", "88%", "98%"],
                "Interpretasi": ["Tidak relevan", "Netral", "Relevan", "Sangat relevan"]
            })
            st.dataframe(conversion_table, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Interpretation thresholds
        st.markdown("**Interpretasi Probabilitas Relevansi:**")
        col1, col2, col3, col4 = st.columns(4)
        col1.success("🟢 **≥90%** Sangat Relevan")
        col2.info("🔵 **70-89%** Relevan")
        col3.warning("🟡 **50-69%** Cukup Relevan")
        col4.error("🔴 **<50%** Kurang Relevan")
        
        st.markdown("---")
        
        # Query Test Section
        st.markdown("### 📝 100 Query Test untuk Evaluasi RAG")
        st.info("**100 query** (20 per mata kuliah) digunakan untuk menguji efektivitas retrieval RAG system.")
        
        # Define queries per subject
        queries = {
            "Algoritma dan Pemrograman": [
                "Bagaimana cara menggunakan Git untuk version control?",
                "Apa perbedaan antara Git dan GitHub?",
                "Jelaskan cara membuat repository baru di GitHub",
                "Bagaimana cara menggunakan if-else di Python?",
                "Apa fungsi dari elif dalam struktur kondisi?",
                "Jelaskan konsep nested if dalam pemrograman",
                "Bagaimana cara membuat loop for di Python?",
                "Apa perbedaan antara while dan for loop?",
                "Jelaskan penggunaan break dan continue dalam loop",
                "Bagaimana cara membuat function di Python?",
                "Apa itu error handling dengan try-except?",
                "Jelaskan penggunaan regular expression di Python",
                "Apa itu class dan object dalam OOP Python?",
                "Bagaimana cara menggunakan constructor di Python?",
                "Jelaskan konsep inheritance di Python",
                "Apa perbedaan list, tuple, dan dictionary?",
                "Bagaimana cara menggunakan operator aritmatika?",
                "Jelaskan tipe data dasar di Python",
                "Apa itu encapsulation dalam OOP?",
                "Bagaimana cara menggunakan method dalam class?"
            ],
            "Pemrograman Website": [
                "Bagaimana cara membuat struktur HTML dasar?",
                "Apa perbedaan antara div dan span?",
                "Jelaskan penggunaan CSS untuk styling",
                "Bagaimana cara menggunakan Tailwind CSS?",
                "Apa itu responsive design dengan CSS?",
                "Jelaskan konsep flexbox dan grid layout",
                "Bagaimana cara membuat event handler di JavaScript?",
                "Apa itu DOM manipulation?",
                "Jelaskan cara menggunakan getElementById",
                "Bagaimana setup koneksi database dengan PHP?",
                "Apa itu prepared statement di PDO?",
                "Jelaskan cara mencegah SQL injection",
                "Bagaimana cara install Laravel?",
                "Apa itu MVC pattern di Laravel?",
                "Jelaskan konsep routing di Laravel",
                "Bagaimana cara membuat controller di Laravel?",
                "Apa itu Eloquent ORM?",
                "Jelaskan cara membuat relasi antar tabel",
                "Bagaimana implementasi authentication di Laravel?",
                "Apa fungsi middleware di Laravel?"
            ],
            "Basis Data": [
                "Bagaimana cara membuat tabel dengan CREATE TABLE?",
                "Apa perbedaan PRIMARY KEY dan FOREIGN KEY?",
                "Jelaskan penggunaan constraint NOT NULL",
                "Bagaimana cara menggunakan SELECT dengan WHERE?",
                "Apa fungsi ORDER BY dalam query?",
                "Jelaskan penggunaan LIKE untuk pencarian",
                "Bagaimana cara menggunakan INSERT INTO?",
                "Apa perbedaan UPDATE dan DELETE?",
                "Jelaskan penggunaan TRUNCATE vs DELETE",
                "Bagaimana cara menggunakan operator AND dan OR?",
                "Apa itu operator BETWEEN dan IN?",
                "Jelaskan penggunaan operator LIKE dengan wildcard",
                "Bagaimana cara membuat JOIN antar tabel?",
                "Apa perbedaan INNER JOIN dan LEFT JOIN?",
                "Jelaskan konsep foreign key relationship",
                "Bagaimana menggunakan fungsi agregat SUM dan COUNT?",
                "Apa fungsi GROUP BY dan HAVING?",
                "Jelaskan penggunaan AVG dan MAX",
                "Bagaimana cara membuat subquery?",
                "Apa itu UNION, INTERSECT, dan EXCEPT?"
            ],
            "OOP Java": [
                "Bagaimana cara membuat class di Java?",
                "Apa perbedaan class dan object?",
                "Jelaskan struktur dasar program Java",
                "Bagaimana cara membuat constructor?",
                "Apa itu overloading constructor?",
                "Jelaskan penggunaan this keyword",
                "Bagaimana implementasi encapsulation?",
                "Apa itu getter dan setter?",
                "Jelaskan access modifier public dan private",
                "Bagaimana cara menggunakan inheritance?",
                "Apa itu keyword extends?",
                "Jelaskan konsep method overriding",
                "Bagaimana implementasi polymorphism?",
                "Apa perbedaan compile-time dan runtime polymorphism?",
                "Jelaskan penggunaan abstract class",
                "Bagaimana cara menggunakan interface?",
                "Apa itu Thread di Java?",
                "Jelaskan cara membuat Thread dengan Runnable",
                "Bagaimana synchronization bekerja?",
                "Apa itu thread-safe programming?"
            ],
            "Pemrograman Mobile": [
                "Bagaimana cara membuat Activity di Android?",
                "Apa itu lifecycle Activity?",
                "Jelaskan method onCreate dan onPause",
                "Bagaimana cara menggunakan Intent?",
                "Apa perbedaan implicit dan explicit Intent?",
                "Jelaskan cara mengirim data antar Activity",
                "Bagaimana cara membuat RecyclerView?",
                "Apa itu ViewHolder pattern?",
                "Jelaskan penggunaan Adapter di RecyclerView",
                "Bagaimana cara membuat Fragment?",
                "Apa perbedaan Activity dan Fragment?",
                "Jelaskan komunikasi antar Fragment",
                "Bagaimana menggunakan background thread?",
                "Apa itu AsyncTask di Android?",
                "Jelaskan penggunaan Coroutines",
                "Bagaimana cara request API dengan Retrofit?",
                "Apa itu JSON parsing?",
                "Jelaskan penggunaan Volley untuk networking",
                "Bagaimana menyimpan data dengan SharedPreferences?",
                "Bagaimana menggunakan SQLite di Android?"
            ]
        }
        
        # Display queries per subject
        for subject, query_list in queries.items():
            with st.expander(f"📚 {subject} (20 Query)"):
                for i, q in enumerate(query_list, 1):
                    st.markdown(f"**{i}.** {q}")
        
        st.markdown("---")
        
        # RAG Summary
        if not rag_effectiveness.empty:
            st.markdown("### 📊 Ringkasan Efektivitas RAG")
            st.dataframe(rag_effectiveness, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Retrieval per Subject
        st.markdown("### 📈 Performa Retrieval per Mata Kuliah")
        if not retrieval_data.empty:
            st.dataframe(
                retrieval_data.style.format({
                    "P(relevant)": "{:.1f}%",
                    "Response Time (ms)": "{:.0f}"
                }).background_gradient(
                    cmap="Blues", subset=["P(relevant)"], vmin=0, vmax=100
                ),
                use_container_width=True,
                hide_index=True
            )
            
            st.bar_chart(
                retrieval_data.set_index("Mata Kuliah")[["P(relevant)"]]
            )
        
        st.markdown("---")
        
        # Sigmoid Analysis
        st.markdown("### 🔬 Perbandingan Skor Retrieval")
        if not sigmoid_data.empty:
            # Summary metrics - keep as decimal 0-1
            avg_faiss = sigmoid_data["faiss_sigmoid"].mean()
            avg_rerank = sigmoid_data["rerank_sigmoid"].mean() * 100
            avg_top1 = sigmoid_data["rerank_top1_sigmoid"].mean() * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("FAISS (Cosine Similarity)", f"{avg_faiss:.2f}")
            col2.metric("Rerank (Probabilitas)", f"{avg_rerank:.1f}%")
            col3.metric("Top-1 Terbaik", f"{avg_top1:.1f}%")
            
            st.info("💡 **FAISS** = Cosine similarity (0-1), **Rerank/Top-1** = Probabilitas relevansi dari CrossEncoder (%).")
            
            st.markdown("---")
            
            # Detailed FAISS vs Reranking table
            st.markdown("### 📊 Detail Skor Retrieval per Query")
            
            # Filter by subject
            subjects = sorted(sigmoid_data["mata_kuliah"].unique().tolist())
            selected_subject = st.selectbox("Filter Mata Kuliah:", ["Semua"] + subjects, key="sigmoid_filter")
            
            filtered_sigmoid = sigmoid_data.copy()
            if selected_subject != "Semua":
                filtered_sigmoid = filtered_sigmoid[filtered_sigmoid["mata_kuliah"] == selected_subject]
            
            # Prepare display dataframe
            display_df = filtered_sigmoid.copy()
            display_df["FAISS"] = display_df["faiss_sigmoid"].round(2)
            display_df["P(relevant)"] = (display_df["rerank_sigmoid"] * 100).round(1).astype(str) + "%"
            display_df["P(relevant) Top-1"] = (display_df["rerank_top1_sigmoid"] * 100).round(1).astype(str) + "%"
            
            # Shorten query text for display
            display_df["Query"] = display_df["query"].str[:50] + "..."
            
            # Display table
            display_table = display_df[["mata_kuliah", "Query", "FAISS", "P(relevant)", "P(relevant) Top-1"]].copy()
            display_table.columns = ["Mata Kuliah", "Query", "FAISS", "P(relevant)", "P(relevant) Top-1"]
            
            st.dataframe(
                display_table,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            st.info(f"Menampilkan **{len(filtered_sigmoid)}** query")
    
    # ==================== HASIL GENERATE SOAL ====================
    elif section == "📄 Hasil Generate Soal":
        st.markdown("## 📄 Hasil Generate Soal Sistem")
        
        if assessments:
            # Statistics
            total_soal = len(assessments)
            subjects = sorted(set(a["mata_kuliah"] for a in assessments))
            
            complete_count = sum(1 for a in assessments if a.get("metrics", {}).get("structure_compliance", 0) == 1.0)
            compliance_rate = (complete_count / total_soal * 100) if total_soal > 0 else 0
            avg_time = sum(a.get("metrics", {}).get("processing_time_s", 0) for a in assessments) / total_soal if total_soal > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Soal", total_soal)
            col2.metric("Mata Kuliah", len(subjects))
            col3.metric("Structure Compliance", f"{compliance_rate:.0f}%")
            col4.metric("Avg Processing Time", f"{avg_time:.1f}s")
            
            st.markdown("---")
            
            # Distribution
            st.markdown("### 📈 Distribusi per Mata Kuliah")
            subject_counts = {}
            for a in assessments:
                subj = a["mata_kuliah"]
                subject_counts[subj] = subject_counts.get(subj, 0) + 1
            
            dist_df = pd.DataFrame({
                "Mata Kuliah": list(subject_counts.keys()),
                "Jumlah Soal": list(subject_counts.values())
            }).sort_values("Jumlah Soal", ascending=False)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(dist_df, use_container_width=True, hide_index=True)
            with col2:
                st.bar_chart(dist_df.set_index("Mata Kuliah"))
            
            st.markdown("---")
            
            # View soal
            st.markdown("### 📄 Lihat Detail Soal")
            
            detail_subject = st.selectbox("Pilih Mata Kuliah:", subjects)
            detail_filtered = [a for a in assessments if a["mata_kuliah"] == detail_subject]
            
            st.info(f"Menampilkan **{len(detail_filtered)} soal** untuk {detail_subject}")
            
            for i, assessment in enumerate(detail_filtered, 1):
                topic = assessment.get("topic", "Unknown")
                difficulty = assessment["difficulty"]
                
                with st.expander(f"📄 Soal {i}: {topic} ({difficulty})"):
                    if "metrics" in assessment:
                        metrics = assessment["metrics"]
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Processing Time", f"{metrics.get('processing_time_s', 0):.1f}s")
                        col2.metric("Has Soal", "✅" if metrics.get('has_soal') else "❌")
                        col3.metric("Has Kunci", "✅" if metrics.get('has_kunci_jawaban') else "❌")
                    
                    st.markdown("---")
                    st.markdown(assessment.get("content", "No content"))
        else:
            st.error("Data assessments tidak dapat dimuat.")
    
    # ==================== DATA MENTAH ====================
    elif section == "📈 Data Mentah":
        st.markdown("## 📈 Data Mentah Penelitian")
        
        # Get list of files in hasil directory
        data_files = sorted([f for f in BASE_PATH.iterdir() if f.suffix in ['.csv', '.json']], key=lambda x: x.name)
        
        if not data_files:
            st.warning("Belum ada data mentah yang tersedia.")
        else:
            # Display file list summary
            files_data = []
            for f in data_files:
                size_kb = f.stat().st_size / 1024
                files_data.append({
                    "Nama File": f.name,
                    "Tipe": f.suffix.upper().replace('.', ''),
                    "Ukuran": f"{size_kb:.2f} KB"
                })
                
            st.dataframe(pd.DataFrame(files_data), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader("🛠️ Preview & Download")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_file_name = st.selectbox("Pilih file untuk dilihat:", [f.name for f in data_files])
            
            selected_file_path = BASE_PATH / selected_file_name
            
            # Prepare data for download and preview
            data_to_download = None
            mime_type = "text/plain"
            
            try:
                if selected_file_path.suffix == '.csv':
                    df = pd.read_csv(selected_file_path)
                    st.dataframe(df, use_container_width=True)
                    data_to_download = df.to_csv(index=False).encode('utf-8')
                    mime_type = "text/csv"
                    
                elif selected_file_path.suffix == '.json':
                    with open(selected_file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    st.json(json_data, expanded=False)
                    data_to_download = json.dumps(json_data, indent=2).encode('utf-8')
                    mime_type = "application/json"
                
                with col2:
                    st.write("") # Spacer to align button
                    st.write("") 
                    if data_to_download:
                        st.download_button(
                            label="⬇️ Download File",
                            data=data_to_download,
                            file_name=selected_file_name,
                            mime=mime_type,
                            type="primary",
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Gagal memuat file: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><strong>RAG-LLM Assessment Generator</strong> | Skripsi 2025</p>
        <p>Mahendra - Departemen Matematika, Universitas Hasanuddin</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
