"""
RAG-LLM Assessment Generator - Hasil Penelitian Dashboard
Dashboard untuk menampilkan hasil penelitian skripsi secara interaktif untuk sidang ujian.
"""

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Hasil Penelitian - RAG-LLM Assessment Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        color: #a0aec0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #a0aec0;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Section headers */
    .section-header {
        color: #e2e8f0;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.5);
    }
    
    /* Assessment card */
    .assessment-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Interpretation badges */
    .badge-excellent {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    
    .badge-good {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    
    /* Statics improvement */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Determine base path
BASE_PATH = Path(__file__).parent / "hasil"

@st.cache_data
def load_evaluations():
    """Load expert evaluation data"""
    try:
        with open(BASE_PATH / "evaluations.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading evaluations: {e}")
        return []

@st.cache_data
def load_assessments():
    """Load generated assessments"""
    try:
        with open(BASE_PATH / "assessmentss.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading assessments: {e}")
        return []

@st.cache_data
def load_retrieval_data():
    """Load retrieval analysis data"""
    try:
        return pd.read_csv(BASE_PATH / "tabel_3_9_retrieval.csv")
    except Exception as e:
        st.error(f"Error loading retrieval data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_rag_effectiveness():
    """Load RAG effectiveness summary"""
    try:
        return pd.read_csv(BASE_PATH / "rag_effectiveness_summary.csv")
    except Exception as e:
        st.error(f"Error loading RAG effectiveness: {e}")
        return pd.DataFrame()

@st.cache_data
def load_sigmoid_analysis():
    """Load sigmoid analysis for retrieval"""
    try:
        df = pd.read_csv(BASE_PATH / "retrieval_sigmoid_analysis.csv")
        return df
    except Exception as e:
        st.error(f"Error loading sigmoid analysis: {e}")
        return pd.DataFrame()

def calculate_evaluation_stats(evaluations):
    """Calculate statistics from evaluations"""
    if not evaluations:
        return {}
    
    df = pd.DataFrame(evaluations)
    
    stats = {
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
    
    return stats

def render_metric_card(value, label, prefix="", suffix=""):
    """Render a styled metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{prefix}{value}{suffix}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

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
        ["🏠 Overview", "📋 Evaluasi Expert", "🔍 Efektivitas RAG", "📄 Hasil Generate Soal Sistem Final", "📈 Data Mentah"]
    )
    
    # ==================== OVERVIEW ====================
    if section == "🏠 Overview":
        st.markdown('<h2 class="section-header">📊 Ringkasan Hasil Penelitian</h2>', unsafe_allow_html=True)
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Evaluasi Expert",
                value=eval_stats.get("total_evaluations", 0),
                delta=f"{eval_stats.get('unique_evaluators', 0)} evaluator"
            )
        
        with col2:
            st.metric(
                label="Skor Rata-rata",
                value=f"{eval_stats.get('avg_overall', 0):.2f}/5.00",
                delta="Sangat Baik" if eval_stats.get('avg_overall', 0) >= 4.0 else "Baik"
            )
        
        with col3:
            st.metric(
                label="Soal Generated",
                value=len(assessments),
                delta="5 Mata Kuliah"
            )
        
        with col4:
            st.metric(
                label="RAG Success Rate",
                value="74.1%",
                delta="100 Query Test"
            )
        
        st.divider()
        
        # Evaluation Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Distribusi Skor Evaluasi")
            if evaluations:
                df_eval = pd.DataFrame(evaluations)
                
                # Create interpretation counts
                interpretation_counts = df_eval["interpretation"].value_counts()
                
                # Bar chart
                st.bar_chart(interpretation_counts)
                
                st.info(f"""
                **Keterangan:**
                - 🌟 Sangat Baik (≥4.25): {eval_stats.get('excellent_count', 0)} evaluasi
                - ✅ Baik (3.5-4.24): {eval_stats.get('good_count', 0)} evaluasi
                - ⚠️ Perlu Perbaikan (<3.5): {eval_stats.get('needs_improvement', 0)} evaluasi
                """)
        
        with col2:
            st.markdown("### 📈 Skor Per Aspek Evaluasi")
            
            aspect_scores = {
                "Relevansi Materi": eval_stats.get("avg_relevance", 0),
                "Kesesuaian Tingkat Kesulitan": eval_stats.get("avg_difficulty_match", 0),
                "Struktur Soal": eval_stats.get("avg_structure", 0),
                "Nilai Pedagogis": eval_stats.get("avg_pedagogical", 0),
                "Skor Keseluruhan": eval_stats.get("avg_overall", 0),
            }
            
            aspect_df = pd.DataFrame({
                "Aspek": list(aspect_scores.keys()),
                "Skor": list(aspect_scores.values())
            })
            
            st.dataframe(
                aspect_df.style.format({"Skor": "{:.2f}"}).background_gradient(cmap="Blues", subset=["Skor"]),
                use_container_width=True,
                hide_index=True
            )
            
            # Show existing image if available
            img_path = BASE_PATH / "grafik_skor_per_aspek.png"
            if img_path.exists():
                st.image(str(img_path), caption="Grafik Skor Expert Evaluation per Aspek")
        
        st.divider()
        
        # RAG Retrieval Summary
        st.markdown("### 🔍 Efektivitas Retrieval RAG")
        
        if not retrieval_data.empty:
            st.dataframe(
                retrieval_data.style.format({
                    "Relevance Score (avg)": "{:.4f}",
                    "Top-1 Score (avg)": "{:.4f}",
                    "Avg Response Time (ms)": "{:.2f}"
                }).background_gradient(cmap="Greens", subset=["Relevance Score (avg)", "Top-1 Score (avg)"]),
                use_container_width=True,
                hide_index=True
            )
    
    # ==================== EVALUASI EXPERT ====================
    elif section == "📋 Evaluasi Expert":
        st.markdown('<h2 class="section-header">📋 Hasil Evaluasi Expert</h2>', unsafe_allow_html=True)
        
        if evaluations:
            df_eval = pd.DataFrame(evaluations)
            
            # Filters
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
                    "Filter Tingkat Kesulitan:",
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
            
            # Summary stats for filtered data
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
                }).background_gradient(cmap="RdYlGn", subset=["Overall"]),
                use_container_width=True,
                hide_index=True
            )
            
            # Comments section
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
        st.markdown('<h2 class="section-header">🔍 Efektivitas Retrieval RAG</h2>', unsafe_allow_html=True)
        
        # Sigmoid Formula Explanation
        st.markdown("### 📐 Normalisasi Skor dengan Fungsi Sigmoid")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            CrossEncoder menghasilkan skor dalam bentuk **logit** (tanpa batas). 
            Untuk menginterpretasi sebagai probabilitas relevansi, digunakan **fungsi sigmoid**:
            """)
            
            st.latex(r"P(relevant) = \sigma(x) = \frac{1}{1 + e^{-x}}")
            
            st.markdown("""
            **Keterangan:**
            - **x** = skor CrossEncoder (raw logit)
            - **e** = bilangan Euler (≈ 2,71828)
            - **σ** = fungsi sigmoid
            - **P(relevant)** = probabilitas relevansi (0-1)
            """)
        
        with col2:
            st.markdown("**Tabel Konversi Referensi:**")
            conversion_table = pd.DataFrame({
                "CrossEncoder Score": [-2.0, 0.0, 2.0, 4.0, 6.0],
                "Sigmoid": [0.12, 0.50, 0.88, 0.98, 0.997],
                "P(relevant)": ["12%", "50%", "88%", "98%", "99.7%"],
                "Interpretasi": [
                    "Hampir pasti tidak relevan",
                    "Netral (threshold)",
                    "Kemungkinan besar relevan",
                    "Hampir pasti relevan",
                    "Sangat relevan"
                ]
            })
            st.dataframe(conversion_table, use_container_width=True, hide_index=True)
        
        # Interpretation thresholds
        st.markdown("**Interpretasi Probabilitas Relevansi:**")
        interpretation_df = pd.DataFrame({
            "Range P(relevant)": ["≥ 90%", "70% - 89%", "50% - 69%", "< 50%"],
            "Kategori": ["🟢 Sangat Relevan", "🔵 Relevan", "🟡 Cukup Relevan", "🔴 Kurang Relevan"],
            "Interpretasi": [
                "Konteks sangat sesuai dengan query",
                "Konteks sesuai dengan query",
                "Konteks cukup sesuai, masih dapat digunakan",
                "Konteks kurang sesuai dengan query"
            ]
        })
        st.dataframe(interpretation_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # RAG Summary
        if not rag_effectiveness.empty:
            st.markdown("### 📊 Ringkasan Efektivitas RAG")
            st.dataframe(rag_effectiveness, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Retrieval per Subject
        st.markdown("### 📈 Performa Retrieval per Mata Kuliah")
        if not retrieval_data.empty:
            st.dataframe(
                retrieval_data.style.format({
                    "Relevance Score (avg)": "{:.4f}",
                    "Top-1 Score (avg)": "{:.4f}",
                    "Avg Response Time (ms)": "{:.2f}"
                }).background_gradient(cmap="Blues", subset=["Relevance Score (avg)", "Top-1 Score (avg)"]),
                use_container_width=True,
                hide_index=True
            )
            
            # Chart
            st.bar_chart(
                retrieval_data.set_index("Mata Kuliah")[["Relevance Score (avg)", "Top-1 Score (avg)"]]
            )
        
        st.divider()
        
        # Sigmoid Analysis Sample with Percentage
        st.markdown("### 🔬 Analisis Sigmoid dengan Persentase Relevansi")
        if not sigmoid_data.empty:
            st.success("✅ Reranking dengan CrossEncoder meningkatkan probabilitas relevansi secara signifikan.")
            
            # Create sample with percentage columns
            sample_df = sigmoid_data.head(20).copy()
            
            # Add percentage columns
            sample_df["FAISS P(relevant)"] = sample_df["faiss_sigmoid"] * 100
            sample_df["Rerank P(relevant)"] = sample_df["rerank_sigmoid"] * 100
            sample_df["Top-1 P(relevant)"] = sample_df["rerank_top1_sigmoid"] * 100
            
            # Select and rename columns for display
            display_df = sample_df[[
                "mata_kuliah", "query", 
                "faiss_sigmoid", "FAISS P(relevant)",
                "rerank_sigmoid", "Rerank P(relevant)",
                "rerank_top1_sigmoid", "Top-1 P(relevant)"
            ]].copy()
            
            display_df.columns = [
                "Mata Kuliah", "Query",
                "FAISS (σ)", "FAISS %",
                "Rerank (σ)", "Rerank %",
                "Top-1 (σ)", "Top-1 %"
            ]
            
            # Truncate query for display
            display_df["Query"] = display_df["Query"].str[:50] + "..."
            
            st.dataframe(
                display_df.style.format({
                    "FAISS (σ)": "{:.4f}",
                    "FAISS %": "{:.1f}%",
                    "Rerank (σ)": "{:.4f}",
                    "Rerank %": "{:.1f}%",
                    "Top-1 (σ)": "{:.4f}",
                    "Top-1 %": "{:.1f}%"
                }).background_gradient(cmap="RdYlGn", subset=["Rerank %", "Top-1 %"], vmin=0, vmax=100),
                use_container_width=True,
                hide_index=True
            )
            
            # Show improvement statistics
            st.markdown("#### 📈 Peningkatan dari FAISS ke Reranking")
            avg_faiss = sigmoid_data["faiss_sigmoid"].mean() * 100
            avg_rerank = sigmoid_data["rerank_sigmoid"].mean() * 100
            avg_top1 = sigmoid_data["rerank_top1_sigmoid"].mean() * 100
            improvement = avg_rerank - avg_faiss
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg FAISS P(relevant)", f"{avg_faiss:.1f}%")
            m2.metric("Avg Rerank P(relevant)", f"{avg_rerank:.1f}%", delta=f"+{improvement:.1f}%")
            m3.metric("Avg Top-1 P(relevant)", f"{avg_top1:.1f}%")
            m4.metric("Peningkatan Reranking", f"+{improvement:.1f}%", delta="signifikan" if improvement > 10 else "moderat")
    
    # ==================== HASIL GENERATE SOAL SISTEM FINAL ====================
    elif section == "📄 Hasil Generate Soal Sistem Final":
        st.markdown('<h2 class="section-header">📄 Hasil Generate Soal Sistem Final</h2>', unsafe_allow_html=True)
        
        if assessments:
            # Statistics Overview
            st.markdown("### 📊 Statistik Hasil Generasi")
            
            total_soal = len(assessments)
            subjects = sorted(set(a["mata_kuliah"] for a in assessments))
            topics = sorted(set(a.get("topic", "Unknown") for a in assessments))
            
            # Calculate structure compliance
            complete_count = sum(1 for a in assessments if a.get("metrics", {}).get("structure_compliance", 0) == 1.0)
            compliance_rate = (complete_count / total_soal * 100) if total_soal > 0 else 0
            
            # Calculate avg processing time
            avg_time = sum(a.get("metrics", {}).get("processing_time_s", 0) for a in assessments) / total_soal if total_soal > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Soal", total_soal)
            col2.metric("Mata Kuliah", len(subjects))
            col3.metric("Structure Compliance", f"{compliance_rate:.1f}%")
            col4.metric("Avg Processing Time", f"{avg_time:.1f}s")
            
            st.divider()
            
            # Distribution per Subject
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
            
            st.divider()
            
            # Filters for viewing
            st.markdown("### 🔎 Lihat Soal")
            col1, col2, col3 = st.columns(3)
            
            difficulties = ["Semua", "Mudah", "Sedang", "Sulit"]
            
            with col1:
                selected_subject = st.selectbox("Pilih Mata Kuliah:", ["Semua"] + subjects)
            with col2:
                selected_diff = st.selectbox("Pilih Tingkat Kesulitan:", difficulties)
            with col3:
                # Get available topics for selected subject
                if selected_subject == "Semua":
                    available_topics = sorted(set(a.get("topic", "Unknown") for a in assessments))
                else:
                    available_topics = sorted(set(a.get("topic", "Unknown") for a in assessments if a["mata_kuliah"] == selected_subject))
                selected_topic = st.selectbox("Pilih Topik:", ["Semua"] + available_topics)
            
            # Filter assessments
            filtered = assessments.copy()
            if selected_subject != "Semua":
                filtered = [a for a in filtered if a["mata_kuliah"] == selected_subject]
            if selected_diff != "Semua":
                filtered = [a for a in filtered if a["difficulty"] == selected_diff]
            if selected_topic != "Semua":
                filtered = [a for a in filtered if a.get("topic", "Unknown") == selected_topic]
            
            st.markdown(f"### 📝 Menampilkan {len(filtered)} soal")
            
            if filtered:
                # Create a table view first
                table_data = []
                for a in filtered:
                    metrics = a.get("metrics", {})
                    table_data.append({
                        "Mata Kuliah": a["mata_kuliah"],
                        "Topik": a.get("topic", "-"),
                        "Kesulitan": a["difficulty"],
                        "Processing (s)": metrics.get("processing_time_s", 0),
                        "Struktur": "✅" if metrics.get("structure_compliance", 0) == 1.0 else "⚠️",
                        "Soal": "✅" if metrics.get("has_soal") else "❌",
                        "Kunci": "✅" if metrics.get("has_kunci_jawaban") else "❌"
                    })
                
                table_df = pd.DataFrame(table_data)
                st.dataframe(
                    table_df.style.format({"Processing (s)": "{:.1f}"}),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.divider()
                
                # Show detailed content - require mata kuliah selection first
                st.markdown("### 📄 Detail Soal")
                
                # Require mata kuliah selection for detail view
                detail_subject = st.selectbox(
                    "Pilih Mata Kuliah untuk melihat detail soal:", 
                    subjects,
                    key="detail_subject_select"
                )
                
                # Filter assessments for selected subject only
                detail_filtered = [a for a in assessments if a["mata_kuliah"] == detail_subject]
                
                st.markdown(f"**Menampilkan {len(detail_filtered)} soal untuk {detail_subject}**")
                
                # Show all soal for selected mata kuliah
                for i, assessment in enumerate(detail_filtered, 1):
                    topic = assessment.get("topic", "Unknown Topic")
                    difficulty = assessment["difficulty"]
                    
                    with st.expander(f"📄 Soal {i}: {topic} ({difficulty})"):
                        # Metrics row
                        if "metrics" in assessment:
                            metrics = assessment["metrics"]
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Processing Time", f"{metrics.get('processing_time_s', 0):.1f}s")
                            m2.metric("Structure Score", f"{metrics.get('structure_compliance', 0)*100:.0f}%")
                            m3.metric("Has Soal", "✅" if metrics.get('has_soal') else "❌")
                            m4.metric("Has Kunci", "✅" if metrics.get('has_kunci_jawaban') else "❌")
                        
                        st.divider()
                        
                        # Content
                        content = assessment.get("content", "No content available")
                        st.markdown(content)
            else:
                st.warning("Tidak ada soal untuk kombinasi filter yang dipilih.")
        else:
            st.error("Data assessments tidak dapat dimuat.")
    
    # ==================== DATA MENTAH ====================
    elif section == "📈 Data Mentah":
        st.markdown('<h2 class="section-header">📈 Data Mentah Penelitian</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Berikut adalah akses langsung ke file-file data mentah yang digunakan dalam penelitian:
        """)
        
        files_info = [
            ("evaluations.json", "Hasil evaluasi expert (50+ evaluasi)", "26 KB"),
            ("assessmentss.json", "Soal-soal yang di-generate", "787 KB"),
            ("tabel_3_9_retrieval.csv", "Hasil pengukuran retrieval per mata kuliah", "294 bytes"),
            ("rag_effectiveness_summary.csv", "Ringkasan efektivitas RAG", "182 bytes"),
            ("retrieval_sigmoid_analysis.csv", "Analisis sigmoid retrieval (100 query)", "12 KB"),
            ("chunking_results_*.json", "Hasil proses chunking dokumen", "-"),
            ("extraction_results_*.json", "Hasil ekstraksi dokumen", "-"),
        ]
        
        df_files = pd.DataFrame(files_info, columns=["Nama File", "Deskripsi", "Ukuran"])
        st.dataframe(df_files, use_container_width=True, hide_index=True)
        
        # Quick data preview
        st.divider()
        st.markdown("### 🔎 Preview Data")
        
        preview_option = st.selectbox(
            "Pilih data untuk di-preview:",
            ["Evaluations", "RAG Effectiveness", "Retrieval per Mata Kuliah"]
        )
        
        if preview_option == "Evaluations":
            st.json(evaluations[:5] if len(evaluations) > 5 else evaluations)
        elif preview_option == "RAG Effectiveness":
            st.dataframe(rag_effectiveness, use_container_width=True)
        elif preview_option == "Retrieval per Mata Kuliah":
            st.dataframe(retrieval_data, use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #718096; padding: 1rem;">
        <p>RAG-LLM Assessment Generator | Skripsi 2026</p>
        <p style="font-size: 0.85rem;">Dashboard untuk presentasi sidang ujian</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
