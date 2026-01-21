# RAG-LLM Assessment Generator - Research Results Dashboard

Interactive Streamlit dashboard displaying research results from RAG-LLM Assessment Generator thesis project.

## 📊 Features

- **Overview**: Key metrics, evaluation distribution, and aspect scores
- **Expert Evaluation**: Filterable table of expert assessments with comments
- **RAG Effectiveness**: Sigmoid analysis, retrieval metrics, and improvement statistics  
- **Generated Assessments**: All 120 generated questions with filters by subject/difficulty/topic
- **Raw Data**: Access to underlying data files

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Live Demo

[View Dashboard on Streamlit Cloud](https://rag-llm-results-dashboard.streamlit.app)

## 📁 Data Files

- `evaluations.json` - Expert evaluation results (50+ evaluations)
- `assessmentss.json` - Generated assessments (120 questions)
- `retrieval_sigmoid_analysis.csv` - RAG retrieval analysis with sigmoid scores
- `tabel_3_9_retrieval.csv` - Retrieval metrics per subject
- `rag_effectiveness_summary.csv` - RAG effectiveness summary

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Expert Evaluations | 50 |
| Average Score | 4.30/5.00 |
| Generated Questions | 120 |
| Structure Compliance | 99.2% |
| RAG Success Rate | 74.1% |

---

*Research Dashboard for Thesis 2026*
