# 🚆 RailResolve AI
### AI-Powered Railway Complaint Intelligence & Resolution Support System

**Transforming passenger complaints into structured insights, policy-backed recommendations, and actionable case intelligence.**

RailResolve AI is an AI-assisted railway grievance management prototype built using Python, Streamlit, Natural Language Processing (NLP), hybrid information retrieval, and SQLite.

The application helps organize passenger complaints, identify relevant policy information, generate suggested responses, and support staff in reviewing and managing cases through an integrated interface.

Rather than functioning as a basic complaint classification tool, RailResolve AI brings together complaint intelligence, policy evidence retrieval, response assistance, case tracking, and grievance analytics in one platform.

> **Project status:** Demonstration and learning prototype. It is not an official Indian Railways grievance portal and is not connected to live railway systems.

---

## ✨ Key Features

### 1. Intelligent Complaint Analysis
- Analyse passenger complaints using NLP-based classification.
- Identify complaint categories and relevant information.
- Organize unstructured complaint text into structured case information.

### 2. Hybrid Policy Retrieval
- Retrieve relevant information from railway policy documents.
- Combine **FAISS dense vector search** with **BM25 keyword retrieval**.
- Provide policy-related evidence to support response preparation.

### 3. AI-Assisted Response Drafting
- Prepare suggested complaint responses using complaint details and retrieved policy evidence.
- Support staff with context-aware response recommendations.
- Keep human review as part of the response workflow.

### 4. Case Management
- Store complaint records using SQLite.
- Inspect individual case details.
- Update complaint statuses and review case history.

### 5. Grievance Analytics Dashboard
- Explore complaint categories and case statuses.
- Review priorities and staff decisions.
- Analyse complaint trends and management insights.

### 6. Advanced Grievance Intelligence
- Explainable priority recommendations.
- Potential duplicate complaint detection.
- Configurable demonstration SLA tracking.
- Suggested department routing.
- Location-based complaint hotspot summaries.
- Management-level grievance insights.

**Important:** Priority, department routing, and SLA values are prototype recommendations. They do not represent official railway rules or service-level commitments. Suggested responses and recommendations require human review.

---

## 🧠 How It Works

The application follows an integrated complaint intelligence workflow:

1. **Complaint Input** – A passenger complaint is entered into the application.
2. **Complaint Analysis** – NLP-based processing identifies relevant complaint information and category.
3. **Policy Retrieval** – Hybrid retrieval searches railway policy documents using FAISS and BM25.
4. **Response Assistance** – The system prepares a suggested response using the complaint context and retrieved evidence.
5. **Staff Review** – Staff can inspect the case, review recommendations, and record decisions.
6. **Analytics** – Complaint records contribute to case-level dashboards and management insights.

### System Architecture

```text
          Passenger Complaint
                  |
                  v
        Complaint Intelligence
                  |
          NLP Classification
                  |
                  v
        Hybrid Policy Retrieval
           /            \
          v              v
     FAISS Search     BM25 Search
          \              /
           \            /
                  |
                  v
          Retrieved Evidence
                  |
                  v
        Response Drafting
                  |
                  v
            Staff Review
                  |
                  v
          SQLite Case Storage
                  |
                  v
        Grievance Analytics
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn |
| NLP & Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Search | FAISS |
| Keyword Retrieval | BM25 |
| Database | SQLite |
| PDF Processing | PyMuPDF |
| Data Visualization | Streamlit |

---

## 📂 Project Structure

```text
RailResolve-AI/
│
├── app.py
├── create_data.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── advanced_features.py
│   ├── complaint_classifier.py
│   ├── database.py
│   ├── document_processor.py
│   ├── hybrid_retriever.py
│   ├── missing_info.py
│   ├── resolution_agent.py
│   └── retriever.py
│
├── data/
│   ├── complaints.csv
│   └── classification_evaluation.csv
│
├── documents/
│   └── policies/
│       ├── citizens_charter_passenger_services.pdf
│       └── refund_rules.pdf
│
└── artifacts/
    └── vector_index/
        ├── chunks.json
        └── policy.faiss
```

The SQLite database (`data/railresolve.db`) is generated locally when the application runs. A personal or populated database does not need to be committed to GitHub.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/zebasayyed17/RailResolve-AI.git
cd RailResolve-AI
```

### 2. Create a Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare Policy Documents

Ensure the following documents are available:

```text
documents/policies/citizens_charter_passenger_services.pdf
documents/policies/refund_rules.pdf
```

The application also requires the generated retrieval index:

```text
artifacts/vector_index/chunks.json
artifacts/vector_index/policy.faiss
```

If these files are not included in your repository checkout, generate them using:

```bash
python -m src.document_processor
python -m src.retriever --build
```

The first run may download the `sentence-transformers/all-MiniLM-L6-v2` embedding model.

### 5. Generate Sample Data (Optional)

```bash
python create_data.py
```

This generates:

- `data/complaints.csv`
- `data/classification_evaluation.csv`

### 6. Launch the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

---

## 📊 Data & Privacy

- Sample complaint records generated by `create_data.py` are synthetic demonstration data.
- Case information is stored locally in SQLite.
- Analytics reflect only the cases entered into the application's local database.
- Do not upload real passenger personal information, credentials, API keys, or private complaint records to a public repository.
- Share policy documents only where their terms and permissions allow.

---

## ⚠️ Limitations & Responsible Use

RailResolve AI is an educational prototype and has several limitations:

- It is not integrated with live Indian Railways systems.
- Retrieved policy passages may be incomplete or may not cover every complaint scenario.
- Suggested responses and recommendations may require correction.
- Priority, routing, and SLA features are configurable demonstration features, not official railway commitments.
- Analytics should not be interpreted as railway-wide complaint statistics.
- Staff should verify policy evidence and approve responses before use.

The application supports grievance handling; it does not independently submit or resolve passenger complaints.

---

## 🚀 Future Enhancements

Potential improvements include:

- Secure, role-based staff authentication.
- Deployment with a production-ready database.
- More extensive classification and retrieval evaluation.
- Improved multilingual complaint understanding.
- Integration with approved railway grievance systems.
- Enhanced reporting and operational dashboards.

---

## 👩‍💻 Author

**Zeba Sayyed**

Data Science & Analytics | Machine Learning | NLP | Generative AI

- **GitHub:** [@zebasayyed17](https://github.com/zebasayyed17)
- **LinkedIn:** [Connect with me](https://www.linkedin.com/in/zeba-sayyed-28766941b)

