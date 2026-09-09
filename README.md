# HireWise — AI-Powered Hiring & Onboarding Agent

> **"One intelligent agent. Three trusted workflows."**  
> An enterprise-grade Agentic AI application designed for high-precision HR operations, intelligent resume screening, and automated interview scheduling.

---

## 🌟 Executive Summary & Hackathon Pitch

Modern hiring workflows suffer from data cross-contamination and hallucinated candidate assessments. **HireWise** solves this by enforcing **strict architectural boundaries** across three distinct human resource workflows:
1. **Candidate FAQ (Policy RAG):** Grounded exclusively in the official company handbook.
2. **Resume Screening (Evidence-Based Evaluation):** Structured matching of candidate skills against Job Descriptions, completely isolated from corporate policy data.
3. **Interview Scheduling (Autonomous Tool Calling):** Real-time calendar availability lookup and slot reservation with instant confirmation.

Built on local **ChatOllama (`qwen3:0.6b`)** with **ChromaDB**, HireWise demonstrates the complete end-to-end LangChain Agentic AI workshop pipeline without external LLM dependencies or data leakage.

---

## 🏗️ Technical Architecture & Workshop Pipeline

The application adheres to the mandatory workshop pipeline:

```text
               User Inquiry / Dashboard Interaction
                                 ↓
                   HireWise ReAct Agent Engine
            ┌────────────────────┼────────────────────┐
            ↓                    ↓                    ↓
     Candidate FAQ        Resume Screening     Interview Slots
            ↓                    ↓                    ↓
       policy_qa        resume_to_jd_match  interview_slot_lookup
      Custom Tool          Custom Tool           Custom Tool
            ↓                    ↓                    ↓
     Policy Retriever     Dual Retrievers       Mock Calendar
   (hirewise_policy)   (resume + job_desc)     (Slot Registry)
            ↓                    ↓                    ↓
     Handbook Chunks      Evidence Chunks       Real-time Slots
            ↓                    ↓                    ↓
   ChatOllama (qwen3)   ChatOllama (qwen3)     Slot Booking Tool
            ↓                    ↓                    ↓
    Grounded Policy      Structured Match      Confirmation &
         Answer          JSON (82% Match)       Booking ID
```

### Complete Workshop Pipeline Components
- **LLM Setup:** `ChatOllama` utilizing local `qwen3:0.6b`
- **Document Loading:** `PyPDFLoader` loading `data/handbook.pdf`, `data/job_description.pdf`, `data/resume_candidate_1.pdf`
- **Text Splitting:** `RecursiveCharacterTextSplitter` preserving distinct metadata tags
- **Embeddings:** `OllamaEmbeddings` powered by local `qwen3-embedding:0.6b` (1024-dimensional vectors)
- **Vector Store:** `Chroma` with **3 persistent, isolated collections**:
  - `hirewise_policy`
  - `hirewise_resume`
  - `hirewise_job_description`
- **Dedicated Retrievers:** `policy_retriever`, `resume_retriever`, `jd_retriever` guaranteeing zero cross-contamination
- **Custom Tools:** LangChain `@tool` decorated functions (`policy_qa`, `resume_to_jd_match`, `interview_slot_lookup`, `interview_slot_booking`)
- **Agent:** ReAct Agent loop with transparent execution tracing
- **Frontend:** Streamlit enterprise SaaS dashboard with responsive layout and clean typography

---

## 📂 Project Structure

```text
HireWise/
├── app.py                      # Main Streamlit SaaS Application
├── verify_setup.py             # Pre-flight diagnostic verification script
├── test_scenarios.py           # Automated end-to-end demo test suite
├── generate_data.py            # PDF generator producing realistic mock assets
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & pitch
├── .env.example                # Template for optional Adzuna credentials
├── .gitignore                  # Git ignore rules (protects .env and chroma_db)
│
├── data/                       # Mock PDF Documents
│   ├── handbook.pdf            # HireWise Technologies Employee Handbook & Benefits
│   ├── job_description.pdf     # Software Engineer Role Specification
│   └── resume_candidate_1.pdf  # Candidate 1 Fictional Resume
│
├── src/                        # Modular Source Code
│   ├── __init__.py
│   ├── config.py               # Central configuration & constants
│   ├── llm.py                  # ChatOllama initialization & health checks
│   ├── loaders.py              # PyPDFLoader wrapper with metadata tags
│   ├── text_splitters.py       # RecursiveCharacterTextSplitter logic
│   ├── embeddings.py           # OllamaEmbeddings setup (1024-dim vectors)
│   ├── vectorstore.py          # Chroma multi-collection persistent storage
│   ├── retrievers.py           # Dedicated isolated retrievers
│   ├── prompts.py              # Prompts enforcing strict data boundaries
│   ├── tools.py                # @tool decorated custom functions
│   ├── agent.py                # ReAct agent & transparent execution tracer
│   └── adzuna.py               # Optional Adzuna Job Search API client
│
└── chroma_db/                  # Local persistent Chroma vector database
```

---

## 🔒 Data Isolation & Zero Data Bleed Guarantee

| Workflow | Allowed Collection / Tool | Disallowed Data | Protection Mechanism |
| :--- | :--- | :--- | :--- |
| **Candidate FAQ** | `hirewise_policy` (`handbook.pdf`) | Resumes, JD, Calendar | Isolated Chroma collection; system prompt prohibits candidate retrieval |
| **Resume Screening** | `hirewise_resume` + `hirewise_job_description` | Company Handbook Policies | Tool queries only resume/JD collections; never cites company policy |
| **Interview Scheduling** | `interview_slot_lookup` / `interview_slot_booking` | All Vector Stores | Tool interacts exclusively with calendar store; zero vector querying |

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- [Ollama](https://ollama.com/) running locally

### 2. Pull Required Ollama Models
Ensure Ollama is running, then pull the LLM and embedding models:
```bash
ollama pull qwen3:0.6b
ollama pull qwen3-embedding:0.6b
```
*(Optional alternative embedding model: `ollama pull nomic-embed-text`)*

### 3. Clone Repository & Install Dependencies
```bash
# Clone the repository
git clone <your-repo-url>
cd "hackathon agentic ai"

# Create virtual environment (optional but recommended)
python -m venv .venv

# Windows activation:
.venv\Scripts\activate
# Linux/macOS activation:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Generate Mock PDFs & Run Diagnostics
```bash
# Generate mock handbook, job description, and resume PDFs
python generate_data.py

# Run the complete diagnostic pre-flight verification
python verify_setup.py
```

Expected diagnostic output:
```text
============================================================
      HireWise — Pre-Flight Diagnostic Verification
============================================================
[OK] Ollama Service is running (3 models found)
[OK] LLM (qwen3:0.6b): operational
[OK] Embeddings (qwen3-embedding:0.6b): operational (1024-dim vectors)
[OK] PDF Asset (Handbook): handbook.pdf exists
[OK] PDF Asset (Job Description): job_description.pdf exists
[OK] PDF Asset (Resume Candidate 1): resume_candidate_1.pdf exists
[OK] PyPDFLoader & Metadata Tagging: policy, job_description, resume verified
[OK] RecursiveCharacterTextSplitter: produced 9 handbook chunks
[OK] Chroma Vector Store (3 Isolated Collections):
     • hirewise_policy: 9 chunks indexed
     • hirewise_resume: 6 chunks indexed
     • hirewise_job_description: 6 chunks indexed
[OK] Dedicated Retrievers & Data Isolation: Zero cross-collection data bleed verified
[OK] Custom Tool (interview_slot_lookup): 3 slots found for 2026-09-10
[OK] Custom Tool (resume_to_jd_match): Structured match produced (82% match)
[OK] Custom Tool (policy_qa RAG): operational
[OK] HireWise Agent & Execution Tracer: Correctly routed to Candidate FAQ
============================================================
 [SUCCESS] ALL PRE-FLIGHT CHECKS PASSED! HIREWISE IS READY TO RUN.
============================================================
```

### 5. Launch the Streamlit SaaS Dashboard
```bash
streamlit run app.py
# Or if 'streamlit' is not in your global PATH:
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎯 Demo Scenarios & Judging Guide

HireWise includes built-in one-click demo triggers directly on the **Dashboard** page for effortless presentations:

### Scenario 1 — Candidate FAQ (Policy RAG)
- **Question:** *"What is the company's work-from-home policy?"*
- **Agent Route:** `Candidate FAQ` → `policy_qa` → `policy_retriever` → `hirewise_policy`
- **Output:** *"The company's work-from-home policy allows eligible full-time employees to work remotely up to 2 days per week with manager approval."*
- **Badge:** `Source: Company Handbook`

### Scenario 2 — Resume Screening (Structured Match)
- **Prompt:** *"Compare Candidate 1's resume with the Software Engineer job description."*
- **Agent Route:** `Resume Screening` → `resume_to_jd_match` → Dual Retrievers (`hirewise_resume` + `hirewise_job_description`)
- **Output:**
  - **Match Score:** 82%
  - **Matched Skills:** Python, Java, SQL, Git, REST APIs, Data Structures & Algorithms
  - **Missing Skills:** Docker, AWS, CI/CD
  - **Relevant Experience:** 6-month Software Engineering Internship with FastAPI backend development and PostgreSQL optimization.
  - **Summary:** High alignment with core backend fundamentals; gaps identified in cloud/containerization requirements.

### Scenario 3 — Interview Scheduling & Booking
- **Lookup Query:** *"What interview slots are available tomorrow?"*
  - **Output:** Available slots for `2026-09-10`: `10:00 AM`, `2:00 PM`, `4:00 PM`.
- **Booking Query:** *"Book the 2:00 PM slot tomorrow for Candidate 1."*
  - **Output:** *"✓ Interview booked successfully for Candidate 1 on 2026-09-10 at 2:00 PM."* (Assigned Booking ID: `HW-BOOK-101`).

---

## 🌐 Optional: Adzuna Live Job Discovery Integration

HireWise supports optional live market job discovery via the Adzuna API without requiring it for core operations:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Populate your Adzuna credentials:
   ```env
   ADZUNA_APP_ID=your_adzuna_app_id
   ADZUNA_APP_KEY=your_adzuna_app_key
   ADZUNA_COUNTRY=in
   ```
3. Navigate to **Live Job Discovery** in the dashboard to search live open positions across industries.

*(If credentials are not provided, the page displays a helpful configuration guide and does not crash.)*

---

## 🔍 Agent Transparency Drawer

During judging demonstrations, each response includes an expandable drawer titled **"How HireWise handled this request"**, showing:
- **Intent Identified:** (e.g. `Candidate FAQ`, `Resume Screening`, `Interview Scheduling`)
- **Tool Invoked:** (`policy_qa`, `resume_to_jd_match`, `interview_slot_lookup`, `interview_slot_booking`)
- **Retriever Bound:** Specific Chroma collection queried
- **Sources Cited:** Verifiable PDF file basenames
- **Execution Steps:** Step-by-step trace of agent reasoning

---

## 🛠️ Troubleshooting

- **Ollama connection failed:**
  Make sure Ollama is started: `ollama serve` in a terminal window.
- **Model not found:**
  Run `ollama pull qwen3:0.6b` and `ollama pull qwen3-embedding:0.6b`.
- **Port 8501 in use:**
  Run `streamlit run app.py --server.port 8502`.
- **Reset Calendar:**
  Click the "Reset Calendar to Default Slots" button on the Interview Scheduling tab.

---

## 👥 Hackathon Team & Acknowledgments

Developed for the **1-Day Agentic AI Hackathon**.  
Powered by **LangChain**, **ChromaDB**, **Ollama**, and **Streamlit**.
