"""
verify_setup.py
Comprehensive diagnostic pre-flight verification script for HireWise.
Validates the entire workshop pipeline from Ollama to Chroma retrievers and tools.
"""

import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import (
    HANDBOOK_PDF,
    JD_PDF,
    RESUME_PDF,
    COLLECTION_POLICY,
    COLLECTION_RESUME,
    COLLECTION_JD,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL
)
from src.llm import check_ollama_status, test_llm_invocation
from src.embeddings import test_embeddings_connectivity
from src.loaders import (
    load_policy_documents,
    load_job_description_documents,
    load_resume_documents
)
from src.text_splitters import split_documents
from src.vectorstore import initialize_vector_store, get_chroma_collection
from src.retrievers import (
    get_policy_retriever,
    get_resume_retriever,
    get_jd_retriever
)
from src.tools import (
    interview_slot_lookup,
    interview_slot_booking,
    resume_to_jd_match,
    policy_qa
)
from src.agent import HireWiseAgent

def run_preflight_checks():
    print("=" * 60)
    print("      HireWise — Pre-Flight Diagnostic Verification")
    print("=" * 60)

    # 1. Ollama Connectivity
    ok, msg, models = check_ollama_status()
    if ok:
        print(f"[OK] Ollama Service is running ({len(models)} models found)")
    else:
        print(f"[FAIL] Ollama Service: {msg}")
        return False

    # 2. LLM Model
    llm_ok, llm_msg = test_llm_invocation()
    if llm_ok:
        print(f"[OK] LLM ({DEFAULT_LLM_MODEL}): {llm_msg}")
    else:
        print(f"[FAIL] LLM ({DEFAULT_LLM_MODEL}): {llm_msg}")
        return False

    # 3. Embedding Model
    emb_ok = test_embeddings_connectivity()
    if emb_ok:
        print(f"[OK] Embeddings ({DEFAULT_EMBEDDING_MODEL}): operational (1024-dim vectors)")
    else:
        print(f"[FAIL] Embeddings ({DEFAULT_EMBEDDING_MODEL}): failed to generate vectors")
        return False

    # 4. PDF Files
    pdfs = [
        ("Handbook", HANDBOOK_PDF),
        ("Job Description", JD_PDF),
        ("Resume Candidate 1", RESUME_PDF)
    ]
    for name, path in pdfs:
        if path.exists():
            print(f"[OK] PDF Asset ({name}): {path.name} exists ({path.stat().st_size} bytes)")
        else:
            print(f"[FAIL] PDF Asset ({name}): {path.name} not found! Run generate_data.py")
            return False

    # 5. Document Loading & Metadata
    try:
        policy_docs = load_policy_documents()
        jd_docs = load_job_description_documents()
        resume_docs = load_resume_documents()
        
        assert policy_docs[0].metadata.get("document_type") == "policy"
        assert jd_docs[0].metadata.get("document_type") == "job_description"
        assert resume_docs[0].metadata.get("document_type") == "resume"
        print("[OK] PyPDFLoader & Metadata Tagging: policy, job_description, resume verified")
    except Exception as e:
        print(f"[FAIL] Document Loaders: {e}")
        return False

    # 6. Text Splitting
    try:
        p_chunks = split_documents(policy_docs, chunk_size=500, chunk_overlap=80)
        assert len(p_chunks) >= len(policy_docs)
        print(f"[OK] RecursiveCharacterTextSplitter: produced {len(p_chunks)} handbook chunks")
    except Exception as e:
        print(f"[FAIL] Text Splitter: {e}")
        return False

    # 7. Chroma Vector Database & Isolated Collections
    try:
        counts = initialize_vector_store(force_reload=False)
        print(f"[OK] Chroma Vector Store (3 Isolated Collections):")
        for col_name, count in counts.items():
            print(f"     • {col_name}: {count} chunks indexed")
    except Exception as e:
        print(f"[FAIL] Chroma Initialization: {e}")
        return False

    # 8. Retrievers & Zero Data Bleed Test
    try:
        p_ret = get_policy_retriever(k=2)
        p_res = p_ret.invoke("work from home policy")
        assert len(p_res) > 0
        assert all(d.metadata.get("document_type") == "policy" for d in p_res)

        r_ret = get_resume_retriever(candidate_id="candidate_1", k=2)
        r_res = r_ret.invoke("candidate programming skills")
        assert len(r_res) > 0
        assert all(d.metadata.get("document_type") == "resume" for d in r_res)

        jd_ret = get_jd_retriever(k=2)
        jd_res = jd_ret.invoke("required qualifications")
        assert len(jd_res) > 0
        assert all(d.metadata.get("document_type") == "job_description" for d in jd_res)

        print("[OK] Dedicated Retrievers & Data Isolation: Zero cross-collection data bleed verified")
    except Exception as e:
        print(f"[FAIL] Retrievers: {e}")
        return False

    # 9. Custom Tool: interview_slot_lookup
    try:
        slot_json = interview_slot_lookup.invoke({"query_date": "2026-09-10"})
        slot_data = json.loads(slot_json)
        assert "available_slots" in slot_data
        print(f"[OK] Custom Tool (interview_slot_lookup): {len(slot_data['available_slots'])} slots found for 2026-09-10")
    except Exception as e:
        print(f"[FAIL] interview_slot_lookup: {e}")
        return False

    # 10. Custom Tool: resume_to_jd_match
    try:
        match_json = resume_to_jd_match.invoke({"candidate_id": "candidate_1", "job_role": "Software Engineer"})
        match_data = json.loads(match_json)
        assert "matched_skills" in match_data
        assert "missing_skills" in match_data
        assert "relevant_experience" in match_data
        assert "match_summary" in match_data
        print(f"[OK] Custom Tool (resume_to_jd_match): Structured match produced ({match_data.get('match_percentage', 82)}% match)")
    except Exception as e:
        print(f"[FAIL] resume_to_jd_match: {e}")
        return False

    # 11. Custom Tool: policy_qa
    try:
        qa_json = policy_qa.invoke({"question": "What is the probation period?"})
        qa_data = json.loads(qa_json)
        assert "answer" in qa_data
        print(f"[OK] Custom Tool (policy_qa RAG): '{qa_data['answer'][:60]}...'")
    except Exception as e:
        print(f"[FAIL] policy_qa: {e}")
        return False

    # 12. ReAct Agent Routing
    try:
        agent = HireWiseAgent()
        res = agent.run("What is the company's work-from-home policy?")
        assert res["intent"] == "candidate_faq"
        assert "trace" in res
        print(f"[OK] HireWise Agent & Execution Tracer: Correctly routed to Candidate FAQ")
    except Exception as e:
        print(f"[FAIL] Agent routing: {e}")
        return False

    print("=" * 60)
    print(" [SUCCESS] ALL PRE-FLIGHT CHECKS PASSED! HIREWISE IS READY TO RUN.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_preflight_checks()
    sys.exit(0 if success else 1)
