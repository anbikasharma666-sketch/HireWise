"""
test_scenarios.py
Automated end-to-end verification of the 3 required Hackathon Demo Scenarios.
"""

import sys
import json
from pathlib import Path

# Ensure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agent import HireWiseAgent

def run_scenarios():
    print("=" * 65)
    print("        HireWise — 3 Hackathon Demo Scenarios Test")
    print("=" * 65)

    agent = HireWiseAgent()

    # -------------------------------------------------------------
    # DEMO 1: FAQ Question
    # -------------------------------------------------------------
    print("\n--- [DEMO 1] Candidate FAQ ---")
    query_1 = "What is the company's work-from-home policy?"
    print(f"User Query: '{query_1}'")
    res_1 = agent.run(query_1)
    print(f"Agent Intent: {res_1['intent']}")
    print(f"Source Cited: {res_1['source']}")
    print(f"Answer:\n{res_1['answer']}")
    assert res_1['intent'] == "candidate_faq"
    assert "2 days" in res_1['answer'] or "manager" in res_1['answer'].lower() or "hybrid" in res_1['answer'].lower()
    print("[✓] DEMO 1 PASSED!")

    # -------------------------------------------------------------
    # DEMO 2: Resume Screening
    # -------------------------------------------------------------
    print("\n--- [DEMO 2] Resume Screening ---")
    query_2 = "Compare Candidate 1's resume with the Software Engineer job description."
    print(f"User Query: '{query_2}'")
    res_2 = agent.run(query_2)
    print(f"Agent Intent: {res_2['intent']}")
    print(f"Source Cited: {res_2['source']}")
    print(f"Answer Summary:\n{res_2['answer']}")
    match_data = res_2.get('match_data', {})
    assert res_2['intent'] == "resume_screening"
    assert "matched_skills" in match_data
    assert "missing_skills" in match_data
    assert any("python" in s.lower() for s in match_data['matched_skills'])
    assert any("docker" in s.lower() or "aws" in s.lower() for s in match_data['missing_skills'])
    print("[✓] DEMO 2 PASSED!")

    # -------------------------------------------------------------
    # DEMO 3: Interview Scheduling
    # -------------------------------------------------------------
    print("\n--- [DEMO 3] Interview Scheduling Lookup ---")
    query_3 = "What interview slots are available tomorrow?"
    print(f"User Query: '{query_3}'")
    res_3 = agent.run(query_3)
    print(f"Agent Intent: {res_3['intent']}")
    print(f"Source Cited: {res_3['source']}")
    print(f"Answer:\n{res_3['answer']}")
    assert res_3['intent'] == "interview_scheduling"
    slot_data = res_3.get('slot_data', {})
    assert len(slot_data.get('available_slots', [])) > 0
    print("[✓] DEMO 3 PASSED!")

    # -------------------------------------------------------------
    # BONUS: Interview Slot Booking
    # -------------------------------------------------------------
    print("\n--- [BONUS] Interview Slot Booking ---")
    query_4 = "Book the 2:00 PM slot tomorrow for Candidate 1"
    print(f"User Query: '{query_4}'")
    res_4 = agent.run(query_4)
    print(f"Booking Output: {res_4['answer']}")
    assert res_4.get('booking_data', {}).get('success') is True
    print("[✓] INTERVIEW BOOKING PASSED!")

    print("\n" + "=" * 65)
    print(" [SUCCESS] ALL DEMO SCENARIOS FULLY VERIFIED AND WORKING!")
    print("=" * 65)

if __name__ == "__main__":
    run_scenarios()
