"""
src/adzuna.py
Optional Adzuna Job Discovery API integration.
Ensures zero secrets are hardcoded and returns structured job postings.
"""

import os
import requests
from typing import Dict, Any, List, Optional
from src.config import ADZUNA_APP_ID, ADZUNA_APP_KEY, ADZUNA_COUNTRY

def get_adzuna_credentials():
    """Returns the current Adzuna app_id and app_key dynamically."""
    app_id = (os.getenv("ADZUNA_APP_ID") or ADZUNA_APP_ID or "").strip()
    app_key = (os.getenv("ADZUNA_APP_KEY") or ADZUNA_APP_KEY or "").strip()
    return app_id, app_key

def is_adzuna_configured() -> bool:
    """Checks if Adzuna credentials are provided in environment."""
    app_id, app_key = get_adzuna_credentials()
    return bool(app_id and app_key)

def search_adzuna_jobs(
    query: str = "Software Engineer",
    location: str = "India",
    country: Optional[str] = None,
    results_per_page: int = 5
) -> Dict[str, Any]:
    """
    Queries the Adzuna API for live job opportunities.
    Gracefully handles missing keys and HTTP errors.
    """
    app_id, app_key = get_adzuna_credentials()
    if not (app_id and app_key):
        return {
            "success": False,
            "configured": False,
            "message": "Live job discovery is unavailable. Add ADZUNA_APP_ID and ADZUNA_APP_KEY to .env to enable it.",
            "jobs": []
        }

    target_country = country or os.getenv("ADZUNA_COUNTRY") or ADZUNA_COUNTRY or "in"
    url = f"https://api.adzuna.com/v1/api/jobs/{target_country}/search/1"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": query,
        "where": location,
        "results_per_page": results_per_page,
        "content-type": "application/json"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            raw_results = data.get("results", [])
            formatted_jobs = []
            for item in raw_results:
                salary_min = item.get("salary_min")
                salary_max = item.get("salary_max")
                salary_str = "Not specified"
                if salary_min and salary_max:
                    salary_str = f"₹{salary_min:,.0f} - ₹{salary_max:,.0f}" if target_country == "in" else f"${salary_min:,.0f} - ${salary_max:,.0f}"
                elif salary_min:
                    salary_str = f"From ₹{salary_min:,.0f}" if target_country == "in" else f"From ${salary_min:,.0f}"

                formatted_jobs.append({
                    "id": item.get("id"),
                    "title": item.get("title", "Software Engineer").replace("<strong>", "").replace("</strong>", ""),
                    "company": item.get("company", {}).get("display_name", "Tech Company"),
                    "location": item.get("location", {}).get("display_name", location),
                    "description": item.get("description", "No description provided.")[:280] + "...",
                    "salary": salary_str,
                    "redirect_url": item.get("redirect_url", "#"),
                    "created": item.get("created", "")[:10]
                })

            return {
                "success": True,
                "configured": True,
                "total_count": data.get("count", len(formatted_jobs)),
                "message": f"Found {len(formatted_jobs)} live job postings matching '{query}'.",
                "jobs": formatted_jobs
            }
        else:
            return {
                "success": False,
                "configured": True,
                "message": f"Adzuna API returned status {response.status_code}: {response.text}",
                "jobs": []
            }
    except Exception as e:
        return {
            "success": False,
            "configured": True,
            "message": f"Error contacting Adzuna API: {str(e)}",
            "jobs": []
        }
