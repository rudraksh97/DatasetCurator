"""
Dataset Search using DuckDuckGo (ddgs)

SEARCH LOGIC:
1. User asks "find population data of India"
2. We search DuckDuckGo with CSV-focused queries
3. Filter results to prioritize direct CSV links and GitHub repos
4. User selects by number
5. We download and validate the CSV
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import requests

# Cache: dataset_id -> list of search results
_search_cache: Dict[str, List[Dict[str, Any]]] = {}


def search_datasets(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """Search for CSV datasets using DuckDuckGo."""
    results = []
    seen_urls = set()
    
    print(f"[SEARCH] Starting search for: {query}")
    
    try:
        from ddgs import DDGS
        
        search_queries = [
            f"{query} csv github",
            f"{query} dataset download csv",
            f"{query} data csv",
        ]
        
        ddgs = DDGS()
        
        for search_query in search_queries:
            print(f"[SEARCH] Query: {search_query}")
            
            try:
                search_results = ddgs.text(search_query, max_results=10)
                
                if search_results:
                    for r in search_results:
                        url = r.get("href", "") or r.get("link", "")
                        title = r.get("title", "Unknown")
                        body = r.get("body", "") or r.get("snippet", "")
                        
                        if not url or url in seen_urls:
                            continue
                        seen_urls.add(url)
                        
                        # Check if it's a direct CSV or GitHub
                        is_github_csv = ("github.com" in url and ".csv" in url)
                        is_raw_csv = "raw.githubusercontent.com" in url
                        is_direct_csv = url.endswith(".csv") or is_github_csv or is_raw_csv
                        is_github = "github.com" in url
                        is_kaggle = "kaggle.com" in url
                        
                        if is_direct_csv or is_github or is_kaggle or "data" in url.lower():
                            source = "github" if is_github else "kaggle" if is_kaggle else "web"
                            
                            results.append({
                                "title": title[:100],
                                "url": url,
                                "description": body[:200] if body else "",
                                "source": source,
                                "is_direct_csv": is_direct_csv,
                            })
                            print(f"[SEARCH] Added: {title[:50]}... (direct_csv={is_direct_csv})")
                        
                        if len(results) >= max_results:
                            break
                            
            except Exception as e:
                print(f"[SEARCH] Query error: {e}")
                continue
            
            if len(results) >= max_results:
                break
                
    except Exception as e:
        print(f"[SEARCH] DuckDuckGo error: {e}")
        return _fallback_search(query, max_results)
    
    # Sort: direct CSVs first
    results.sort(key=lambda x: (0 if x["is_direct_csv"] else 1))
    
    if not results:
        return _fallback_search(query, max_results)
    
    return results[:max_results]


def _fallback_search(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """Fallback: Search GitHub API directly for CSV files."""
    results = []
    print(f"[FALLBACK] Searching GitHub for: {query}")
    
    try:
        headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "DatasetCurator/1.0"}
        params = {"q": f"{query} extension:csv", "per_page": max_results}
        
        response = requests.get("https://api.github.com/search/code", headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            for item in response.json().get("items", []):
                repo = item.get("repository", {})
                path = item.get("path", "")
                full_name = repo.get("full_name", "")
                
                # Build raw URL
                raw_url = f"https://raw.githubusercontent.com/{full_name}/main/{path}"
                
                results.append({
                    "title": f"{repo.get('name', 'Unknown')}: {path}",
                    "url": raw_url,
                    "description": repo.get("description", f"CSV from {full_name}")[:200],
                    "source": "github",
                    "is_direct_csv": True,
                })
                
                if len(results) >= max_results:
                    break
                    
    except Exception as e:
        print(f"[FALLBACK] Error: {e}")
    
    return results


def cache_search_results(dataset_id: str, results: List[Dict[str, Any]]) -> None:
    """Store results so user can select by number later."""
    _search_cache[dataset_id] = results


def get_cached_results(dataset_id: str) -> Optional[List[Dict[str, Any]]]:
    """Retrieve cached search results."""
    return _search_cache.get(dataset_id)


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """Format results as a numbered list for the user."""
    if not results:
        return "No datasets found. Try simpler keywords like 'weather data' or 'stock prices'."
    
    output = "**Found these datasets:**\n\n"
    
    for i, r in enumerate(results, 1):
        csv_badge = " `CSV`" if r["is_direct_csv"] else ""
        title = r["title"][:55] + "..." if len(r["title"]) > 55 else r["title"]
        
        output += f"**{i}.**{csv_badge} {title}\n"
        
        if r["description"]:
            desc = r["description"][:70] + "..." if len(r["description"]) > 70 else r["description"]
            output += f"   _{desc}_\n"
        output += "\n"
    
    output += "*Reply with a number (e.g., '1') to load that dataset.*"
    
    return output


def _github_to_raw_url(url: str) -> str:
    """
    Convert ANY GitHub URL to a raw download URL.
    
    Examples:
    - github.com/user/repo/blob/main/file.csv -> raw.githubusercontent.com/user/repo/main/file.csv
    - github.com/user/repo/tree/main/data -> (repo URL, need to search)
    """
    # Already a raw URL
    if "raw.githubusercontent.com" in url:
        return url
    
    # GitHub blob URL (direct file view)
    if "/blob/" in url:
        # Extract the path after blob and convert
        url = url.replace("github.com", "raw.githubusercontent.com")
        url = url.replace("/blob/", "/")
        return url
    
    return url


def _try_download_url(url: str) -> Optional[str]:
    """Try to download from a URL, return content if successful."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 DatasetCurator/1.0"}
        response = requests.get(url, timeout=15, headers=headers)
        
        if response.status_code == 200:
            content = response.text
            # Make sure it's not HTML
            if not content.strip().startswith("<!") and not content.strip().startswith("<html"):
                return content
    except Exception:
        pass
    return None


def _find_csv_in_github_repo(url: str) -> Optional[str]:
    """Given a GitHub repo URL, try to find a CSV file inside it."""
    match = re.search(r"github\.com/([^/]+)/([^/\?#]+)", url)
    if not match:
        return None
    
    owner, repo = match.groups()
    repo = repo.rstrip("/")
    
    print(f"[DOWNLOAD] Searching repo {owner}/{repo} for CSV files...")
    
    for branch in ["main", "master"]:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
        
        try:
            resp = requests.get(api_url, params={"ref": branch}, timeout=10)
            
            if resp.status_code != 200:
                continue
                
            files = resp.json()
            if not isinstance(files, list):
                continue
            
            # Look for CSV files in root
            for f in files:
                if f.get("name", "").endswith(".csv"):
                    download_url = f.get("download_url")
                    print(f"[DOWNLOAD] Found: {f.get('name')}")
                    return download_url
            
            # Check common data folders
            for folder_name in ["data", "dataset", "datasets", "csv", "Data"]:
                for f in files:
                    if f.get("name", "").lower() == folder_name.lower() and f.get("type") == "dir":
                        folder_resp = requests.get(f.get("url", ""), timeout=10)
                        if folder_resp.status_code == 200:
                            folder_files = folder_resp.json()
                            if isinstance(folder_files, list):
                                for ff in folder_files:
                                    if ff.get("name", "").endswith(".csv"):
                                        return ff.get("download_url")
                                        
        except Exception as e:
            print(f"[DOWNLOAD] API error: {e}")
            continue
    
    return None


async def download_from_search_result(result: Dict[str, Any], save_path: Path) -> Dict[str, Any]:
    """Download a dataset from a search result."""
    url = result["url"]
    print(f"[DOWNLOAD] Original URL: {url}")
    
    content = None
    download_url = url
    
    # Strategy 1: If it looks like a direct CSV link, convert and download
    if result["is_direct_csv"]:
        raw_url = _github_to_raw_url(url)
        print(f"[DOWNLOAD] Trying raw URL: {raw_url}")
        content = _try_download_url(raw_url)
        
        # Try alternate branch if main doesn't work
        if not content and "/main/" in raw_url:
            alt_url = raw_url.replace("/main/", "/master/")
            print(f"[DOWNLOAD] Trying master branch: {alt_url}")
            content = _try_download_url(alt_url)
            if content:
                download_url = alt_url
        elif content:
            download_url = raw_url
    
    # Strategy 2: If it's a GitHub repo, search for CSV inside
    if not content and result["source"] == "github":
        csv_url = _find_csv_in_github_repo(url)
        if csv_url:
            print(f"[DOWNLOAD] Found CSV in repo: {csv_url}")
            content = _try_download_url(csv_url)
            if content:
                download_url = csv_url
    
    # Strategy 3: Try the original URL directly
    if not content:
        print(f"[DOWNLOAD] Trying original URL directly...")
        content = _try_download_url(url)
        if content:
            download_url = url
    
    # Handle Kaggle
    if not content and result["source"] == "kaggle":
        return {
            "success": False,
            "error": "Kaggle requires login. Please download manually and upload via the attachment button.",
        }
    
    # No content found
    if not content:
        return {
            "success": False,
            "error": "Could not download CSV from this URL. Try a different result.",
        }
    
    # Validate CSV
    try:
        df = pd.read_csv(io.StringIO(content))
        if len(df.columns) < 2:
            return {"success": False, "error": "File has too few columns to be a dataset."}
        print(f"[DOWNLOAD] Valid CSV: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        return {"success": False, "error": f"Could not parse CSV: {str(e)[:80]}"}
    
    # Save to disk
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(content)
    
    return {
        "success": True,
        "name": result["title"],
        "url": download_url,
        "path": str(save_path),
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
    }
