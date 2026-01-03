"""Fetch datasets from various sources."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import requests

# Sample datasets that can be fetched
SAMPLE_DATASETS = {
    "iris": {
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        "description": "Classic Iris flower dataset (150 rows, 5 columns)",
    },
    "titanic": {
        "url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "description": "Titanic passenger data (891 rows, 12 columns)",
    },
    "tips": {
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
        "description": "Restaurant tips dataset (244 rows, 7 columns)",
    },
    "mpg": {
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv",
        "description": "Auto MPG dataset (398 rows, 9 columns)",
    },
    "penguins": {
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
        "description": "Palmer Penguins dataset (344 rows, 7 columns)",
    },
    "diamonds": {
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv",
        "description": "Diamonds dataset (53940 rows, 10 columns)",
    },
    "flights": {
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv",
        "description": "Monthly airline passengers (144 rows, 3 columns)",
    },
    "gapminder": {
        "url": "https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv",
        "description": "Gapminder 2007 world data (142 rows, 6 columns)",
    },
    "stocks": {
        "url": "https://raw.githubusercontent.com/plotly/datasets/master/stockdata.csv",
        "description": "Stock market data",
    },
    "covid": {
        "url": "https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv",
        "description": "COVID-19 cases by country",
    },
}


def list_available_datasets() -> str:
    """Return a formatted list of available datasets."""
    result = "📚 **Available Datasets:**\n\n"
    for name, info in SAMPLE_DATASETS.items():
        result += f"- **{name}**: {info['description']}\n"
    result += "\n*Say 'fetch titanic' or 'get iris dataset' to load one.*"
    return result


async def fetch_dataset(name: str, save_path: Path) -> Dict[str, Any]:
    """Fetch a dataset and save it to disk."""
    name = name.lower().strip()
    
    if name not in SAMPLE_DATASETS:
        return {
            "success": False,
            "error": f"Dataset '{name}' not found. Available: {', '.join(SAMPLE_DATASETS.keys())}",
        }
    
    info = SAMPLE_DATASETS[name]
    
    try:
        response = requests.get(info["url"], timeout=30)
        response.raise_for_status()
        
        # Save to disk
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(response.content)
        
        # Get basic stats
        df = pd.read_csv(io.StringIO(response.text))
        
        return {
            "success": True,
            "name": name,
            "description": info["description"],
            "path": str(save_path),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
        }
    except requests.RequestException as e:
        return {"success": False, "error": f"Failed to fetch: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Error: {str(e)}"}
