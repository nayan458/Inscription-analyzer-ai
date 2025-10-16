# utils.py
import os
import wikipediaapi
import json
import logging
import time
from pathlib import Path
from requests.exceptions import RequestException
from retry import retry

# Create a portable logs directory relative to the script
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "wikipedia_errors.log"

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Wikipedia API
wiki = wikipediaapi.Wikipedia(
    user_agent='MyWikiApp/1.0 (abc@gmail.com)',
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

def sections_to_dict(sections, level=0):
    """Recursively convert sections to dictionary format."""
    section_list = []
    for section in sections:
        section_dict = {
            'title': section.title,
            'text': section.text[:200] + '...' if len(section.text) > 200 else section.text,
            'level': level,
            'subsections': sections_to_dict(section.sections, level + 1)
        }
        section_list.append(section_dict)
    return section_list

@retry(RequestException, tries=3, delay=2, backoff=2)
def fetch_page(title: str):
    """Fetch a Wikipedia page with retry logic."""
    try:
        page = wiki.page(title)
        if page.exists():
            return page
        else:
            logging.error(f"Page '{title}' does not exist")
            return None
    except Exception as e:
        logging.error(f"Error fetching page '{title}': {type(e).__name__}: {e}")
        raise

def get_wiki_data(title: str, save_to_file: bool = False, output_dir: str = None):
    """Main function to fetch and format Wikipedia data."""
    page = fetch_page(title)
    if page and page.exists():
        try:
            data = {
                'title': page.title,
                'summary': page.summary,
                'url': page.fullurl,
                'sections': sections_to_dict(page.sections),
                'links': list(page.links)[:5],
                'categories': list(page.categories.keys())
            }
            result = {"status": "success", "data": data}
            
            if save_to_file:
                # Use a portable data directory
                data_dir = Path(__file__).parent / "data"
                data_dir.mkdir(parents=True, exist_ok=True)
                save_to_json(result, title.replace(' ', '_') + '.json', str(data_dir))
            
            return result
        except Exception as e:
            logging.error(f"Error preparing data for '{title}': {type(e).__name__}: {e}")
            return {"status": "error", "message": f"Failed to prepare data for {title}"}
    else:
        return {"status": "error", "message": f"Page '{title}' does not exist"}

# Save to JSON file with fallback
def save_to_json(data, filename, output_dir=None, retries=3):
    if output_dir:
        base_dir = Path(output_dir)
    else:
        # Default to a portable data directory
        base_dir = Path(__file__).parent / "data"
    fallback_dir = Path(__file__).parent / "temp_data"
    output_file = base_dir / filename

    for attempt in range(retries):
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if not os.access(output_file.parent, os.W_OK):
                raise PermissionError(f"No write access to {output_file.parent}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logging.info(f"Data successfully saved to {output_file}")
            return True
        except (PermissionError, OSError) as e:
            logging.error(f"Attempt {attempt + 1}/{retries} - Error saving to {output_file}: {type(e).__name__}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                output_file = fallback_dir / filename
                try:
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                    logging.info(f"Data successfully saved to fallback {output_file}")
                    return True
                except Exception as e2:
                    logging.error(f"Error saving to fallback {output_file}: {type(e2).__name__}: {e2}")
                    return False

# Respect rate limits
def respect_rate_limit():
    time.sleep(1)