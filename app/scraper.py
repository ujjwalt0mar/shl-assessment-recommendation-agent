from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict
import argparse

import requests
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)


def scrape_shl_catalog(catalog_url: str, timeout: int = 30) -> List[Dict[str, str]]:
    """
    Scrape SHL catalog page and return normalized assessment records.

    Note: HTML structure may change over time; selectors are intentionally defensive.
    """
    response = requests.get(catalog_url, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    cards = soup.select("article, .product-card, .catalog-card, .assessment-card")

    results: List[Dict[str, str]] = []
    for card in cards:
        title_el = card.select_one("h2, h3, .title, .product-title")
        link_el = card.select_one("a[href]")
        desc_el = card.select_one("p, .description, .summary")

        if not title_el or not link_el:
            continue

        name = title_el.get_text(" ", strip=True)
        url = link_el.get("href", "").strip()
        if url.startswith("/"):
            from urllib.parse import urljoin

            url = urljoin(catalog_url, url)

        description = desc_el.get_text(" ", strip=True) if desc_el else ""

        skills = []
        skills_el = card.select(".skills li, .tags li, .skill")
        for node in skills_el:
            text = node.get_text(" ", strip=True)
            if text:
                skills.append(text)

        duration_el = card.select_one(".duration, .time")
        test_type_el = card.select_one(".test-type, .category")

        item = {
            "name": name,
            "url": url,
            "description": description,
            "skills": skills,
            "duration": duration_el.get_text(" ", strip=True) if duration_el else None,
            "test_type": test_type_el.get_text(" ", strip=True) if test_type_el else "Unknown",
        }
        results.append(item)

    logger.info("Scraped %d catalog items", len(results))
    return results


def save_catalog(items: List[Dict[str, str]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=2, ensure_ascii=False)
    logger.info("Saved catalog to %s", destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape SHL catalog and save JSON dataset.")
    parser.add_argument("--url", required=True, help="SHL catalog URL")
    parser.add_argument(
        "--output",
        default="data/assessments.json",
        help="Output JSON path for cleaned catalog data",
    )
    args = parser.parse_args()

    items = scrape_shl_catalog(args.url)
    save_catalog(items, Path(args.output))


if __name__ == "__main__":
    main()
