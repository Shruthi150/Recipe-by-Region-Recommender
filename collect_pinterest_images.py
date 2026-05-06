from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import shutil
import tempfile

from icrawler.builtin import BingImageCrawler
from recipe_recommender import load_recipe_data, RecipeRegionRecommender
from recipe_image_service import slugify, style_image


# --- Define the new output folder for Pinterest images ---
BASE_DIR = Path(__file__).resolve().parent
PINTEREST_CACHE_DIR = BASE_DIR / "assets" / "pinterest_images"
# ---------------------------------------------------------


def _crawl_first_image(crawler_class: type, keyword: str, temp_dir: str) -> Path | None:
    crawler = crawler_class(storage={"root_dir": temp_dir})
    crawler.crawl(keyword=keyword, max_num=1)
    downloaded = sorted(Path(temp_dir).glob("*"))
    return downloaded[0] if downloaded else None


def collect_one_image(recipe_name: str, course: str, region: str) -> tuple[bool, Path, str]:
    slug = slugify(recipe_name)
    output_path = PINTEREST_CACHE_DIR / f"{slug}.jpg"

    # Skip if we already downloaded it successfully in a previous run
    if output_path.exists() and output_path.stat().st_size > 10000:
        return True, output_path, "cached"

    with tempfile.TemporaryDirectory() as temp_dir:
        # Force the crawler to only look at Pinterest
        queries = [
            f"{recipe_name} site:in.pinterest.com",
            f"{recipe_name} dish site:in.pinterest.com",
            f"{recipe_name} aestheticsite:in.pinterest.com"
        ]

        source = None
        provider = "none"

        for query in queries:
            try:
                source = _crawl_first_image(BingImageCrawler, query, temp_dir)
                if source is not None:
                    provider = "pinterest_search"
                    break
            except Exception:
                continue

        if source is None:
            # Fallback to a generic Pinterest food aesthetic if specific recipe fails
            try:
                source = _crawl_first_image(BingImageCrawler, "indian food aesthetic site:pinterest.com", temp_dir)
                if source is not None:
                    provider = "bing"
            except Exception:
                source = None

        if source is None:
            return False, output_path, "failed"

        # Save and style the image
        shutil.copyfile(source, output_path)
        style_image(output_path)
        return True, output_path, provider


def main() -> None:
    logging.getLogger("icrawler").setLevel(logging.ERROR)

    # Create the new pinterest_images directory
    PINTEREST_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_recipe_data()
    model = RecipeRegionRecommender.fit(raw)
    
    total = len(model.data)
    print(f"Collecting images from Pinterest for {total} recipes...")

    completed = 0
    success = 0
    from_search = 0
    from_generic = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_map = {
            executor.submit(collect_one_image, row["name"], row["course"], row["region_clean"]): (index, row["name"])
            for index, (_, row) in enumerate(model.data.iterrows(), start=1)
        }
        for future in as_completed(future_map):
            completed += 1
            index, recipe_name = future_map[future]
            try:
                downloaded, path, provider = future.result()
                if downloaded:
                    success += 1
                    if provider == "pinterest_search":
                        from_search += 1
                    if provider == "pinterest_generic":
                        from_generic += 1
                    print(f"[{completed}/{total}] #{index}: {path.name} ({provider})")
                else:
                    failed += 1
                    print(f"[{completed}/{total}] #{index}: {path.name} (failed)")
            except Exception as error:
                failed += 1
                print(f"[{completed}/{total}] #{index}: failed - {error}")

    print(
        f"\nFinished! Collected {completed} recipe images ({success} downloaded to {PINTEREST_CACHE_DIR.name}).\n"
        f"Targeted Pinterest Search: {from_search}, Generic Pinterest Fallback: {from_generic}, Failed: {failed}."
    )


if __name__ == "__main__":
    main()