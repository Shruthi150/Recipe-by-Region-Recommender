from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import shutil
import tempfile

from icrawler.builtin import BingImageCrawler
from recipe_recommender import load_recipe_data, RecipeRegionRecommender
from recipe_image_service import IMAGE_CACHE_DIR, slugify, style_image


def _crawl_first_image(crawler_class: type, keyword: str, temp_dir: str) -> Path | None:
    crawler = crawler_class(storage={"root_dir": temp_dir})
    crawler.crawl(keyword=keyword, max_num=1)
    downloaded = sorted(Path(temp_dir).glob("*"))
    return downloaded[0] if downloaded else None


def collect_one_image(recipe_name: str, course: str, region: str) -> tuple[bool, Path, str]:
    slug = slugify(recipe_name)
    output_path = IMAGE_CACHE_DIR / f"{slug}.jpg"

    with tempfile.TemporaryDirectory() as temp_dir:
        bing_queries = [
            f"{recipe_name} indian dish",
            f"{recipe_name} {course} indian food",
            f"{recipe_name} {region} food",
            f"{recipe_name} recipe photo",
        ]

        source = None
        provider = "none"

        for query in bing_queries:
            try:
                source = _crawl_first_image(BingImageCrawler, query, temp_dir)
                if source is not None:
                    provider = "bing"
                    break
            except Exception:
                continue

        if source is None:
            try:
                source = _crawl_first_image(BingImageCrawler, "indian food platter", temp_dir)
                if source is not None:
                    provider = "generic"
            except Exception:
                source = None

        if source is None:
            return False, output_path, "failed"

        shutil.copyfile(source, output_path)
        style_image(output_path)
        return True, output_path, provider


def main() -> None:
    logging.getLogger("icrawler").setLevel(logging.ERROR)

    raw = load_recipe_data()
    model = RecipeRegionRecommender.fit(raw)
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for existing in IMAGE_CACHE_DIR.glob("*.jpg"):
        existing.unlink()

    total = len(model.data)
    print(f"Collecting images for {total} recipes...")

    completed = 0
    success = 0
    from_bing = 0
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
                    if provider == "bing":
                        from_bing += 1
                    if provider == "generic":
                        from_generic += 1
                    print(f"[{completed}/{total}] #{index}: {path.name} ({provider})")
                else:
                    failed += 1
                    print(f"[{completed}/{total}] #{index}: {path.name} (failed)")
            except Exception as error:
                failed += 1
                print(f"[{completed}/{total}] #{index}: failed - {error}")

    print(
        f"Collected {completed} recipe images ({success} downloaded). "
        f"Bing: {from_bing}, Generic fallback: {from_generic}, Failed: {failed}."
    )


if __name__ == "__main__":
    main()