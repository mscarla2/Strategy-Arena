import asyncio
from playwright.async_api import async_playwright
import json

async def scrape_user_watchlists(username):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        url = f"https://www.reddit.com/user/{username}/submitted/"
        await page.goto(url)

        watchlists = []
        seen_ids = set()

        print(f"Waiting for posts from u/{username} (complete captcha if prompted)...")

        # Wait indefinitely until at least one shreddit-post appears
        await page.wait_for_selector('shreddit-post', timeout=0)

        print("Posts detected — beginning incremental scrape...")

        last_height = -1

        while True:
            posts = await page.query_selector_all('shreddit-post')

            for post in posts:
                title = await post.get_attribute('post-title')

                if not (title and "WATCHLIST" in title.upper()):
                    continue

                # Deduplicate by post id, falling back to title::timestamp
                post_id = await post.get_attribute('id') or await post.get_attribute('post-id')
                if not post_id:
                    timestamp = await post.get_attribute('created-timestamp') or ""
                    post_id = f"{title}::{timestamp}"

                if post_id in seen_ids:
                    continue
                seen_ids.add(post_id)

                content_element = await post.query_selector('div[id$="-post-rtjson-content"]')
                if not content_element:
                    content_element = await post.query_selector('.md')

                content_text = await content_element.inner_text() if content_element else ""

                entry = {
                    "title": title,
                    "content": content_text.strip(),
                    "timestamp": await post.get_attribute('created-timestamp'),
                }
                watchlists.append(entry)
                print(f"  Found: {title}")

            # Scroll to load more posts
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(0.5)

            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                # No new content loaded — we've reached the end
                break
            last_height = new_height

        await browser.close()
        return watchlists


if __name__ == "__main__":
    import os

    OUTPUT = "scraped_watchlists.json"
    new_data = asyncio.run(scrape_user_watchlists("Prestigious_Garlic_9"))

    # Load existing records so we never lose historical posts
    existing: list = []
    if os.path.exists(OUTPUT):
        try:
            with open(OUTPUT) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = []

    # Deduplicate: key = (title, timestamp) — same logic as the scraper
    def _key(entry: dict) -> str:
        return f"{entry.get('title','')}::{entry.get('timestamp','')}"

    seen = {_key(e) for e in existing}
    added = 0
    for entry in new_data:
        k = _key(entry)
        if k not in seen:
            existing.append(entry)
            seen.add(k)
            added += 1

    with open(OUTPUT, "w") as f:
        json.dump(existing, f, indent=4)

    print(f"Done! {added} new watchlist(s) added. Total: {len(existing)}.")
