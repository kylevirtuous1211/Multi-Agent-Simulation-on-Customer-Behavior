import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode

async def search_online(user_prompt, max_results=3):
    """
    Search online based on user's prompt using Crawl4AI.
    
    Args:
        user_prompt (str): The search query from the user
        max_results (int): Maximum number of search results to return
        
    Returns:
        list: List of dictionaries containing search results with titles and content
    """
    # Configure browser settings
    browser_conf = BrowserConfig(
        headless=True,  # Run browser in headless mode
        verbose=False   # Minimize logging
    )
    
    # Configure crawler run settings
    run_conf = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,  # Use cache for faster results
        wait_for_timeout=3000,         # Wait 3 seconds for content to load
        output_formats=['markdown']    # Get results in markdown format
    )
    
    # Format the search query for Google
    search_url = f"https://www.google.com/search?q={user_prompt.replace(' ', '+')}"
    
    results = []
    
    # Initialize the crawler
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        # First get the search results page
        search_result = await crawler.arun(url=search_url, config=run_conf)
        
        if not search_result or not search_result.success:
            return [{"title": "Search failed", "content": "Unable to retrieve search results."}]
        
        # Extract links from search results
        links_to_visit = []
        for link in search_result.links.external[:max_results]:
            if "google.com" not in link.url:  # Filter out Google-related links
                links_to_visit.append(link.url)
        
        # Visit each link and extract content
        for url in links_to_visit:
            try:
                result = await crawler.arun(url=url, config=run_conf)
                if result and result.success:
                    # Extract title and content
                    title = result.metadata.title if result.metadata and result.metadata.title else "No title"
                    content = result.markdown.fit_markdown[:1000] + "..." if len(result.markdown.fit_markdown) > 1000 else result.markdown.fit_markdown
                    
                    results.append({
                        "title": title,
                        "url": url,
                        "content": content
                    })
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                continue
    
    return results

# Example usage
async def main():
    user_prompt = "latest developments in AI"
    search_results = await search_online(user_prompt)
    
    print(f"Search results for: '{user_prompt}'")
    for i, result in enumerate(search_results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Content Preview: {result['content'][:200]}...")

if __name__ == "__main__":
    asyncio.run(main())
