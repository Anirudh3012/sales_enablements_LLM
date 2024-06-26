import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

def scrape_website_content(url, max_depth=2):
    visited_urls = set()
    aggregated_text = ""
    url_queue = deque([(url, 0)])  # Queue of (url, depth)

    def format_text(url, headings, paragraphs):
        formatted_text = f"\n\nURL: {url}\n"
        formatted_text += "\n".join([f"{heading[0]}: {heading[1]}" for heading in headings])
        formatted_text += "\n\n" + "\n".join(paragraphs)
        return formatted_text

    while url_queue:
        current_url, depth = url_queue.popleft()
        if depth > max_depth or current_url in visited_urls:
            continue

        try:
            visited_urls.add(current_url)
            response = requests.get(current_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract headings and paragraphs
            headings = [(tag.name, tag.text.strip()) for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']) if tag.text.strip()]
            paragraphs = [p.text.strip() for p in soup.find_all('p') if p.text.strip()]
            aggregated_text += format_text(current_url, headings, paragraphs)

            # Extract links from <a>, <link>, <iframe> tags
            links = set()
            for tag in soup.find_all(['a', 'link', 'iframe'], href=True):
                links.add(tag['href'])
            for tag in soup.find_all('iframe', attrs={'src': True}):
                links.add(tag['src'])
            
            # Process each link
            for link in links:
                backlink_url = urljoin(current_url, link)
                parsed_url = urlparse(backlink_url)
                if parsed_url.netloc == urlparse(url).netloc:  # Ensure we stay within the same domain
                    url_queue.append((backlink_url, depth + 1))

        except Exception as e:
            print(f"Error scraping {current_url}: {e}")

    return aggregated_text
