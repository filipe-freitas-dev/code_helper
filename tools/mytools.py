import requests
from bs4 import BeautifulSoup

def get_urls_content(urls: list[str]) -> str:
    return "\n".join(
        line 
        for url in urls 
        for line in BeautifulSoup(requests.get(url).text, "html.parser")\
            .get_text()\
                .splitlines() 
        if line.strip()
    )


