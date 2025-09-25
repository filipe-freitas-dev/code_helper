import requests
from bs4 import BeautifulSoup

def get_url_content(url: str) -> str:
    return "\n".join(
        line 
        for line in BeautifulSoup(requests.get(url).text, "html.parser")\
            .get_text()\
                .splitlines() 
        if line.strip()
    )


