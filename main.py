#!/usr/bin/env python

# from chatbots import OpenAiChatBot

# if __name__ == "__main__":
#     chatbot = OpenAiChatBot()
#     chatbot.run()

import requests
import json

def searxng_search(query: str, searx_host: str, num_results: int = 10):
    """
    Faz uma busca no SearxNG e retorna JSON dos resultados.
    """
    url = f"{searx_host.rstrip('/')}/search"
    params = {
        "q": query,
        "format": "json",
        "pageno": 1,
        "results_on_new_tab": 0,
        "num_results": num_results,
        "language": "pt-br",
        "engines": "google,duckduckgo"
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

# Exemplo de uso:
searx_host = "http://localhost:8888"  # ou sua instância SearxNG
resultado = searxng_search("Qual foi a ultima fala do trump nos ultimos dias?", searx_host)

json_formatado = json.dumps(resultado, indent=4, ensure_ascii=False)
print()