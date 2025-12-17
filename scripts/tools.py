import requests

def youtube_search(query):
    return [f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"]

def arxiv_search(query):
    return [f"https://arxiv.org/search/?query={query.replace(' ', '+')}&searchtype=all"]