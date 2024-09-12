
import os
import time
import requests
from typing import Union, List, Dict
import backoff





def on_backoff(details):
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


class SuperScholarSearcher:
    """
    One search interface that directly give you all the answers
    """
    def __init__(self,api_key):
        self.api_key=api_key


    def __forward__(self,query,details=None):
        """
        Query: query for search papers
        Details: what content you want to find
        """
        pass    


    @backoff.on_exception(
        backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
    )
    def search_s2(query, result_limit=10) -> Union[None, List[Dict]]:
        # https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/post_graph_get_papers
        # should also search from the internal base or KGs
        if not query:
            return None
        S2_API_KEY=os.environ['S2_API_KEY']
        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers={"X-API-KEY": S2_API_KEY},
            params={
                "query": query,
                "limit": result_limit,
                "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
                'fieldsOfStudy': 'Computer Science,Mathematics,Physics'
            },
        )
        rsp.raise_for_status()
        results = rsp.json()
        total = results["total"]
        if not total:
            return None

        papers = results["data"]
        return papers

    def search_arxiv(query, result_limit=10) -> Union[None, List[Dict]]:
        pass

    def search_web(query, result_limit=10) -> Union[None, List[Dict]]:
        pass

    #### Internal Knowledge base

    def search_lib_primary(query, result_limit=10) -> Union[None, List[Dict]]:
        # the selected ~300 model arch papers
        pass

    def search_lib_secondary(query, result_limit=10) -> Union[None, List[Dict]]:
        # the papers that are cited by the primary library, where their ideas come from  
        pass
