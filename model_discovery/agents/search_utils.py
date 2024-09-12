
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
    def __init__(self,ptree):
        self.ptree=ptree
        self.result_limits={
            's2':10,
            'arxiv':5,
            'pwc':5,
        }


    def __forward__(self,query):
        """
        Query: query for search papers
        Details: what content you want to find
        """
        s2_results=self.search_s2(query,self.result_limits['s2'])
        arxiv_results=self.search_arxiv(query,self.result_limits['arxiv'])
        pwc_results=self.search_pwc(query,self.result_limits['pwc'])


    #### External Sources, give you mostly the abstract and title

    @backoff.on_exception(
        backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
    )
    def search_s2(self,query, result_limit=10,start_year=2015, top_conf_only=True, **fields) -> Union[None, List[Dict]]:
        # https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/post_graph_get_papers
        # should also search from the internal base or KGs
        if not query:
            return None
        S2_API_KEY=os.environ['S2_API_KEY']
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,tldr,abstract,citationCount,influentialCitationCount",
            'fieldsOfStudy': 'Computer Science',
            'publicationDateOrYear': f'{start_year}:',
            **fields    
        }
        if top_conf_only:
            params['venue'] = 'NeurIPS,ICML,ICLR,ACL,EMNLP,NAACL' # top3 DL/NLP conferences
        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers={"X-API-KEY": S2_API_KEY},
            params=params
        )
        rsp.raise_for_status()
        results = rsp.json()
        total = results["total"]
        if not total:
            return None
        papers = results["data"]
        return papers


    def search_arxiv(self,query, result_limit=10, category='cat:cs.LG+OR+cat:cs.CL'):#, start_year=2015):
        """
        Searches arXiv for papers in the specified category with the user query and time restriction.

        :param user_query: A string query to search for.
        :param category: The arXiv category to search in (default: 'cs.LG' for Machine Learning).
        :param max_results: The maximum number of results to return (default: 10).
        :param start_year: The minimum year of paper submission (default: 2015).
        :return: List of paper titles and summaries.
        """
        base_url = 'http://export.arxiv.org/api/query?'
        user_query=urllib.parse.quote(query, safe=":/?&=")
        search_query = f'{category}+AND+all:"{user_query}"'
        # date_filter = f'AND+submittedDate:[{start_year}0101+TO+*]' # not working
        query_string = f"search_query={search_query}&start=0&max_results={result_limit}"
        query_url = base_url + query_string
        response = feedparser.parse(query_url)
        if not response.entries:
            return
        papers = {}
        for entry in response.entries:
            papers[entry.id] = {
                'title': entry.title,
                'summary': entry.summary,
                'published': entry.published,
                'authors': entry.authors,
            }
            if hasattr(entry, 'arxiv_comment'):
                papers[entry.id]['arxiv_comment'] = entry.arxiv_comment

        return papers


    def search_pwc(self,query, result_limit=10) -> Union[None, List[Dict]]:
        # https://paperswithcode-client.readthedocs.io/en/latest/index.html
        client = PapersWithCodeClient()
        ret=client.search(query,items_per_page=result_limit)
        rets={}
        for i in ret.results:
            paper=i.paper
            repo=i.repository
            rets[paper.id]={
                'title':paper.title,
                'abstract':paper.abstract,
                'authors':paper.authors,
                'published':paper.published,
                'proceeding':paper.proceeding,
            }
            if repo:
                rets[paper.id]['repo_owner']=repo.owner
                rets[paper.id]['repo_name']=repo.name
                rets[paper.id]['repo_description']=repo.description
                rets[paper.id]['repo_stars']=repo.stars
        return rets


    #### Internal Sources

    def search_lib_primary(query, result_limit=10) -> Union[None, List[Dict]]:
        # the selected ~300 model arch papers
        pass

    def search_lib_secondary(query, result_limit=10) -> Union[None, List[Dict]]:
        # the papers that are cited by the primary library, where their ideas come from  
        pass

    def search_lib_plus(query, result_limit=10) -> Union[None, List[Dict]]:
        # the papers recommended by S2 for the primary library  
        pass
