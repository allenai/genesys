
import os
import time
import requests
from typing import Union, List, Dict, Optional
import feedparser
import urllib.request
import urllib.parse
from paperswithcode import PapersWithCodeClient


import model_discovery.utils as U



class SuperScholarSearcher:
    """
    One search interface that directly give you all the answers
    """
    def __init__(self,ptree,stream,cfg={}):
        self.ptree=ptree
        self.files_dir=U.pjoin(ptree.lib_dir,'..','files')
        DEFAULT_SEARCH_LIMITS={
            's2':10,
            'arxiv':5,
            'pwc':5,
        }
        self.result_limits=U.safe_get_cfg_dict(cfg,'result_limits',DEFAULT_SEARCH_LIMITS)
        self.stream=stream
        self.pwc_client=PapersWithCodeClient()



    def __forward__(self,title_abstract,content):
        """
        title_abstract: for search papers by title and abstract
        content: for matching the related content from papers
        """
        pass


    
    #### Get Paper Files by URL or ArXiv ID

    def get_html(self,url,save_path):
        if U.pexists(save_path):
            return U.load_file(save_path)
        response = requests.get(url)
        if response.status_code == 200:
            U.save_file(save_path,response.text)
            return response.text
        else:
            return None

    def download_pdf(self,url,save_path):
        if U.pexists(save_path):
            return U.load_file(save_path)
        try:
            pdf=urllib.request.urlretrieve(url, save_path)
            return pdf
        except:
            return None

    def get_paper_file(self,arxiv_id=None,pdf_url=None):
        paper_file=None
        htmlst_dir=U.pjoin(self.files_dir,'htmlst')
        pdfst_dir=U.pjoin(self.files_dir,'pdfst')
        U.mkdir(htmlst_dir)
        U.mkdir(pdfst_dir)
        if arxiv_id:
            ar5iv_url=f'https://ar5iv.labs.arxiv.org/html/{arxiv_id}'
            save_path=U.pjoin(htmlst_dir,f'{arxiv_id}.html')
            paper_file=self.get_html(ar5iv_url,save_path)
        else:
            if pdf_url:
                save_path=U.pjoin(pdfst_dir,f'{arxiv_id}.pdf')
                paper_file=self.download_pdf(pdf_url,save_path)
        return paper_file

    def get_paper_file_from_results(self,results):
        for r in results:
            if 'arxiv_id' in r:
                return self.get_paper_file(arxiv_id=r['arxiv_id'])
            elif 'openAccessPdf' in r:
                return self.get_paper_file(pdf_url=r['openAccessPdf'])

    #### External Sources, give you mostly the abstract and title


    def search_external(self,query,pretty=True,pure=True) -> Union[None, List[Dict]]:
        # search for external papers
        s2_results=self.safe_search(self.search_s2,query,self.result_limits['s2'])
        arxiv_results=self.safe_search(self.search_arxiv,query,self.result_limits['arxiv'])
        pwc_results=self.safe_search(self.search_pwc,query,self.result_limits['pwc'])

        aggregated_results=s2_results
        for r in arxiv_results+pwc_results:
            if not self.exist_in_set(r,aggregated_results):
                aggregated_results.append(r)
        self.stream.write(f'##### *Found {len(aggregated_results)} related papers*')
        if pure:
            aggregated_results=self.pure_results(aggregated_results)
        if pretty:
            return aggregated_results, self.pretty_print(aggregated_results)
        else:
            return aggregated_results

    def pure_results(self,results):
        pure_results=[]
        for r in results:
            for key in ['arxiv_id','s2_id','pwc_id','repository','openAccessPdf']:
                if key in r:
                    r.pop(key)
            pure_results.append(r)
        return pure_results

    def pretty_print(self,results):
        ppr=''
        for i,r in enumerate(results):
            ppr+=f'##### {i+1}. {r["title"]}\n\n'
            ppr+=f'*{", ".join(r["authors"]) if r["authors"] else "Anonymous"}*\n\n'
            if 'tldr' in r and r['tldr']:
                ppr+=f'**TL;DR:** {r["tldr"]}\n\n'
            ppr+=f'**Abstract:** {r["abstract"]}\n\n'
            if 'venue' in r and r['venue']:
                ppr+=f'**Venue:** {r["venue"]}\n\n'
            elif 'conference' in r and r['conference']:
                ppr+=f'**Conference:** {r["conference"]}\n\n'
            elif 'proceeding' in r and r['proceeding']:
                ppr+=f'**Proceeding:** {r["proceeding"]}\n\n'
            if 'year' in r:
                ppr+=f'**Year:** {r["year"]}\n\n'
            if 'published' in r:
                ppr+=f'**Published:** {r["published"]}'
                if 'updated' in r:
                    ppr+=f'  (*Updated: {r["updated"]}*)'
                ppr+='\n\n'
            if 'citationCount' in r:
                ppr+=f'**Citations:** {r["citationCount"]}'
                if 'influentialCitationCount' in r:
                    ppr+=f'  (*Influential: {r["influentialCitationCount"]}*)'
            ppr+='\n\n'
            if 's2_id' in r:
                ppr+=f'**Semantic Scholar ID:** {r["s2_id"]}\n\n'
            if 'pwc_id' in r:
                ppr+=f'**PaperswithCode ID:** {r["pwc_id"]}\n\n'
            if 'arxiv_id' in r and r['arxiv_id']:
                ppr+=f'**ArXiv ID:** {r["arxiv_id"]}\n\n'
            if 'repository' in r:
                ppr+=f'**Repository:**\n\n'
                ppr+=f'  - *Owner:* {r["repository"]["owner"]}\n\n'
                ppr+=f'  - *Name:* {r["repository"]["name"]}\n\n'
                ppr+=f'  - *Description:* {r["repository"]["description"]}\n\n'
                ppr+=f'  - *Stars:* {r["repository"]["stars"]}\n\n'
                ppr+=f'  - *URL:* {r["repository"]["url"]}\n\n'
            if 'openAccessPdf' in r and r['openAccessPdf']:
                ppr+=f'**Open Access PDF:** {r["openAccessPdf"]}\n\n'
            ppr+=f'\n\n'
        return ppr


    def exist_in_set(self,paper,paper_set):
        for p in paper_set:
            if self.check_duplicate(paper,p):
                return True
        return False

    def check_duplicate(self,paper1,paper2):
        if paper1['arxiv_id'] and paper2['arxiv_id']:
            if paper1['arxiv_id'].strip()==paper2['arxiv_id'].strip():
                return True
        if paper1['title'].strip()==paper2['title'].strip():
            return True
        if paper1['abstract'].strip()==paper2['abstract'].strip():
            return True
        return False

    def safe_search(self,fn,query,result_limit=10):
        # try:
        return fn(query,result_limit)
        # except Exception as e:
        #     print(f"Error searching {fn.__name__}: {e}")
        #     return []

    def search_s2(self,query, result_limit=10,start_year=2010, top_conf_only=True, **fields) -> Union[None, List[Dict]]:
        # https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/post_graph_get_papers
        
        S2_API_KEY=os.environ['S2_API_KEY']
        info=f'Searching Semantic Scholar for "{query}"'
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,tldr,abstract,citationCount,influentialCitationCount,openAccessPdf,externalIds",
            'fieldsOfStudy': 'Computer Science',
            **fields    
        }
        if top_conf_only:
            params['venue'] = 'NeurIPS,ICML,ICLR,ACL,EMNLP,NAACL' # top3 DL/NLP conferences
            info+=f' from {", ".join(params["venue"].split(","))}'
        if start_year:
            params['publicationDateOrYear'] = f'{start_year}:'
            info+=f' after {start_year}'
        self.stream.write(info+'...')
        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers={"X-API-KEY": S2_API_KEY},
            params=params
        )
        rsp.raise_for_status()
        results = rsp.json()
        total = results["total"]
        if not total:
            return []
        papers=[]
        for r in results["data"]:
            arxiv_id=None
            if 'externalIds' in r and 'ArXiv' in r['externalIds']:
                arxiv_id=r['externalIds']['ArXiv']
            paper={
                'arxiv_id':arxiv_id,
                's2_id':r['paperId'],
                'title':r['title'],
                'abstract':r['abstract'],
                'authors':[i['name'] for i in r['authors']],
                'venue':r['venue'],
                'year':r['year'],
                'tldr':r['tldr']['text'] if r['tldr'] else None,
                'citationCount':r['citationCount'],
                'influentialCitationCount':r['influentialCitationCount'],
                'openAccessPdf':r['openAccessPdf']['url'] if r['openAccessPdf'] else None,
            }
            papers.append(paper)
        return papers

    def search_arxiv(self,query, result_limit=10, category=['cs.LG','cs.CL']):#, start_year=2015):
        """
        Searches arXiv for papers in the specified category with the user query and time restriction.

        :param user_query: A string query to search for.
        :param category: The arXiv category to search in (default: 'cs.LG' for Machine Learning).
        :param max_results: The maximum number of results to return (default: 10).
        :param start_year: The minimum year of paper submission (default: 2015).
        :return: List of paper titles and summaries.
        """
        self.stream.write(f'Searching arXiv for "{query}" in {", ".join(category)}...')
        base_url = 'http://export.arxiv.org/api/query?'
        user_query=urllib.parse.quote(query, safe=":/?&=")
        category_str='+OR+'.join(['cat:'+i for i in category])
        search_query = f'{category_str}+AND+all:"{user_query}"'
        # date_filter = f'AND+submittedDate:[{start_year}0101+TO+*]' # not working
        query_string = f"search_query={search_query}&start=0&max_results={result_limit}"
        query_url = base_url + query_string
        response = feedparser.parse(query_url)
        papers = []
        for entry in response['entries']:
            paper={
                'arxiv_id': entry['id'].split('/')[-1].split('v')[0],
                'title': entry['title'],
                'abstract': entry['summary'],
                'published': entry['published'],
                'updated': entry['updated'],
                'authors': [i['name'] for i in entry['authors']],
            }
            if 'arxiv_comment' in entry:
                paper['arxiv_comment'] = entry['arxiv_comment']
            papers.append(paper)
        return papers


    def search_pwc(self,query, result_limit=10) -> Union[None, List[Dict]]:
        # https://paperswithcode-client.readthedocs.io/en/latest/api/client.html?highlight=search#paperswithcode.client.PapersWithCodeClient.search
        self.stream.write(f'Searching Papers with Code for "{query}"...')
        ret=user_input(self.pwc_client,query,items_per_page=result_limit)
        results=ret['results']
        papers=[]
        for i in results:
            paper=i['paper']
            repo=i['repository']
            paper={
                'pwc_id':paper['id'],
                'title':paper['title'],
                'abstract':paper['abstract'],
                'authors':paper['authors'],
                'published':paper['published'],
                'proceeding':paper['proceeding'],
                'conference':paper['conference'],
                'openAccessPdf':paper['url_pdf'],
                'arxiv_id':paper['arxiv_id'],
                'repository':None,
            }
            if repo:
                paper['repository']={
                    'owner':repo['owner'],
                    'name':repo['name'],
                    'description':repo['description'],
                    'stars':repo['stars'],
                    'url':repo['url'],
                }   
            papers.append(paper)
        return papers


    #### Internal Sources

    def _load_libs(self): # used for building the search library
        # load the primary and secondary libraries
        self.lib={}
        for i in os.listdir(self.ptree.lib_dir):
            if i.endswith('.json'):
                self.lib[i.split('.')[0]]=U.load_json(U.pjoin(self.ptree.lib_dir,i))
        lib_est_dir=U.pjoin(self.ptree.lib_dir,'..','tree_ext')
        self.lib2={}
        lib2_dir=U.pjoin(lib_est_dir,'secondary')
        for i in os.listdir(lib2_dir):
            if i.endswith('.json'):
                self.lib2[i.split('.')[0]]=U.load_json(U.pjoin(lib2_dir,i))
        self.libp={}
        libp_dir=U.pjoin(lib_est_dir,'plus')
        for i in os.listdir(libp_dir):
            if i.endswith('.json'):
                self.libp[i.split('.')[0]]=U.load_json(U.pjoin(libp_dir,i))
        
    def search_lib_primary(query, result_limit=10) -> Union[None, List[Dict]]:
        # the selected ~300 model arch papers
        pass

    def search_lib_secondary(query, result_limit=10) -> Union[None, List[Dict]]:
        # the papers that are cited by the primary library, where their ideas come from  
        pass

    def search_lib_plus(query, result_limit=10) -> Union[None, List[Dict]]:
        # the papers recommended by S2 for the primary library  
        pass


    
    #### Design Artifacts (TODO later)

    def search_design(query, result_limit=10) -> Union[None, List[Dict]]:
        # search for design artifacts
        pass





### Paper With Code Patches, it depends on old version of pydantic, httpx, and typing_extensions

from paperswithcode.client import handler

@handler
def user_input(
    cls,
    q: Optional[str] = None,
    page: int = 1,
    items_per_page: int = 50,
    **kwargs
):
    """Search in a similar fashion to the frontpage search.

    Args:
        q: Filter papers by querying the paper title and abstract.
        page: Desired page.
        items_per_page: Desired number of items per page.

    Returns:
        PaperRepos object.
    """
    params = {key: str(value) for key, value in kwargs.items()}
    params["page"] = str(page)
    params["items_per_page"] = str(items_per_page)
    timeout = None
    if q is not None:
        params["q"] = q
    return cls.http.get("/search/", params=params, timeout=timeout)