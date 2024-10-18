
import os
import time
import requests
from typing import Union, List, Dict, Optional
import feedparser
import urllib.request
import urllib.parse
from paperswithcode import PapersWithCodeClient
from requests.exceptions import RequestException
from openai import OpenAI
import asyncio
import random
import cohere
import numpy as np

from langchain_community.document_loaders import MathpixPDFLoader,UnstructuredHTMLLoader
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_together import TogetherEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.evaluation import load_evaluator,EmbeddingDistance


from tqdm import tqdm
import pypdf



import model_discovery.utils as U


PERPLEXITY_SYSTEM = """
You are an AI research assistant who helps a language model researcher. Your
goal is to help the researcher to discover the best novel autoregressive LM
block that can defeat the existing state-of-the-art models, measured in low
perplexity in corpora, high accuracy in downstream tasks, robustness to variant
inputs, efficiency in training and inference, and most importantly, good
scalability that providing better overall performance with more data and larger
models.

You task is to search for the information that can help the researchers to
achieve this goal. You will be given a set of keywords that points to the topic
they are interested in, and details about the exact contents they are looking
for. Search for the informaiton based on the keywords and details that align
with the goal.
"""



PERPLEXITY_PROMPT = """
Here is a set of keywords: 
{query}

Here is a detail: 
{detail}
{analysis}

Search for the information that can help the researchers to achieve the goal of
improving autoregressive language model design.
"""


PERPLEXITY_SYSTEM_INSTRUCT = """
You are an AI research assistant who helps a language model researcher gather information for discovering the best novel autoregressive LM block that can defeat the existing state-of-the-art models.

## Background

Modern LMs are typically structured as a stack of repeating blocks. The goal is to design a novel LM block that outperforms current state-of-the-art models, aiming for:
- Low perplexity on corpora,
- High accuracy on downstream tasks,
- Robustness to varied inputs,
- Efficiency in both training and inference,
- Excellent scalability with more data and larger models.

You will be provided with the researcher's thoughts, analysis, and descriptions, and your task is to understand the intent of the researcher and search for the information that can best help the intent.
"""

PERPLEXITY_PROMPT_INSTRUCT = """
Here are the information from the researcher: 
{instruct}

Understand the goal, idea and intents of the researcher. Find the most useful information that can best help the researcher to achieve the goal.
"""



### Paper With Code Patches, it depends on old version of pydantic, httpx, and typing_extensions

from paperswithcode.client import handler

@handler
def pwc_search_patched(
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



DEFAULT_SEARCH_LIMITS={
    's2':5,
    'arxiv':3,
    'pwc':3,
    'lib':5,
    'lib2':3,
    'libp':3,
}
DEFAULT_PERPLEXITY_SETTINGS={
    'model_size':'large',
    'max_tokens':2000,
}
DEFAULT_PROPOSAL_SEARCH_CFG={
    'top_k':3,
    'cutoff':0.5,
    'sibling':2,
}

DEFAULT_UNIT_SEARCH_CFG={
    'top_k':4,
    'cutoff':0.5,
}
DEFAULT_RERANK_RATIO=0.2


OPENAI_EMBEDDING_MODELS=[
    'text-embedding-3-large',
    'text-embedding-3-small',
    'text-embedding-ada-002',
]

TOGETHER_EMBEDDING_MODELS=[
    'togethercomputer/m2-bert-80M-2k-retrieval',
    'togethercomputer/m2-bert-80M-8k-retrieval',
    'togethercomputer/m2-bert-80M-32k-retrieval',
    'WhereIsAI/UAE-Large-V1',
    'BAAI/bge-large-en-v1.5',
    'BAAI/bge-base-en-v1.5',
    'sentence-transformers/msmarco-bert-base-dot-v5',
    'bert-base-uncased',
]

COHERE_EMBEDDING_MODELS=[
    'embed-english-v3.0',
    'embed-multilingual-v3.0',
    'embed-english-light-v3.0',
    'embed-multilingual-light-v3.0',
    'embed-english-v2.0',
    'embed-english-light-v2.0',
    'embed-multilingual-v2.0',
]

DEFAULT_EMBEDDING_MODELS = {
    'vectorstore':'text-embedding-3-large',
    'proposal':'text-embedding-3-large',
    'unitcode':'text-embedding-3-large',
}

DEFAULT_EMBEDDING_DISTANCES = {
    'proposal':EmbeddingDistance.COSINE.value,
    'unitcode':EmbeddingDistance.COSINE.value,
}

SUPPORTED_EMBEDDING_MODELS=[
    *OPENAI_EMBEDDING_MODELS,
    *TOGETHER_EMBEDDING_MODELS,
    *COHERE_EMBEDDING_MODELS,
]

DEFAULT_VS_INDEX_NAME='modis-library-v0'

class SuperScholarSearcher:
    """
    One search interface that directly give you all the answers
    """
    def __init__(self,ptree,stream,cfg={}):
        self.ptree=ptree
        self.files_dir=U.pjoin(ptree.lib_dir,'..','files')
        self.libfiles_dir=U.pjoin(ptree.lib_dir,'..','lib_files') # files for the library requests, including indices and splits
        self.co=None
        try:
            self.co=cohere.Client(os.environ['COHERE_API_KEY'])
        except Exception as e:
            print(f'Unable to initialize Cohere: {e}')
        self.pc=None
        try:
            self.pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        except Exception as e:
            print(f'Unable to initialize Pinecone: {e}')
        
        self.s2_key_set='S2_API_KEY' in os.environ
        self.ppl_key_set='PERPLEXITY_API_KEY' in os.environ

        self.index_name=None
        self.reconfig(cfg)

        self.stream=stream
        self.pwc_client=PapersWithCodeClient()

        self.client = OpenAI(api_key=os.environ['MY_OPENAI_KEY'])
        self.lib=None
        self.texts=None
        self.splits=None
        self.vectors=None

        self.design_proposals={}
        self.unit_codes={}
        self.embedding_cache={
            'design_proposals':{},
            'unit_codes':{},
        }
        self._refresh_db()

    def _refresh_db(self):
        for acronym in self.ptree.filter_by_type(['DesignArtifact','DesignArtifactImplemented']):
            if acronym not in self.design_proposals:
                design=self.ptree.get_node(acronym)
                self.design_proposals[acronym]=design.proposal.proposal
        
        for name in self.ptree.GD.terms:
            if name not in self.unit_codes:
                self.unit_codes[name]=self.ptree.GD.get_unit(name).code # randomly get one

    def _get_embedding_model(self,model_name):
        if model_name in OPENAI_EMBEDDING_MODELS:
            return OpenAIEmbeddings(openai_api_key=os.environ['MY_OPENAI_KEY'],model=model_name)    
        elif model_name in TOGETHER_EMBEDDING_MODELS:
            return TogetherEmbeddings(model=model_name,api_key=os.environ['TOGETHER_API_KEY'])
        elif model_name in COHERE_EMBEDDING_MODELS:
            return CohereEmbeddings(model=model_name,cohere_api_key=os.environ['COHERE_API_KEY'])
        else:
            raise ValueError(f'Unsupported embedding model: {model_name}')

    def _setup_embedding_models(self):
        self.embedding_vs=self._get_embedding_model(self.embedding_models['vectorstore'])
        self.embedding_proposal=self._get_embedding_model(self.embedding_models['proposal'])
        self.embedding_unit=self._get_embedding_model(self.embedding_models['unitcode'])
        self.emb_evaluator_proposal = load_evaluator(
            "embedding_distance", 
            embeddings=self.embedding_proposal,
            distance_metric=EmbeddingDistance(self.embedding_distances['proposal'])
        )
        self.emb_evaluator_unit = load_evaluator(
            "embedding_distance", 
            embeddings=self.embedding_unit,
            distance_metric=EmbeddingDistance(self.embedding_distances['unitcode'])
        )


    def reconfig(self,cfg,stream=None):
        self.embedding_models=U.safe_get_cfg_dict(cfg,'embedding_models',DEFAULT_EMBEDDING_MODELS)
        self.embedding_distances=U.safe_get_cfg_dict(cfg,'embedding_distances',DEFAULT_EMBEDDING_DISTANCES)
        self.result_limits=U.safe_get_cfg_dict(cfg,'result_limits',DEFAULT_SEARCH_LIMITS)
        self.rerank_ratio=cfg.get('rerank_ratio',DEFAULT_RERANK_RATIO)
        self.perplexity_settings=U.safe_get_cfg_dict(cfg,'perplexity_settings',DEFAULT_PERPLEXITY_SETTINGS)
        self.proposal_search_cfg=U.safe_get_cfg_dict(cfg,'proposal_search',DEFAULT_PROPOSAL_SEARCH_CFG)
        self.unit_search_cfg=U.safe_get_cfg_dict(cfg,'unit_search',DEFAULT_UNIT_SEARCH_CFG)
        index_name=cfg.get('index_name',DEFAULT_VS_INDEX_NAME) # change it to your index name
        assert index_name, 'Index name is required'
        if index_name!=self.index_name and self.pc is not None: # always do it in init
            self.index_name=index_name
            self.index=self.get_index(index_name)
        self.cfg={
            'index_name':self.index_name,
            'result_limits':self.result_limits,
            'rerank_ratio':self.rerank_ratio,
            'perplexity_settings':self.perplexity_settings,
            'proposal_search':self.proposal_search_cfg,
            'embedding_models':self.embedding_models,
            'embedding_distances':self.embedding_distances,
            'unit_search':self.unit_search_cfg,
        }
        self._setup_embedding_models()
        if stream:
            self.stream=stream

    def __call__(self,query=None,detail=None,analysis=None,instruct=None,raw=False,prompt=True):
        """
        query: for search papers in S2, ArXiv, and Papers with Code...
        detail: for search papers in the internal library vector stores
        """
        if detail:
            internal_results,internal_pp=self.search_internal(detail,pretty=True,prompt=prompt)
        else:
            internal_results,internal_pp=None,None

        if query:
            external_results,external_pp=self.search_external(query,pretty=True,prompt=prompt)
        else:
            external_results,external_pp=None,None

        perplexity_results, perplexity_pp = None, None
        if self.perplexity_settings['model_size']!='none' and self.ppl_key_set:
            if instruct:
                perplexity_results, perplexity_pp = self.search_perplexity(
                    str(query),detail,analysis,instruct,
                    size=self.perplexity_settings['model_size'],
                    max_tokens=self.perplexity_settings['max_tokens'])
            elif query or detail or analysis:
                perplexity_results, perplexity_pp = self.search_perplexity(
                    str(query),detail,analysis,instruct,
                    size=self.perplexity_settings['model_size'],
                    max_tokens=self.perplexity_settings['max_tokens'])

        self.stream.write(f'Concluding search results...')

        pp=''
        if internal_pp:
            pp+=internal_pp+'\n'
        if external_pp:
            pp+=external_pp+'\n'
        if perplexity_pp:
            pp+=perplexity_pp+'\n'
        if raw:
            raw_ret={
                'external_rets':external_results,
                'internal_rets':internal_results,
                'perplexity_rets':perplexity_results,
            }
            return pp,raw_ret
        else:
            return pp
        

    def has_index(self,index_name):
        existing_indexes=[i['name'] for i in self.pc.list_indexes()]
        return index_name in existing_indexes


    #### Search design base

    def _search_designs(self,query,parents=None):
        topk,prt=self.query_design_proposals(query)
        RET={
            'topk':topk,
            'proposal_prt':prt,
        }
        if parents:
            siblings,prts=self.query_sibling_designs(parents)
            prt+='\n\n---\n\n'+prts
            RET['sibling_designs']=siblings
            RET['sibling_prt']=prts
        return RET,prt

    def query_sibling_designs(self,parents,pp=True):
        top_k=self.proposal_search_cfg['sibling']
        if top_k<=0:
            return [],'Sibling proposal search not available.' if pp else []
        siblings=self.ptree.find_sibling_designs(parents)
        if pp:
            prt='**Siblings Design Proposals from Previous Designs**:\n\n'
            abstracts,reviews,ratings=self.ptree.get_abstracts(siblings)
            if not siblings:
                prt+='### No siblings found from the previous designs with same seeds.\n\n'
            else:
                top_k=min(top_k,len(siblings))
                siblings = random.sample(siblings,top_k) if top_k>0 else []
                prt+=f'### Found {len(siblings)} siblings from the previous designs with same seeds:\n\n'
                for i,p in enumerate(siblings):
                    prt+=f'#### Sibling {i+1}. {p}\n\n```\n\n{abstracts[i]}\n\n```\n\n'
                    prt+=f'##### Review\n\n```\n\n{reviews[i]}\n\n```\n\n'
                    prt+=f'##### Rating\n\n```\n\n{ratings[i]} out of 5\n\n```\n\n'
            return siblings,prt
        else:
            return siblings
    
    def _get_score(self,emb_pred,emb_ref,evaluator):
        vectors = np.array([emb_pred,emb_ref])
        return evaluator._compute_score(vectors)
    
    def _get_embeddings(self,key,text,embeddings,cache_key):
        assert cache_key in self.embedding_cache, f'Cache key {cache_key} not found in the embedding cache.'
        if key not in self.embedding_cache[cache_key]:
            self.embedding_cache[cache_key][key]=embeddings.embed_documents([text])[0]
        return self.embedding_cache[cache_key][key]

    def emb_evaluate(self,query,references,evaluator,embeddings,top_k,cutoff,cache_key):
        scores={}
        emb_pred=embeddings.embed_documents([query])[0]
        for i in references: # TODO: parallelize this, but may be errornous API side
            emb_ref=self._get_embeddings(i,references[i],embeddings,cache_key)
            scores[i]=self._get_score(emb_pred,emb_ref,evaluator)
        filtered_scores={i:s for i,s in scores.items() if s>cutoff}
        pps=list(sorted(filtered_scores.items(),key=lambda x:x[1]))[:top_k]
        return pps,scores

    def query_design_proposals(self,query,pp=True):
        top_k=self.proposal_search_cfg['top_k']
        if top_k<=0:
            return [],'' if pp else []
        cutoff=self.proposal_search_cfg['cutoff']
        pps,scores=self.emb_evaluate(query,self.design_proposals,self.emb_evaluator_proposal,self.embedding_proposal,top_k,cutoff,cache_key='design_proposals')
        if pp:
            prt='**Similar Design Proposals from Previous Designs**:\n\n'
            if not pps:
                prt+='### No similar design proposals found from the previous designs for the given proposal.\n\n'
            else:
                prt+=f'### Found {len(pps)} similar design proposals from the previous designs for the given proposal:\n\n'
                for i,p in pps:
                    prt+=f'#### {i} (Score: {scores[i]:.2f})\n\n<details><summary>Click to Expand</summary>\n\n```\n{self.design_proposals[i]}\n```\n\n</details>\n\n'
            return pps,prt
        else:
            return pps

    def query_unit_codes(self,query,pp=True):
        top_k=self.unit_search_cfg['top_k']
        if top_k<=0:
            return [],'Unit code search not available.' if pp else []
        cutoff=self.unit_search_cfg['cutoff']
        pps,scores=self.emb_evaluate(query,self.unit_codes,self.emb_evaluator_unit,self.embedding_unit,top_k,cutoff,cache_key='unit_codes')
        if pp:
            prt='**Similar Unit Codes from Previous Designs**:\n\n'
            if not pps:
                prt+='### No similar unit codes found from the previous designs for the given unit.\n\n'
            else:
                prt+=f'### Found {len(pps)} similar unit codes from the previous designs for the given unit:\n\n'
                for i,p in pps:
                    prt+=f'#### {i} (Score: {scores[i]:.2f})\n\n<details><summary>Click to Expand</summary>\n\n```\n{self.unit_codes[i]}\n```\n\n</details>\n\n'
            return pps,prt
        else:
            return pps



    #### Get Paper Files by URL or ArXiv ID

    def get_html(self,url,save_path):
        if U.pexists(save_path):
            return U.read_file(save_path)
        response = requests.get(url)
        if response.status_code == 200:
            U.write_file(save_path,response.text)
            return response.text
        else:
            return None

    def download_pdf(self,url,save_path):
        if U.pexists(save_path):
            return U.read_file(save_path)
        try:
            pdf=urllib.request.urlretrieve(url, save_path)
            return pdf
        except:
            return None

    def get_paper_file(self,arxiv_id=None,pdf_url=None):
        # TODO: try also downloading the pdf from the arxiv_id first, pdf quality is better
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

    def search_external_details(self,query):
        # TODO: search for the contents in the external papers
        pass

    #### External Sources, give you mostly the abstract and title


    def _search_external(self,query,pretty=True,prompt=True,div=1) -> Union[None, List[Dict]]:
        # search for external papers
        s2_results=self.safe_search(self.search_s2,query,max(self.result_limits['s2']//div,3))
        arxiv_results=self.safe_search(self.search_arxiv,query,max(self.result_limits['arxiv']//div,2))
        pwc_results=self.safe_search(self.search_pwc,query,max(self.result_limits['pwc']//div,2))
        aggregated_results=self.safe_append_results(s2_results,arxiv_results+pwc_results)
        return aggregated_results
    
    def safe_append_results(self,results,aggregated_results,query=None):
        for r in results:
            if not self.exist_in_set(r,aggregated_results):
                if query:
                    r['search_query']=query
                aggregated_results.append(r)
        return aggregated_results

    def search_external(self,query,pretty=True,prompt=True,split_keywords=True) -> Union[None, List[Dict]]:
        # search for external papers
        if split_keywords:
            kws=[]
            keywords = [query] if isinstance(query,str) else query
            for kw in keywords:
                kw=kw.replace('keywords','').replace('keyword','').strip()
                for k in kw.split(','):
                    for kk in k.split('\n'):
                        kws.append(kk.strip())  
            query=kws
        if isinstance(query,str):
            aggregated_results=self._search_external(query,pretty,prompt)
        else:
            assert isinstance(query,list), 'query must be a string or a list of keywords'
            aggregated_results=[]
            for q in query:
                aggregated_results=self.safe_append_results(self._search_external(q,pretty,prompt,div=len(query)),aggregated_results,q)
        self.stream.write(f'##### *Found {len(aggregated_results)} related papers from external sources*')
        grouped_results=self.group_by_sources(aggregated_results)
        if pretty:
            return grouped_results, self.pretty_print(grouped_results,query,prompt)
        else:
            return grouped_results

    def group_by_sources(self,results):
        # group the results by the source
        grouped_results={}
        for r in results:
            if 's2_id' in r:
                if 'Semantic Scholar' not in grouped_results:
                    grouped_results['Semantic Scholar']=[]
                grouped_results['Semantic Scholar'].append(r)
            elif 'pwc_id' in r:
                if 'Papers with Code' not in grouped_results:
                    grouped_results['Papers with Code']=[]
                grouped_results['Papers with Code'].append(r)
            else:
                if 'ArXiv' not in grouped_results:
                    grouped_results['ArXiv']=[]
                grouped_results['ArXiv'].append(r)
        return grouped_results

    def pretty_print(self,grouped_results,query,prompt=True):
        num_results=sum([len(group) for group in grouped_results.values()])
        ppr=f'\n---\n## Found {num_results} related papers from {len(grouped_results)} external sources\n\n'
        if isinstance(query,str):
            ppr+=f'\n\nYour raw search query input to the search frame: {query}\n\n'
        else:
            ppr+=f'\n\nYour {len(query)} raw search queries input to the search frame: {", ".join(query)}\n\n'
        ppr+=f'Considering refining your search by improving the query keywords input.\n\n'
        for source in ['Semantic Scholar','ArXiv','Papers with Code']:
            if source not in grouped_results:
                continue
            group=grouped_results[source]
            ppr+=f'### {len(group)} related papers from {source}\n\n'
            for i,r in enumerate(group):
                ppr+=f'#### {i+1}. {r["title"]}\n\n'
                if 'search_query' in r:
                    ppr+=f'*From Search Query: {r["search_query"]}*\n\n'
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
                if not prompt:
                    if 's2_id' in r:
                        s2_url=f'https://www.semanticscholar.org/paper/{r["s2_id"]}'
                        ppr+=f'**Semantic Scholar URL:** {s2_url}\n\n'
                    if 'pwc_id' in r:
                        ppr+=f'**Papers with Code URL:** {r["url_abs"]}\n\n'
                    if 'arxiv_id' in r and r['arxiv_id']:
                        arxiv_url=f'https://arxiv.org/abs/{r["arxiv_id"]}'
                        ppr+=f'**ArXiv URL:** {arxiv_url}\n\n'
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
        if paper1['abstract'] and paper2['abstract']:
            if paper1['abstract'].strip()==paper2['abstract'].strip():
                return True
        return False

    def safe_search(self,fn,query,result_limit=10):
        # try:
        if result_limit<=0:
            return []
        return fn(query,result_limit)
        # except Exception as e:
        #     print(f"Error searching {fn.__name__}: {e}")
        #     return []

    def search_s2(self,query, result_limit=10,start_year=2010, top_conf_only=True, **fields) -> Union[None, List[Dict]]:
        # https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/post_graph_get_papers
        
        if self.s2_key_set:
            headers={"X-API-KEY": os.environ['S2_API_KEY']}
        else:
            headers={}
        info=f'*Searching Semantic Scholar for "{query}"*'
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
            headers=headers,
            params=params
        )
        try:
            rsp.raise_for_status()
        except Exception as e:
            self.stream.write(f'Error searching Semantic Scholar: {e}')
            return []
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

    def search_arxiv(self,query, result_limit=10, category=['cs.LG','cs.CL'], max_retries=3, retry_delay=1):#, start_year=2015):
        """
        Searches arXiv for papers in the specified category with the user query and time restriction.

        :param user_query: A string query to search for.
        :param category: The arXiv category to search in (default: 'cs.LG' for Machine Learning).
        :param max_results: The maximum number of results to return (default: 10).
        :param start_year: The minimum year of paper submission (default: 2015).
        :return: List of paper titles and summaries.
        """
        self.stream.write(f'*Searching arXiv for "{query}" in {", ".join(category)}...*')
        base_url = 'http://export.arxiv.org/api/query?'
        user_query=urllib.parse.quote(query, safe=":/?&=")
        category_str='+OR+'.join(['cat:'+i for i in category])
        search_query = f'{category_str}+AND+all:"{user_query}"'
        # date_filter = f'AND+submittedDate:[{start_year}0101+TO+*]' # not working
        query_string = f"search_query={search_query}&start=0&max_results={result_limit}"
        query_url = base_url + query_string

        for attempt in range(max_retries):
            try:
                response = feedparser.parse(query_url)
            except (ConnectionResetError, RequestException) as e:
                if attempt < max_retries - 1:
                    self.stream.write(f"*ArXiv search failed. Retrying in {retry_delay} seconds...*")
                    time.sleep(retry_delay)
                else:
                    self.stream.write(f"*ArXiv search failed after {max_retries} attempts: {str(e)}*")
                    return []

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
        self.stream.write(f'*Searching Papers with Code for "{query}"...*')
        ret=pwc_search_patched(self.pwc_client,query,items_per_page=result_limit)
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
                'url_abs':paper['url_abs'],
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


    ##### Perplexity.ai Web Search

    def search_perplexity(self,query,detail,analysis=None, instruct=None, size='large', max_tokens=2000): # perplexity search
        self.stream.write(f'*Searching web with Perplexity...*')
        url = "https://api.perplexity.ai/chat/completions"

        if analysis:
            analysis=f'\nHere is an analysis of the model that the researcher is trying to improve that may help you better understand the researcher\'s intent:\n{analysis}'
        else:
            analysis=''
        
        if instruct:
            content=PERPLEXITY_PROMPT_INSTRUCT.format(instruct=instruct)
        else:
            content=PERPLEXITY_PROMPT.format(query=query, detail=detail, analysis=analysis)
        payload = {
            "model": f"llama-3.1-sonar-{size}-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": PERPLEXITY_SYSTEM_INSTRUCT if instruct else PERPLEXITY_SYSTEM
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "top_p": 0.9,
            "return_citations": True,
            "search_domain_filter": ["perplexity.ai"],
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": "month",
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1
        }

        headers = {
            "Authorization": f"Bearer {os.environ['PERPLEXITY_API_KEY']}",
            "Content-Type": "application/json"
        }

        response = requests.request("POST", url, json=payload, headers=headers)
        try:
            RET = response.json()
        except Exception as e:
            self.stream.write(f'Error searching web with Perplexity: {e}')
            return None, None
        if 'error' in RET:
            self.stream.write(f'Error searching web with Perplexity: {RET["error"]}')
            return None, None
        else:
            self.stream.write(f'##### *Results collected from web search*')
            ret = RET['choices'][0]['message']['content']
            return RET,f'\n---\n## Web search results\n\n {ret}'

    #### Internal Sources

    def _load_texts(self,index_only=False):
        # try to load pdf-converted texts, if not exist, load html-converted texts, otherwise skip
        if index_only:
            if U.pexists(U.pjoin(self.libfiles_dir,'texts_index.json')):
                self.texts=U.load_json(U.pjoin(self.libfiles_dir,'texts_index.json'))
            if U.pexists(U.pjoin(self.libfiles_dir,'texts2_index.json')):
                self.texts2=U.load_json(U.pjoin(self.libfiles_dir,'texts2_index.json'))
            if U.pexists(U.pjoin(self.libfiles_dir,'textsp_index.json')):
                self.textsp=U.load_json(U.pjoin(self.libfiles_dir,'textsp_index.json'))
            if self.texts and self.texts2 and self.textsp:
                return

        self.stream.write('Loading raw_texts in internal library...')
        self.texts={}
        self.texts2={}
        self.textsp={}
        for name,lib,tail in [
                ('Primary',self.texts,''),
                ('Secondary',self.texts2,'2'),
                ('Plus',self.textsp,'p')]:
            
            ## XXX: REMOVE THIS
            if name!='Primary':
                continue

            text_dir=U.pjoin(self.files_dir,'texts'+tail)
            htext_dir=U.pjoin(self.files_dir,'htexts'+tail)
            if U.pexists(text_dir):  # pdfs first, overwrite htmls
                for i in tqdm(os.listdir(text_dir),desc=f'Loading splits {name}'):
                    if i.endswith('.txt'):
                        if index_only:
                            lib[i.split('.')[0]]=None
                        else:
                            lib[i.split('.')[0]]=U.read_file(U.pjoin(text_dir,i))
            if U.pexists(htext_dir): 
                for i in tqdm(os.listdir(htext_dir),desc=f'Loading splits {name}'):
                    if i.endswith('.txt'):
                        if i.split('.')[0] in lib:
                            continue
                        if index_only:
                            lib[i.split('.')[0]]=None
                        else:
                            lib[i.split('.')[0]]=U.read_file(U.pjoin(htext_dir,i))
        if index_only:
            U.save_json(self.texts,U.pjoin(self.libfiles_dir,'texts_index.json'))
            U.save_json(self.texts2,U.pjoin(self.libfiles_dir,'texts2_index.json'))
            U.save_json(self.textsp,U.pjoin(self.libfiles_dir,'textsp_index.json'))

    def _load_libs(self): # used for building the search library
        # load the primary and secondary libraries
        files_dir=U.pjoin(self.ptree.lib_dir,'..','files')
        lib_est_dir=U.pjoin(self.ptree.lib_dir,'..','tree_ext')
        self.lib={}
        self.lib2={}
        self.libp={}

        for name,lib,dir,tail in [
                ('Primary',self.lib,self.ptree.lib_dir,''),
                ('Secondary',self.lib2,U.pjoin(lib_est_dir,'secondary'),'2'),
                ('Plus',self.libp,U.pjoin(lib_est_dir,'plus'),'p')]:

            ### XXX: REMOVE THIS
            if name!='Primary':
                continue

            for i in os.listdir(dir):
                if i.endswith('.json'):
                    lib[i.split('.')[0]]=U.load_json(U.pjoin(dir,i))
            if U.pexists(U.pjoin(files_dir,'htmls'+tail))       :
                for i in os.listdir(U.pjoin(files_dir,'htmls'+tail)):   
                    if i.endswith('.html'):
                        lib[i.split('.')[0]]['html_path']=U.pjoin(files_dir,'htmls'+tail,i)
            if U.pexists(U.pjoin(files_dir,'pdfs'+tail)):
                for i in os.listdir(U.pjoin(files_dir,'pdfs'+tail)):    
                    if i.endswith('.pdf'):
                        lib[i.split('.')[0]]['pdf_path']=U.pjoin(files_dir,'pdfs'+tail,i)

    def _convert_libs_html(self):
        # convert the htmls in the libraries to text
        if self.lib is None:
            self._load_libs()
        for name,lib,folder in [
                ('Primary',self.lib,'htexts'),
                ('Secondary',self.lib2,'htexts2'),
                ('Plus',self.libp,'htextsp')]:
            save_dir=U.pjoin(self.files_dir,folder)
            U.mkdir(save_dir)
            for i in tqdm(lib,desc=f'Converting library {name} HTMLs'):
                save_path=U.pjoin(save_dir,f'{i}.txt')
                if 'html_path' in lib[i] and not U.pexists(save_path):
                    try:
                        self._convert_html_to_text(lib[i]['html_path'],save_path)
                    except Exception as e:
                        print(f'Error converting {lib[i]["html_path"]}: {e}')

    def _convert_libs_pdf(self,max_pages=0): # max_pages=0 means no limit, only for non-primary papers
        # convert the pdfs in the libraries to text, PDF conversion is expensive but have higher quality then htmls
        if self.lib is None:
            self._load_libs()
        for name,lib,folder in [
                ('Primary',self.lib,'texts'),
                ('Secondary',self.lib2,'texts2'),
                ('Plus',self.libp,'textsp')]:
            
            # XXX: only convert primary papers for now
            if name!='Primary':
                continue

            save_dir=U.pjoin(self.files_dir,folder)
            U.mkdir(save_dir)
            for i in tqdm(self.lib,desc=f'Converting library {name} PDFs'):
                save_path=U.pjoin(save_dir,f'{i}.txt')
                if 'pdf_path' in self.lib[i] and not U.pexists(save_path):
                    try:
                        max_pages=0 if name=='Primary' else max_pages
                        self._convert_pdf_to_text(self.lib[i]['pdf_path'],save_path,max_pages)
                    except Exception as e:
                        print(f'Error converting {self.lib[i]["pdf_path"]}: {e}')

    def _convert_pdf_to_text(self,file_path,save_path,max_pages=0):
        # use mathpix to convert the pdf to text https://mathpix.com/ 
        if max_pages>0:
            pdf=pypdf.PdfReader(file_path)
            if len(pdf.pages)>max_pages:
                return None # skip the paper
        
        loader=MathpixPDFLoader(file_path,mathpix_api_id=os.environ['MATHPIX_API_KEY'])
        data=loader.load()
        text=''
        for i in data:
            text+=i.page_content
        with open(save_path,'w',encoding='utf-8') as f:
            f.write(text)
        return text

    def _convert_html_to_text(self,file_path,save_path):
        # convert html to text using unstructured
        loader = UnstructuredHTMLLoader(file_path)
        data = loader.load()
        text=''
        for i in data:
            text+=i.page_content
        with open(save_path,'w',encoding='utf-8') as f:
            f.write(text)
        return text

    def _upload_file(self,file_path):
        RET=self.client.files.create(
            file=open(file_path, "rb",encoding='utf-8'),
            purpose="assistants"
        )
        return RET.id

    def get_index(self,index_name=None):
        if not index_name:
            index_name=self.index_name
        if not self.has_index(index_name):
            self.pc.create_index(
                name=index_name,
                dimension=3072, # assume openai text-embedding-3-large	
                metric="cosine", 
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-west-2'
                ) 
            ) 
        return self.pc.Index(index_name)

    def split_text(self,text,id,text_splitter=None):
        if not text_splitter:
            text_splitter = SemanticChunker(
                self.embedding_vs, breakpoint_threshold_type="gradient"
            )
        docs = text_splitter.create_documents([text])
        txts=[i.page_content for i in docs]
        embs=self.embedding_vs.embed_documents(txts)
        vectors=[]
        splits={}
        for idx,i in enumerate(docs):
            vectors.append({
                'id':f'{id}-{idx}',
                'values':embs[idx],
                'metadata':{'id':id,'splits':len(docs)},
            })
            splits[f'{id}-{idx}']=i.page_content
        return vectors,splits

    def _load_splits(self,load_vectors=False):
        if self.texts is None: # just load index for now
            self._load_texts(index_only=True)

        self.splits={}
        self.splits2={}
        self.splitsp={}
        self.vectors={}
        self.vectors2={}
        self.vectorsp={}
        for name,lib,tail,splits,vectors in [
                ('Primary',self.texts,'',self.splits,self.vectors),
                ('Secondary',self.texts2,'2',self.splits2,self.vectors2),
                ('Plus',self.textsp,'p',self.splitsp,self.vectorsp)]:
            
            ## XXX: REMOVE THIS
            if name!='Primary':
                continue
            
            for i in tqdm(lib,desc=f'Loading splits {name}'):
                U.mkdir(U.pjoin(self.libfiles_dir,'splits'+tail))
                if U.pexists(U.pjoin(self.libfiles_dir,'splits'+tail,f'{i}.json')):
                    split=U.load_json(U.pjoin(self.libfiles_dir,'splits'+tail,f'{i}.json'))
                    if load_vectors:
                        U.mkdir(U.pjoin(self.files_dir,'vectors'+tail))
                        vector=U.load_json(U.pjoin(self.files_dir,'vectors'+tail,f'{i}.json'))
                else:
                    if not lib[i]:
                        print(f'{i} not found in {name}')
                        self._load_texts()
                        if tail=='': lib=self.texts
                        elif tail=='2': lib=self.texts2
                        elif tail=='p': lib=self.textsp
                    # try:
                    vector,split=self.split_text(lib[i],i)
                    # except Exception as e:
                    #     print(f'Error splitting {i}: {e}')
                    #     vector,split={},{}
                    U.save_json(vector,U.pjoin(self.files_dir,'vectors'+tail,f'{i}.json'))
                    U.save_json(split,U.pjoin(self.libfiles_dir,'splits'+tail,f'{i}.json'))
                splits.update(split)
                if load_vectors:
                    vectors[i]=vector

    def _build_vector_stores(self):
        if self.vectors is None:
            self._load_splits(load_vectors=True)
        for namespace,vectors in [
                ('primary',self.vectors),
                ('secondary',self.vectors),
                ('plus',self.vectors)]:
            
            ## XXX: REMOVE THIS
            if namespace!='primary':
                continue

            for id in tqdm(vectors,desc=f'Upserting texts {namespace}'):
                if vectors[id]:
                    self.index.upsert(vectors=vectors[id],namespace=namespace)

    def _rerank(self,query,docs,result_limit):
        response = self.co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=docs,
            top_n=result_limit,
        )
        indices=[i.index for i in response.results] # 0-indexed
        relevance_scores=[i.relevance_score for i in response.results]
        return indices,relevance_scores

    def _get_split_by_id(self,id,namespace):
        if self.splits is None:
            self._load_splits()
        if namespace=='primary':
            return self.splits[id]
        elif namespace=='secondary':
            return self.splits2[id]
        elif namespace=='plus':
            return self.splitsp[id]
        else:
            raise ValueError(f'Unknown namespace: {namespace}')

    def _get_metainfo_by_id(self,id,namespace):
        if self.lib is None:
            self._load_libs()
        if namespace=='primary':
            return self.lib[id]
        elif namespace=='secondary':
            return self.lib2[id]
        elif namespace=='plus':
            return self.libp[id]
        else:
            raise ValueError(f'Unknown namespace: {namespace}')

    def _query_index(self,query,namespace,result_limit=10):
        embed=self.embedding_vs.embed_query(query)
        if self.rerank_ratio>0:
            top_k=int(result_limit//self.rerank_ratio)
        else:
            top_k=result_limit
        ret=self.index.query(
            namespace=namespace,
            vector=embed,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        matches=ret.matches
        split_ids=[i.id for i in matches]
        scores=[i.score for i in matches]
        docs=[self._get_split_by_id(i,namespace) for i in split_ids]
        ids=[i.metadata['id'] for i in matches]
        splits=[i.metadata['splits'] for i in matches]
        if self.rerank_ratio>0 and self.co is not None:
            indices,relevance_scores=self._rerank(query,docs,result_limit)
            scores=relevance_scores
            ids=[ids[i] for i in indices]
            docs=[docs[i] for i in indices]
            splits=[splits[i] for i in indices]
            split_ids=[split_ids[i] for i in indices]
        # group by id
        grouped_docs={}
        for i in range(len(ids)):
            if ids[i] not in grouped_docs:
                grouped_docs[ids[i]]=[]
            grouped_docs[ids[i]].append({
                'score':scores[i],
                'doc':docs[i],
                'split_id':split_ids[i],
                'splits':splits[i],
            })
            grouped_docs[ids[i]].sort(key=lambda x: x['score'], reverse=True)
        # metainfo per id
        metainfo={}
        for id in grouped_docs:
            metainfo[id]=self._get_metainfo_by_id(id,namespace)
        return grouped_docs,metainfo


    def search_lib_primary(self,query, result_limit=10) -> Union[None, List[Dict]]:
        # the selected ~300 model arch papers
        if result_limit<=0:
            return {},{}
        self.stream.write('*Searching primary library...*')
        grouped_docs,metainfo=self._query_index(query,'primary',result_limit)
        return grouped_docs,metainfo

    def search_lib_secondary(self,query, result_limit=10) -> Union[None, List[Dict]]:
        # the papers that are cited by the primary library, where their ideas come from  
        if result_limit<=0:
            return {},{}
        self.stream.write('*Searching secondary library...*')
        grouped_docs,metainfo=self._query_index(query,'secondary',result_limit)
        return grouped_docs,metainfo

    def search_lib_plus(self,query, result_limit=10) -> Union[None, List[Dict]]:
        # the papers recommended by S2 for the primary library  
        if result_limit<=0:
            return {},{}
        self.stream.write('*Searching library plus...*')
        grouped_docs,metainfo=self._query_index(query,'plus',result_limit)
        return grouped_docs,metainfo

    def search_internal(self,query,pretty=True,prompt=True) -> Union[None, List[Dict]]:
        # search for papers in the internal library
        sources={}
        if self.pc is not None:
            sources['Internal Library']=self.search_lib_primary(query,self.result_limits['lib'])
        # only primary now, testing
        # sources['Referencces of Library']=self.search_lib_secondary(query,self.result_limits['lib2'])
        # sources['Recommanded Papers of Library']=self.search_lib_plus(query,self.result_limits['libp'])

        total_chunks=0
        for source in sources:
            total_chunks+=sum([len(sources[i][0]) for i in sources])
        rerankinfo='' if self.rerank_ratio==0 else f' after reranking {int(total_chunks//self.rerank_ratio)} candidates'
        self.stream.write(f'##### *Found {total_chunks} related contents{rerankinfo}...*')
        if pretty:
            return sources,self.vs_pretty_print(sources,query,prompt)
        else:
            return sources
    
    def vs_pretty_print(self,sources,query,prompt=True):
        total_chunks=0
        for source in sources:
            total_chunks+=sum([len(sources[i][0]) for i in sources])
        total_papers=sum([len(sources[i][0]) for i in sources])
        ppr=f'\n---\n## Found {total_chunks} related chunks from {len(sources)} internal sources\n\n'
        ppr+=f'Your raw search query input to the search frame: \n\n{query}\n\n'
        ppr+=f'Considering refining your search by improving the query keywords input.\n\n'
        for source in sources:
            grouped_docs=sources[source][0]
            metainfo=sources[source][1]
            count=sum([len(grouped_docs[i]) for i in grouped_docs])
            ppr+=f'### {count} related chunks from {len(grouped_docs)} papers in {source}\n\n'
            group_scores={}
            for i in grouped_docs:
                group=grouped_docs[i]
                group_scores[i]=np.mean([j['score'] for j in group])
            group_scores=sorted(group_scores.items(),key=lambda x:x[1],reverse=True)

            for idx,id_score in enumerate(group_scores):
                id,score=id_score
                r=metainfo[id]
                ppr+=f'#### {idx+1}. {r["title"]} (Avg. Score: {score:.2f})\n\n'
                ppr+=f'*{", ".join(r["authors"]) if r["authors"] else "Anonymous"}*\n\n'
                ppr+=f'**Published in:** {r["venue"]} ({r["year"]})'
                ppr+=f'\t**Cited by** {r["citationCount"]}'
                ppr+=f'  (*Influential: {r["influentialCitationCount"]}*)\n\n'
                ppr+=f'**TL;DR:** {r["tldr"]}\n\n'
                if prompt:
                    ppr+=f'**Abstract:** {r["abstract"]}\n\n'
                group=grouped_docs[id]
                for did,doc in enumerate(group):
                    ppr+=f'##### *Relevant Chunk: No. {int(doc["split_id"].split("-")[1])+1}/{int(doc["splits"])} (Score: {doc["score"]:.2f})*\n\n'
                    ppr+=f'```\n{doc["doc"]}\n```\n\n'
                if not prompt:
                    ppr+=f'<details><summary>Show details</summary>\n\n'
                    ppr+=f'**Semantic Scholar ID:** {r["s2id"]}\n\n'
                    ppr+=f'**Abstract:** {r["abstract"]}\n\n'
                    if r['code'] is not None:
                        ppr+=f'###### Reference Code\n\n'
                        ppr+=f'```python\n{r["code"]}\n```\n\n'
                    ppr+=f'</details>\n\n'
        return ppr


