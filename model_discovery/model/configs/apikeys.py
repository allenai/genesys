from dataclasses import dataclass


@dataclass
class APIKeys:
    '''API keys for external services.'''

    openai: str = ''
    openai_org: str = ''
    openai_proj: str = 'Default project'
    s2: str = ''

