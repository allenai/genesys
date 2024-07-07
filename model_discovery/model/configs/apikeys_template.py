from dataclasses import dataclass

# Replace the placeholders with your API keys and rename this file to apikeys.py
@dataclass
class APIKeys:
    '''API keys for external services.'''

    openai: str = 'YOUR-APIKEY'
    openai_org: str = 'YOUR-ORG-ID'
    openai_proj: str = 'YOUR-PROJ-ID' # seems optional

