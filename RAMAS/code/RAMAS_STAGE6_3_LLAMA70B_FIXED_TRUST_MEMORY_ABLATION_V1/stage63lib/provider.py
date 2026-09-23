"""Stateless Ollama calls with explicit shared serving settings."""
import json
import time
import urllib.request
from .legacy import SYSTEM_PROMPT
from .journal import digest

class ProviderIdentityError(RuntimeError):
    pass

class OllamaProvider:
    kind='ollama'
    def __init__(self, config, agent_config):
        self.config=config
        self.agent=agent_config
        if config['model']!='llama3.3:70b':
            raise ProviderIdentityError('Only llama3.3:70b is allowed')
        self.root=config['endpoint'].removesuffix('/api/chat')
        if self.root==config['endpoint']:
            raise ValueError('Ollama endpoint must end with /api/chat')
        self.pinned=self.inspect()

    def request(self, suffix, value=None):
        request=urllib.request.Request(self.root+suffix,
            data=None if value is None else json.dumps(value).encode(),
            headers={'Content-Type':'application/json'},
            method='GET' if value is None else 'POST')
        with urllib.request.urlopen(request,timeout=self.config['timeout_seconds']) as response:
            return json.loads(response.read())

    def inspect(self):
        tags=self.request('/api/tags')
        matches=[x for x in tags.get('models',[]) if x.get('name')==self.config['model']]
        if len(matches)!=1 or not matches[0].get('digest'):
            raise ProviderIdentityError('llama3.3:70b is missing or has no digest; no automatic model substitution')
        version=self.request('/api/version')
        show=self.request('/api/show',{'model':self.config['model']})
        stable={k:show.get(k) for k in ['parameters','template','system','model_info','details','capabilities']}
        return dict(model=self.config['model'],model_digest=matches[0]['digest'],
                    details=matches[0].get('details'),ollama_version=version.get('version'),
                    serving_metadata_sha256=digest(stable),options=self.options())

    def options(self):
        return {k:self.config[k] for k in ['temperature','seed','num_ctx','num_predict']}

    def assert_identity(self):
        if self.inspect()!=self.pinned:
            raise ProviderIdentityError('Ollama model or serving metadata changed during this run')

    def complete(self,payload):
        self.assert_identity()
        schema={'type':'object','additionalProperties':False,
            'required':['action','confidence','reason_codes','cited_memory_ids'],
            'properties':{
                'action':{'type':'string','enum':self.agent['actions']},
                'confidence':{'type':'number','minimum':0,'maximum':1},
                'reason_codes':{'type':'array','minItems':1,'items':{'type':'string','enum':self.agent['reason_codes']}},
                'cited_memory_ids':{'type':'array','items':{'type':'string'}}}}
        body={'model':self.config['model'],'stream':False,'format':schema,
              'messages':[{'role':'system','content':SYSTEM_PROMPT},
                          {'role':'user','content':json.dumps(payload,sort_keys=True,allow_nan=False)}],
              'options':self.options()}
        started=time.monotonic()
        outer=self.request('/api/chat',body)
        if outer.get('error'):
            raise RuntimeError('Ollama error: '+str(outer['error']))
        if outer.get('model')!=self.config['model']:
            raise ProviderIdentityError('Response model does not match llama3.3:70b')
        if not outer.get('done'):
            raise RuntimeError('Ollama response did not complete')
        raw=outer.get('message',{}).get('content')
        if not isinstance(raw,str):
            raise RuntimeError('Ollama did not provide message.content')
        return raw,outer,time.monotonic()-started
