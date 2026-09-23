"""Durable per-call journal. Rebuild state; never resample a saved response."""
from pathlib import Path
import hashlib
import json
import os

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def atomic_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.pending')
    with tmp.open('w') as f:
        f.write(canonical(value)+'\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    fd=os.open(str(path.parent), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

class Journal:
    def __init__(self, output, provider, identity):
        self.output=Path(output)
        self.provider=provider
        self.identity=identity
        self.new_calls=0
        self.reused_calls=0

    def complete(self, key, payload):
        # Keys are engine-owned; this is not an arbitrary-path API.
        if not all(c.isalnum() or c in '-_' for c in key):
            raise ValueError('Invalid journal key')
        path=self.output/'calls'/(key+'.json')
        pending=self.output/'pending_calls'/(key+'.json')
        request_hash=digest(payload)
        if path.exists():
            entry=json.loads(path.read_text())
            checksum=entry.pop('record_sha256')
            if digest(entry)!=checksum or entry['identity']!=self.identity or entry['request_sha256']!=request_hash:
                raise RuntimeError('Journal identity/request/hash mismatch: '+key)
            if entry['request']!=payload or entry['response_sha256']!=digest(entry['raw_response']):
                raise RuntimeError('Journal payload mismatch: '+key)
            if pending.exists():
                prior=json.loads(pending.read_text())
                if prior['request_sha256']!=request_hash or prior['identity']!=self.identity:
                    raise RuntimeError('Pending marker differs from durable response: '+key)
                pending.unlink()
            self.reused_calls+=1
            return entry['raw_response'],entry['provider_response'],entry['latency_seconds']
        # Save the exact intended request before sending it. A process may die
        # after the server responds but before journalling; only that unsaved
        # response can require a retry. Already durable responses never rerun.
        if pending.exists():
            prior=json.loads(pending.read_text())
            if prior['request_sha256']!=request_hash or prior['identity']!=self.identity:
                raise RuntimeError('Pending call does not match restored state: '+key)
        atomic_json(pending,dict(identity=self.identity,request_sha256=request_hash,request=payload))
        try:
            raw,outer,latency=self.provider.complete(payload)
        except Exception as exc:
            atomic_json(self.output/'last_transport_failure.json',dict(key=key,error_type=type(exc).__name__,message=str(exc)))
            raise  # Stop before scoring this decision. Never trade on an outage.
        entry=dict(identity=self.identity,request=payload,request_sha256=request_hash,
                   raw_response=raw,response_sha256=digest(raw),provider_response=outer,
                   latency_seconds=latency)
        entry['record_sha256']=digest(entry)
        atomic_json(path,entry)
        pending.unlink(missing_ok=True)
        self.new_calls+=1
        return raw,outer,latency
