"""Reacquire the selected CC0 road scan from Poly Haven with hash checks."""
import json,hashlib,urllib.request,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
UA={'User-Agent':'IstanaOpen-EducationalSubmission/1.0'}
def fetch(url):return urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=120).read()
def main():
    metadata=json.loads(fetch('https://api.polyhaven.com/files/asphalt_02'))
    def channel(pair):
        key,label=pair;entry=metadata[key]['4k']['jpg']
        out=ROOT/'SourceAssets/Surfaces'/('PH_asphalt_02_'+label+'_4k.jpg')
        data=out.read_bytes() if out.exists() else fetch(entry['url'])
        if hashlib.md5(data).hexdigest()!=entry['md5']:raise ValueError('Provider checksum mismatch: '+entry['url'])
        out.write_bytes(data)
        return dict(path=out.relative_to(ROOT).as_posix(),source=entry['url'],url='https://polyhaven.com/a/asphalt_02',license='CC0-1.0',sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(channel,[('Diffuse','Diffuse'),('nor_dx','NormalDX'),('Rough','Roughness')]))
    m=ROOT/'SourceAssets/manifest.json';old=json.loads(m.read_text())
    m.write_text(json.dumps([x for x in old if x['path'] not in {r['path'] for r in records}]+records,indent=2))
    print('Downloaded and verified',len(records),'4K channels',sum(x['bytes'] for x in records),'bytes')
if __name__=='__main__':main()
