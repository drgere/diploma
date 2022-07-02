import random
from pathlib import Path

from analytics.utils import loading_json, writing_json

def data_model(modelout, refdata, rndseed=816223):
    mt = loading_json(modelout)
    mt_id = {it['id']:it for it in mt}
    rf = loading_json(refdata)
    rf_id = {it['id']:it for it in rf}
    id = sorted(list(set(mt_id)), key=lambda id:int(id.split('.')[-1]))
    out_items = []
    for i in id:
        itm = {'id':i}
        itm['word'] = rf_id[i]['word']
        itm['gloss-ref']=rf_id[i]['gloss']
        itm['gloss-out']=mt_id[i]['gloss']
        itm['example']=rf_id[i]['example']
        out_items.append(itm)
    file_over = Path(modelout).with_suffix('.refalign.json')
    writing_json(file_over, out_items)
    random.seed(rndseed)
    random.shuffle(out_items)
    file_over = Path(modelout).with_suffix('.refalign.rndshuff.json')
    writing_json(file_over, out_items)

if __name__ == '__main__':
    data_model('/home/olygina/project/semeval2022/submission/defmod/defmod.en.sgns.json',
               '/home/olygina/codwoe/reference_data/en.test.defmod.complete.json')
