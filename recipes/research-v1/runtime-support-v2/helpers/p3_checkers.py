"""Pure evaluation callback helper. Caller supplies private row payloads separately."""
import json
import re

def check(row,text):
    family=row['family'];p=row['payload']
    if family=='JSON':
        duplicate=[]
        def object_pairs(pairs):
            obj={}
            for key,value in pairs:
                if key in obj:duplicate.append(key)
                obj[key]=value
            return obj
        try:
            obj=json.loads(text,object_pairs_hook=object_pairs,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            valid=True
        except (ValueError,TypeError):obj=None;valid=False
        shape=isinstance(obj,dict) and not duplicate and set(obj)==set(p['keys'])
        values=bool(isinstance(obj,dict) and all(k in obj and isinstance(obj[k],str) and
                   obj[k]==next(v for key,v in p['fields'] if key==k) for k in p['keys']))
        return {'pass':bool(valid and shape and values),'valid_json':valid,'duplicate_keys':bool(duplicate),
                'exact_key_set':bool(shape),'correct_values':values}
    if family=='MEMBERSHIP':
        present=any(item==p['target'] for item in p['items'])
        expected=p['present_label'] if present else p['absent_label']
        predicted='positive' if text==p['present_label'] else 'negative' if text==p['absent_label'] else 'invalid'
        return {'pass':text==expected,'actual_class':'positive' if present else 'negative','predicted_class':predicted}
    expected=p['value'] if family=='COPY' else next(v for k,v in p['fields'] if k==p['key'])
    exact=text==expected
    included=bool(re.search(r'(?<!\w)'+re.escape(expected)+r'(?!\w)',text))
    return {'pass':exact,'correct_value_exact':exact,
            'extra_content_with_expected_value':included and not exact,
            'wrong_or_missing_value':not included,
            'value_presence_is_lexical_diagnostic_only':True}
