"""Fixed-snapshot, CPU-only pilot proposal selection. Never executes corpus code.

Text normalization is exclusively for grouping/overlap keys. Stored messages
and tokenizer input retain their original strings. No teacher or trainer runs.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
import unicodedata

REPO = Path('.')
HANDOFF = REPO / 'runs/D4_completion_and_SFT_pilot_handoff'
PREP = HANDOFF / 'pilot_v2_preparation'
SNAPSHOT = PREP / 'snapshot'
D1_ROOT = Path('artifacts/instruction_preparation/d1_2_targeted_content_quality_cleanup_v1/output')
TOKENIZER = REPO / 'runs/g_production_2026-08-21/release/tokenizer.json'
TOKENIZER_SHA = 'd8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce'
SUITE = PREP / 'evaluation/DEVELOPMENT_SUITE.jsonl'
SUITE_SHA = '9633d0d916293610b9664b5f6565144c154f485bd3fcf9a51c8932ef771227fd'
SEED = 'D4-PILOT-V2-FIXED-2026-09-06-1300'
MIX = {'FOUNDATION': .65, 'TEXT_TRANSFORM': .20, 'PYTHON_BASIC': .08, 'PRACTICAL_CHAT': .07}
ALLOWED_SOURCES = {'openhermes-50k', 'smol-magpie-ultra-short', 'self-oss-instruct'}


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode('utf-8')).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def save_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(canonical(row) + '\n')


def norm(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def rank(identity, purpose):
    return digest(SEED + '\0' + purpose + '\0' + str(identity))


def words(text):
    return re.findall(r"[\w']+", norm(text))


def ngrams(text, n=13):
    tokens = words(text)
    return {' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


GENERIC = re.compile(r'^(?:hi|hello|hey|thanks|thank you|good (?:morning|afternoon|evening)|how are you)[!.? ]*$', re.I)


def grouping_keys(row):
    messages = row['messages']
    normalized = [(m['role'], norm(m['content'])) for m in messages]
    keys = ['conversation:' + canonical(normalized)]
    substantive = [m for m in messages if m['role'] == 'user' and not GENERIC.fullmatch(m['content'].strip())]
    if substantive:
        # Full user task context is conservative across differing source boilerplate.
        keys.append('substantive_users:' + canonical([norm(m['content']) for m in substantive]))
        non_assistant = [(m['role'], norm(m['content'])) for m in messages if m['role'] != 'assistant' and
                         not (m['role'] == 'user' and GENERIC.fullmatch(m['content'].strip()))]
        keys.append('full_nonassistant_context:' + canonical(non_assistant))
        keys.append('full_generation_context:' + canonical(normalized[:-1]))
    # Every supervised turn has its own complete preceding context. A single
    # turn task and a conversation beginning with that task share a split.
    for index, message in enumerate(messages):
        if message['role'] != 'assistant': continue
        prefix = messages[:index]
        prefix_users = [norm(m['content']) for m in prefix if m['role']=='user' and not GENERIC.fullmatch(m['content'].strip())]
        if prefix_users:
            keys.append('supervised_prompt_prefix:' + canonical([(m['role'],norm(m['content'])) for m in prefix]))
            keys.append('supervised_substantive_users_prefix:' + canonical(prefix_users))
    return keys


class UnionFind:
    def __init__(self, n): self.parent = list(range(n))
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a, b)] = min(a, b)


def family_indices(value):
    """Use explicit existing D1 family associations, including truncated evidence."""
    result = []
    if isinstance(value, dict):
        if 'retained_original_row_index' in value and 'rejected_original_row_indices' in value:
            result.append([value['retained_original_row_index']] + value['rejected_original_row_indices'])
        for child in value.values(): result.extend(family_indices(child))
    elif isinstance(value, list):
        for child in value: result.extend(family_indices(child))
    return result


def assign_groups(rows, prior_summary):
    uf, owner, links = UnionFind(len(rows)), {}, Counter()
    for index, row in enumerate(rows):
        for key in grouping_keys(row):
            if key in owner:
                uf.union(index, owner[key]); links[key.split(':', 1)[0]] += 1
            else: owner[key] = index
    by_original = {row['original_row_index']: i for i, row in enumerate(rows)}
    families = family_indices(prior_summary)
    for family in families:
        present = [by_original[i] for i in family if i in by_original]
        for index in present[1:]:
            uf.union(present[0], index); links['known_D1_family'] += 1
    members = defaultdict(list)
    for i in range(len(rows)): members[uf.find(i)].append(i)
    for group in members.values():
        identity = digest('\n'.join(sorted(rows[i]['audit_id'] for i in group)))
        split = 'validation' if int(rank(identity, 'split')[:16], 16) / 2**64 < 1 / 21 else 'train'
        for index in group:
            rows[index]['group_id'] = identity
            rows[index]['assigned_split'] = split
    return {'groups': len(members), 'duplicate_groups': sum(len(g) > 1 for g in members.values()),
            'largest_group': max(map(len, members.values()), default=0), 'links': dict(links),
            'prior_family_records_available': len(families),
            'prior_family_limit': 'Existing summary contains top groups, with some truncated memberships; no complete semantic family registry exists in candidate records.'}


NONPY = re.compile(r'\b(?:javascript|typescript|ruby|java|php|golang|rust|kotlin|swift|perl|haskell|scala|lua|fortran|matlab|powershell|bash|sql|html|css)\b|c\+\+|c#|\b(?:c|r) (?:language|program|code)\b', re.I)
PY = re.compile(r'\bpython(?:\s?3)?\b|(?:^|\n)\s*(?:async\s+)?def\s+\w+\s*\(|(?:^|\n)\s*(?:from\s+\w+\s+import|import\s+\w+)', re.I)
PROGRAM = re.compile(r'```|\b(?:write|implement|create|debug|develop|design)\b.{0,80}\b(?:function|program|method|script|algorithm|code)\b|\bprogramming\b', re.I | re.S)
LIBRARY = re.compile(r'\b(?:pytorch|torch|tensorflow|keras|numpy|pandas|scipy|scikit|sklearn|opencv|matplotlib|selenium|django|flask|fastapi|sympy|boto3|sqlalchemy|beautifulsoup|requests|cuda|tensor|tensors|neural network|machine learning|distributed|multiprocessing|multithreading|asynchronous|socket|database|webscrap|web scrap|http server|rest api|api endpoint)\b', re.I)
SYSTEM_CODE = re.compile(r'\b(?:file system|filesystem|directory|directories|file path|read (?:a |the )?file|write (?:a |the )?file|network|subprocess|shell command|operating system|environment variable|command.line|download|upload|serialize.*file|permission)\b', re.I)
ADVANCED = re.compile(r'\b(?:calculus|integral|integration by parts|derivative|differential equation|eigenvalue|eigenvector|linear algebra|matrix multiplication|matrices|topology|group theory|number theory|modular arithmetic|prove that|formal proof|theorem|lemma|induction proof|combinatorial|binomial distribution|standard deviation|bayesian|regression|hypothesis test|markov|fourier|laplace|hamiltonian|dijkstra|bellman.ford|shortest path|dynamic programming|competitive programming|time complexity|space complexity|np.hard|np.complete|traveling salesman|minimum spanning|binary search tree|red.black|avl tree|suffix array|suffix tree|segment tree|fenwick|backtracking|optimization algorithm|recursive descent|regular language|turing machine|chromatic|asymptotic|O\(n|O\(log)\b|\\(?:frac|sum|int|lim|prod)\b', re.I)
NICHE = re.compile(r'\b(?:clinical trial|diagnos(?:is|e)|prescri(?:be|ption)|dosage|pharmacology|oncology|litigation|jurisdiction|case law|contract law|tax liability|financial derivative|stock valuation|options pricing|quantum mechanics|quantum field|molecular orbital|spectroscopy|thermodynamic cycle|fluid dynamics|semiconductor fabrication|genomic|proteomic|biochemical pathway)\b', re.I)
TRANSLATION = re.compile(r'\btranslat(?:e|ion|ing)\b|\b(?:in|into|to) (?:french|german|spanish|italian|russian|chinese|japanese|korean|arabic|hindi|bulgarian|portuguese|latin|turkish|hebrew|urdu|bengali)\b', re.I)
ROLE_ASSIGNMENT = re.compile(r"(?:^|\n)(?:you are|you're)\s+(?!given\b|provided\b|asked\b|tasked\b|required\b|allowed\b|going\b|to\b)|\b(?:assume the role|playing the role|i want you to play|let's.*(?:setting|scenario)|imagine yourself as)\b", re.I)
FOLLOWUP = re.compile(r"\b(?:that|this|it|those|these|more|also|instead|why|can i|can you|how can|what about|what if)\b", re.I)
ROLEPLAY = re.compile(r'\b(?:role.?play|pretend (?:to be|you are)|act as (?:a |an |the )?(?:character|wizard|pirate|vampire|doctor|lawyer)|fantasy world|fanfiction|fan fiction|erotic|screenplay|novel chapter|epic poem)\b', re.I)
TRANSFORM = re.compile(r'\b(?:rewrite|rewriting|paraphrase|summari[sz]e|summari[sz]ation|proofread|sentiment|classify|classification|extract|extraction|rephrase|grammar|grammatical|alphabeti[sz]e|reorder|sort these words|convert.*(?:uppercase|lowercase)|positive or negative|negative or positive|topic label)\b|\b(?:following|given|provided) (?:text|paragraph|passage|review|sentence|context)\b', re.I)
CHAT = re.compile(r'\b(?:how are you|how was your|feeling (?:sad|happy|nervous)|i feel|thank you|thanks for|nice to meet|good morning|good evening|help me (?:plan|decide)|what should i (?:do|say)|conversation|follow.up|can you (?:make it|explain that|clarify|simplify)|tell me more)\b', re.I)
ISSUE = re.compile(r'\b(?:incorrect|wrong|inaccurate|invalid|hallucinated|misleading|unsupported|contradicts)\b|(?:example|sample).{0,40}(?:error|mismatch)|fails to|does not (?:handle|match|respect)|missing (?:input|required)', re.I)
MISSING = re.compile(r'(?:summari[sz]e|rewrite|paraphrase|proofread|analy[sz]e|translate|extract).{0,60}(?:following|below)\s+(?:text|paragraph|passage|article|document|sentence)\s*[:.]?\s*$', re.I | re.S)
CHECKABLE = re.compile(r'\b(?:exactly|at least|at most|no more than|only|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:words?|sentences?|bullet points?|paragraphs?|items?|lines?)\b|\b(?:start|end|begin) with\b|\b(?:uppercase|lowercase|alphabetical)\b', re.I)
ALLOWED_IMPORTS = {'math', 're', 'string', 'collections', 'itertools', 'functools', 'statistics', 'operator', 'typing', 'copy', 'decimal', 'fractions', 'bisect'}
FORBIDDEN_CALLS = {'eval', 'exec', 'compile', '__import__', 'open', 'input', 'getattr', 'setattr', 'delattr', 'globals', 'locals', 'breakpoint', 'exit', 'quit'}


def code_blocks(text):
    return re.findall(r'```([^\n`]*)\n(.*?)```', text, flags=re.S)


def python_scope(text):
    blocks = code_blocks(text)
    assistant_code = [body for language, body in blocks if not language.strip() or language.strip().lower() in ('python', 'python3', 'py')]
    if sum(len(body.splitlines()) for body in assistant_code) > 65:
        return 'python_code_too_long'
    for code in assistant_code:
        # Parse only; no compile/eval/exec/import of dataset programs.
        try: tree = ast.parse(code)
        except SyntaxError: return 'python_fence_not_parseable_defer'
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.With, ast.AsyncWith)):
                return 'python_advanced_structure'
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [alias.name.split('.')[0] for alias in node.names] if isinstance(node, ast.Import) else [(node.module or '').split('.')[0]]
                if any(module not in ALLOWED_IMPORTS for module in modules): return 'python_external_or_system_import'
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                return 'python_system_or_dynamic_execution'
            if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
                return 'python_introspection_defer'
            if isinstance(node, ast.FunctionDef) and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == node.name for c in ast.walk(node)):
                return 'python_recursive_complexity_defer'
    return None


def scope_decision(row, explicit_exclusions):
    if row['audit_id'] in explicit_exclusions: return explicit_exclusions[row['audit_id']], None
    messages = row['messages']
    if any(set(m) != {'role', 'content'} or not isinstance(m['content'], str) for m in messages):
        return 'unexpected_message_schema', None
    users = '\n'.join(m['content'] for m in messages if m['role'] == 'user')
    assistants = '\n'.join(m['content'] for m in messages if m['role'] == 'assistant')
    text = users + '\n' + assistants
    if TRANSLATION.search(users): return 'translation_centered_or_nonenglish_target', None
    letters = [c for c in assistants if c.isalpha()]
    if letters and sum(ord(c) > 127 for c in letters) / len(letters) > .08:
        return 'nonenglish_assistant_content', None
    if ROLEPLAY.search(users) or (ROLE_ASSIGNMENT.search(users) and (row.get('task_type')=='CONVERSATION' or sum(m['role']=='assistant' for m in messages)>1)):
        return 'elaborate_persona_or_story', None
    if ADVANCED.search(text): return 'advanced_math_or_algorithm', None
    if NICHE.search(users): return 'specialized_professional_scope', None
    if LIBRARY.search(text): return 'specialized_library_or_system_task', None
    if ISSUE.search((row.get('judgment') or {}).get('reason', '')):
        return 'existing_judge_rationale_flags_content_issue', None
    if any(MISSING.search(m['content']) for m in messages if m['role'] == 'user'):
        return 'missing_input_cue_defer', None
    if sum(m['role'] == 'assistant' for m in messages) > 4:
        return 'too_many_supervised_turns_for_focused_pilot', None
    if any(len(m['content'].split()) > 500 for m in messages if m['role'] == 'assistant'):
        return 'long_or_elaborate_answer', None
    if len(re.findall(r'\d+(?:\.\d+)?', users)) >= 8 and re.search(r'\b(?:calculate|compute|solve|probability|equation)\b', users, re.I):
        return 'calculation_heavy', None
    programming = bool(PROGRAM.search(text) or row.get('task_type') == 'PYTHON_BASIC')
    is_python = bool(PY.search(text))
    if programming and NONPY.search(text): return 'non_python_programming', None
    fences = [lang.strip().lower() for lang, _ in code_blocks(text)]
    if any(lang not in ('', 'python', 'python3', 'py', 'text', 'plaintext', 'json') for lang in fences):
        return 'non_python_or_unrecognized_code_fence', None
    if programming and SYSTEM_CODE.search(text): return 'file_network_system_heavy_code', None
    if programming and not is_python: return 'programming_language_unclear_or_not_python', None
    if is_python:
        issue = python_scope(assistants)
        if issue: return issue, None
        return None, 'PYTHON_BASIC'
    if TRANSFORM.search(users): return None, 'TEXT_TRANSFORM'
    later_users = [m['content'] for m in messages if m['role']=='user'][1:]
    if any(len(text.split()) <= 35 and FOLLOWUP.search(text) for text in later_users):
        return None, 'PRACTICAL_CHAT'
    if (row.get('task_type') == 'CONVERSATION' and CHAT.search(users)) or (len([m for m in messages if m['role']=='assistant']) > 1 and CHAT.search(users)):
        return None, 'PRACTICAL_CHAT'
    if row.get('task_type') in {'GENERAL_QA', 'INSTRUCTION', 'TEXT_TRANSFORM', 'CONVERSATION'}:
        return None, 'FOUNDATION'
    return 'unclear_basic_task_fit', None


def turn_category(user, assistant, row_category, previous):
    pair = user + '\n' + assistant
    if PY.search(pair): return 'PYTHON_BASIC'
    if row_category == 'PYTHON_BASIC' and previous == 'PYTHON_BASIC' and len(user.split()) < 30:
        return 'PYTHON_BASIC'
    if TRANSFORM.search(user): return 'TEXT_TRANSFORM'
    if row_category == 'PRACTICAL_CHAT': return 'PRACTICAL_CHAT'
    return 'FOUNDATION'


def frozen_suite_index(suite_rows):
    exact, gram = set(), set()
    for row in suite_rows:
        for message in row['messages']:
            if message['role'] == 'user': exact.add(norm(message['content']))
            gram.update(ngrams(message['content']))
        reference = row.get('reference')
        if isinstance(reference, str): gram.update(ngrams(reference))
    return exact, gram


def suite_overlap(row, index):
    exact, gram = index
    for message in row['messages']:
        if message['role'] == 'user' and norm(message['content']) in exact:
            return 'development_suite_exact_user_prompt'
        if message['role'] != 'system' and ngrams(message['content']) & gram:
            return 'development_suite_13gram_prompt_context_or_reference'
    return None


def format_metadata(row, tok):
    from src.chat_template import encode_chat, IGNORE_INDEX, EOS_ID, BOS_ID, PAD_ID, USER_ID, SYSTEM_ID, ASSISTANT_ID
    ids, labels = encode_chat(tok, row['messages'], default_system=None)
    if len(ids) > 2048: raise ValueError('overlength_reject_without_truncation')
    supervised = sum(label != IGNORE_INDEX for label in labels[1:])
    if supervised == 0: raise ValueError('all_masked')
    if ids[0] != BOS_ID or labels[0] != IGNORE_INDEX or len(ids) != len(labels): raise ValueError('formatter_contract')
    counts, turn_rows, user, previous = Counter(), [], '', None
    for i, message in enumerate(row['messages']):
        if message['role'] == 'user': user = message['content']
        elif message['role'] == 'assistant':
            number = len(tok.encode(message['content']).ids) + 1
            category = turn_category(user, message['content'], row['pilot_task'], previous)
            previous = category
            counts[category] += number
            turn_rows.append({'message_index': i, 'task': category, 'supervised_tokens_with_eos': number,
                              'python_explicit_content_cue': bool(PY.search(user + '\n' + message['content']))})
    if sum(counts.values()) != supervised: raise ValueError('shifted_supervision_attribution_mismatch')
    if sum(label == EOS_ID for label in labels[1:]) != len(turn_rows): raise ValueError('assistant_eos_mismatch')
    if any(label in (BOS_ID, PAD_ID, USER_ID, SYSTEM_ID, ASSISTANT_ID) for label in labels[1:] if label != IGNORE_INDEX):
        raise ValueError('supervised_control_token')
    return {'sequence_tokens': len(ids), 'shifted_supervised_tokens': supervised,
            'assistant_turns': len(turn_rows), 'per_turn': turn_rows, 'task_supervised_tokens': dict(counts),
            'python_lower_tokens': sum(t['supervised_tokens_with_eos'] for t in turn_rows if t['python_explicit_content_cue']),
            'python_upper_tokens': supervised if any(t['task']=='PYTHON_BASIC' for t in turn_rows) else 0,
            'has_system': row['messages'][0]['role'] == 'system',
            'checkable_instruction_cues': CHECKABLE.findall('\n'.join(m['content'] for m in row['messages'] if m['role']=='user'))}


def solve_linear(matrix, values):
    augmented = [list(row) + [value] for row, value in zip(matrix, values)]
    n = len(values)
    for i in range(n):
        pivot = max(range(i,n), key=lambda j: abs(augmented[j][i]))
        augmented[i], augmented[pivot] = augmented[pivot], augmented[i]
        denominator = augmented[i][i]
        if abs(denominator) < 1e-10: raise ValueError('task mix unavailable: singular attribution matrix')
        augmented[i] = [x/denominator for x in augmented[i]]
        for j in range(n):
            if j != i:
                factor = augmented[j][i]
                augmented[j] = [a-factor*b for a,b in zip(augmented[j],augmented[i])]
    return [line[-1] for line in augmented]


def choose_token_mix(rows, target_count):
    """One simple actual-token vector correction, then deterministic prefixes."""
    tasks = list(MIX)
    buckets = {task: [] for task in tasks}
    for row in rows: buckets[row['pilot_task']].append(row)
    for task in buckets: buckets[task].sort(key=lambda r: rank(r['group_id'], 'quota-' + task))
    supply = {task:sum(r['shifted_supervised_tokens'] for r in pool) for task,pool in buckets.items()}
    if any(not value for value in supply.values()): raise ValueError('requested task family has no eligible supply')
    means = {task:supply[task]/len(buckets[task]) for task in tasks}
    # Columns are observed per-assistant token composition of each scope bucket.
    matrix = [[sum(r['task_supervised_tokens'].get(actual,0) for r in buckets[task])/supply[task]
               for task in tasks] for actual in tasks]
    weights = dict(zip(tasks, solve_linear(matrix,[MIX[task] for task in tasks])))
    if any(value <= 0 for value in weights.values()): raise ValueError('target vector infeasible without additional content selection')
    total_for_count = target_count / sum(weights[task]/means[task] for task in tasks)
    max_by_supply = {task:supply[task]/weights[task] for task in tasks}
    total = min(total_for_count,*max_by_supply.values())
    for _ in range(5):
        selected, quotas = [], {}
        for task,pool in buckets.items():
            quota,used,chosen=total*weights[task],0,[]
            for row in pool:
                number=row['shifted_supervised_tokens']
                if used+number > quota:
                    if abs(used+number-quota) < abs(used-quota):
                        chosen.append(row);used+=number
                    break
                chosen.append(row);used+=number
            selected.extend(chosen)
            quotas[task]={'available_rows':len(pool),'available_tokens':supply[task],
                          'token_target':quota,'selected_rows':len(chosen),
                          'selected_tokens_by_conversation_category':used}
        if len(selected)<=target_count: break
        total *= target_count/len(selected)*.999
    if len(selected)>target_count: raise ValueError('bounded quota count correction failed')
    selected.sort(key=lambda r: rank(r['audit_id'],'dataset-order'))
    return selected,{'requested_rows':target_count,'selected_rows':len(selected),
                     'planning_total_tokens':total,'quotas':quotas,
                     'one_pass_actual_token_attribution_matrix':matrix,
                     'corrected_bucket_token_weights':weights,
                     'shortage_rows':max(0,target_count-len(selected)),
                     'limiting_supply_task':min(max_by_supply,key=max_by_supply.get)}


def describe(rows):
    total = sum(r['shifted_supervised_tokens'] for r in rows)
    task_tokens, source_rows, source_tokens, tasks, turns, length = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    lengths = sorted(r['sequence_tokens'] for r in rows)
    for row in rows:
        task_tokens.update(row['task_supervised_tokens']); source_rows[row['source']] += 1
        source_tokens[row['source']] += row['shifted_supervised_tokens']; tasks[row['pilot_task']] += 1
        turns[str(row['assistant_turns'])] += 1
        length['<=256' if row['sequence_tokens'] <=256 else '257-512' if row['sequence_tokens']<=512 else '513-1024' if row['sequence_tokens']<=1024 else '1025-2048'] += 1
    return {'rows': len(rows), 'total_sequence_tokens': sum(lengths), 'actual_shifted_supervised_tokens_with_eos': total,
            'actual_task_supervised_tokens': dict(task_tokens),
            'actual_task_supervised_fraction': {task: task_tokens[task]/total if total else 0 for task in MIX},
            'source_rows': dict(source_rows), 'source_supervised_tokens': dict(source_tokens),
            'conversation_task_rows': dict(tasks), 'assistant_turn_count_rows': dict(turns), 'sequence_length_bins': dict(length),
            'sequence_length_p50': lengths[len(lengths)//2] if lengths else None,
            'sequence_length_p95': lengths[min(len(lengths)-1, math.ceil(len(lengths)*.95)-1)] if lengths else None,
            'max_sequence_length': max(lengths, default=0),
            'python_lower_tokens': sum(r['python_lower_tokens'] for r in rows),
            'python_upper_tokens': sum(r['python_upper_tokens'] for r in rows),
            'python_lower_fraction': sum(r['python_lower_tokens'] for r in rows)/total if total else 0,
            'python_upper_fraction': sum(r['python_upper_tokens'] for r in rows)/total if total else 0,
            'mixed_python_conversations': sum(0<r['python_lower_tokens']<r['python_upper_tokens'] for r in rows),
            'rows_without_system_preserved': sum(not r['has_system'] for r in rows)}


def stratified_preview(rows, count=200):
    buckets = defaultdict(list)
    for row in rows: buckets[(row['source'], row['pilot_task'])].append(row)
    for key in buckets: buckets[key].sort(key=lambda r: rank(r['audit_id'], 'fresh-preview'))
    chosen, counts = [], Counter()
    # Round-robin strata guarantees manageable per-source/task coverage; it is
    # intentionally not a population-proportional quality estimator.
    while len(chosen) < min(count, len(rows)):
        for key in sorted(buckets):
            if buckets[key] and len(chosen) < count:
                chosen.append(buckets[key].pop(0)); counts[' / '.join(key)] += 1
    return chosen, dict(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope-only', action='store_true')
    args = ap.parse_args()
    sys.path.insert(0, str(REPO))
    source_manifest = json.loads((SNAPSHOT/'SNAPSHOT_MANIFEST.json').read_text())
    verified = {}
    for name in ['MAIN_RESULTS.jsonl', 'references/MAIN_KEY.jsonl', 'review_overlay/CONFIRMED_CONTENT_HOLDS.jsonl']:
        actual = sha(SNAPSHOT/name)
        if actual != source_manifest['source_files'][name]['sha256']: raise ValueError('snapshot hash mismatch: '+name)
        verified[name] = actual
    if sha(SUITE) != SUITE_SHA: raise ValueError('frozen development suite changed')
    if sha(TOKENIZER) != TOKENIZER_SHA: raise ValueError('accepted tokenizer changed')
    all_rows = read_jsonl(SNAPSHOT/'MAIN_RESULTS.jsonl')
    rows = [row for row in all_rows if row['route']=='KEEP_PROVISIONAL']
    if len(all_rows) !=55000 or len(rows)!=22775: raise ValueError('fixed baseline membership changed')
    keys = {r['audit_id']:r for r in read_jsonl(SNAPSHOT/'references/MAIN_KEY.jsonl')}
    by_id = {r['audit_id']:r for r in all_rows}
    holds = read_jsonl(HANDOFF/'CONFIRMED_CONTENT_HOLDS.jsonl')
    overlay = read_jsonl(SNAPSHOT/'review_overlay/CONFIRMED_CONTENT_HOLDS.jsonl')
    if {h['audit_id'] for h in holds} != {h['audit_id'] for h in overlay}: raise ValueError('hold identities disagree')
    exclusions, hold_report = {}, []
    for hold in holds:
        original = by_id[hold['audit_id']]
        key = keys[hold['audit_id']]
        for field in ('source','original_row_index'):
            if original[field] != hold[field] or key[field] != hold[field]: raise ValueError('hold source/index mismatch')
        if digest(canonical(original['messages'])) != hold['messages_sha256']: raise ValueError('hold messages mismatch')
        exclusions[hold['audit_id']] = 'confirmed_content_hold_overlay'
        hold_report.append({**hold, 'verified_audit_source_original_index_messages': True})
    examples = read_jsonl(HANDOFF/'PILOT_SELECTION_EXAMPLES.jsonl')
    for example in examples:
        if example['suggested_pilot_action']=='DEFER_REFERENCE_OUTPUT_ERROR_FOR_PILOT':
            original = by_id[example['audit_id']]
            if original['messages'] != example['original_messages']: raise ValueError('selection example identity mismatch')
            exclusions[example['audit_id']] = 'handoff_reference_output_error_already_noted_by_Flash'
    prior_summary = json.loads((D1_ROOT/'D1_2_SUMMARY.json').read_text())
    protected_path = Path('artifacts/instruction_preparation/d0_smol_smoltalk_census_v1/output/BENCHMARK_OVERLAP_HITS.jsonl')
    if sha(protected_path) != prior_summary['benchmark_filter']['d0_benchmark_hits_sha256']:
        raise ValueError('inherited protected benchmark evidence changed')
    protected = {(h['sft_source'],h['sft_row_index']) for h in read_jsonl(protected_path)}
    protected_matches = [r['audit_id'] for r in rows if (r['source'],r['original_row_index']) in protected]
    for audit_id in protected_matches: exclusions[audit_id] = 'inherited_protected_benchmark_hit'
    grouping = assign_groups(rows, prior_summary)  # BEFORE quota or hash row sampling.
    suite_index = frozen_suite_index(read_jsonl(SUITE))
    decisions, eligible, counts = [], [], Counter()
    for row in rows:
        key = keys[row['audit_id']]
        if any(row[field] != key[field] for field in ('source','original_row_index')): raise ValueError('candidate identity mismatch')
        if row['source'] not in ALLOWED_SOURCES: raise ValueError('unapproved source')
        reason, task = scope_decision(row, exclusions)
        reason = reason or suite_overlap(row, suite_index)
        record = {'audit_id': row['audit_id'], 'source': row['source'], 'original_row_index': row['original_row_index'],
                  'messages_sha256': digest(canonical(row['messages'])), 'original_d4_route': row['route'],
                  'original_d4_quality': row['quality'], 'original_d4_task_type': row['task_type'],
                  'group_id': row['group_id'], 'assigned_split_before_quota': row['assigned_split'],
                  'selection_decision': 'DEFER' if reason else 'SCOPE_ELIGIBLE_PENDING_FORMATTER_QUOTA',
                  'selection_reason': reason, 'pilot_task': task}
        decisions.append(record)
        counts[reason or 'scope_eligible'] += 1
        if not reason:
            eligible.append({'audit_id': row['audit_id'], 'source': row['source'], 'original_row_index': row['original_row_index'],
                             'messages': row['messages'], 'messages_sha256': record['messages_sha256'],
                             'group_id': row['group_id'], 'assigned_split': row['assigned_split'], 'pilot_task': task,
                             'original_d4_quality': row['quality'], 'original_d4_task_type': row['task_type']})
    save_json(PREP/'selection/SCOPE_COUNTS.json', {'fixed_keep_pool':22775, 'decisions':dict(counts),
              'eligible_by_task':dict(Counter(r['pilot_task'] for r in eligible)), 'grouping':grouping})
    print('scope', dict(counts), 'eligible tasks', dict(Counter(r['pilot_task'] for r in eligible)), flush=True)
    save_json(PREP/'selection/CONFIRMED_HOLDS_EXCLUSION_VERIFICATION.json', hold_report)
    if args.scope_only:
        save_jsonl(PREP/'selection/SELECTION_DECISIONS.jsonl', decisions)
        return
    from src.chat_template import load_chat_tokenizer, build_example
    tok = load_chat_tokenizer(str(TOKENIZER))
    decision_by_id = {r['audit_id']:r for r in decisions}
    formatted, failures = [], Counter()
    for index, row in enumerate(eligible):
        try:
            row.update(format_metadata(row, tok))
            formatted.append(row)
        except ValueError as error:
            failures[str(error)] += 1
            decision_by_id[row['audit_id']].update(selection_decision='DEFER', selection_reason='formatter:' + str(error))
        if (index+1)%2000==0: print('formatted', index+1, '/', len(eligible), flush=True)
    # Keep one representative per group; all members already received a split.
    representatives = {}
    for row in formatted:
        gid = row['group_id']
        if gid not in representatives or rank(row['audit_id'],'representative') < rank(representatives[gid]['audit_id'],'representative'):
            representatives[gid] = row
    pool = list(representatives.values())
    train, train_quota = choose_token_mix([r for r in pool if r['assigned_split']=='train'], 10000)
    val, val_quota = choose_token_mix([r for r in pool if r['assigned_split']=='validation'], 500)
    selected = train + val
    selected_by_id = {r['audit_id']:r for r in selected}
    for record in decisions:
        if record['audit_id'] in selected_by_id:
            record.update(selection_decision='SELECTED_PROPOSAL', selection_reason='grouped split then deterministic supervised-token quota')
        elif record['selection_decision']=='SCOPE_ELIGIBLE_PENDING_FORMATTER_QUOTA':
            record.update(selection_decision='NOT_SELECTED_QUOTA_OR_GROUP_REPRESENTATIVE', selection_reason='eligible; retained outside this small proposal')
    if {r['group_id'] for r in train} & {r['group_id'] for r in val}: raise ValueError('group split leakage')
    train_keys = set(k for row in train for k in grouping_keys(row))
    val_keys = set(k for row in val for k in grouping_keys(row))
    if train_keys & val_keys: raise ValueError('exact conversation/prompt split leakage')
    if any(suite_overlap(row, suite_index) for row in selected): raise ValueError('development leakage')
    if any(r['audit_id'] in exclusions for r in selected): raise ValueError('known hold selected')
    # The actual trainer builder pads/returns causal-label tensors. Verify every
    # selected row against the same labels[:,1:] operation used by masked_ce_loss.
    for index, row in enumerate(selected):
        x, y, weight = build_example(row, tok, 2048, None, 1.0, [], 'none')
        if int((y[1:] != -100).sum().item()) != row['shifted_supervised_tokens']:
            raise ValueError('actual trainer shifted token count differs')
        if float(weight) != 1.0: raise ValueError('unexpected example loss weight')
        if row['messages'] != by_id[row['audit_id']]['messages']: raise ValueError('training text changed')
        if (index+1)%2000==0: print('trainer formatter verified', index+1, '/', len(selected), flush=True)
    dataset_names = {'PILOT_V2_CANDIDATE.jsonl':selected, 'PILOT_V2_TRAIN.jsonl':train, 'PILOT_V2_VALIDATION.jsonl':val}
    for name, data in dataset_names.items(): save_jsonl(PREP/'dataset'/name, data)
    save_jsonl(PREP/'selection/SELECTION_DECISIONS.jsonl', decisions)
    preview, preview_counts = stratified_preview(selected)
    save_jsonl(PREP/'preview/PILOT_V2_FRESH200_REVIEW.jsonl', preview)
    save_json(PREP/'preview/PREVIEW_MANIFEST.json', {'seed':SEED, 'rows':len(preview), 'strata':preview_counts,
        'selection':'deterministic round-robin source/substantive task strata after proposal freeze',
        'all_messages_preserved':True, 'content_review_status':'CONTENT_REVIEW_PENDING',
        'population_accuracy_estimate':False, 'training_started':False})
    checkable = [r for r in selected if r['checkable_instruction_cues'] or r['pilot_task']=='PYTHON_BASIC']
    checkable.sort(key=lambda r: rank(r['audit_id'],'bounded-check-review'))
    save_jsonl(PREP/'preview/CHECKABLE_INSTRUCTIONS_AND_PYTHON.jsonl', checkable[:100])
    examples_report=[]
    for ex in examples:
        dec = decision_by_id.get(ex['audit_id'])
        examples_report.append({'audit_id':ex['audit_id'], 'suggested_action':ex['suggested_pilot_action'],
                               'selection_decision':dec, 'is_population_quality_estimate':False})
    save_json(PREP/'selection/HANDOFF_EXAMPLES_DISPOSITIONS.json', examples_report)
    stats = {'status':'PILOT_CANDIDATE_PREPARED', 'formatter_and_split_status':'FORMATTER_AND_SPLIT_CHECKS_PASSED',
             'content_status':'CONTENT_REVIEW_PENDING', 'TRAINING_STARTED':False,
             'candidate':describe(selected), 'train':describe(train), 'validation':describe(val),
             'eligible_after_scope_and_formatter':describe(pool), 'scope_counts':dict(counts),
             'formatter_deferrals':dict(failures), 'train_quota':train_quota, 'validation_quota':val_quota,
             'grouping':grouping, 'normalized_conversation_or_prompt_keys_cross_split':0,
             'development_exact_or_13gram_overlap_selected':0, 'confirmed_holds_excluded':len(holds),
             'all_selected_rows_actual_trainer_shift_verified':len(selected),
             'mix_method':'Actual preserved-message encode_chat plus actual build_example labels[1:] != -100, including each assistant EOS; per-assistant content task attribution, with conservative Python mixed-conversation bounds.',
             'protected_benchmark_evidence':{'summary_path':str(D1_ROOT/'D1_2_SUMMARY.json'), 'summary_sha256':sha(D1_ROOT/'D1_2_SUMMARY.json'),
                'inherited_filter':prior_summary['benchmark_filter'], 'new_whole_corpus_regrading':False,
                'protected_hits_file_sha256_verified':True,'fixed_keep_pool_protected_hits':protected_matches},
             'limitations':['Provisional D4 labels and deterministic scope rules do not establish all-answer correctness.',
                           'Scope keyword/AST rules can defer suitable rows and miss ambiguous content; full-message preview remains pending.',
                           'No candidate code executed; syntax inspection is not program correctness testing.',
                           'Exact/full-context grouping and 13-gram development checks do not prove absence of semantic or public-benchmark contamination.',
                           'Group validation assignment preceded token quotas; validation and fresh review are development material, not untouched final benchmarks.',
                           'Python lower bound counts explicitly identifiable Python turns; upper bound includes all supervised turns of any Python-containing conversation.',
                           'Requested row counts and token weights are planning targets; shortages were not filled from complex/other-source rows.']}
    save_json(PREP/'selection/PILOT_V2_STATISTICS.json', stats)
    manifest = {'status':'PILOT_CANDIDATE_PREPARED', 'CONTENT_REVIEW_PENDING':True, 'TRAINING_STARTED':False,
                'snapshot_cut_time_utc':source_manifest['snapshot_cut_time_utc'], 'fixed_source_hashes':verified,
                'tokenizer_sha256':TOKENIZER_SHA, 'development_suite_sha256':SUITE_SHA,
                'development_suite_path':str(SUITE), 'selection_seed':SEED, 'script_sha256':sha(Path(__file__)),
                'formatter_sha256':sha(REPO/'src/chat_template.py'), 'trainer_sha256':sha(REPO/'sft/train_sft.py'),
                'formatter_policy':{'default_system':None,'preserve_messages':True,'max_length':2048,'overlength':'reject','causal_shift':'labels[1:]'},
                'files':{name:{'sha256':sha(PREP/'dataset'/name),'rows':len(data)} for name,data in dataset_names.items()},
                'preview_sha256':sha(PREP/'preview/PILOT_V2_FRESH200_REVIEW.jsonl')}
    save_json(PREP/'dataset/PILOT_V2_MANIFEST.json', manifest)
    print('COMPLETE', json.dumps({k:stats[k] for k in ('train','validation')},ensure_ascii=False), flush=True)


if __name__=='__main__': main()
