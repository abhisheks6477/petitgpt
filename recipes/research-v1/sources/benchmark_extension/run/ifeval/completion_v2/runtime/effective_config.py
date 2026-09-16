"""Task-local HF config compatibility; no weights, package or snapshot mutation."""
import copy, hashlib, json
from types import SimpleNamespace
from transformers.generation.utils import GenerationMixin

OVERRIDES = dict(do_sample=False, num_beams=1, num_return_sequences=1,
                 max_new_tokens=1280, use_cache=True, temperature=0.0,
                 disable_compile=True)
NEUTRAL = dict(repetition_penalty=1.0, encoder_repetition_penalty=1.0,
               no_repeat_ngram_size=0, encoder_no_repeat_ngram_size=0,
               min_length=0, min_new_tokens=0, length_penalty=1.0,
               diversity_penalty=0.0, num_beam_groups=1,
               early_stopping=False, token_healing=False, use_mtp=False,
               is_assistant=False, renormalize_logits=False, remove_invalid_values=False,
               output_scores=False, output_logits=False, output_attentions=False,
               output_hidden_states=False, return_dict_in_generate=False)
ABSENT = ('forced_bos_token_id', 'forced_eos_token_id', 'bad_words_ids',
          'suppress_tokens', 'begin_suppress_tokens', 'sequence_bias', 'force_words_ids',
          'constraints', 'stop_strings', 'watermarking_config',
          'exponential_decay_length_penalty', 'guidance_scale', 'max_time',
          'prompt_lookup_num_tokens', 'assistant_early_exit', 'speculation_type',
          'compile_config', 'continuous_batching_config', 'penalty_alpha', 'dola_layers',
          'prefill_chunk_size', 'cache_implementation', 'cache_config', 'max_cache_len')
CANONICAL = dict(bos_token_id=1, eos_token_id=2, pad_token_id=2)

def canon(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

def identity(x):
    return hashlib.sha256(canon(x).encode()).hexdigest()

def checked_update(config, changes):
    unused = config.update(**changes)
    if unused:
        raise ValueError('Unsupported generation settings: ' + repr(unused))

def validate(config, effective=True):
    values = config.to_dict()
    for key, neutral in NEUTRAL.items():
        value = values.get(key)
        if (value is None and not effective):
            continue
        if value != neutral:
            raise ValueError(f'Nonneutral {key}={value!r}; required {neutral!r}')
    for key in ABSENT:
        if values.get(key) is not None:
            raise ValueError(f'Enabled forbidden control {key}={values[key]!r}')
    for key, required in CANONICAL.items():
        if values.get(key) != required:
            raise ValueError(f'Canonical token mismatch {key}={values.get(key)!r}')
    if effective:
        for key, required in OVERRIDES.items():
            if values.get(key) != required:
                raise ValueError(f'Frozen override mismatch {key}={values.get(key)!r}')
        if config.get_generation_mode().value != 'greedy_search':
            raise ValueError('Effective generation mode is not greedy_search')
    return True

def resolve_actual(receiver, config, **kwargs):
    return GenerationMixin._prepare_generation_config(receiver, config, **kwargs)

def prepare(raw):
    original = copy.deepcopy(raw.to_dict())
    validate(raw, effective=False)  # reject explicit nonneutral controls first
    effective, unused = resolve_actual(SimpleNamespace(generation_config=raw), copy.deepcopy(raw))
    if unused:
        raise ValueError('Unexpected resolver leftovers: ' + repr(unused))
    defaults_resolved = copy.deepcopy(effective.to_dict())
    # The installed resolver leaves some inactive/unset controls as None.
    # Make the already-frozen neutral policy explicit, never overwrite nonneutral values.
    for key, value in NEUTRAL.items():
        if getattr(effective, key, None) is None:
            checked_update(effective, {key: value})
    checked_update(effective, OVERRIDES)
    validate(effective)
    assert raw.to_dict() == original
    evidence = dict(raw=original, installed_defaults=raw._get_default_generation_params(),
                    after_installed_resolution=defaults_resolved, overrides=OVERRIDES,
                    neutral_unset_resolution=NEUTRAL, canonical=CANONICAL,
                    effective=effective.to_dict(), effective_config_id=identity(effective.to_dict()),
                    raw_config_id=identity(original), policy='frozen greedy; sampling-only options inactive')
    return effective, evidence

def invoke(receiver, effective, evidence, *, input_ids, attention_mask):
    """This is the only production HF generate call; no custom generation kwargs."""
    validate(effective)
    if identity(effective.to_dict()) != evidence['effective_config_id']:
        raise ValueError('Prepared config changed after validation')
    if identity(receiver.generation_config.to_dict()) != evidence['raw_config_id']:
        raise ValueError('Receiver raw generation config changed')
    resolved, leftover = resolve_actual(receiver, effective)
    validate(resolved)
    if leftover or resolved.to_dict() != effective.to_dict():
        raise ValueError('Production resolver diverged from recorded effective config')
    return receiver.generate(input_ids=input_ids, attention_mask=attention_mask,
                             generation_config=copy.deepcopy(effective))
