#!/usr/bin/env python3
"""Generate paraphrased task strings (programmatic word substitution) and
rebuild the T5 text cache so each task maps to MULTIPLE T5 embeddings.

Why programmatic: there are ~21k unique task strings across local OXE (~20k
of those from Bridge alone — crowd-sourced and unique). Manual paraphrasing
isn't feasible at that scale. Word-level substitution covers the common
manipulation verbs and prepositions, multiplying the effective phrasings
~3-5× without changing semantics.

Output: rebuilt t5 cache where each original task → 1 original + N paraphrases,
all keyed under the original string. At train time the cache returns the original
embedding (existing path); the new entries are added under composite keys so a
future per-batch sampler can pick a paraphrase. (For this first pass we just
generate the paraphrased strings and embed them; integration with the train loop
to actually sample paraphrases comes next.)
"""
import os, sys, json, re, time, argparse, hashlib
import torch

# Word-substitution rules. Each entry: (canonical, [alternatives]).
# Goal: cover common manipulation verbs, common prepositions/articles, common objects.
# Each tuple's first entry is the "from"; alternatives are valid substitutes.
SUB_GROUPS = [
    # action verbs (most useful — covers most task variations)
    ('pick',  ['grab', 'lift', 'take', 'pickup']),
    ('place', ['put', 'set', 'lay', 'position', 'drop']),
    ('put',   ['place', 'set', 'lay']),
    ('move',  ['shift', 'slide', 'reposition', 'relocate']),
    ('open',  ['pull open', 'slide open']),
    ('close', ['shut', 'push closed', 'slide closed']),
    ('push',  ['shove', 'slide']),
    ('pull',  ['draw', 'tug']),
    ('grasp', ['grab', 'grip', 'hold']),
    ('rotate',['turn', 'spin']),
    ('stack', ['pile', 'put on top of']),
    ('insert',['put inside', 'place into']),
    # prepositions / determiners
    ('into',  ['inside', 'in']),
    ('onto',  ['on top of', 'on']),
    ('from',  ['out of', 'off of']),
    # color synonyms (occasionally)
    ('red',     ['crimson']),
    ('blue',    ['navy']),
    ('yellow',  ['gold']),
    # objects (common in OXE)
    ('block',   ['cube', 'box']),
    ('drawer',  ['compartment']),
    ('cup',     ['mug']),
    ('bowl',    ['dish']),
]


def make_paraphrases(text: str, max_paraphrases: int = 4) -> list:
    """Produce up to max_paraphrases distinct paraphrased variants by substituting
    one rule at a time. Case-preserving (capitalized at sentence start).
    """
    text = text.strip()
    if not text: return []
    base = text
    out = set()
    base_l = base.lower()
    for canonical, alts in SUB_GROUPS:
        # word-boundary, case-insensitive
        pat = re.compile(r'\b' + re.escape(canonical) + r'\b', re.IGNORECASE)
        if not pat.search(base_l): continue
        for alt in alts:
            def repl(m):
                # preserve case of first char
                w = m.group(0)
                if w[:1].isupper():
                    return alt[:1].upper() + alt[1:]
                return alt
            new = pat.sub(repl, base)
            if new != base and new.lower() not in {o.lower() for o in out} | {base.lower()}:
                out.add(new)
                if len(out) >= max_paraphrases: return list(out)
    return list(out)


def gather_task_strings(t5_cache_path: str) -> list:
    """Return list of all task strings present in the existing t5 cache."""
    t5 = torch.load(t5_cache_path, map_location='cpu', weights_only=False)
    return list(t5['embeddings'].keys()), t5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-cache', default='data/cache/t5_text_cache.pt')
    ap.add_argument('--out-cache', default='data/cache/t5_text_cache_paraphrased.pt')
    ap.add_argument('--max-paraphrases', type=int, default=4)
    ap.add_argument('--t5-model', default='google/flan-t5-small')
    ap.add_argument('--max-len', type=int, default=24)
    ap.add_argument('--dry-run', action='store_true',
                    help="Only generate paraphrase strings; do NOT embed (skip T5 load).")
    args = ap.parse_args()

    print(f"loading task strings from {args.in_cache} ...")
    keys, t5_old = gather_task_strings(args.in_cache)
    print(f"  {len(keys)} task strings; t5 dim={t5_old['dim']}, layers={t5_old['n_layers']}")

    # generate paraphrases for each
    print(f"\ngenerating paraphrases (max {args.max_paraphrases}/task)...")
    t0 = time.time()
    paraphrased = {}                          # original -> [paraphrase, ...]
    n_with_any = 0; n_total = 0
    for k in keys:
        ps = make_paraphrases(k, max_paraphrases=args.max_paraphrases)
        if ps:
            paraphrased[k] = ps
            n_with_any += 1
            n_total += len(ps)
    print(f"  {n_with_any}/{len(keys)} tasks got at least one paraphrase ({n_with_any/len(keys)*100:.1f}%)")
    print(f"  {n_total} paraphrases total (avg {n_total/max(n_with_any,1):.2f} per paraphrased task)")
    print(f"  [{time.time()-t0:.1f}s] paraphrase generation done")

    # show 6 random examples
    import random
    random.seed(0)
    samples = random.sample(list(paraphrased.items()), min(6, len(paraphrased)))
    print(f"\nsample paraphrases:")
    for orig, ps in samples:
        print(f"  original: {orig[:80]}")
        for p in ps[:3]: print(f"    → {p[:80]}")

    if args.dry_run:
        print("\n[dry-run] skipping T5 embedding")
        return

    # embed the new paraphrases via T5
    print(f"\nloading T5 model {args.t5_model} for embedding ...")
    from transformers import AutoTokenizer, T5EncoderModel
    tok = AutoTokenizer.from_pretrained(args.t5_model, use_fast=True)
    enc = T5EncoderModel.from_pretrained(args.t5_model, output_hidden_states=True).cuda().eval()

    # Build the new cache: include original entries + paraphrase entries
    new_cache = {
        'embeddings': dict(t5_old['embeddings']),     # keep originals
        'dim': t5_old['dim'],
        'n_layers': t5_old['n_layers'],
        'paraphrase_map': {},                          # original -> [paraphrase keys]
    }

    all_para_strings = []
    for orig, ps in paraphrased.items():
        new_cache['paraphrase_map'][orig] = ps
        all_para_strings.extend(ps)
    # dedupe paraphrases that might collide with originals or each other
    unique_para = sorted(set(all_para_strings) - set(new_cache['embeddings'].keys()))
    print(f"  {len(unique_para)} unique new strings to embed (after deduping vs originals)")

    BATCH = 32
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(unique_para), BATCH):
            chunk = unique_para[i:i+BATCH]
            ids = tok(chunk, return_tensors='pt', padding='max_length',
                      truncation=True, max_length=args.max_len)
            ids = {k: v.cuda() for k, v in ids.items()}
            out = enc(**ids)
            # hidden_states is tuple of length n_layers+1 (input embed + N encoder layers).
            # Original cache was built keeping ALL of them — include them here too so shapes match.
            hs = torch.stack(list(out.hidden_states), dim=0)            # (L+1, B, T, D)
            for b, txt in enumerate(chunk):
                new_cache['embeddings'][txt] = {'hidden': hs[:, b].cpu()}
            if i % (BATCH * 20) == 0 and i > 0:
                done = i + len(chunk)
                eta = (time.time() - t0) * (len(unique_para) - done) / max(done, 1)
                print(f"  embedded {done}/{len(unique_para)} ({eta:.0f}s remaining)")

    torch.save(new_cache, args.out_cache)
    print(f"\nsaved {args.out_cache}")
    print(f"  total embedded entries: {len(new_cache['embeddings'])}  "
          f"(originals: {len(t5_old['embeddings'])}, +{len(new_cache['embeddings'])-len(t5_old['embeddings'])} new)")
    print(f"  paraphrase_map: {len(new_cache['paraphrase_map'])} originals with paraphrases")


if __name__ == '__main__':
    main()
