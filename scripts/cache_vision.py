#!/usr/bin/env python3
"""Pre-compute and cache InternVL3 hidden states for every (episode, chunk).

Two augmentation knobs (both off by default — pass flags to enable):

  --n-vis-aug N
      For each chunk, additionally cache N augmented variants. Each variant
      uses one consistent random photometric transform (brightness, contrast,
      saturation, hue, blur, small crop) applied to all frames in that chunk
      so temporal coherence is preserved. The augmented features go through
      InternVL3 ONCE here — the training loop only pays the disk-read cost.

  --llm-augment-prompts (+ --n-prompt-paraphrases K)
      Generate K paraphrases of every base task prompt via the Anthropic API
      (claude-haiku-4-5 by default). Each cached variant draws a random
      paraphrase from this pool, so the LLM and the policy see prompt-level
      diversity. Falls back to a static paraphrase bank if the API key isn't
      set or the SDK isn't installed.

Output layout per episode:
    ep_NNN.pt = list of (q_int8, scales_fp16) entries, length n_chunks*(1+n_vis_aug):
        first n_chunks       = original (variant 0)
        next  n_chunks        = aug variant 1
        ...
        next  n_chunks        = aug variant n_vis_aug

meta.json carries:
    n_episodes, total_chunks, n_vis_aug,
    prompts_per_variant[ep_i][variant_idx] = "the prompt used for that entry"
    paraphrase_pool: { base_prompt: [paraphrase, ...] }

Usage:
  # plain (no augmentation; matches the old behavior):
  python -m scripts.cache_vision --cache-dir vision_cache

  # OXE with 3 visual variants per chunk + LLM-augmented prompts:
  python -m scripts.cache_vision \
      --dataset oxe --oxe-dataset-id lerobot/svla_so101_pickplace \
      --cache-dir oxe_vision_cache \
      --n-vis-aug 3 \
      --llm-augment-prompts --n-prompt-paraphrases 25
"""
import os, sys, gc, time, random, argparse, json

THIS = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(THIS))

import torch
import psutil

from babygroot_strm import (InternVL3Vision, load_so101_episodes,
                            load_lerobot_episodes, TASK_PROMPTS,
                            visual_augment_chunk, build_paraphrase_pool)


def _quantize_int8(stacked):
    """Per-channel symmetric int8 quantization of (25, N_tok, D)."""
    scales = stacked.abs().amax(dim=(0, 1), keepdim=True).clamp(min=1e-8) / 127.0
    q = (stacked / scales).round().clamp(-128, 127).to(torch.int8)
    return q, scales.squeeze(0).squeeze(0).half()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', default='vision_cache')
    ap.add_argument('--dataset', choices=['so101', 'oxe'], default='so101',
                    help="`so101` = original 78-episode bundle. "
                         "`oxe` = a single LeRobot dataset (default "
                         "lerobot/svla_so101_pickplace, the v5 target).")
    ap.add_argument('--oxe-dataset-id', type=str,
                    default='lerobot/svla_so101_pickplace')
    ap.add_argument('--oxe-camera', type=str, default='observation.images.front')
    ap.add_argument('--data-dir', default=None,
                    help="Local SO-101 dataset dir (only used for --dataset=so101).")
    # Augmentation
    ap.add_argument('--n-vis-aug', type=int, default=0,
                    help="Augmented vision variants per chunk (0 = off).")
    ap.add_argument('--llm-augment-prompts', action='store_true',
                    help="Use the Anthropic API to paraphrase task prompts.")
    ap.add_argument('--n-prompt-paraphrases', type=int, default=20,
                    help="How many paraphrases per base prompt to generate.")
    ap.add_argument('--llm-model', default='claude-haiku-4-5')
    # Sub-sampling for testing
    ap.add_argument('--n-eps-cap', type=int, default=None,
                    help="Cap episodes processed (useful for smoke tests).")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    random.seed(0)

    print("Loading InternVL3 ...")
    vision = InternVL3Vision()

    print(f"Loading episodes ({args.dataset}, with video) ...")
    if args.dataset == 'oxe':
        episodes = load_lerobot_episodes(args.oxe_dataset_id,
                                          camera_key=args.oxe_camera,
                                          load_video=True,
                                          n_episodes=args.n_eps_cap)
    else:
        kwargs = dict(load_video=True)
        if args.data_dir is not None:
            kwargs['data_dir'] = args.data_dir
        episodes = load_so101_episodes(**kwargs)
        if args.n_eps_cap is not None:
            episodes = episodes[:args.n_eps_cap]

    total_chunks = sum(ep[0].shape[0] for ep in episodes)
    n_variants = 1 + args.n_vis_aug
    print(f"  {len(episodes)} episodes, {total_chunks} chunks  "
          f"(variants per chunk: {n_variants})")

    # ── Build the paraphrase pool (one query per unique base prompt) ──
    base_prompts = []
    for (_, _, _, task) in episodes:
        base_prompts.append(task)
    print(f"\nBuilding paraphrase pool over {len(set(base_prompts))} unique "
          f"base prompts (use_llm={args.llm_augment_prompts}) ...")
    paraphrase_pool = build_paraphrase_pool(
        base_prompts,
        n=args.n_prompt_paraphrases,
        use_llm=args.llm_augment_prompts,
        model=args.llm_model,
    )
    for bp, plist in list(paraphrase_pool.items())[:3]:
        print(f"  '{bp[:50]}' → {len(plist)} paraphrases  (sample: {plist[0][:50]!r})")

    print(f"\nCaching to {args.cache_dir}/")
    t0 = time.perf_counter(); total_bytes = 0
    meta = {
        'n_episodes': len(episodes),
        'total_chunks': total_chunks,
        'n_vis_aug': args.n_vis_aug,
        'n_variants_per_chunk': n_variants,
        'dataset': args.dataset,
        'oxe_dataset_id': args.oxe_dataset_id if args.dataset == 'oxe' else None,
        'paraphrase_pool': paraphrase_pool,
        'prompts_per_variant': {},
    }

    for ep_i, (action_chunks, _, per_chunk_frames, task) in enumerate(episodes):
        n_chunks = action_chunks.shape[0]
        pool = paraphrase_pool.get(task) or paraphrase_pool[next(iter(paraphrase_pool))]

        # Pick one prompt per variant for this episode (constant within a
        # variant, varies across variants — keeps the variant signal clean).
        rng = random.Random(0xC0FFEE + ep_i)
        variant_prompts = [rng.choice(pool) for _ in range(n_variants)]
        meta['prompts_per_variant'][str(ep_i)] = variant_prompts

        ep_cache: list = []  # length = n_chunks * n_variants
        # Variant 0 = original; variants 1..n_vis_aug = augmented
        for v in range(n_variants):
            prompt = variant_prompts[v]
            for ch_i in range(n_chunks):
                frames = per_chunk_frames[ch_i]
                if v > 0:
                    frames = visual_augment_chunk(
                        frames, seed=0xBEEF + ep_i * 10_000 + ch_i * 100 + v)
                with torch.no_grad():
                    all_hidden = vision([frames], prompt=prompt)
                layers_f = [h.squeeze(0).float().cpu() for h in all_hidden]
                stacked = torch.stack(layers_f)        # (25, N_tok, D)
                q, scale = _quantize_int8(stacked)
                ep_cache.append((q, scale))

        out_path = os.path.join(args.cache_dir, f'ep_{ep_i:03d}.pt')
        torch.save(ep_cache, out_path)
        size = os.path.getsize(out_path); total_bytes += size
        elapsed = time.perf_counter() - t0
        ram = psutil.virtual_memory().used / 1e9
        print(f"  ep {ep_i:3d}/{len(episodes)}  chunks={n_chunks:3d}  "
              f"variants={n_variants}  "
              f"size={size/1e6:.0f}MB  total={total_bytes/1e9:.1f}GB  "
              f"RAM={ram:.1f}GB  [{elapsed:.0f}s]")
        if (ep_i + 1) % 5 == 0:
            gc.collect(); torch.cuda.empty_cache()

    with open(os.path.join(args.cache_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. {total_chunks} chunks × {n_variants} variants cached in "
          f"{time.perf_counter()-t0:.0f}s, total {total_bytes/1e9:.1f} GB.")


if __name__ == '__main__':
    main()
