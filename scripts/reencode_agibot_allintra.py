#!/usr/bin/env python3
"""Re-encode AgiBot episode videos to 224x224 ALL-INTRA (GOP=1) H.264 in place.

Why: training was GPU-starved (3-8% util) because the dataset decodes ONE frame per
sample from inter-coded 640x480 mp4s — seeking through P/B-frame deltas costs ~19 ms/frame.
All-intra (every frame a keyframe) makes random seek exact + cheap (~5.5 ms/frame) AND, at
224x224, is ~3.7 KB/frame = 2.8x SMALLER than the source and 4.4x smaller than a JPEG cache.

Each episode mp4 in data/oxe/agibot_task*/videos/.../episode_*.mp4 is currently a SYMLINK to
the raw obs head_color.mp4. We decode the source, re-encode to a real 224 all-intra file, and
atomically replace the symlink. Loader paths are unchanged — it just gets faster. After this,
data/agibot_raw/obs* can be deleted (the all-intra copies are self-contained).

  python -m scripts.reencode_agibot_allintra --workers 48 --crf 20 --size 224
"""
import os, sys, glob, argparse, tempfile, traceback
from multiprocessing import Pool
import av


def reencode_one(args):
    path, size, crf = args
    try:
        if not os.path.islink(path):
            return (path, 'skip-real', 0, 0)
        src = os.path.realpath(path)
        if not os.path.exists(src):
            return (path, 'missing-src', 0, 0)
        d = os.path.dirname(path)
        fd, tmp = tempfile.mkstemp(suffix='.mp4', dir=d); os.close(fd)
        n_in = n_out = 0
        inp = av.open(src); ins = inp.streams.video[0]
        out = av.open(tmp, 'w')
        os_ = out.add_stream('libx264', rate=30, options={'crf': str(crf)})
        os_.width = os_.height = size; os_.pix_fmt = 'yuv420p'
        os_.codec_context.gop_size = 1                       # all-intra: every frame a keyframe
        for fr in inp.decode(ins):
            n_in += 1
            fr2 = fr.reformat(width=size, height=size)
            for p in os_.encode(fr2):
                out.mux(p)
        for p in os_.encode():
            out.mux(p)
        out.close(); inp.close()
        # verify frame count preserved (alignment with proprio/action is by frame index)
        with av.open(tmp) as c:
            n_out = c.streams.video[0].frames or n_in
        if abs(n_out - n_in) > 1:
            os.remove(tmp)
            return (path, f'frame-mismatch {n_in}->{n_out}', n_in, n_out)
        os.replace(tmp, path)                                # atomic: symlink -> real all-intra file
        return (path, 'ok', n_in, os.path.getsize(path))
    except Exception:
        return (path, 'ERR ' + traceback.format_exc().splitlines()[-1][:80], 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='data/oxe/agibot_task*/videos/**/episode_*.mp4')
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--crf', type=int, default=20)
    ap.add_argument('--workers', type=int, default=48)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob, recursive=True))
    todo = [(p, args.size, args.crf) for p in paths]
    print(f"{len(paths)} episode videos; {sum(1 for p in paths if os.path.islink(p))} are symlinks (to do). "
          f"workers={args.workers} size={args.size} crf={args.crf}", flush=True)

    ok = skip = err = 0; bytes_out = 0; frames = 0
    with Pool(args.workers) as pool:
        for i, (path, status, n_in, sz) in enumerate(pool.imap_unordered(reencode_one, todo, chunksize=4)):
            if status == 'ok':
                ok += 1; bytes_out += sz; frames += n_in
            elif status.startswith('skip'):
                skip += 1
            else:
                err += 1; print(f"  [{status}] {path}", flush=True)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(paths)}  ok={ok} skip={skip} err={err}  "
                      f"({bytes_out/1e9:.2f} GB, {frames} frames so far)", flush=True)
    print(f"DONE ok={ok} skip={skip} err={err}  total {bytes_out/1e9:.2f} GB over {frames} frames "
          f"({bytes_out/max(frames,1):.0f} B/frame)", flush=True)


if __name__ == '__main__':
    main()
