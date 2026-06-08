#!/usr/bin/env python3
"""Convert one extracted AgiBot task → our LeRobot-style layout (what multi_oxe reads).

Reads (extracted) AgiBot raw:
  observations/<task>/<ep>/videos/head_color.mp4     (primary RGB = top_head)
  <somewhere>/<ep>/proprio_stats.h5                  (arms action+state)
Writes (our format, matches data/oxe/bridge_orig_lerobot):
  meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl
  data/chunk-{cc}/episode_{ep:06d}.parquet           (observation.state[16], action[16], ...)
  videos/chunk-{cc}/observation.images.top_head/episode_{ep:06d}.mp4   (symlink to head_color.mp4)

Arms-only 16-D = joint(14)+effector(2), joint-space. Frame-aligned (proprio & video both 30Hz).
mp4 is SYMLINKED (not copied) to save disk; pass --copy to hard-copy instead.
"""
import os, sys, glob, json, argparse, shutil
import numpy as np, h5py
import pyarrow as pa, pyarrow.parquet as pq

CAM_KEY = 'observation.images.top_head'
RAW_VIDEO = 'head_color.mp4'
CHUNKS_SIZE = 1000
FPS = 30


def find_proprio(raw_dir, ep):
    hits = glob.glob(os.path.join(raw_dir, '**', str(ep), 'proprio_stats.h5'), recursive=True)
    return hits[0] if hits else None


def read_arms(h5_path):
    with h5py.File(h5_path, 'r') as f:
        aj = np.asarray(f['action/joint/position'], dtype=np.float32)      # (T,14)
        ae = np.asarray(f['action/effector/position'], dtype=np.float32)   # (T,2)
        sj = np.asarray(f['state/joint/position'], dtype=np.float32)
        se = np.asarray(f['state/effector/position'], dtype=np.float32)
    action = np.hstack([aj, ae])                                          # (T,16)
    state = np.hstack([sj, se])                                           # (T,16)
    return action, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obs-dir', required=True, help='dir containing <ep>/videos/head_color.mp4 (extracted obs tar)')
    ap.add_argument('--proprio-dir', required=True, help='dir containing (recursive) <ep>/proprio_stats.h5')
    ap.add_argument('--task-json', required=True, help='task_info/task_<id>.json for the instruction')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--copy', action='store_true', help='hard-copy mp4 instead of symlink')
    ap.add_argument('--max-eps', type=int, default=None)
    args = ap.parse_args()

    obs_root = args.obs_dir
    eps = sorted(int(d) for d in os.listdir(obs_root) if d.isdigit()) if os.path.isdir(obs_root) else []
    if args.max_eps: eps = eps[:args.max_eps]
    if not eps:
        print(f"No episodes under {obs_root}"); return

    task_info = json.load(open(args.task_json))
    instruction = f"{task_info[0]['task_name']}. {task_info[0].get('init_scene_text','')}".strip()

    os.makedirs(os.path.join(args.out_dir, 'meta'), exist_ok=True)
    ep_meta, idx_global, n_ok = [], 0, 0
    for ep in eps:
        h5p = find_proprio(args.proprio_dir, ep)
        mp4 = os.path.join(obs_root, str(ep), 'videos', RAW_VIDEO)
        if not h5p or not os.path.isfile(mp4):
            continue
        try:
            action, state = read_arms(h5p)
        except Exception:
            continue
        T = min(len(action), len(state))
        if T < 34:                                                        # need lookback+chunk
            continue
        action, state = action[:T], state[:T]
        cc = ep // CHUNKS_SIZE
        # parquet
        pdir = os.path.join(args.out_dir, f'data/chunk-{cc:03d}'); os.makedirs(pdir, exist_ok=True)
        tbl = pa.table({
            'observation.state': [state[i].tolist() for i in range(T)],
            'action': [action[i].tolist() for i in range(T)],
            'timestamp': [float(i) / FPS for i in range(T)],
            'frame_index': list(range(T)),
            'episode_index': [ep] * T,
            'index': list(range(idx_global, idx_global + T)),
            'task_index': [0] * T,
        })
        pq.write_table(tbl, os.path.join(pdir, f'episode_{ep:06d}.parquet'))
        # video symlink
        vdir = os.path.join(args.out_dir, f'videos/chunk-{cc:03d}/{CAM_KEY}'); os.makedirs(vdir, exist_ok=True)
        dst = os.path.join(vdir, f'episode_{ep:06d}.mp4')
        if os.path.lexists(dst): os.remove(dst)
        (shutil.copyfile if args.copy else os.symlink)(os.path.abspath(mp4), dst)
        ep_meta.append({'episode_index': ep, 'length': T, 'tasks': [instruction]})
        idx_global += T; n_ok += 1

    # meta files
    info = {
        'codebase_version': 'v2.0', 'robot_type': 'agibot', 'fps': FPS, 'chunks_size': CHUNKS_SIZE,
        'total_episodes': n_ok, 'total_frames': idx_global,
        'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
        'features': {
            CAM_KEY: {'dtype': 'video', 'shape': [480, 640, 3], 'names': ['height', 'width', 'rgb'],
                      'info': {'video.fps': float(FPS), 'video.codec': 'av1', 'video.pix_fmt': 'yuv420p',
                               'video.is_depth_map': False, 'has_audio': False}},
            'observation.state': {'dtype': 'float32', 'shape': [16]},
            'action': {'dtype': 'float32', 'shape': [16],
                       'names': {'motors': [f'joint_{i}' for i in range(14)] + ['gripper_left', 'gripper_right']}},
        },
    }
    json.dump(info, open(os.path.join(args.out_dir, 'meta/info.json'), 'w'), indent=1)
    with open(os.path.join(args.out_dir, 'meta/episodes.jsonl'), 'w') as f:
        for r in ep_meta: f.write(json.dumps(r) + '\n')
    with open(os.path.join(args.out_dir, 'meta/tasks.jsonl'), 'w') as f:
        f.write(json.dumps({'task_index': 0, 'task': instruction}) + '\n')
    print(f"converted {n_ok}/{len(eps)} episodes → {args.out_dir}  ({idx_global} frames)  instr=<{instruction[:50]}>")


if __name__ == '__main__':
    main()
