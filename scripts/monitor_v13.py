#!/usr/bin/env python3
"""Watch v13 training log; emit one stdout line per significant event.

Events emitted:
  - PROGRESS every 5000 steps  (routine status)
  - PLATEAU when EMA loss stops improving by >0.005 nats over a 5000-step window
  - SPIKE on any grad-hard-clip or grad-spike event
  - NaN on any non-finite loss / param
  - CKPT when ckpt-save lines appear (rare; helps map to disk state)
  - TERMINATED when the v13 process exits

Each emitted line is one Monitor notification, so phrasing is terse.
"""
import os, time, re, sys

LOG = os.environ.get('V13_LOG', '/tmp/v13_train.log')
PID = int(os.environ.get('V13_PID', 1191760))
REPORT_EVERY = 5000          # steps between routine progress
PLATEAU_WIN = 5000           # steps of no-improvement before plateau alert
PLATEAU_TOL = 0.005          # nats improvement threshold
ALERT_COOLDOWN = 10000       # steps between repeated plateau alerts
EMA_ALPHA = 0.1              # smoothing on per-200-step loss readings

step_re = re.compile(r'step\s+(\d+)/\d+\s+loss=([\d.+-]+|nan|inf)\s+acc=([\d.]+)%.*rmax=([\d.]+).*ρ_L=([\d.]+).*ρ_H=([\d.]+)')


def pid_alive(pid):
    try:
        os.kill(pid, 0); return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main():
    last_report = 0
    ema = None
    best_ema = float('inf'); best_step = 0
    last_plateau_alert = -ALERT_COOLDOWN
    start_step = None
    start_t = time.time()
    last_pid_check = time.time()
    last_step_seen = 0

    print(f"START watching {LOG} (PID {PID}); report every {REPORT_EVERY} steps, "
          f"plateau if no Δ>{PLATEAU_TOL} over {PLATEAU_WIN} steps", flush=True)
    # wait for log file to exist
    for _ in range(60):
        if os.path.exists(LOG): break
        time.sleep(2)
    else:
        print(f"ERROR: {LOG} never appeared", flush=True); return

    f = open(LOG, 'r')
    f.seek(0, 2)                              # tail mode: only new lines

    while True:
        line = f.readline()
        if not line:
            time.sleep(3)
            # liveness check every 60s
            if time.time() - last_pid_check > 60:
                last_pid_check = time.time()
                if not pid_alive(PID):
                    print(f"TERMINATED: v13 PID {PID} exited at last_step={last_step_seen}", flush=True)
                    return
            continue

        # parse step line
        m = step_re.search(line)
        if m:
            step = int(m.group(1))
            try: loss = float(m.group(2))
            except: loss = float('nan')
            acc = float(m.group(3))
            rmax = float(m.group(4))
            rho_L = float(m.group(5))
            rho_H = float(m.group(6))
            last_step_seen = step
            if start_step is None:
                start_step = step
                start_t = time.time()

            # NaN check
            if loss != loss:                  # NaN
                print(f"NaN at step {step}: loss=NaN, acc={acc:.1f}%", flush=True)
                continue

            # EMA + plateau detection
            ema = loss if ema is None else (1 - EMA_ALPHA) * ema + EMA_ALPHA * loss
            if ema < best_ema - PLATEAU_TOL:
                best_ema = ema; best_step = step
            elif (step - best_step >= PLATEAU_WIN
                  and step - last_plateau_alert >= ALERT_COOLDOWN
                  and step > 5000):                                 # ignore warmup phase
                print(f"PLATEAU step {step}: EMA loss {ema:.3f} (best EMA {best_ema:.3f} @ step {best_step}; "
                      f"{step-best_step} steps no improvement); acc={acc:.1f}%, ρ_L={rho_L:.3f} ρ_H={rho_H:.3f}",
                      flush=True)
                last_plateau_alert = step

            # routine progress
            if step - last_report >= REPORT_EVERY:
                rate = (step - start_step) / max(time.time() - start_t, 1)
                eta_h = (200000 - step) / max(rate, 0.01) / 3600
                print(f"PROGRESS step {step}/200k: loss={loss:.3f} (EMA {ema:.3f}, best {best_ema:.3f}) "
                      f"acc={acc:.1f}% rmax={rmax:.2f} ρ_L={rho_L:.3f} ρ_H={rho_H:.3f} "
                      f"rate={rate:.2f}/s ETA={eta_h:.1f}h", flush=True)
                last_report = step

        # spike / clip events
        if 'grad-hard-clip' in line:
            print(f"SPIKE: {line.strip()[:140]}", flush=True)
        elif 'grad-spike' in line:
            print(f"SPIKE: {line.strip()[:140]}", flush=True)
        elif 'PARAM-NaN' in line or 'non-finite' in line:
            print(f"NaN: {line.strip()[:140]}", flush=True)
        elif 'saved ' in line and '.pt' in line:
            print(f"CKPT: {line.strip()[:140]}", flush=True)


if __name__ == '__main__':
    main()
