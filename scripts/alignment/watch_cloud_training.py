"""Mirror this run's remote log locally without saving SSH passwords."""
import argparse
from datetime import datetime
import getpass
import json
from pathlib import Path
import sys
import time
import paramiko


def format_number(value, *, percent=False):
    if not isinstance(value, (int, float)):
        return 'pending'
    return f'{value:.2%}' if percent else f'{value:.6g}'


def progress_message(status, config, events, last_dev):
    train = next((e for e in reversed(events) if e.get('event') == 'train'), {})
    # The tail may contain no evaluation event between infrequent evaluations.
    # last_dev_metrics.json persists the latest actual evaluation in that case.
    dev = (last_dev or status.get('latest_dev') or
           next((e for e in reversed(events) if e.get('event') == 'dev'), {}))
    return (f"{datetime.now().strftime('%H:%M:%S')}  {status.get('state')}  "
            f"step={status.get('step', 0)}/{config.get('max_steps', '?')}  "
            f"loss={format_number(train.get('loss'))}  "
            f"dev_phone_PER={format_number(dev.get('phone_error_rate'), percent=True)}  "
            f"dev_group_PER={format_number(dev.get('group_macro_per'), percent=True)}  "
            f"best_group_PER={format_number(status.get('best_dev_group_macro_per'), percent=True)}  "
            f"elapsed={format_number(status.get('elapsed_seconds', 0))}s" +
            (f"  stop={status['stop_reason']}" if status.get('stop_reason') else ''))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--user', default='root')
    p.add_argument('--remote-run', required=True)
    p.add_argument('--local-run', type=Path, required=True)
    p.add_argument('--password-stdin', action='store_true')
    p.add_argument('--max-seconds', type=int, default=7200)
    args = p.parse_args()
    password = sys.stdin.readline().rstrip('\r\n') if args.password_stdin else getpass.getpass('SSH password: ')
    client = paramiko.SSHClient()
    client.load_host_keys(str(Path.home() / '.ssh' / 'known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(args.host, port=args.port, username=args.user, password=password,
                   look_for_keys=False, allow_agent=False, timeout=20)
    del password
    args.local_run.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    sftp.get_channel().settimeout(20)
    watch = args.local_run / 'live_progress.log'
    seen = -1
    snapshots = {}
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.max_seconds:
            status = None
            for name in ['status.json', 'metrics.jsonl', 'run_config.json',
                         'last_dev_metrics.json', 'final_model.json']:
                try:
                    with sftp.open(f'{args.remote_run}/{name}', 'rb') as f:
                        size = f.stat().st_size
                        offset = max(0, size - 24000) if name == 'metrics.jsonl' else 0
                        f.seek(offset)
                        content = f.read(size - offset)
                        if offset:
                            content = content.split(b'\n', 1)[-1]
                    local_name = 'metrics_tail.jsonl' if name == 'metrics.jsonl' else name
                    (args.local_run / local_name).write_bytes(content)
                    if name.endswith('.json'):
                        snapshots[name] = json.loads(content)
                        if name == 'status.json':
                            status = snapshots[name]
                except (OSError, json.JSONDecodeError):
                    continue
            if status and (status.get('step') != seen or status.get('state') != 'training'):
                seen = status.get('step', 0)
                metrics = args.local_run / 'metrics_tail.jsonl'
                events = []
                if metrics.exists():
                    for line in metrics.read_text(encoding='utf-8').splitlines():
                        try: events.append(json.loads(line))
                        except json.JSONDecodeError: pass
                message = progress_message(status, snapshots.get('run_config.json', {}),
                                           events, snapshots.get('last_dev_metrics.json', {}))
                print(message, flush=True)
                with watch.open('a', encoding='utf-8') as f: f.write(message + '\n')
                if status.get('state') in ('finished', 'failed'):
                    break
            time.sleep(5)
    finally:
        sftp.close()
        client.close()


if __name__ == '__main__':
    main()
