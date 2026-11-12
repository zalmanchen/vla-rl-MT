"""Make small, traceable LIBERO datasets for engineering checks only.

Select complete episodes by language and success flag. Rewrite only identifiers;
states, actions, timestamps and video bytes remain those in the released data.
Never use this subset for the reproduction's reported training or evaluation.
"""

import argparse
import json
import shutil
from pathlib import Path

TASK = "put both the alphabet soup and the tomato sauce in the basket"
SPLITS = ("libero10_task0_sft", "libero10_task0_train", "libero10_task0_eval")


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_episodes(episodes, sft):
    candidates = [ep for ep in episodes if TASK in ep["tasks"]]
    if sft:
        selected = candidates[:2]
    else:
        selected = [next(ep for ep in candidates if bool(ep["is_success"]) == flag)
                    for flag in (False, True)]
    if len(selected) != 2:
        raise ValueError("Need two complete target-task episodes")
    return selected


def prepare(source, output):
    import pyarrow as pa
    import pyarrow.parquet as pq

    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "engineering smoke only; not a training/evaluation result", "splits": {}}
    for name in SPLITS:
        src, dst = source / name, output / name
        selected = select_episodes(read_lines(src / "meta/episodes.jsonl"), name.endswith("sft"))
        shutil.copytree(src / "meta", dst / "meta")
        # Published sidecars refer to the original identifiers and must not leak in.
        for path in (dst / "meta").glob("*.parquet"):
            path.unlink()
        info = json.loads((src / "meta/info.json").read_text())
        videos = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
        offset, remapped, records = 0, [], []
        id_map = {}
        for new_id, episode in enumerate(selected):
            old_id = episode["episode_index"]
            id_map[old_id] = new_id
            old_vars = {"episode_chunk": old_id // info["chunks_size"], "episode_index": old_id}
            new_vars = {"episode_chunk": 0, "episode_index": new_id}
            source_path = src / info["data_path"].format(**old_vars)
            table = pq.read_table(source_path)
            n = len(table)
            if n != episode["length"]:
                raise ValueError(f"Episode length mismatch: {source_path}")
            for key, values in {"episode_index": [new_id] * n,
                                "task_index": [0] * n,
                                "index": list(range(offset, offset + n))}.items():
                column = table.schema.get_field_index(key)
                table = table.set_column(column, table.schema.field(column), pa.array(values, type=pa.int64()))
            dest = dst / info["data_path"].format(**new_vars)
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dest)
            for key in videos:
                original = src / info["video_path"].format(**old_vars, video_key=key)
                linked = dst / info["video_path"].format(**new_vars, video_key=key)
                linked.parent.mkdir(parents=True, exist_ok=True)
                linked.symlink_to(original.resolve())
            remapped.append({**episode, "episode_index": new_id})
            records.append({"source_episode": old_id, "episode_index": new_id,
                            "frames": n, "success": episode.get("is_success", True)})
            offset += n
        info.update(total_episodes=2, total_frames=offset, total_tasks=1,
                    total_chunks=1, total_videos=2 * len(videos), splits={"train": "0:2"})
        (dst / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
        (dst / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": TASK}) + "\n")
        (dst / "meta/episodes.jsonl").write_text("".join(json.dumps(ep) + "\n" for ep in remapped))
        stats_path = dst / "meta/episodes_stats.jsonl"
        if stats_path.exists():
            stats = [{**entry, "episode_index": id_map[entry["episode_index"]]}
                     for entry in read_lines(src / "meta/episodes_stats.jsonl")
                     if entry["episode_index"] in id_map]
            stats_path.write_text("".join(json.dumps(entry) + "\n" for entry in stats))
        report["splits"][name] = {"source": str(src.resolve()), "episodes": records, "frames": offset}
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
