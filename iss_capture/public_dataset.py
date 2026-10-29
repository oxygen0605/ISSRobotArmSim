"""Pinned public dataset retrieval and observed-input episode replay."""
from pathlib import Path, PurePosixPath
import hashlib, json, os, shutil, subprocess, sys, tarfile, tempfile
import numpy as np
from PIL import Image
from .config import ROOT

DEFAULT_DIRECTORY = ROOT / "datasets/task_scenes_v1"
SOURCE_FILE = ROOT / "config/dataset_source.json"

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def source_config():
    config = json.loads(SOURCE_FILE.read_text())
    if not isinstance(config.get("revision"), str) or len(config["revision"]) != 40:
        raise RuntimeError("The public dataset must be pinned to a published commit.")
    return config

def verify_episode(directory, files):
    for item in files:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest path")
        path = directory.parent / Path(*relative.parts)
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise RuntimeError("Dataset file verification failed: " + str(path))

def extract_episode(archive, destination, episode, checksums):
    """Reject links and paths outside the single expected episode directory."""
    if digest(archive) != episode["sha256"]:
        raise RuntimeError("Archive hash mismatch: " + episode["archive"])
    target = destination / episode["id"]
    expected = {row["path"]: row for row in checksums}
    if target.exists():
        verify_episode(target, checksums)
        return
    with tempfile.TemporaryDirectory(prefix=".dataset-extract-", dir=destination) as temporary:
        stage = Path(temporary)
        seen = set()
        with tarfile.open(archive, "r:") as tar:
            for member in tar:
                name = PurePosixPath(member.name)
                if (not member.isfile() or name.is_absolute() or ".." in name.parts
                        or len(name.parts) != 2 or name.parts[0] != episode["id"]
                        or member.name not in expected or member.name in seen):
                    raise ValueError("Unexpected archive member: " + member.name)
                if member.size != expected[member.name]["bytes"]:
                    raise ValueError("Unexpected member size: " + member.name)
                output = stage / Path(*name.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as inp, output.open("wb") as out:
                    shutil.copyfileobj(inp, out)
                seen.add(member.name)
        if seen != set(expected):
            raise RuntimeError("Archive has missing files")
        verify_episode(stage / episode["id"], checksums)
        (stage / episode["id"]).rename(target)

def download_dataset(destination=DEFAULT_DIRECTORY, episodes=None, split=None):
    # This function is called by the dedicated online CLI process.
    from huggingface_hub import hf_hub_download
    config = source_config()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    def fetch(name):
        return Path(hf_hub_download(config["repo_id"], name, repo_type="dataset",
                                   revision=config["revision"]))
    manifest_file = fetch("metadata/dataset_manifest.json")
    if digest(manifest_file) != config["manifest_sha256"]:
        raise RuntimeError("Dataset manifest hash mismatch")
    manifest = json.loads(manifest_file.read_text())
    checksum_file = fetch("metadata/file_checksums.jsonl")
    if digest(checksum_file) != config["checksums_sha256"]:
        raise RuntimeError("Dataset checksums hash mismatch")
    checksums = [json.loads(line) for line in checksum_file.read_text().splitlines()]
    wanted = set(episodes or [])
    known = {row["id"] for row in manifest["episodes"]}
    if wanted - known:
        raise ValueError("Unknown episode IDs: " + str(sorted(wanted-known)))
    selected = [row for row in manifest["episodes"]
                if (not wanted or row["id"] in wanted) and (split is None or row["split"] == split)]
    if not selected:
        raise ValueError("No episodes match the selection")
    for index, episode in enumerate(selected, 1):
        files = [row for row in checksums if row["path"].split("/")[0] == episode["id"]]
        assert len(files) == episode["files"]
        print(f"[{index}/{len(selected)}] {episode['id']}", flush=True)
        if (destination / episode["id"]).exists():
            verify_episode(destination / episode["id"], files)
        else:
            extract_episode(fetch(episode["archive"]), destination, episode, files)
    (destination/"metadata").mkdir(exist_ok=True)
    shutil.copy2(manifest_file, destination/"metadata/dataset_manifest.json")
    shutil.copy2(checksum_file, destination/"metadata/file_checksums.jsonl")
    provenance = {**config, "downloaded_episodes": sorted(
        name for name in known if (destination/name/"result.json").is_file())}
    (destination/".dataset_source.json").write_text(json.dumps(provenance, indent=2)+"\n")
    return destination

def ensure_dataset(directory=None, episodes=None):
    directory = Path(directory or DEFAULT_DIRECTORY).resolve()
    if directory != DEFAULT_DIRECTORY.resolve():
        if not directory.is_dir() or not list(directory.glob("*/result.json")):
            raise FileNotFoundError("No dataset episodes at " + str(directory))
        if episodes and any(not (directory/name/"result.json").is_file() for name in episodes):
            raise FileNotFoundError("Requested episode missing from " + str(directory))
        return directory
    config = source_config()
    marker = directory/".dataset_source.json"
    if marker.exists():
        current = json.loads(marker.read_text())
        downloaded = set(current.get("downloaded_episodes", []))
        needed = set(episodes or [])
        if (current.get("revision") == config["revision"]
                and (needed <= downloaded if needed else len(downloaded) == config["episode_count"])
                and all((directory/name/"result.json").is_file() for name in downloaded)):
            return directory
    command = [sys.executable, str(ROOT/"scripts/download_dataset.py"), "--destination", str(directory)]
    for name in episodes or []:
        command += ["--episode", name]
    subprocess.run(command, check=True, env={**os.environ, "HF_HUB_OFFLINE": "0"})
    return directory

def dataset_provenance(directory):
    marker = Path(directory) / ".dataset_source.json"
    if marker.is_file():
        data = json.loads(marker.read_text())
        return {key: data[key] for key in ("repo_id", "repo_type", "revision",
                "manifest_sha256", "checksums_sha256", "format") if key in data}
    return {"kind": "local", "directory": str(Path(directory))}

def load_observation(directory, record):
    """Build policy input from recorded sensors; ignore labels and object truth."""
    directory = Path(directory)
    frame = record["frame"]
    if not frame.isdigit():
        raise ValueError("Invalid frame name")
    observation = {key: np.asarray(record[key], dtype=np.float64)
                   for key in ("q", "q_velocity", "previous_command")}
    observation.update({key: float(record[key])
                        for key in ("image_time", "controller_time", "sim_time")})
    for name in ("front", "side", "previous_front"):
        with Image.open(directory/(frame+"_"+name+".png")) as image:
            observation[name] = np.asarray(image.convert("RGB"))
    with np.load(directory/(frame+"_depth.npz"), allow_pickle=False) as depth:
        for name in ("front", "side"):
            observation[name+"_depth"] = depth[name].astype(np.float32)
    return observation
