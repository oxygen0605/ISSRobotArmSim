#!/usr/bin/env python3
"""Copy real camera MP4s and create four-view GIFs at simulation-time speed."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path

VIEWS=("front","opposite","side","overview")
LABELS=("First person","Opposite (recording only)","Side","Overview (recording only)")
ROOT=Path(__file__).resolve().parents[1]
FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def sha(path):
    with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()

def probe(path):
    return json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries",
        "format=duration:stream=width,height,nb_frames","-of","json",str(path)],text=True))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run",type=Path,required=True)
    p.add_argument("--output",type=Path,default=ROOT/"docs/results/media")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    published=json.loads((ROOT/"artifacts/smolvla_one_policy/capture_config.json").read_text())
    manifest={"source_run":args.source_run.name,"timing":"1x simulation time; GIF 10 fps; no additional end pause",
              "grid":["front, opposite","side, overview"],"tasks":[]}
    for task in ("task1","task2","task3"):
        source=args.source_run/task;result=json.loads((source/"result.json").read_text())
        assert result["success"] and not result["teacher_used"]
        hashes={Path(k).name:v for k,v in result["checkpoint_hashes"].items()}
        assert hashes["adapter_model.safetensors"]==published["adapter_sha256"]
        assert hashes["shared_head.safetensors"]==published["head_sha256"]
        row={"task":task,"seed":result["case"]["seed"],"success":True,"steps":result["steps"],
             "instruction":result["instruction"],"adapter_sha256":hashes["adapter_model.safetensors"],
             "head_sha256":hashes["shared_head.safetensors"],"videos":{}}
        for view in (*VIEWS,"front_depth","side_depth"):
            original=source/("policy_"+view+".mp4");dest=args.output/(task+"-"+view+".mp4")
            shutil.copy2(original,dest)
            row["videos"][view]={"file":dest.name,"sha256":sha(dest),"duration_seconds":float(probe(dest)["format"]["duration"])}
        cmd=["ffmpeg","-hide_banner","-loglevel","error","-y","-filter_complex_threads","2"]
        for view in VIEWS:cmd+=["-i",str(args.output/(task+"-"+view+".mp4"))]
        filters=[]
        for i,label in enumerate(LABELS):
            filters.append(f"[{i}:v]setpts=PTS-STARTPTS,fps=10,scale=512:512:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad=512:512:(ow-iw)/2:(oh-ih)/2:color=0x101827,setsar=1,pad=512:544:0:32:color=0x101827,"
                f"drawtext=fontfile={FONT}:text='{label}':fontcolor=white:fontsize=18:x=10:y=7[v{i}]")
        filters.append("[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|512_0|0_544|512_544:shortest=1[grid]")
        filters.append("[grid]split[p][q];[p]palettegen=stats_mode=diff:reserve_transparent=0[palette];"
                       "[q][palette]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle[out]")
        gif=args.output/(task+"-four-views-1x.gif")
        subprocess.run(cmd+["-filter_complex",";".join(filters),"-map","[out]","-loop","0",str(gif)],check=True)
        row["gif"]={"file":gif.name,"sha256":sha(gif),"bytes":gif.stat().st_size}
        manifest["tasks"].append(row)
        print(json.dumps({"task":task,"gif_MB":round(gif.stat().st_size/1e6,3)},ensure_ascii=False),flush=True)
    (args.output.parent/"media-manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
if __name__=="__main__":main()
