"""Pace actual console UI captures into one short model-switching MP4.

These are real browser captures of selection/apply/result states, not a
reconstructed interface. Holds are editorial; not a latency benchmark.
"""
import argparse
import json
from pathlib import Path
import imageio_ffmpeg
import numpy as np
from PIL import Image

SHOTS = [("01-select-rl.png", 2), ("02-rl-applied.png", 5),
         ("03-select-greedy.png", 2), ("04-applying-greedy.png", 1),
         ("05-greedy-applied.png", 5), ("06-select-initial.png", 2),
         ("07-applying-initial.png", 1), ("08-initial-applied.png", 5)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    for name, _ in SHOTS:
        if not (root/name).is_file(): raise FileNotFoundError(root/name)
    movie = root/"actual-console-switching.mp4"
    if movie.exists(): raise FileExistsError(movie)
    with Image.open(root/SHOTS[0][0]) as first: width,height=first.size
    size = (width+width%2,height+height%2)
    writer = imageio_ffmpeg.write_frames(str(movie),size,fps=30,codec='libx264',
        quality=8,macro_block_size=2,output_params=['-movflags','+faststart'])
    writer.send(None)
    try:
        for name,seconds in SHOTS:
            with Image.open(root/name) as shot:
                if shot.size != (width,height): raise ValueError('Browser viewport changed during capture')
                frame = np.zeros((size[1],size[0],3), dtype=np.uint8)
                frame[:height,:width] = np.asarray(shot.convert('RGB'))
            for _ in range(seconds*30): writer.send(frame)
    finally: writer.close()
    (root/"provenance.json").write_text(json.dumps(dict(
        source="Actual simulation console browser captures",
        saved_outputs=True, newly_run_inference=False, editorial_holds=True,
        duration_s=sum(t for _,t in SHOTS), shots=SHOTS), indent=2))
    print(movie)


if __name__ == "__main__": main()
