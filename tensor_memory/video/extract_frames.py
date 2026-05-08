"""
One-time frame extraction for UCF-101.
Uses PyAV with direct timestamp seeking — does NOT decode the full video.

Output layout:
  <out_dir>/<ClassName>/<video_stem>/frame_0000.jpg  ...

Usage:
  python extract_frames.py --video_dir UCF-101 --out_dir UCF-101-frames --num_frames 8 --workers 8
"""
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm


def extract_one(video_path: Path, out_root: Path, num_frames: int) -> str:
    out_dir = out_root / video_path.parent.name / video_path.stem
    if len(list(out_dir.glob("frame_*.jpg"))) >= num_frames:
        return "skip"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import av
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"   # key-frame seek only

            duration = float(stream.duration or 0) * float(stream.time_base)
            if duration <= 0:
                # fallback: use container duration
                duration = float(container.duration or 1) / 1_000_000

            # evenly-spaced target timestamps
            times = np.linspace(0, duration * 0.999, num_frames)
            frames_saved = 0

            for i, t in enumerate(times):
                pts = int(t / float(stream.time_base))
                container.seek(pts, stream=stream, backward=True, any_frame=False)
                for pkt in container.decode(stream):
                    img = pkt.to_image().convert("RGB")
                    img.save(out_dir / f"frame_{i:04d}.jpg", quality=95)
                    frames_saved += 1
                    break   # one frame per seek

            # pad with copies of last frame if we got fewer than expected
            existing = sorted(out_dir.glob("frame_*.jpg"))
            while len(existing) < num_frames:
                last = existing[-1]
                dst  = out_dir / f"frame_{len(existing):04d}.jpg"
                Image.open(last).save(dst, quality=95)
                existing.append(dst)

        return "ok"
    except Exception as e:
        # fallback: torchvision.io (slower but more codec coverage)
        try:
            import torchvision.io as tvio
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vframes, _, _ = tvio.read_video(str(video_path), pts_unit="sec",
                                                output_format="TCHW")
            if vframes.shape[0] == 0:
                return f"empty: {video_path.name}"
            total = vframes.shape[0]
            idxs  = np.linspace(0, total - 1, num_frames, dtype=int)
            for i, idx in enumerate(idxs):
                frame = vframes[idx].permute(1, 2, 0).numpy()
                Image.fromarray(frame).save(out_dir / f"frame_{i:04d}.jpg", quality=95)
            return "ok(tv)"
        except Exception as e2:
            return f"error: {e2}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir",  default="UCF-101")
    p.add_argument("--out_dir",    default="UCF-101-frames")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--workers",    type=int, default=8)
    args = p.parse_args()

    video_root = Path(args.video_dir)
    out_root   = Path(args.out_dir)
    videos     = sorted(video_root.rglob("*.avi"))
    print(f"Found {len(videos)} videos  ->  extracting {args.num_frames} frames each")
    print(f"Output: {out_root.resolve()}")

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_one, v, out_root, args.num_frames): v for v in videos}
        for fut in tqdm(as_completed(futs), total=len(videos)):
            result = fut.result()
            if result not in ("ok", "ok(tv)", "skip"):
                errors.append((futs[fut].name, result))

    print(f"\nDone. Errors: {len(errors)}")
    for name, e in errors[:20]:
        print(f"  {name}: {e}")


if __name__ == "__main__":
    main()
