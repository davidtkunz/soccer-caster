"""Fetch a short clip from a video URL for pipeline testing.

Grabs a *section*, not a whole match. A few minutes at 720p is plenty to check
whether segmentation thresholds are anywhere near right, and it keeps the
download small enough to iterate on.

    python tools/fetch_clip.py "<url>" --start 00:12:00 --duration 180

Note on sourcing: YouTube's Terms of Service prohibit downloading, and
broadcast football is copyrighted. For a local research prototype that is a
judgement call you own. If you want footage that is unambiguously licensed for
this *and* comes with ground-truth event labels -- which is what the evaluation
set actually needs -- SoccerNet is the purpose-built option.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def ffmpeg_path() -> str | None:
    """Prefer a bundled ffmpeg so no system install is required."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def parse_timestamp(value: str) -> float:
    """Accept SS, MM:SS, or HH:MM:SS."""
    parts = [float(p) for p in value.split(":")]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def fetch(url: str, out: Path, start: float | None, duration: float, height: int):
    import yt_dlp

    out.parent.mkdir(parents=True, exist_ok=True)

    options = {
        "format": f"bestvideo[height<={height}]+bestaudio/best[height<={height}]",
        "outtmpl": str(out.with_suffix("")) + ".%(ext)s",
        "merge_output_format": "mp4",
        "quiet": False,
        "noprogress": False,
        "overwrites": True,
    }

    ffmpeg = ffmpeg_path()
    if ffmpeg:
        options["ffmpeg_location"] = ffmpeg

    if start is not None:
        if not ffmpeg:
            print(
                "warning: no ffmpeg available, so --start/--duration are ignored "
                "and the whole video will be downloaded.",
                file=sys.stderr,
            )
        else:
            end = start + duration
            options["download_ranges"] = lambda info, ydl: [
                {"start_time": start, "end_time": end}
            ]
            options["force_keyframes_at_cuts"] = True

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)

    print()
    print(f"title:  {info.get('title')}")
    print(f"fps:    {info.get('fps')}")
    print(f"size:   {info.get('width')}x{info.get('height')}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--out", default="data/clip.mp4", type=Path)
    parser.add_argument("--start", default=None,
                        help="section start, e.g. 00:12:00")
    parser.add_argument("--duration", default=180.0, type=float,
                        help="seconds to grab from --start (default 180)")
    parser.add_argument("--height", default=720, type=int,
                        help="max vertical resolution (default 720)")
    args = parser.parse_args()

    start = parse_timestamp(args.start) if args.start else None
    path = fetch(args.url, args.out, start, args.duration, args.height)
    print(f"\nsaved: {path}")
    print("\nnext:")
    print(f"  python -m caster.segmentation {path} --dump out/segmentation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
