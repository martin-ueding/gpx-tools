import argparse
import os
import glob
import datetime
import pprint
import shutil

import gpxpy


def merge_track(bucket):
    """
    Merges multiple GPX tracks when the gap between them is not too long.

    :param tracks:
    :return:
    """
    assert len(bucket) >= 1
    if len(bucket) == 1:
        return bucket[0]

    for elem in bucket[1:]:
        bucket[0].tracks[0].segments[0].points += elem.tracks[0].segments[0].points
    return bucket[0]


def main():
    options = _parse_args()

    merged_dir = os.path.join(options.basedir, "Merged")
    os.makedirs(merged_dir, exist_ok=True)
    processed_raw_dir = os.path.join(options.basedir, "Processed Raw")
    os.makedirs(processed_raw_dir, exist_ok=True)

    raw_gpx_paths = glob.glob(os.path.join(options.basedir, "*.gpx"))
    raw_gpx_paths.sort()
    raw_gpx = []
    for raw_gpx_path in raw_gpx_paths:
        print(f"Loading {raw_gpx_path} …")
        with open(raw_gpx_path) as f:
            raw_gpx.append(gpxpy.parse(f))

    raw_gpx.sort(key=lambda gpx: gpx.tracks[0].segments[0].points[0].time)

    max_gap = datetime.timedelta(minutes=options.max_gap_minutes)

    buckets = []
    if len(raw_gpx) > 0:
        bucket = []
        for i in range(1, len(raw_gpx)):
            first = raw_gpx[i - 1]
            second = raw_gpx[i]
            gap = (
                second.tracks[0].segments[0].points[0].time
                - first.tracks[-1].segments[-1].points[-1].time
            )
            print(gap)
            bucket.append(first)
            if gap > max_gap:
                buckets.append(bucket)
                bucket = []
        bucket.append(raw_gpx[-1])
        buckets.append(bucket)

    pprint.pprint(buckets)

    for i, bucket in enumerate(buckets):
        print(f"Merging bucket {i} of {len(buckets)} …")
        merged = merge_track(bucket)
        with open(f"{merged_dir}/{merged.tracks[0].name}.gpx", "w") as f:
            f.write(merged.to_xml())

    for raw_gpx_path in raw_gpx_paths:
        shutil.move(raw_gpx_path, processed_raw_dir)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Utilities to work with GPX data and create a heatmap from it."
    )
    parser.add_argument(
        "--basedir",
        default=os.path.expanduser("~/Dokumente/Karten/Tracks"),
        help="Path to directory structure. Default: %(default)s",
    )
    parser.add_argument("--max_gap_minutes", default=5, type=int)
    options = parser.parse_args()

    return options


if __name__ == "__main__":
    main()
