#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2021 Martin Ueding <mu@martin-ueding.de>

import argparse
import json
import os
import glob
import datetime
import pprint
import shutil

import gpxpy


def find_start(gpx):
    for track in gpx.tracks:
        for segment in track.segments:
            points = segment.points
            if len(points) > 0:
                return points[0]
    raise RuntimeError('Could not find a start.')


def find_end(gpx):
    for track in reversed(gpx.tracks):
        for segment in reversed(track.segments):
            points = segment.points
            if len(points) > 0:
                return points[-1]
    raise RuntimeError('Could not find an end.')


def main():
    options = _parse_args()

    cache_file = 'duplicate-cache.json'
    if os.path.isfile(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
    else:
        cache = {}

    for dirpath, dirnames, filenames in os.walk('.'):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            if not filename.endswith('.gpx'):
                continue
            path = os.path.join(dirpath, filename)
            if path in cache:
                continue
            print(path)
            with open(path) as f:
                gpx = gpxpy.parse(f)
            start = find_start(gpx).time.timestamp()
            end = find_end(gpx).time.timestamp()
            cache[path] = [start, end]

    for path in list(cache.keys()):
        if not os.path.isfile(path):
            del cache[path]

    with open(cache_file, 'w') as f:
        json.dump(cache, f)

    items = list(cache.items())
    items.sort()
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            path1, (start1, end1) = items[i]
            path2, (start2, end2) = items[j]

            if start1 < start2 < end1 or start2 < start1 < end2:
                print(f'gpxsee "{path1}" "{path2}"')


def _parse_args():
    parser = argparse.ArgumentParser()
    options = parser.parse_args()

    return options


if __name__ == '__main__':
    main()
