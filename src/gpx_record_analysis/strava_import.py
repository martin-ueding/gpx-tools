import csv
import shutil
import os

import dateutil.parser

with open("activities.csv") as f:
    reader = csv.reader(f)
    next(reader)
    for line in reader:
        number = line[0]
        kind = line[3]
        name = line[2]
        start = dateutil.parser.parse(line[1])
        new_name = f"{kind}/{start.year}-{start.month:02d}-{start.day:02d} {start.hour:02d}:{start.minute:02d} {name}.gpx"
        if not os.path.isdir(kind):
            os.mkdir(kind)
        old_name = os.path.basename(line[10])
        if old_name.endswith(".gz"):
            old_name = old_name[:-3]
        if os.path.isfile(old_name):
            print(old_name, "→", new_name)
            os.rename(old_name, new_name)
