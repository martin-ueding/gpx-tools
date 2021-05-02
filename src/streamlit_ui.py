import glob
import os
import urllib.error
import urllib.request
import time

import altair as alt
import gpxpy
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


##############################################

# globals
PLT_COLORMAP = "hot"  # matplotlib color map
MAX_TILE_COUNT = 2000  # maximum number of tiles to download
MAX_HEATMAP_SIZE = (2160, 3840)  # maximum heatmap size in pixel

OSM_TILE_SERVER = "https://maps.wikimedia.org/osm-intl/{}/{}/{}.png"  # OSM tile url from https://wiki.openstreetmap.org/wiki/Tile_servers
OSM_TILE_SIZE = 256  # OSM tile size in pixel
OSM_MAX_ZOOM = 19  # OSM maximum zoom level


def deg2xy(lat_deg, lon_deg, zoom):
    # returns OSM coordinates (x,y) from (lat,lon) in degree
    # from https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    #
    # input: lat_deg = float
    #        long_deg = float
    #        zoom = int
    # output: x = float
    #         y = float

    lat_rad = np.radians(lat_deg)
    n = 2.0 ** zoom
    x = (lon_deg + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n

    return x, y


def xy2deg(x, y, zoom):
    # returns (lat, lon) in degree from OSM coordinates (x,y)
    # from https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    #
    # input: x = float
    #        y = float
    #        zoom = int
    # output: lat_deg = float
    #         lon_deg = float

    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1.0 - 2.0 * y / n)))
    lat_deg = np.degrees(lat_rad)

    return lat_deg, lon_deg


def gaussian_filter(image, sigma):
    # returns image filtered with a gaussian function of variance sigma**2
    #
    # input: image = numpy.ndarray
    #        sigma = float
    # output: image = numpy.ndarray

    i, j = np.meshgrid(
        np.arange(image.shape[0]), np.arange(image.shape[1]), indexing="ij"
    )

    mu = (int(image.shape[0] / 2.0), int(image.shape[1] / 2.0))

    gaussian = (
        1.0
        / (2.0 * np.pi * sigma * sigma)
        * np.exp(-0.5 * (((i - mu[0]) / sigma) ** 2 + ((j - mu[1]) / sigma) ** 2))
    )

    gaussian = np.roll(gaussian, (-mu[0], -mu[1]), axis=(0, 1))

    image_fft = np.fft.rfft2(image)
    gaussian_fft = np.fft.rfft2(gaussian)

    image = np.fft.irfft2(image_fft * gaussian_fft)

    return image


def download_tile(tile_url, tile_file):
    # download tile from url, save to file and wait 0.1s
    #
    # input: tile_url = str
    #        tile_file = str
    # output: bool

    request = urllib.request.Request(tile_url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urllib.request.urlopen(request) as response:
            data = response.read()

    except urllib.error.URLError:
        return False

    with open(tile_file, "wb") as file:
        file.write(data)

    time.sleep(0.1)

    return True


def heatmap_from_gpx(lat_lon_data, num_files=1, args_orange=False, args_sigma=1):
    # find tiles coordinates
    lat_min, lon_min = np.min(lat_lon_data, axis=0)
    lat_max, lon_max = np.max(lat_lon_data, axis=0)

    zoom = OSM_MAX_ZOOM

    while True:
        x_tile_min, y_tile_max = map(int, deg2xy(lat_min, lon_min, zoom))
        x_tile_max, y_tile_min = map(int, deg2xy(lat_max, lon_max, zoom))

        if (x_tile_max - x_tile_min + 1) * OSM_TILE_SIZE <= MAX_HEATMAP_SIZE[0] and (
            y_tile_max - y_tile_min + 1
        ) * OSM_TILE_SIZE <= MAX_HEATMAP_SIZE[1]:
            break

        zoom -= 1

    tile_count = (x_tile_max - x_tile_min + 1) * (y_tile_max - y_tile_min + 1)

    if tile_count > MAX_TILE_COUNT:
        exit("ERROR zoom value too high, too many tiles to download")

    # download tiles
    os.makedirs("tiles", exist_ok=True)

    supertile = np.zeros(
        (
            (y_tile_max - y_tile_min + 1) * OSM_TILE_SIZE,
            (x_tile_max - x_tile_min + 1) * OSM_TILE_SIZE,
            3,
        )
    )
    print("supertile.shape:", supertile.shape)

    n = 0
    for x in range(x_tile_min, x_tile_max + 1):
        for y in range(y_tile_min, y_tile_max + 1):
            n += 1

            tile_url = OSM_TILE_SERVER.format(zoom, x, y)

            tile_dir = os.path.expanduser(f'~/.cache/heatmap-tiles/{zoom}/{x}')
            if not os.path.isdir(tile_dir):
                os.makedirs(tile_dir)
            tile_file = f"{tile_dir}/{y}.png"

            if not glob.glob(tile_file):
                print("downloading tile {}/{}".format(n, tile_count))

                if not download_tile(tile_url, tile_file):
                    print(
                        "ERROR downloading tile {} failed, using blank tile".format(
                            tile_url
                        )
                    )

                    tile = np.ones((OSM_TILE_SIZE, OSM_TILE_SIZE, 3))

                    plt.imsave(tile_file, tile)

            tile = plt.imread(tile_file)

            i = y - y_tile_min
            j = x - x_tile_min

            supertile[
                i * OSM_TILE_SIZE : (i + 1) * OSM_TILE_SIZE,
                j * OSM_TILE_SIZE : (j + 1) * OSM_TILE_SIZE,
                :,
            ] = tile[:, :, :3]

    if not args_orange:
        supertile = np.sum(supertile * [0.2126, 0.7152, 0.0722], axis=2)  # to grayscale
        supertile = 1.0 - supertile  # invert colors
        supertile = np.dstack((supertile, supertile, supertile))  # to rgb

    print("supertile.shape:", supertile.shape)


    # fill trackpoints
    sigma_pixel = args_sigma if not args_orange else 1

    data = np.zeros(supertile.shape[:2])
    print("data.shape:", data.shape)

    xy_data = deg2xy(lat_lon_data[:, 0], lat_lon_data[:, 1], zoom)

    xy_data = np.array(xy_data).T
    print("xy_data.shape:", xy_data.shape)
    xy_data = np.round(
        (xy_data - [x_tile_min, y_tile_min]) * OSM_TILE_SIZE
    )  # to supertile coordinates
    print("xy_data.shape:", xy_data.shape)

    print(xy_data)
    print("xy_data.shape:", xy_data.shape)

    for j, i in xy_data.astype(int):
        data[
            i - sigma_pixel : i + sigma_pixel, j - sigma_pixel : j + sigma_pixel
        ] += 1.0

    print(data)
    print(data.min(), data.max())

    # threshold to max accumulation of trackpoint
    if not args_orange:
        res_pixel = (
            156543.03 * np.cos(np.radians(np.mean(lat_lon_data[:, 0]))) / (2.0 ** zoom)
        )  # from https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
        print(res_pixel)

        # trackpoint max accumulation per pixel = 1/5 (trackpoint/meter) * res_pixel (meter/pixel) * activities
        # (Strava records trackpoints every 5 meters in average for cycling activites)
        m = np.round((1.0 / 5.0) * res_pixel * num_files)
    else:
        m = 1.0

    m = 1.0

    print(data.min(), data.max())
    data[data > m] = m
    print(data.min(), data.max())

    # equalize histogram and compute kernel density estimation
    if not args_orange:
        data_hist, _ = np.histogram(data, bins=int(m + 1))

        data_hist = np.cumsum(data_hist) / data.size  # normalized cumulated histogram

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j] = m * data_hist[int(data[i, j])]  # histogram equalization

        data = gaussian_filter(
            data, float(sigma_pixel)
        )  # kernel density estimation with normal kernel

        data = (data - data.min()) / (data.max() - data.min())  # normalize to [0,1]

    # colorize
    if not args_orange:
        cmap = plt.get_cmap(PLT_COLORMAP)

        data_color = cmap(data)
        data_color[data_color == cmap(0.0)] = 0.0  # remove background color

        for c in range(3):
            supertile[:, :, c] = (1.0 - data_color[:, :, c]) * supertile[
                :, :, c
            ] + data_color[:, :, c]

    else:
        color = np.array([255, 82, 0], dtype=float) / 255  # orange

        for c in range(3):
            supertile[:, :, c] = np.minimum(
                supertile[:, :, c] + gaussian_filter(data, 1.0), 1.0
            )  # white

        data = gaussian_filter(data, 0.5)
        data = (data - data.min()) / (data.max() - data.min())

        for c in range(3):
            supertile[:, :, c] = (1.0 - data) * supertile[:, :, c] + data * color[c]

    return supertile


##############################################


def gpx_to_df(path: str, gpx: gpxpy.gpx.GPX) -> pd.DataFrame:
    points = gpx.tracks[0].segments[0].points

    time = [point.time for point in points]
    latitude = [point.latitude for point in points]
    longitude = [point.longitude for point in points]
    elevation = [point.elevation for point in points]
    step_size = [0] + [p1.distance_2d(p2) for p1, p2 in zip(points[:-1], points[1:])]
    distance = np.cumsum(step_size)

    df = pd.DataFrame(
        {
            "filename": os.path.basename(path),
            "time": time,
            "latitude": latitude,
            "longitude": longitude,
            "elevation": elevation,
            "step_size": step_size,
            "distance": distance,
        }
    )

    lag = 1
    diff = df[["time", "distance", "elevation"]].diff(periods=lag)
    diff_df = pd.DataFrame(
        {
            "time": df["time"],
            "speed": 3.6 * diff["distance"] / [x.seconds for x in diff["time"]],
            "slope": diff["elevation"],
        }
    )
    df["speed"] = diff_df["speed"].ewm(span=12).mean()
    df["slope"] = diff_df["slope"].ewm(span=12).mean()

    return df


def gpx_to_summary(gpx: gpxpy.gpx.GPX) -> dict:
    length = gpx.tracks[0].segments[0].length_2d()

    start = gpx.tracks[0].segments[0].points[0].time
    end = gpx.tracks[0].segments[0].points[-1].time
    date_format = "%a %Y-%m-%d %H:%M"
    duration = int((end - start).total_seconds())
    duration_str = f"{duration % 3600 // 60:02d}:{duration % 60:02d}"
    if duration >= 3600:
        duration_str = f"{duration // 3600:d}:" + duration_str
    avg_speed = 3.6 * length / duration

    result = {
        "Length / km": f"{length / 1000:.2f} km",
        "Start": start.strftime(date_format),
        "End": end.strftime(date_format),
        "Duration": duration_str,
        "Average speed": f"{avg_speed:.1f} km/h",
    }
    return result


base = os.path.expanduser("~/Dokumente/Karten/Tracks")

paths = {}
for dirpath, dirnames, filenames in os.walk(base):
    gpx_files = [file for file in filenames if file.lower().endswith(".gpx")]
    if len(gpx_files) > 0:
        paths[dirpath] = gpx_files

chosen_directory = st.selectbox("Directory", sorted(paths.keys()))

all_dfs = []
for filename in paths[chosen_directory]:
    with open(os.path.join(chosen_directory, filename)) as f:
        gpx = gpxpy.parse(f)
    df = gpx_to_df(filename, gpx)
    all_dfs.append(df)
dir_df = pd.concat(all_dfs)

img = heatmap_from_gpx(dir_df[["latitude", "longitude"]].to_numpy(), len(all_dfs))
st.image(img)


chosen_file = st.selectbox("File", sorted(paths[chosen_directory]))

with open(os.path.join(chosen_directory, chosen_file)) as f:
    gpx = gpxpy.parse(f)
df = gpx_to_df(chosen_file, gpx)

summary = gpx_to_summary(gpx)
st.json(summary)

chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        latitude="latitude",
        longitude="longitude",
        color=alt.Color(
            "elevation",
            title="Elevation",
            scale=alt.Scale(scheme="redyellowgreen", reverse=True),
        ),
    )
)
st.altair_chart(chart)

chart = (
    alt.Chart(df)
    .mark_line()
    .encode(
        x="time",
        y="elevation",
    )
)
st.altair_chart(chart)


chart = (
    alt.Chart(df)
    .mark_line()
    .encode(
        x="distance",
        y="elevation",
    )
)
st.altair_chart(chart)


chart = (
    alt.Chart(df)
    .mark_line()
    .encode(
        x="time",
        y="distance",
    )
)
st.altair_chart(chart)

chart = (
    alt.Chart(df)
    .mark_line()
    .encode(
        x="time",
        y="speed",
    )
)
st.altair_chart(chart)
