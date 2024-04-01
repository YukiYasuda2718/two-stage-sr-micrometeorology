import sys
import xml.etree.ElementTree as ET
from logging import getLogger

import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray
from shapely.geometry import Point, Polygon

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


logger = getLogger()


def make_longitude_latitude_grids(
    *,
    center_lon: float,
    center_lat: float,
    width_x_km: float,
    width_y_km: float,
    num_x_grids: int,
    num_y_grids: int,
    earth_radius_km: float = 6371.0,
    endpoint: bool = True,
) -> tuple[NDArray, NDArray]:
    width_lon = np.rad2deg(
        width_x_km / (earth_radius_km * np.cos(np.deg2rad(center_lat)))
    )
    width_lat = np.rad2deg(width_y_km / earth_radius_km)

    min_lon, max_lon, min_lat, max_lat = (
        center_lon - width_lon / 2,
        center_lon + width_lon / 2,
        center_lat - width_lat / 2,
        center_lat + width_lat / 2,
    )

    lons = np.linspace(min_lon, max_lon, num_x_grids, endpoint=endpoint)
    lats = np.linspace(min_lat, max_lat, num_y_grids, endpoint=endpoint)

    return np.meshgrid(lons, lats, indexing="ij")


def make_domain_polygon_from_grids(lons: NDArray, lats: NDArray) -> Polygon:
    min_lon, max_lon = np.min(lons.flatten()), np.max(lons.flatten())
    min_lat, max_lat = np.min(lats.flatten()), np.max(lats.flatten())

    logger.info(f"min_lon = {min_lon}, max_lon = {max_lon}")
    logger.info(f"min_lat = {min_lat}, max_lat = {max_lat}")

    domain = Polygon(
        [(min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat)]
    )
    assert domain.is_valid

    return domain


def get_tile_box(gml_file_path: str) -> Polygon:
    context = ET.iterparse(gml_file_path, events=("start", "end"))

    _, root = next(context)

    min_lat, max_lat = None, None
    min_lon, max_lon = None, None

    for event, elem in context:
        if event == "end" and elem.tag.endswith("lowerCorner"):
            min_lat, min_lon, _ = elem.text.split(" ")

        if event == "end" and elem.tag.endswith("upperCorner"):
            max_lat, max_lon, _ = elem.text.split(" ")

        if min_lat is not None and max_lat is not None:
            break

    outline = {
        "min_lat": float(min_lat),
        "max_lat": float(max_lat),
        "min_lon": float(min_lon),
        "max_lon": float(max_lon),
    }

    return shapely.geometry.box(
        outline["min_lon"],
        outline["min_lat"],
        outline["max_lon"],
        outline["max_lat"],
        ccw=True,
    )


def get_lon_lat_indices_covering_box(
    box: Polygon, grid_lons: np.ndarray, grid_lats: np.ndarray
) -> tuple[int, int, int, int]:
    assert grid_lats.ndim == grid_lons.ndim == 2
    assert grid_lats.shape == grid_lons.shape

    min_lon, max_lon = min(box.boundary.xy[0]), max(box.boundary.xy[0])
    min_lat, max_lat = min(box.boundary.xy[1]), max(box.boundary.xy[1])

    min_lon_idx = np.argmin(np.abs(grid_lons[:, 0] - min_lon))
    max_lon_idx = np.argmin(np.abs(grid_lons[:, 0] - max_lon))

    min_lat_idx = np.argmin(np.abs(grid_lats[0, :] - min_lat))
    max_lat_idx = np.argmin(np.abs(grid_lats[0, :] - max_lat))

    min_lon_idx = max(0, min_lon_idx - 1)
    min_lat_idx = max(0, min_lat_idx - 1)

    max_lon_idx = min(grid_lons.shape[0], max_lon_idx + 1)
    max_lat_idx = min(grid_lats.shape[1], max_lat_idx + 1)

    return min_lon_idx, max_lon_idx, min_lat_idx, max_lat_idx


def get_lat_lon_height_list(
    str_poslist: str,
) -> tuple[list[tuple[float, float]], float]:
    pos = str_poslist.split(" ")  # lat, lon, height

    poslist = []
    height_from_sea_surf = None

    for lat, lon, height in zip(pos[0::3], pos[1::3], pos[2::3]):
        poslist.append((float(lon), float(lat)))
        height = float(height)

        if height_from_sea_surf is None:
            height_from_sea_surf = height
        else:
            assert height_from_sea_surf == height, f"{height_from_sea_surf} vs {height}"

    return poslist, height_from_sea_surf


def get_lat_lon_list_and_averaged_height(
    str_poslist: str,
) -> tuple[list[tuple[float, float]], float]:
    pos = str_poslist.split(" ")  # lat, lon, height

    poslist = []
    heights_from_sea_surf = []

    for lat, lon, height in zip(pos[0::3], pos[1::3], pos[2::3]):
        poslist.append((float(lon), float(lat)))
        heights_from_sea_surf.append(float(height))

    return poslist, np.mean(heights_from_sea_surf)


def get_lat_lon_list(str_poslist: str) -> list[tuple[float, float]]:
    pos = str_poslist.split(" ")  # lat, lon, height

    poslist = []
    for lat, lon in zip(pos[0::3], pos[1::3]):
        poslist.append((float(lon), float(lat)))

    return poslist


def get_building_tag_count(file_path) -> int:
    context = ET.iterparse(file_path, events=("start", "end"))
    _, root = next(context)

    cnt = 0
    for event, elem in context:
        if event == "end" and elem.tag.endswith("Building"):
            cnt += 1

    return cnt


def update_building_information(
    dict_build_info: dict,
    all_grid_info: dict,
    bldg_outer_shapes: list[Polygon],
    heights_from_sea: list[float],
    target_domain: Polygon,
):
    for bldg, h in zip(bldg_outer_shapes, heights_from_sea):
        deleted_keys = []

        if bldg.disjoint(target_domain):
            continue

        for key, grid_info in all_grid_info.items():
            point = grid_info["point"]
            mssg_index = (grid_info["i+1"], grid_info["j+1"])

            if bldg.covers(point):
                dict_build_info[mssg_index] = {
                    "Longitude": point.coords[0][0],
                    "Latitude": point.coords[0][1],
                    "i+1": grid_info["i+1"],
                    "j+1": grid_info["j+1"],
                    "i": grid_info["i+1"] - 1,
                    "j": grid_info["j+1"] - 1,
                    "BuildingHiehtFromSeaSurf": h,
                }
                deleted_keys.append(key)

        for key in deleted_keys:
            if key in all_grid_info:
                del all_grid_info[key]


def update_building_information_reverse(
    dict_build_info: dict,
    all_grid_info: dict,
    bldg_outer_shapes: list[Polygon],
    heights_from_sea: list[float],
    target_domain: Polygon,
):
    target_bldgs, target_heights = [], []

    for bldg, h in zip(bldg_outer_shapes, heights_from_sea):
        if bldg.disjoint(target_domain):
            continue

        target_bldgs.append(bldg)
        target_heights.append(h)

    for _, grid_info in all_grid_info.items():
        point = grid_info["point"]
        mssg_index = (grid_info["i+1"], grid_info["j+1"])

        for bldg, h in zip(target_bldgs, target_heights):
            if bldg.covers(point):
                dict_build_info[mssg_index] = {
                    "Longitude": point.coords[0][0],
                    "Latitude": point.coords[0][1],
                    "i+1": grid_info["i+1"],
                    "j+1": grid_info["j+1"],
                    "i": grid_info["i+1"] - 1,
                    "j": grid_info["j+1"] - 1,
                    "BuildingHiehtFromSeaSurf": h,
                }
                break


def write_dataframe_for_mssg(
    df: pd.DataFrame,
    file_name: str = "Index_.txt",
    cols: list = ["i", "j", "lu"],
):
    df[cols].to_csv(file_name, sep=" ", index=False)

    with open(file_name, "r") as f:
        lines = f.readlines()
        lines[0] = f"#  {lines[0]}"  # add `#` to header

    with open(file_name, "w") as f:
        f.writelines(lines)


def get_hrefs_in_lod2_solid(context):
    list_hrefs = []
    for event, elem in context:
        if event == "end" and elem.tag.endswith("surfaceMember"):
            for k, v in elem.attrib.items():
                if k.endswith("href"):
                    list_hrefs.append(v.replace("#", ""))
        if event == "end" and elem.tag.endswith("lod2Solid"):
            break
    return list_hrefs


def get_roof_poslist_from_lod2_surface(context, target_hrefs: set[str]):
    poslist = None

    for event, elem in context:
        if event == "start" and elem.tag.endswith("boundedBy"):
            _, bldg_elem = next(context)

            if bldg_elem.tag.endswith("WallSurface") or bldg_elem.tag.endswith(
                "GroundSurface"
            ):
                break

        if event == "start" and elem.tag.endswith("Polygon"):
            if not set(elem.attrib.values()).intersection(target_hrefs):
                break

        if event == "end" and elem.tag.endswith("posList"):
            poslist = elem.text
            break

    for event, elem in context:
        if event == "end" and elem.tag.endswith("boundedBy"):
            return poslist


def get_bldg_shapes_and_heights(
    dict_building_shapes: dict[float, Polygon]
) -> tuple[list[float], list[Polygon]]:
    heights_from_sea = []
    bldg_outer_shapes = []

    for h, bldg in sorted(dict_building_shapes.items(), reverse=True):
        is_added = True
        for _bldg in bldg_outer_shapes:
            try:
                if _bldg.contains(bldg):
                    is_added = False
                    break
            except Exception as e:
                # An error can occur when the original geometric data is invalid.
                logger.error(e)
        if is_added:
            heights_from_sea.append(h)
            bldg_outer_shapes.append(bldg)

    return heights_from_sea, bldg_outer_shapes


def convert_table_to_grid_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    return pd.pivot_table(
        data=df[["i", "j", target_col]],
        values=target_col,
        index="i",
        columns="j",
        aggfunc="max",
    )
