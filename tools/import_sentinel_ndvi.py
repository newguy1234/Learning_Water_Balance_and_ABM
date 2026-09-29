"""Import an existing Sentinel-2 NDVI cube into this project's annual input file.

Example:
    python tools/import_sentinel_ndvi.py SOURCE.nc --year 2022
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def import_ndvi(source, year, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Source and destination must differ")
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with xr.open_dataset(source) as ds:
        times = pd.to_datetime(ds.time.values)
        keep = np.flatnonzero(times.year == year)
        if not keep.size:
            raise ValueError(f"No imagery for {year}")
        subset = ds.isel(time=keep).sortby("time")
        if not pd.Index(subset.time.values).is_unique:
            raise ValueError("Duplicate acquisition dates")
        ndvi = subset.ndvi.transpose("time", "lat", "lon").load()
        valid = np.isfinite(ndvi) & (ndvi >= -1) & (ndvi <= 1)
        invalid_count = int((np.isfinite(ndvi) & ~valid).sum())
        if "valid" in subset:
            valid = valid & subset.valid.load().astype(bool)
        ndvi = ndvi.where(valid).astype("float32")
        if not bool(np.isfinite(ndvi).any()):
            raise ValueError("No valid NDVI after masking")
        out = xr.Dataset({"ndvi": ndvi}, attrs={
            "title": f"Sentinel-2 nominal 10 m NDVI, {year}",
            "source": source.name,
            "source_path": str(source),
            "source_sha256": digest,
            "method": ds.attrs.get("method", ""),
            "quality_control": "Source valid mask retained; nonfinite and NDVI outside [-1, 1] masked, not clipped",
            "invalid_ndvi_pixels_masked": invalid_count,
            "source_resolution_m": 10,
            "crs": "EPSG:4326",
        })
        out.ndvi.attrs.update(units="1", long_name="Cloud-masked normalized difference vegetation index")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".partial.nc")
        out.to_netcdf(temporary, encoding={"ndvi": {"zlib": True, "complevel": 4}})
        with xr.open_dataset(temporary) as check:
            assert check.sizes == out.sizes
            assert bool(((check.ndvi >= -1) & (check.ndvi <= 1) | check.ndvi.isnull()).all())
        temporary.replace(destination)
        print(f"Imported {len(keep)} dates to {destination}; masked {invalid_count:,} out-of-range values", flush=True)
        return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    import_ndvi(args.source, args.year,
                Path(__file__).resolve().parents[1] / "data" / f"paddy_et_s2_{args.year}.nc")
