"""Copernicus Data Space Ecosystem (CDSE) access for the Imbanuma district.

After ``pip install cyclic-irrigation[copernicus]`` (or this repo editable)::

    python -m cyclic copernicus register     # save CDSE username/password locally
    python -m cyclic copernicus status
    python -m cyclic copernicus search 2022
    python -m cyclic copernicus download 2022 --max-cloud 40

Or from Python::

    import cyclic
    cyclic.copernicus_register()             # interactive
    cyclic.copernicus_download(2022)

Credentials are stored under ``~/.cyclic/copernicus.json`` (or ``CYCLIC_CONFIG_DIR``).
You must create a free account at https://dataspace.copernicus.eu/ first; this
module only registers *those* credentials with the local package.

AOI defaults to the Shiroyama–Jinbee / Imbanuma cyclic paddies bbox (Okura 2025).
"""

from __future__ import annotations

import getpass
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

# Imbanuma / Shiroyama–Jinbee paddies (lon/lat WGS84), padded ~0.01°.
IMBANUMA_BBOX = (140.228, 35.750, 140.303, 35.877)  # west, south, east, north

CDSE_SIGNUP = "https://dataspace.copernicus.eu/"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CDSE_CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1"
CDSE_DOWNLOAD = "https://zipper.dataspace.copernicus.eu/odata/v1"


def config_dir():
    root = os.environ.get("CYCLIC_CONFIG_DIR")
    if root:
        return Path(root)
    return Path.home() / ".cyclic"


def credentials_path():
    return config_dir() / "copernicus.json"


def district_bbox(pad_deg=0.01):
    """Bounding box of cyclic paddies, optionally padded (degrees)."""
    try:
        from . import data
        frame = data.paddies()
        west = float(frame["lon"].min()) - pad_deg
        east = float(frame["lon"].max()) + pad_deg
        south = float(frame["lat"].min()) - pad_deg
        north = float(frame["lat"].max()) + pad_deg
        return (west, south, east, north)
    except Exception:
        w, s, e, n = IMBANUMA_BBOX
        return (w - pad_deg + 0.01, s - pad_deg + 0.01, e + pad_deg - 0.01, n + pad_deg - 0.01)


def bbox_polygon_wkt(bbox=None):
    west, south, east, north = bbox or district_bbox()
    return (
        f"POLYGON(({west} {south},{east} {south},{east} {north},"
        f"{west} {north},{west} {south}))"
    )


def load_credentials():
    """Return dict with username/password from file or env, else None."""
    user = os.environ.get("CDSE_USERNAME") or os.environ.get("COPERNICUS_USERNAME")
    password = os.environ.get("CDSE_PASSWORD") or os.environ.get("COPERNICUS_PASSWORD")
    path = credentials_path()
    if path.is_file():
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        user = user or data.get("username")
        password = password or data.get("password")
    if not user or not password:
        return None
    return {"username": user, "password": password, "path": str(path)}


def save_credentials(username, password, path=None):
    path = Path(path) if path else credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "username": username,
        "password": password,
        "signup": CDSE_SIGNUP,
        "note": "CDSE account credentials for cyclic Copernicus downloads",
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def clear_credentials():
    path = credentials_path()
    if path.is_file():
        path.unlink()
        return True
    return False


def register_interactive(username=None, password=None, *, test_login=True):
    """Prompt for CDSE credentials, save locally, optionally verify token."""
    print("Create a free Copernicus Data Space account (if you do not have one):")
    print(f"  {CDSE_SIGNUP}")
    print()
    print("Then enter those credentials to register them with this package.")
    print(f"They will be stored at: {credentials_path()}")
    print("(Or set CDSE_USERNAME / CDSE_PASSWORD in the environment.)")
    print()
    if username is None:
        username = input("CDSE username (email): ").strip()
    if password is None:
        password = getpass.getpass("CDSE password: ")
    if not username or not password:
        raise ValueError("username and password are required")
    path = save_credentials(username, password)
    result = SimpleNamespace(path=str(path), username=username, ok=True, token=None, error=None)
    if test_login:
        try:
            result.token = access_token(username=username, password=password)
            print("Login OK — access token received.")
        except Exception as exc:
            result.ok = False
            result.error = str(exc)
            print(f"Saved credentials, but login test failed: {exc}", file=sys.stderr)
    else:
        print("Saved.", path)
    return result


def access_token(username=None, password=None):
    """OAuth2 password grant against CDSE (client_id=cdse-public)."""
    import requests

    creds = None
    if username is None or password is None:
        creds = load_credentials()
        if creds is None:
            raise RuntimeError(
                "No Copernicus credentials. Run: python -m cyclic copernicus register\n"
                f"Or set CDSE_USERNAME / CDSE_PASSWORD. Sign up at {CDSE_SIGNUP}"
            )
        username = username or creds["username"]
        password = password or creds["password"]
    response = requests.post(
        CDSE_TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": username,
            "password": password,
        },
        timeout=60,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"CDSE auth failed ({response.status_code}): {response.text[:300]}"
        )
    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("CDSE auth response missing access_token")
    return token


def status():
    """Credential / login status for CLI."""
    creds = load_credentials()
    info = SimpleNamespace(
        signup=CDSE_SIGNUP,
        config_path=str(credentials_path()),
        configured=creds is not None,
        username=None if creds is None else creds["username"],
        login_ok=False,
        error=None,
        bbox=district_bbox(),
    )
    if creds is None:
        info.error = "not configured"
        return info
    try:
        access_token()
        info.login_ok = True
    except Exception as exc:
        info.error = str(exc)
    return info


def _season_window(year):
    """Okura irrigation season window used elsewhere in the package."""
    from . import okura
    dates = okura.season_dates(int(year))
    start = dates[0].strftime("%Y-%m-%dT00:00:00.000Z")
    end = dates[-1].strftime("%Y-%m-%dT23:59:59.999Z")
    return start, end, dates[0], dates[-1]


def search_sentinel2(year=None, start=None, end=None, bbox=None,
                     product_type="S2MSI2A", max_cloud=80.0, top=50):
    """Catalogue search for Sentinel-2 products over the district AOI."""
    import requests

    if start is None or end is None:
        if year is None:
            from . import okura
            year = okura.COMPARISON_YEAR
        start, end, _, _ = _season_window(year)
    bbox = bbox or district_bbox()
    poly = bbox_polygon_wkt(bbox)
    filt = (
        f"Collection/Name eq 'SENTINEL-2' and "
        f"Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{product_type}') and "
        f"Attributes/OData.CSC.DoubleAttribute/any("
        f"att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {float(max_cloud)}) and "
        f"ContentDate/Start gt {start} and ContentDate/Start lt {end} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{poly}')"
    )
    url = (
        f"{CDSE_CATALOGUE}/Products"
        f"?$filter={quote(filt)}"
        f"&$orderby=ContentDate/Start asc"
        f"&$top={int(top)}"
    )
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    rows = response.json().get("value", [])
    products = []
    for row in rows:
        cloud = None
        for att in row.get("Attributes") or []:
            if att.get("Name") == "cloudCover":
                cloud = att.get("Value")
        products.append(SimpleNamespace(
            id=row.get("Id"),
            name=row.get("Name"),
            size=row.get("ContentLength"),
            start=(row.get("ContentDate") or {}).get("Start"),
            end=(row.get("ContentDate") or {}).get("End"),
            cloud_cover=cloud,
            online=row.get("Online", True),
            raw=row,
        ))
    return SimpleNamespace(
        year=year, start=start, end=end, bbox=bbox, product_type=product_type,
        max_cloud=float(max_cloud), n=len(products), products=products, url=url,
    )


def download_dir(year=None):
    from . import data
    root = Path(data.DATA) / "copernicus"
    if year is not None:
        root = root / str(int(year))
    root.mkdir(parents=True, exist_ok=True)
    return root


def download_product(product_id, dest=None, token=None, chunk=1024 * 1024):
    """Download one SAFE product zip by catalogue Id."""
    import requests

    token = token or access_token()
    dest = Path(dest) if dest else download_dir()
    dest.mkdir(parents=True, exist_ok=True)
    url = f"{CDSE_DOWNLOAD}/Products({product_id})/$value"
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    with session.get(url, stream=True, allow_redirects=True, timeout=600) as response:
        response.raise_for_status()
        name = None
        disposition = response.headers.get("Content-Disposition") or ""
        if "filename=" in disposition:
            name = disposition.split("filename=")[-1].strip().strip('"')
        if not name:
            name = f"{product_id}.zip"
        path = dest / name
        with open(path, "wb") as handle:
            for block in response.iter_content(chunk_size=chunk):
                if block:
                    handle.write(block)
    return path


def download_season(year, max_cloud=40.0, top=30, product_type="S2MSI2A",
                    skip_existing=True, progress=None):
    """Search + download Sentinel-2 scenes for the Okura season over Imbanuma."""
    catalogue = search_sentinel2(
        year=year, product_type=product_type, max_cloud=max_cloud, top=top)
    out_dir = download_dir(year)
    token = access_token()
    saved = []
    skipped = []
    errors = []
    for i, product in enumerate(catalogue.products):
        if progress:
            progress(f"[{i + 1}/{catalogue.n}] {product.name} cloud={product.cloud_cover}")
        target = out_dir / f"{product.name}.zip"
        if skip_existing and target.is_file() and target.stat().st_size > 0:
            skipped.append(target)
            continue
        try:
            path = download_product(product.id, dest=out_dir, token=token)
            saved.append(path)
        except Exception as exc:
            errors.append(SimpleNamespace(product=product.name, error=str(exc)))
    return SimpleNamespace(
        year=int(year), catalogue=catalogue, directory=str(out_dir),
        saved=saved, skipped=skipped, errors=errors,
        n_saved=len(saved), n_skipped=len(skipped), n_errors=len(errors),
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "copernicus":
        argv = argv[1:]
    cmd = argv[0] if argv else "status"

    if cmd in ("register", "login", "configure", "setup"):
        user = None
        password = None
        if "--username" in argv:
            user = argv[argv.index("--username") + 1]
        if "--password" in argv:
            password = argv[argv.index("--password") + 1]
        result = register_interactive(user, password)
        return 0 if result.ok else 1

    if cmd in ("logout", "clear"):
        cleared = clear_credentials()
        print("cleared" if cleared else "nothing to clear")
        return 0

    if cmd == "status":
        info = status()
        print("signup     ", info.signup)
        print("config     ", info.config_path)
        print("configured ", info.configured, info.username or "")
        print("login_ok   ", info.login_ok)
        print("bbox       ", info.bbox)
        if info.error and not info.login_ok:
            print("error      ", info.error)
        return 0 if (not info.configured) or info.login_ok else 1

    if cmd == "search":
        year = int(argv[1]) if len(argv) > 1 else 2022
        cloud = 80.0
        if "--max-cloud" in argv:
            cloud = float(argv[argv.index("--max-cloud") + 1])
        result = search_sentinel2(year=year, max_cloud=cloud)
        print(f"Sentinel-2 {result.product_type} over Imbanuma, season {year}")
        print(f"window {result.start} .. {result.end}  cloud<={result.max_cloud}  n={result.n}")
        print(f"bbox {result.bbox}")
        for product in result.products:
            print(f"  {product.start[:10]}  cloud={product.cloud_cover}  {product.name}")
        return 0

    if cmd == "download":
        year = int(argv[1]) if len(argv) > 1 else 2022
        cloud = 40.0
        top = 30
        if "--max-cloud" in argv:
            cloud = float(argv[argv.index("--max-cloud") + 1])
        if "--top" in argv:
            top = int(argv[argv.index("--top") + 1])
        result = download_season(
            year, max_cloud=cloud, top=top, progress=lambda msg: print(msg))
        print(f"saved {result.n_saved}  skipped {result.n_skipped}  errors {result.n_errors}")
        print("directory", result.directory)
        for err in result.errors:
            print("error", err.product, err.error, file=sys.stderr)
        return 0 if result.n_errors == 0 else 1

    print(__doc__)
    print("commands: register | status | search [year] | download [year] | clear")
    return 2


# Friendly aliases used from cyclic.__init__
copernicus_register = register_interactive
copernicus_status = status
copernicus_search = search_sentinel2
copernicus_download = download_season
