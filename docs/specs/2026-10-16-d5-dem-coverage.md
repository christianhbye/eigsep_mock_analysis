# D5 terrain coverage extension

Diagnostic follow-up to the nominal convergence cost probe. The 87.5 m and
91 m traces have far-endpoint maxima in southern directions; the old matrix
cannot certify these cells under its unchanged edge-exclusion rule.

Build private candidates extending DEM v0003 by 2 km and 4 km south. Download
16 tiles from the same USGS UT_WestEast_B22 source project, check their TIFF
georeferencing, resolution, raster convention, finite values and nodata, and
retain their URLs/hashes. Preserve all v0003 samples bit for bit, the fixed UTM
origin, height datum, survey offset and East pixel offset. Increase the North
pixel offset by exactly the added number of rows. Do not publish or overwrite
a team product. The XML geographic bounding box remains the legacy frame
metadata; the array/pixel offsets and GeoTIFF coordinates define the extent.

The bounded coverage probe compares both candidates at nominal and +/-3 mm
along E, N and U, at all three heights and 0.1-degree bearing spacing. Retain
all raw traces and flags. Compare the horizon, frozen-response temperatures
and one-sided slope changes between extents. Reject a claim that coverage is
resolved if any sampled trace remains edge limited or the coverage-induced
changes exceed the existing forward/slope criteria. Passing this limited probe
supports launching the numerical matrix; it does not establish global terrain
coverage or physical derivative validity. No stress-position or finer-bearing
coverage claim follows without those calculations.

The full convergence plan gains an optional `dem_sha256`. Without it, the old
serialized plan and input behavior remain unchanged. With it, only `dem.npz`
may differ from the pinned pilot inputs. The sky/beam response retains its
original pilot identity because it is formed before applying the terrain mask.
New terrain changes all trace and dependent evaluation IDs. Coordinates,
criteria and the previously selected diagnostic stress position stay fixed;
the latter is not an independently selected worst point of the expanded DEM.

Use a new input directory, output directory, inventory and frozen resource gate.
Never relabel/reuse old-DEM checkpoints as expanded-DEM results. Native tests,
small coverage/temperature checks and measured costs precede a full-run launch.
CHB explicitly requested notice before that launch; present scope, results,
limits and resource budget and await go-ahead. The old nine-hour projection
is not a budget measurement for the expanded terrain.


The first nominal probe retained 12/14 flagged bearings at +2 km and zero at
+4 km (87.5/91 m). Add an +8 km control before selecting a full-run DEM. The
builder accepts explicit positive integer extensions and can copy earlier
source tiles only after their recorded URL/hash matches. The +2/+4 km probe
and its original source remain immutable. The +4/+8 km comparison uses the
same seven positions and all original criteria, without dropping directions.
