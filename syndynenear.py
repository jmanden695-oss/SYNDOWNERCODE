from random import setstate
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, ICRS
from sbpy.dynamics.syndynes import SynGenerator
from sbpy.dynamics.state import State
from astroquery.jplhorizons import Horizons



# -------------------------------
# 1) Reference epoch
# -------------------------------
epoch = Time("2026-01-14 00:00:00", scale="utc")  # arbitrary reference # yyyy-mm-dd



# -------------------------------
# 2) Get comet state from Horizons
# -------------------------------
comet = Horizons(
    id="C/2025 N1",
    id_type="smallbody",
    location="@0",  # heliocentric
    epochs=epoch.utc.jd
)

vec = comet.vectors()[0]  # take first row
r = np.array([vec['x'], vec['y'], vec['z']]) * u.au
v = np.array([vec['vx'], vec['vy'], vec['vz']]) * (u.au / u.day)

state = State(r=r, v=v, t=epoch)

# -------------------------------
# 3) Set dust parameters
# -------------------------------
#betas = np.array([0.001, 0.01, 0.1, 1])
betas = np.logspace(-4, -1, 10)
ages = np.linspace(0, 70, 20) * u.day

# -------------------------------
# 4) Build SynGenerator
# -------------------------------
syn_gen = SynGenerator(source=state, betas=betas, ages=ages)
synchrones = syn_gen.synchrones()
syndynes = syn_gen.syndynes()

print(len(synchrones), "synchrones")
print(len(syndynes), "syndynes")

# -------------------------------
# 5) Get Earth geocentric position
# -------------------------------
earth = Horizons(id='399', location='@0', epochs=epoch.utc.jd)
earth_vec = earth.vectors()[0]
r_earth = np.array([earth_vec['x'], earth_vec['y'], earth_vec['z']]) * u.au

# -------------------------------
# 6) Helper: convert State -> RA/Dec as seen from Earth
# -------------------------------
def to_geocentric_radec(state, r_earth):
    r_geo = state.r - r_earth  # vector from Earth
    sc = SkyCoord(x=r_geo[0], y=r_geo[1], z=r_geo[2],
                  representation_type='cartesian', frame=ICRS)
    sc_sph = sc.represent_as('spherical')
    ra = sc_sph.lon.to(u.deg).value
    dec = sc_sph.lat.to(u.deg).value
    return ra, dec



# -------------------------------
# 7) Plotting
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 8))

ax.set_aspect('equal', adjustable='datalim')







# -------------------------------
# Plot synchrones (solid)
# -------------------------------
for i, syn in enumerate(synchrones):
    ra_list = []
    dec_list = []

    for s in syn:   # IMPORTANT: iterate over all particles
        ra, dec = to_geocentric_radec(s, r_earth)
        ra_list.append(ra)
        dec_list.append(dec)

    ax.plot(ra_list, dec_list, label=f"age {ages[i].value:.0f} d")

# -------------------------------
# Plot syndynes (dashed)
# -------------------------------
for j, syn in enumerate(syndynes):
    ra_list = []
    dec_list = []

    for s in syn:
        ra, dec = to_geocentric_radec(s, r_earth)
        ra_list.append(ra)
        dec_list.append(dec)

    ax.plot(ra_list, dec_list, '--', label=f"β={betas[j]:.2e}")

ax.set_xlabel("RA (deg)")
ax.set_ylabel("Dec (deg)")
ax.set_title("Syndynes & Synchrones for C/2025 N1 (Geocentric)")
  # RA increases to the left
ax.legend(fontsize=8, loc="best")



# ------------------------------

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.jplhorizons import Horizons

# --- 1) Geocentric vectors (Earth → comet, Earth → Sun) ---

r_comet_geo = state.r - r_earth  # Earth → comet

sun_tab = Horizons(
    id='10',          # Sun
    location='@399',  # Earth
    epochs=epoch.jd
).vectors()

r_sun_geo = np.array([
    sun_tab['x'][0],
    sun_tab['y'][0],
    sun_tab['z'][0]
]) * u.au

# --- 2) Sky coordinates of comet ---

comet_coord = SkyCoord(
    x=r_comet_geo[0],
    y=r_comet_geo[1],
    z=r_comet_geo[2],
    representation_type='cartesian',
    frame='icrs'
)

comet_ra  = comet_coord.spherical.lon.deg
comet_dec = comet_coord.spherical.lat.deg

# --- 3) Direction toward Sun ---

sun_dir = r_sun_geo - r_comet_geo
sun_dir /= np.linalg.norm(sun_dir)

# Construct a temporary Sun point for SkyCoord
eps = 1e-3 * u.au
sun_point = r_comet_geo + eps * sun_dir

sun_point_coord = SkyCoord(
    x=sun_point[0],
    y=sun_point[1],
    z=sun_point[2],
    representation_type='cartesian',
    frame='icrs'
)

sun_ra  = sun_point_coord.spherical.lon.deg
sun_dec = sun_point_coord.spherical.lat.deg

# --- 4) Compute normalized small-angle offsets ---

dra  = (sun_ra - comet_ra) * np.cos(np.deg2rad(comet_dec))
ddec = sun_dec - comet_dec



# Normalize direction vector
norm = np.hypot(dra, ddec)
dra /= norm
ddec /= norm

# --- 5) Compute arrow length based on axis span ---

ra_span = ax.get_xlim()[1] - ax.get_xlim()[0]
dec_span = ax.get_ylim()[1] - ax.get_ylim()[0]

# Arrow is 20% of the smaller axis span
L = 0.1 * min(abs(ra_span), abs(dec_span))

x0, y0 = comet_ra, comet_dec
x1, y1 = x0 + L * dra, y0 + L * ddec

# --- 6) Draw the arrow with annotate (guaranteed visible) ---



ax.annotate(
    '',
    xy=(x1, y1),
    xytext=(x0, y0),
    arrowprops=dict(arrowstyle='-|>', color='gold', lw=2),
    zorder=200
)

# Mark comet
ax.scatter(x0, y0, color='red', s=10, zorder=210)

print("Sun position angle (deg):", np.degrees(np.arctan2(dra, ddec)))
print("RA span:", ax.get_xlim())
print("Dec span:", ax.get_ylim())



from astropy.coordinates import SkyCoord
from astropy import units as u


from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy import units as u
import numpy as np

# Earth velocity (from Horizons Row)
v_earth = np.array([
    earth_vec['vx'],
    earth_vec['vy'],
    earth_vec['vz']
]) * (u.au / u.day)

# Geocentric velocity
v_comet_geo = state.v - v_earth

# Small step along velocity
eps_t = 0.01 * u.day
r_future = r_comet_geo + v_comet_geo * eps_t
future_coord = SkyCoord(
    x=r_future[0],
    y=r_future[1],
    z=r_future[2],
    representation_type='cartesian',
    frame='icrs'
)

comet_coord = SkyCoord(
    x=r_comet_geo[0],
    y=r_comet_geo[1],
    z=r_comet_geo[2],
    representation_type='cartesian',
    frame='icrs'
)

# Tangent plane
offset_frame = SkyOffsetFrame(origin=comet_coord)
vel_offset = future_coord.transform_to(offset_frame)

dx = vel_offset.lon.deg
dy = vel_offset.lat.deg

# Normalize
norm = np.hypot(dx, dy)
dx /= norm
dy /= norm

# Comet sky position
comet = SkyCoord(ra=comet_ra*u.deg, dec=comet_dec*u.deg, frame='icrs')

# Plot arrow
L = (30 * u.arcsec).to(u.deg).value

dra  = dx * L / np.cos(np.deg2rad(comet.dec.deg))
ddec = dy * L

vel_pa = np.degrees(np.arctan2(dra, ddec)) % 360
print("Velocity vector PA (deg):", vel_pa)


ax.annotate(
    '',
    xy=(comet.ra.deg + dra, comet.dec.deg + ddec),
    xytext=(comet.ra.deg, comet.dec.deg),
    xycoords='data',
    textcoords='data',
    arrowprops=dict(
        arrowstyle='-|>',
        color='cyan',
        lw=2
    ),
    zorder=300
)

from astropy.coordinates import SkyOffsetFrame

from astropy.coordinates import SkyCoord, SkyOffsetFrame
import astropy.units as u
import numpy as np

comet_sc = SkyCoord(ra=comet_ra*u.deg, dec=comet_dec*u.deg, frame='icrs')
offset_frame = SkyOffsetFrame(origin=comet_sc)

def offsets_arcsec(ra, dec):
    sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    off = sc.transform_to(offset_frame)
    dx = off.lon.to(u.arcsec).value   # East
    dy = off.lat.to(u.arcsec).value   # North
    return dx, dy

def polyline_pa_and_extent(ra_list, dec_list):
    dx, dy = [], []

    for ra, dec in zip(ra_list, dec_list):
        x, y = offsets_arcsec(ra, dec)
        dx.append(x)
        dy.append(y)

    dx = np.array(dx)
    dy = np.array(dy)

    # Vector from nucleus to furthest point
    r = np.hypot(dx, dy)
    i = np.argmax(r)

    dxm, dym = dx[i], dy[i]

    # PA: North = 0°, East = 90°
    pa = np.degrees(np.arctan2(dxm, dym)) % 360
    extent = r[i]

    return pa, extent

print("\nSynchrones:")
syn_pAs = []

for i, syn in enumerate(synchrones):
    ra_list, dec_list = [], []

    for s in syn:
        ra, dec = to_geocentric_radec(s, r_earth)
        ra_list.append(ra)
        dec_list.append(dec)

    pa, extent = polyline_pa_and_extent(ra_list, dec_list)
    syn_pAs.append(pa)

    print(f"  Age {ages[i].value:6.1f} d : "
          f"PA = {pa:7.2f} deg,  extent = {extent:7.1f} arcsec")
    
    print("\nSyndynes:")
dyn_pAs = []

for j, syn in enumerate(syndynes):
    ra_list, dec_list = [], []

    for s in syn:
        ra, dec = to_geocentric_radec(s, r_earth)
        ra_list.append(ra)
        dec_list.append(dec)

    pa, extent = polyline_pa_and_extent(ra_list, dec_list)
    dyn_pAs.append(pa)

    print(f"  β = {betas[j]:.3e} : "
          f"PA = {pa:7.2f} deg,  extent = {extent:7.1f} arcsec")

def mean_pa(pa_list):
    pa_rad = np.deg2rad(pa_list)
    x = np.sin(pa_rad)
    y = np.cos(pa_rad)
    return np.degrees(np.arctan2(x.mean(), y.mean())) % 360

print("\nMean PA:")
print("  Synchrones :", mean_pa(syn_pAs), "deg")
print("  Syndynes   :", mean_pa(dyn_pAs), "deg")

def pa_spread(pa_list):
    pa = np.deg2rad(pa_list)
    R = np.sqrt((np.sin(pa).mean())**2 + (np.cos(pa).mean())**2)
    return np.degrees(np.sqrt(-2*np.log(R)))

print("\nPA spread (deg):")
print("  Synchrones :", pa_spread(syn_pAs))
print("  Syndynes   :", pa_spread(dyn_pAs))

# -------------------------
# Print comet RA/Dec in sexagesimal format
# -------------------------


comet_sph = comet_coord.spherical

ra_str = comet_sph.lon.to_string(
    unit=u.hour,
    sep=" ",
    precision=0,
    pad=True
)

dec_str = comet_sph.lat.to_string(
    unit=u.deg,
    sep=" ",
    precision=0,
    alwayssign=True,
    pad=True
)

print(f"Comet coordinates: {ra_str} · {dec_str}")


from astropy.coordinates import position_angle
import numpy as np

# Compute position angle from comet to Sun
sun_pa = comet_coord.position_angle(sun_point_coord)

# Convert to degrees in [0, 360)
sun_pa_deg = sun_pa.to(u.deg).value % 360

print(f"Sun position angle (celestial, N=0°): {sun_pa_deg:.3f} deg")

print("Velocity vector PA (deg):", vel_pa)


ax.invert_xaxis()

plt.show()