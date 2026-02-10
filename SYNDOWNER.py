
import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, SkyOffsetFrame, ICRS

from sbpy.dynamics.syndynes import SynGenerator
from sbpy.dynamics.state import State
from astroquery.jplhorizons import Horizons


# ============================================================
# 1) Epoch
# ============================================================

epoch = Time("2025-12-30 04:47:00", scale="utc")


# ============================================================
# 2) Comet heliocentric state (Horizons)
# ============================================================

comet_h = Horizons(
    id="C/2022 QE78",
    id_type="smallbody",
    location="@0",
    epochs=epoch.utc.jd
)

vec = comet_h.vectors()[0]

r = np.array([vec["x"], vec["y"], vec["z"]]) * u.au
v = np.array([vec["vx"], vec["vy"], vec["vz"]]) * (u.au / u.day)

state = State(r=r, v=v, t=epoch)

comet_name = comet_h.id
epoch_str = epoch.utc.strftime("%Y-%m-%d %H:%M UTC")

# ============================================================
# 3) Dust parameters
# ============================================================

#(-5, -1.5, 15)
betas = np.logspace(-5, -1, 10)
ages  = np.linspace(0, 120, 15) * u.day

# ============================================================
# PA zero-point correction (degrees)
# Positive = rotate CCW (Eastward) on the sky
# ============================================================

PA_OFFSET = +18.66401

def apply_pa_offset(pa_deg):
    """
    Apply global PA zero-point correction.
    """
    return (pa_deg + PA_OFFSET) % 360
PLOT_OFFSET = -PA_OFFSET 

# ============================================================
# 4) Syndynes & synchrones
# ============================================================

syn_gen = SynGenerator(source=state, betas=betas, ages=ages)
synchrones = syn_gen.synchrones()
syndynes   = syn_gen.syndynes()

def pa_spread(pa_list):
    """
    Circular standard deviation of PAs (degrees).
    """
    pa = np.deg2rad(pa_list)
    R = np.sqrt((np.sin(pa).mean())**2 + (np.cos(pa).mean())**2)

    # Guard against numerical issues
    R = np.clip(R, 1e-10, 1.0)

    return np.degrees(np.sqrt(-2 * np.log(R)))

def mean_pa(pa_list):
    """
    Circular mean of position angles (degrees).
    PA convention: North=0°, East=90°
    """
    pa = np.deg2rad(pa_list)
    x = np.sin(pa)
    y = np.cos(pa)
    return np.degrees(np.arctan2(x.mean(), y.mean())) % 360


# ============================================================
# 5) Earth geocentric state
# ============================================================

earth = Horizons(id="399", location="@0", epochs=epoch.utc.jd)
earth_vec = earth.vectors()[0]

r_earth = np.array([
    earth_vec["x"],
    earth_vec["y"],
    earth_vec["z"]
]) * u.au

v_earth = np.array([
    earth_vec["vx"],
    earth_vec["vy"],
    earth_vec["vz"]
]) * (u.au / u.day)


# ============================================================
# 6) Comet geocentric SkyCoord + tangent plane
# ============================================================

r_comet_geo = state.r - r_earth

comet_sc = SkyCoord(
    x=r_comet_geo[0],
    y=r_comet_geo[1],
    z=r_comet_geo[2],
    representation_type="cartesian",
    frame=ICRS
)

offset_frame = SkyOffsetFrame(origin=comet_sc)


# ============================================================
# 7) Helper: PA from any SkyCoord
# ============================================================

def pa_from_coord(sc):
    off = sc.transform_to(offset_frame)
    dx = off.lon.to(u.arcsec).value   # East
    dy = off.lat.to(u.arcsec).value   # North
    return np.degrees(np.arctan2(dx, dy)) % 360, dx, dy


# ============================================================
# 8) Sun PA (apparent, of-date)
# ============================================================

sun = Horizons(id="10", location="@399", epochs=epoch.jd).vectors()
r_sun_geo = np.array([sun["x"][0], sun["y"][0], sun["z"][0]]) * u.au

sun_sc = SkyCoord(
    x=r_sun_geo[0],
    y=r_sun_geo[1],
    z=r_sun_geo[2],
    representation_type="cartesian",
    frame=ICRS
)

sun_pa_raw, sun_dx, sun_dy = pa_from_coord(sun_sc)
sun_pa = apply_pa_offset(sun_pa_raw)




# ============================================================
# 9) Velocity PA (matches Horizons Sky_mot_PA)
# ============================================================

v_comet_geo = state.v - v_earth
dt = 0.01 * u.day

future_sc = SkyCoord(
    x=(r_comet_geo + v_comet_geo * dt)[0],
    y=(r_comet_geo + v_comet_geo * dt)[1],
    z=(r_comet_geo + v_comet_geo * dt)[2],
    representation_type="cartesian",
    frame=ICRS
)

vel_pa_raw, vel_dx, vel_dy = pa_from_coord(future_sc)
vel_pa = apply_pa_offset(vel_pa_raw)

# ============================================================
# 11) Plot (visual reference guide)
# ============================================================

def rotate_vector(dx, dy, angle_deg):
    """
    Rotate a vector in the tangent plane.
    dx = East, dy = North
    angle_deg > 0 rotates counterclockwise (toward East)
    """
    a = np.deg2rad(angle_deg)
    xr = dx*np.cos(a) - dy*np.sin(a)
    yr = dx*np.sin(a) + dy*np.cos(a)
    return xr, yr




# ============================================================
# 10) PA of synchrones / syndynes
# ============================================================

def polyline_pa(states):
    dx, dy = [], []

    for s in states:
        sc = SkyCoord(
            x=(s.r - r_earth)[0],
            y=(s.r - r_earth)[1],
            z=(s.r - r_earth)[2],
            representation_type="cartesian",
            frame=ICRS
        )
        off = sc.transform_to(offset_frame)
        dx.append(off.lon.arcsec)
        dy.append(off.lat.arcsec)

    dx = np.array(dx)
    dy = np.array(dy)

    i = np.argmax(np.hypot(dx, dy))
    pa_raw = np.degrees(np.arctan2(dx[i], dy[i])) % 360

    return apply_pa_offset(pa_raw)

def polyline_pa_and_length(syn, r_earth, offset_frame, pa_offset_deg=0.0):
    dx, dy = [], []

    for s in syn:
        sc = SkyCoord(
            x=(s.r - r_earth)[0],
            y=(s.r - r_earth)[1],
            z=(s.r - r_earth)[2],
            representation_type="cartesian",
            frame=ICRS
        )

        off = sc.transform_to(offset_frame)
        dx.append(off.lon.arcsec)   # East
        dy.append(off.lat.arcsec)   # North

    dx = np.array(dx)
    dy = np.array(dy)

    # Furthest particle
    r = np.hypot(dx, dy)
    i = np.argmax(r)

    dxm, dym = dx[i], dy[i]

    # 🔁 APPLY SAME ROTATION AS PLOT
    dxm, dym = rotate_vector(dxm, dym, pa_offset_deg)

    # PA: North=0°, East=90°
    pa = np.degrees(np.arctan2(dxm, dym)) % 360
    length = r[i]

    return pa, length
print("\nSynchrones:")
for i, syn in enumerate(synchrones):
    print(f"  Age {ages[i].value:5.1f} d : PA = {polyline_pa(syn):7.2f} deg")

print("\nSyndynes:")
dyn_pas = []
dyn_lengths = []

for j, syn in enumerate(syndynes):
    pa, length = polyline_pa_and_length(syn, r_earth, offset_frame)
    dyn_pas.append(pa)
    dyn_lengths.append(length)

    print(
        f"  β={betas[j]:.2e} : "
        f"PA = {polyline_pa(syn):7.2f} deg, "
        f"length = {length:8.1f} arcsec"
    )

mean_dyn_pa = mean_pa(dyn_pas)
mean_dyn_pa_FIN = apply_pa_offset(mean_dyn_pa)
spread_dyn_pa = pa_spread(dyn_pas)



print(f"Object: {comet_name}")
print(f"Epoch: {epoch_str}")
print(f"Sun PA (corrected): {sun_pa:.3f} deg")
print(f"Velocity PA (corrected): {vel_pa:.3f} deg")


print("\nSyndyne PA statistics (offset-corrected):")
print(f"  Mean PA   : {mean_dyn_pa_FIN:7.2f} deg")
print(f"  PA spread : {spread_dyn_pa:6.2f} deg")

fig, ax = plt.subplots(figsize=(8, 8))


# -------------------------------------------------
# Plot synchrones (rotated into corrected PA frame)
# -------------------------------------------------
for syn in synchrones:
    xs, ys = [], []

    for s in syn:
        sc = SkyCoord(
            x=(s.r - r_earth)[0],
            y=(s.r - r_earth)[1],
            z=(s.r - r_earth)[2],
            representation_type="cartesian",
            frame=ICRS
        )

        off = sc.transform_to(offset_frame)

        dx = off.lon.arcsec   # East
        dy = off.lat.arcsec   # North

        dxr, dyr = rotate_vector(dx, dy, PLOT_OFFSET)

        xs.append(dxr)
        ys.append(dyr)

    ax.plot(xs, ys, color="gray", lw=0.5)

for syn in syndynes:
    xs, ys = [], []

    for s in syn:
        sc = SkyCoord(
            x=(s.r - r_earth)[0],
            y=(s.r - r_earth)[1],
            z=(s.r - r_earth)[2],
            representation_type="cartesian",
            frame=ICRS
        )

        off = sc.transform_to(offset_frame)

        dx = off.lon.arcsec   # East
        dy = off.lat.arcsec   # North

        dxr, dyr = rotate_vector(dx, dy, PLOT_OFFSET)

        xs.append(dxr)
        ys.append(dyr)

    ax.plot(xs, ys, color="gray", lw=1)


# -------------------------------------------------
# Scale arrow 
# -------------------------------------------------


def scale_arrow(dx, dy, ax, frac=1):
    """
    Scale a direction vector so it is a fraction of the plot size.
    frac = fraction of the smaller axis span
    """
    dx, dy = np.array(dx, float), np.array(dy, float)

    # normalize direction
    n = np.hypot(dx, dy)
    if n == 0:
        return 0.0, 0.0
    dx /= n
    dy /= n

    xspan = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    yspan = abs(ax.get_ylim()[1] - ax.get_ylim()[0])

    L = frac * min(xspan, yspan)
    return dx * L, dy * L

# -------------------------------------------------
# Sun arrow (apply SAME rotation)
# -------------------------------------------------

sun_dx_r, sun_dy_r = rotate_vector(sun_dx, sun_dy, PLOT_OFFSET)
sun_dx_r, sun_dy_r = scale_arrow(sun_dx_r, sun_dy_r, ax, frac=0.05)

ax.arrow(
    0, 0,
    sun_dx_r, sun_dy_r,
    color="orange",
    width=0.1,
    length_includes_head=True,
    label="Sun"
)


# -------------------------------------------------
# Velocity arrow (apply SAME rotation)
# -------------------------------------------------
vel_dx_r, vel_dy_r = rotate_vector(vel_dx, vel_dy, PLOT_OFFSET)
vel_dx_r, vel_dy_r = scale_arrow(vel_dx_r, vel_dy_r, ax, frac=0.05)

ax.arrow(
    0, 0,
    vel_dx_r, vel_dy_r,
    color="cyan",
    width=0.1,
    length_includes_head=True,
    label="Velocity"
)

# -------------------------------------------------
# Comet nucleus
# -------------------------------------------------
ax.scatter(0, 0, color="red", s=30, zorder=10)

# -------------------------------------------------
# Axes & cosmetics
# -------------------------------------------------

ax.set_title(
    f"Syndynes & Synchrones — {comet_name}\nEpoch: {epoch_str}\nPA correction: {PA_OFFSET}",
    fontsize=12
)

beta_min = betas.min()
beta_max = betas.max()

age_min = ages.min().value
age_max = ages.max().value

dust_label = (
    "Dust parameters\n"
    f"β : {beta_min:.1e} → {beta_max:.1e} ({len(betas)})\n"
    f"Age : {age_min:.0f} → {age_max:.0f} d ({len(ages)})"
)


ax.text(
    0.02, 0.98,
    dust_label,
    transform=ax.transAxes,
    fontsize=9,
    va="top",
    ha="left",
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor="gray",
        alpha=0.8
    )
)


ax.set_xlabel("East (arcsec)")
ax.set_ylabel("North (arcsec)")
ax.legend()
ax.invert_xaxis()  # astronomical convention


ax.set_aspect('equal', adjustable='datalim')

plt.show()



