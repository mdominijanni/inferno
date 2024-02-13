from sage.all import *

# Global Setting
saveplots = True
dispres = 300
saveres = 300

# QIF Slope Field
vrest = -60
vcrit = -50
vmin = vrest - 5
vmax = vcrit + 5
labelv = 0.625
labelh = 0.03

qif_slopefield = plot_slope_field(
    lambda t, v: (v - vrest) * (v - vcrit), (0, 1), (vmin, vmax), color="black"
)

vrest_line = line([(0, vrest), (1, vrest)], color="blue")
vrest_label = text(
    r"$V_R$",
    (1 - labelh, vrest - labelv),
    color="blue",
    vertical_alignment="top",
    fontsize=18,
    background_color="white",
)

vcrit_line = line([(0, vcrit), (1, vcrit)], color="red")
vcrit_label = text(
    r"$V_C$",
    (1 - labelh, vcrit + labelv),
    color="red",
    vertical_alignment="bottom",
    fontsize=18,
    background_color="white",
)

P = qif_slopefield + vrest_line + vrest_label + vcrit_line + vcrit_label

if saveplots:
    P.save("qif_slope_field.png", dpi=saveres)
else:
    P.show(dpi=dispres)


# LIF Slope Field
vrest = -60
vmin = vrest - 15
vmax = vrest + 15
hmin = 0
hmax = 2
labelh = 0.03 * (hmax - hmin)
labelv = 0.0

lif_slopefield = plot_slope_field(
    lambda t, v: -(v - vrest), (hmin, hmax), (vmin, vmax), color="black"
)

vrest_line = line([(hmin, vrest), (hmax, vrest)], color="blue")
vrest_label = text(
    r"$V_R$",
    (hmax - labelh, vrest - labelv),
    color="blue",
    vertical_alignment="center",
    fontsize=18,
    background_color="white",
)

P = lif_slopefield + vrest_line + vrest_label

if saveplots:
    P.save("lif_slope_field.png", dpi=saveres)
else:
    P.show(dpi=dispres)


# EIF Slope Field
vrest = -60
vt = -50
D1 = 1
D2 = 2
vmin = vrest - 7
vmax = vt + 7
hmin = 0
hmax = 1
labelh = 0.03 * (hmax - hmin)
labelv = 0.625 * ((vmax - vmin) / 20)

eif_slopefield_d1 = plot_slope_field(
    lambda t, v: -(v - vrest) + D1 * exp((v - vt) / D1), (hmin, hmax), (vmin, vmax), color="black"
)

eif_slopefield_d2 = plot_slope_field(
    lambda t, v: -(v - vrest) + D2 * exp((v - vt) / D2), (hmin, hmax), (vmin, vmax), color="black"
)

vrest_line = line([(hmin, vrest), (hmax, vrest)], color="blue")
vrest_label = text(
    r"$V_R$",
    (hmax - labelh, vrest - labelv),
    color="blue",
    vertical_alignment="top",
    fontsize=18,
    background_color="white",
)

vt_line = line([(hmin, vt), (hmax, vt)], color="red")
vt_label = text(
    r"$V_T$",
    (hmax - labelh, vt + labelv),
    color="red",
    vertical_alignment="bottom",
    fontsize=18,
    background_color="white",
)

P1 = eif_slopefield_d1 + vrest_line + vrest_label + vt_line + vt_label
P2 = eif_slopefield_d2 + vrest_line + vrest_label + vt_line + vt_label

if saveplots:
    P1.save("eif_slope_field_d1.png", dpi=saveres)
    P2.save("eif_slope_field_d2.png", dpi=saveres)
else:
    P1.show(dpi=dispres)
    P2.show(dpi=dispres)
