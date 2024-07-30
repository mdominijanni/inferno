from PIL import Image
from sage.all import plot_slope_field, line, text, exp

# Global Setting
dispres = 300
saveres = 300

# QIF Slope Field
vrest = -60
vcrit = -50
vmin = vrest - 5
vmax = vcrit + 5
labelv = 0.625
labelh = 0.03

# light mode
fn = "qif-slope-field-light.png"

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
P.save(fn, dpi=saveres)

# dark mode
fn = "qif-slope-field-dark.png"

qif_slopefield = plot_slope_field(
    lambda t, v: (v - vrest) * (v - vcrit),
    (0, 1),
    (vmin, vmax),
    color="white",
    transparent=True,
)

vrest_line = line([(0, vrest), (1, vrest)], color="blue")
vrest_label = text(
    r"$V_R$",
    (1 - labelh, vrest - labelv),
    color="blue",
    vertical_alignment="top",
    fontsize=18,
    background_color="black",
)

vcrit_line = line([(0, vcrit), (1, vcrit)], color="red")
vcrit_label = text(
    r"$V_C$",
    (1 - labelh, vcrit + labelv),
    color="red",
    vertical_alignment="bottom",
    fontsize=18,
    background_color="black",
)

P = qif_slopefield + vrest_line + vrest_label + vcrit_line + vcrit_label
P.axes_color("white")
P.axes_label_color("white")
P.tick_label_color("white")

P.save(fn, dpi=saveres)
P = Image.open(fn)
Image.alpha_composite(Image.new("RGB", P.size, "black").convert("RGBA"), P).save(fn)

# LIF Slope Field
vrest = -60
vmin = vrest - 15
vmax = vrest + 15
hmin = 0
hmax = 2
labelh = 0.03 * (hmax - hmin)
labelv = 0.0

# light mode
fn = "lif-slope-field-light.png"

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
P.save(fn, dpi=saveres)

# dark mode
fn = "lif-slope-field-dark.png"

lif_slopefield = plot_slope_field(
    lambda t, v: -(v - vrest),
    (hmin, hmax),
    (vmin, vmax),
    color="white",
    transparent=True,
)

vrest_line = line([(hmin, vrest), (hmax, vrest)], color="blue")
vrest_label = text(
    r"$V_R$",
    (hmax - labelh, vrest - labelv),
    color="blue",
    vertical_alignment="center",
    fontsize=18,
    background_color="black",
)

P = lif_slopefield + vrest_line + vrest_label
P.axes_color("white")
P.axes_label_color("white")
P.tick_label_color("white")

P.save(fn, dpi=saveres)
P = Image.open(fn)
Image.alpha_composite(Image.new("RGB", P.size, "black").convert("RGBA"), P).save(fn)

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

# light mode
fn1 = "eif-slope-field-d1-light.png"
fn2 = "eif-slope-field-d2-light.png"

eif_slopefield_d1 = plot_slope_field(
    lambda t, v: -(v - vrest) + D1 * exp((v - vt) / D1),
    (hmin, hmax),
    (vmin, vmax),
    color="black",
)

eif_slopefield_d2 = plot_slope_field(
    lambda t, v: -(v - vrest) + D2 * exp((v - vt) / D2),
    (hmin, hmax),
    (vmin, vmax),
    color="black",
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

P1.save(fn1, dpi=saveres)
P2.save(fn2, dpi=saveres)

# dark mode
fn1 = "eif-slope-field-d1-dark.png"
fn2 = "eif-slope-field-d2-dark.png"

eif_slopefield_d1 = plot_slope_field(
    lambda t, v: -(v - vrest) + D1 * exp((v - vt) / D1),
    (hmin, hmax),
    (vmin, vmax),
    color="white",
    transparent=True,
)

eif_slopefield_d2 = plot_slope_field(
    lambda t, v: -(v - vrest) + D2 * exp((v - vt) / D2),
    (hmin, hmax),
    (vmin, vmax),
    color="white",
    transparent=True,
)

vrest_line = line([(hmin, vrest), (hmax, vrest)], color="blue")
vrest_label = text(
    r"$V_R$",
    (hmax - labelh, vrest - labelv),
    color="blue",
    vertical_alignment="top",
    fontsize=18,
    background_color="black",
)

vt_line = line([(hmin, vt), (hmax, vt)], color="red")
vt_label = text(
    r"$V_T$",
    (hmax - labelh, vt + labelv),
    color="red",
    vertical_alignment="bottom",
    fontsize=18,
    background_color="black",
)

P1 = eif_slopefield_d1 + vrest_line + vrest_label + vt_line + vt_label
P1.axes_color("white")
P1.axes_label_color("white")
P1.tick_label_color("white")

P2 = eif_slopefield_d2 + vrest_line + vrest_label + vt_line + vt_label
P2.axes_color("white")
P2.axes_label_color("white")
P2.tick_label_color("white")

P1.save(fn1, dpi=saveres)
P1 = Image.open(fn1)
Image.alpha_composite(Image.new("RGB", P1.size, "black").convert("RGBA"), P1).save(fn1)

P2.save(fn2, dpi=saveres)
P2 = Image.open(fn2)
Image.alpha_composite(Image.new("RGB", P2.size, "black").convert("RGBA"), P2).save(fn2)
