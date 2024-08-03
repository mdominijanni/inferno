import os
from PIL import Image, ImageDraw
from sage.all import line, plot, exp, text, var, e

# Global Setting
dispres = 300
saveres = 300

# STDP Hyperparams Viz
dt_min = -80
dt_max = 80
tc_pre = 20
tc_post = 30
eta_pos = 1.0
eta_neg = -0.5

# light mode
fn = "stdp-hyperparam-light.png"

x = var("x")
stdp_plot = plot(
    eta_pos * exp(-x / tc_pre), (x, 0, dt_max), ticks=[[-80, 80], []], color="black"
) + plot(
    eta_neg * exp(x / tc_post), (x, dt_min, 0), ticks=[[-80, 80], []], color="black"
)

eta_pos_line = line([(0, eta_pos), (-20, eta_pos)], color="blue")
eta_pos_label = text(
    r"$A_\text{post}$",
    (-20, eta_pos),
    color="blue",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)

eta_neg_line = line([(0, eta_neg), (20, eta_neg)], color="red")
eta_neg_label = text(
    r"$A_\text{pre}$",
    (20, eta_neg),
    color="red",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)

tc_pos_line = line([(tc_pre, 0), (tc_pre, (eta_pos / e) * 1.75)], color="blue")
tc_pos_label = text(
    r"$\tau_\text{pre}$",
    (tc_pre, (eta_pos / e) * 1.75),
    color="blue",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)

tc_neg_line = line([(-tc_post, 0), (-tc_post, (eta_neg / e) * 1.75)], color="red")
tc_neg_label = text(
    r"$\tau_\text{post}$",
    (-tc_post, (eta_neg / e) * 1.75),
    color="red",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)

tc_pos_e_line = line([(tc_pre, eta_pos / e), (-20, eta_pos / e)], color="blue")
tc_pos_e_label = text(
    r"$A_\text{post} / e$",
    (-20, eta_pos / e),
    color="blue",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)


tc_neg_e_line = line([(-tc_post, eta_neg / e), (20, eta_neg / e)], color="red")
tc_neg_e_label = text(
    r"$A_\text{pre} / e$",
    (20, eta_neg / e),
    color="red",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)

ltp_label = text(
    r"$t^f_\text{post} - t^f_\text{pre} \geq 0$",
    (60, 1.0),
    color="blue",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)

ltd_label = text(
    r"$t^f_\text{post} - t^f_\text{pre} \leq 0$",
    (-60, -0.5),
    color="red",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)

P = (
    eta_pos_line
    + eta_pos_label
    + eta_neg_line
    + eta_neg_label
    + tc_pos_line
    + tc_pos_label
    + tc_neg_line
    + tc_neg_label
    + tc_pos_e_line
    + tc_pos_e_label
    + tc_neg_e_line
    + tc_neg_e_label
    + ltp_label
    + ltd_label
    + stdp_plot
)

P.save(fn, dpi=saveres)

# dark mode
fn = "stdp-hyperparam-dark.png"

x = var("x")
stdp_plot = plot(
    eta_pos * exp(-x / tc_pre),
    (x, 0, dt_max),
    ticks=[[-80, 80], []],
    color="white",
    transparent=True,
) + plot(
    eta_neg * exp(x / tc_post),
    (x, dt_min, 0),
    ticks=[[-80, 80], []],
    color="white",
    transparent=True,
)

eta_pos_line = line([(0, eta_pos), (-20, eta_pos)], color="blue")
eta_pos_label = text(
    r"$A_\text{post}$",
    (-20, eta_pos),
    color="blue",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)

eta_neg_line = line([(0, eta_neg), (20, eta_neg)], color="red")
eta_neg_label = text(
    r"$A_\text{pre}$",
    (20, eta_neg),
    color="red",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)

tc_pos_line = line([(tc_pre, 0), (tc_pre, (eta_pos / e) * 1.75)], color="blue")
tc_pos_label = text(
    r"$\tau_\text{pre}$",
    (tc_pre, (eta_pos / e) * 1.75),
    color="blue",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)

tc_neg_line = line([(-tc_post, 0), (-tc_post, (eta_neg / e) * 1.75)], color="red")
tc_neg_label = text(
    r"$\tau_\text{post}$",
    (-tc_post, (eta_neg / e) * 1.75),
    color="red",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)

tc_pos_e_line = line([(tc_pre, eta_pos / e), (-20, eta_pos / e)], color="blue")
tc_pos_e_label = text(
    r"$A_\text{post} / e$",
    (-20, eta_pos / e),
    color="blue",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)


tc_neg_e_line = line([(-tc_post, eta_neg / e), (20, eta_neg / e)], color="red")
tc_neg_e_label = text(
    r"$A_\text{pre} / e$",
    (20, eta_neg / e),
    color="red",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)

ltp_label = text(
    r"$t^f_\text{post} - t^f_\text{pre} \geq 0$",
    (60, 1.0),
    color="blue",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)

ltd_label = text(
    r"$t^f_\text{post} - t^f_\text{pre} \leq 0$",
    (-60, -0.5),
    color="red",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)

P = (
    eta_pos_line
    + eta_pos_label
    + eta_neg_line
    + eta_neg_label
    + tc_pos_line
    + tc_pos_label
    + tc_neg_line
    + tc_neg_label
    + tc_pos_e_line
    + tc_pos_e_label
    + tc_neg_e_line
    + tc_neg_e_label
    + ltp_label
    + ltd_label
    + stdp_plot
)
P.axes_color("white")
P.axes_label_color("white")
P.tick_label_color("white")

P.save(fn, dpi=saveres)
P = Image.open(fn)
Image.alpha_composite(Image.new("RGB", P.size, "black").convert("RGBA"), P).save(fn)


# STDP Modality Viz
dt_mag = 60
tc_trace = 20
eta_mag = 1.0

# temp files
fn_hebb = "stdp-mode-hebb.png"
fn_ahebb = "stdp-mode-antihebb.png"
fn_ltp = "stdp-mode-ltp.png"
fn_ltd = "stdp-mode-ltd.png"

# light mode
fn = "stdp-mode-light.png"

x = var("x")
hebb_plot = plot(
    eta_mag * exp(-x / tc_trace),
    (x, 0, dt_mag),
    ticks=[[], []],
    color="blue",
    ymin=-eta_mag,
    ymax=eta_mag,
) + plot(
    -eta_mag * exp(x / tc_trace),
    (x, -dt_mag, 0),
    ticks=[[], []],
    color="red",
    ymin=-eta_mag,
    ymax=eta_mag,
)
hebb_x_label = text(
    r"$x = t_\text{post}^f - t_\text{pre}^f$",
    (dt_mag / 2, -0.7 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
hebb_y_label = text(
    r"$y = \Delta W$",
    (dt_mag / 2, -0.9 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
hebb_label = text(
    r"$\text{Hebbian}$",
    (-dt_mag * 0.9, 0.9 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
hebb_eta_post_label = text(
    r"$A_\text{post} > 0$",
    (-dt_mag * 0.9, -0.7 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
hebb_eta_pre_label = text(
    r"$A_\text{pre} < 0$",
    (-dt_mag * 0.9, -0.9 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)

ahebb_plot = plot(
    -eta_mag * exp(-x / tc_trace),
    (x, 0, dt_mag),
    ticks=[[], []],
    color="red",
    ymin=-eta_mag,
    ymax=eta_mag,
) + plot(
    eta_mag * exp(x / tc_trace),
    (x, -dt_mag, 0),
    ticks=[[], []],
    color="blue",
    ymin=-eta_mag,
    ymax=eta_mag,
)
ahebb_x_label = text(
    r"$x = t_\text{post}^f - t_\text{pre}^f$",
    (dt_mag / 2, -0.7 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
ahebb_y_label = text(
    r"$y = \Delta W$",
    (dt_mag / 2, -0.9 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
ahebb_label = text(
    r"$\text{Anti-Hebbian}$",
    (-dt_mag * 0.9, 0.9 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
ahebb_eta_post_label = text(
    r"$A_\text{post} < 0$",
    (-dt_mag * 0.9, -0.7 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
ahebb_eta_pre_label = text(
    r"$A_\text{pre} > 0$",
    (-dt_mag * 0.9, -0.9 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
legend_x_label = text(
    r"$x = t_\text{post}^f - t_\text{pre}^f$",
    (dt_mag / 2, 0.9 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
legend_y_label = text(
    r"$y = \Delta W$",
    (dt_mag / 2, 0.7 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)

ltp_plot = plot(
    eta_mag * exp(-x / tc_trace),
    (x, 0, dt_mag),
    ticks=[[], []],
    color="blue",
    ymin=-eta_mag,
    ymax=eta_mag,
) + plot(
    eta_mag * exp(x / tc_trace),
    (x, -dt_mag, 0),
    ticks=[[], []],
    color="blue",
    ymin=-eta_mag,
    ymax=eta_mag,
)
ltp_x_label = text(
    r"$x = t_\text{post}^f - t_\text{pre}^f$",
    (dt_mag / 2, -0.7 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
ltp_y_label = text(
    r"$y = \Delta W$",
    (dt_mag / 2, -0.9 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
ltp_label = text(
    r"$\text{LTP-Only}$",
    (-dt_mag * 0.9, 0.9 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
ltp_eta_post_label = text(
    r"$A_\text{post} > 0$",
    (-dt_mag * 0.9, -0.7 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
ltp_eta_pre_label = text(
    r"$A_\text{pre} > 0$",
    (-dt_mag * 0.9, -0.9 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)

ltd_plot = plot(
    -eta_mag * exp(-x / tc_trace),
    (x, 0, dt_mag),
    ticks=[[], []],
    color="red",
    ymin=-eta_mag,
    ymax=eta_mag,
) + plot(
    -eta_mag * exp(x / tc_trace),
    (x, -dt_mag, 0),
    ticks=[[], []],
    color="red",
    ymin=-eta_mag,
    ymax=eta_mag,
)
ltd_x_label = text(
    r"$x = t_\text{post}^f - t_\text{pre}^f$",
    (dt_mag / 2, -0.7 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
ltd_y_label = text(
    r"$y = \Delta W$",
    (dt_mag / 2, -0.9 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
ltd_label = text(
    r"$\text{LTD-Only}$",
    (-dt_mag * 0.9, 0.9 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
ltd_eta_post_label = text(
    r"$A_\text{post} < 0$",
    (-dt_mag * 0.9, -0.7 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)
ltd_eta_pre_label = text(
    r"$A_\text{pre} < 0$",
    (-dt_mag * 0.9, -0.9 * eta_mag),
    color="black",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="white",
)

P1 = hebb_plot + hebb_label + hebb_eta_post_label + hebb_eta_pre_label
P2 = (
    ahebb_plot
    + legend_x_label
    + legend_y_label
    + ahebb_label
    + ahebb_eta_post_label
    + ahebb_eta_pre_label
)
P3 = ltp_plot + ltp_label + ltp_eta_post_label + ltp_eta_pre_label
P4 = ltd_plot + ltd_label + ltd_eta_post_label + ltd_eta_pre_label

P1.save(fn_hebb, dpi=(saveres // 2))
P2.save(fn_ahebb, dpi=(saveres // 2))
P3.save(fn_ltp, dpi=(saveres // 2))
P4.save(fn_ltd, dpi=(saveres // 2))

P1 = Image.open(fn_hebb)
P2 = Image.open(fn_ahebb)
P3 = Image.open(fn_ltp)
P4 = Image.open(fn_ltd)

P = Image.new(
    "RGBA",
    (
        max(P1.size[0], P3.size[0]) + max(P2.size[0], P4.size[0]),
        max(P1.size[1], P2.size[1]) + max(P3.size[1], P4.size[1]),
    ),
)
P.paste(P1, (0, 0))
P.paste(P2, (P1.size[0], 0))
P.paste(P3, (0, P1.size[1]))
P.paste(P4, (P3.size[0], P1.size[1]))

draw = ImageDraw.Draw(P)
draw.line((0, P.size[1] / 2, P.size[0], P.size[1] / 2), fill=(0, 0, 0), width=7)
draw.line((P.size[0] / 2, 0, P.size[0] / 2, P.size[1]), fill=(0, 0, 0), width=7)

P.save(fn)

# dark mode
fn = "stdp-mode-dark.png"

x = var("x")
hebb_plot = plot(
    eta_mag * exp(-x / tc_trace),
    (x, 0, dt_mag),
    ticks=[[], []],
    color="blue",
    ymin=-eta_mag,
    ymax=eta_mag,
    transparent=True,
) + plot(
    -eta_mag * exp(x / tc_trace),
    (x, -dt_mag, 0),
    ticks=[[], []],
    color="red",
    ymin=-eta_mag,
    ymax=eta_mag,
    transparent=True,
)
hebb_x_label = text(
    r"$x = t_\text{post}^f - t_\text{pre}^f$",
    (dt_mag / 2, -0.7 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
hebb_y_label = text(
    r"$y = \Delta W$",
    (dt_mag / 2, -0.9 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
hebb_label = text(
    r"$\text{Hebbian}$",
    (-dt_mag * 0.9, 0.9 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
hebb_eta_post_label = text(
    r"$A_\text{post} > 0$",
    (-dt_mag * 0.9, -0.7 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
hebb_eta_pre_label = text(
    r"$A_\text{pre} < 0$",
    (-dt_mag * 0.9, -0.9 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)

ahebb_plot = plot(
    -eta_mag * exp(-x / tc_trace),
    (x, 0, dt_mag),
    ticks=[[], []],
    color="red",
    ymin=-eta_mag,
    ymax=eta_mag,
    transparent=True,
) + plot(
    eta_mag * exp(x / tc_trace),
    (x, -dt_mag, 0),
    ticks=[[], []],
    color="blue",
    ymin=-eta_mag,
    ymax=eta_mag,
    transparent=True,
)
ahebb_x_label = text(
    r"$x = t_\text{post}^f - t_\text{pre}^f$",
    (dt_mag / 2, -0.7 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
ahebb_y_label = text(
    r"$y = \Delta W$",
    (dt_mag / 2, -0.9 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
ahebb_label = text(
    r"$\text{Anti-Hebbian}$",
    (-dt_mag * 0.9, 0.9 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
ahebb_eta_post_label = text(
    r"$A_\text{post} < 0$",
    (-dt_mag * 0.9, -0.7 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
ahebb_eta_pre_label = text(
    r"$A_\text{pre} > 0$",
    (-dt_mag * 0.9, -0.9 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
legend_x_label = text(
    r"$x = t_\text{post}^f - t_\text{pre}^f$",
    (dt_mag / 2, 0.9 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
legend_y_label = text(
    r"$y = \Delta W$",
    (dt_mag / 2, 0.7 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)

ltp_plot = plot(
    eta_mag * exp(-x / tc_trace),
    (x, 0, dt_mag),
    ticks=[[], []],
    color="blue",
    ymin=-eta_mag,
    ymax=eta_mag,
    transparent=True,
) + plot(
    eta_mag * exp(x / tc_trace),
    (x, -dt_mag, 0),
    ticks=[[], []],
    color="blue",
    ymin=-eta_mag,
    ymax=eta_mag,
    transparent=True,
)
ltp_x_label = text(
    r"$x = t_\text{post}^f - t_\text{pre}^f$",
    (dt_mag / 2, -0.7 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
ltp_y_label = text(
    r"$y = \Delta W$",
    (dt_mag / 2, -0.9 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
ltp_label = text(
    r"$\text{LTP-Only}$",
    (-dt_mag * 0.9, 0.9 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
ltp_eta_post_label = text(
    r"$A_\text{post} > 0$",
    (-dt_mag * 0.9, -0.7 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
ltp_eta_pre_label = text(
    r"$A_\text{pre} > 0$",
    (-dt_mag * 0.9, -0.9 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)

ltd_plot = plot(
    -eta_mag * exp(-x / tc_trace),
    (x, 0, dt_mag),
    ticks=[[], []],
    color="red",
    ymin=-eta_mag,
    ymax=eta_mag,
    transparent=True,
) + plot(
    -eta_mag * exp(x / tc_trace),
    (x, -dt_mag, 0),
    ticks=[[], []],
    color="red",
    ymin=-eta_mag,
    ymax=eta_mag,
    transparent=True,
)
ltd_x_label = text(
    r"$x = t_\text{post}^f - t_\text{pre}^f$",
    (dt_mag / 2, -0.7 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
ltd_y_label = text(
    r"$y = \Delta W$",
    (dt_mag / 2, -0.9 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
ltd_label = text(
    r"$\text{LTD-Only}$",
    (-dt_mag * 0.9, 0.9 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
ltd_eta_post_label = text(
    r"$A_\text{post} < 0$",
    (-dt_mag * 0.9, -0.7 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)
ltd_eta_pre_label = text(
    r"$A_\text{pre} < 0$",
    (-dt_mag * 0.9, -0.9 * eta_mag),
    color="white",
    horizontal_alignment="left",
    vertical_alignment="center",
    fontsize=16,
    background_color="black",
)

P1 = hebb_plot + hebb_label + hebb_eta_post_label + hebb_eta_pre_label
P2 = (
    ahebb_plot
    + legend_x_label
    + legend_y_label
    + ahebb_label
    + ahebb_eta_post_label
    + ahebb_eta_pre_label
)
P3 = ltp_plot + ltp_label + ltp_eta_post_label + ltp_eta_pre_label
P4 = ltd_plot + ltd_label + ltd_eta_post_label + ltd_eta_pre_label

P1.axes_color("white")
P1.axes_label_color("white")
P1.tick_label_color("white")
P1.save(fn_hebb, dpi=(saveres // 2))

P2.axes_color("white")
P2.axes_label_color("white")
P2.tick_label_color("white")
P2.save(fn_ahebb, dpi=(saveres // 2))

P3.axes_color("white")
P3.axes_label_color("white")
P3.tick_label_color("white")
P3.save(fn_ltp, dpi=(saveres // 2))

P4.axes_color("white")
P4.axes_label_color("white")
P4.tick_label_color("white")
P4.save(fn_ltd, dpi=(saveres // 2))

P1 = Image.open(fn_hebb)
P1 = Image.alpha_composite(Image.new("RGB", P1.size, "black").convert("RGBA"), P1)

P2 = Image.open(fn_ahebb)
P2 = Image.alpha_composite(Image.new("RGB", P2.size, "black").convert("RGBA"), P2)

P3 = Image.open(fn_ltp)
P3 = Image.alpha_composite(Image.new("RGB", P3.size, "black").convert("RGBA"), P3)

P4 = Image.open(fn_ltd)
P4 = Image.alpha_composite(Image.new("RGB", P4.size, "black").convert("RGBA"), P4)

P = Image.new(
    "RGBA",
    (
        max(P1.size[0], P3.size[0]) + max(P2.size[0], P4.size[0]),
        max(P1.size[1], P2.size[1]) + max(P3.size[1], P4.size[1]),
    ),
)
P.paste(P1, (0, 0))
P.paste(P2, (P1.size[0], 0))
P.paste(P3, (0, P1.size[1]))
P.paste(P4, (P3.size[0], P1.size[1]))

draw = ImageDraw.Draw(P)
draw.line((0, P.size[1] / 2, P.size[0], P.size[1] / 2), fill=(255, 255, 255), width=7)
draw.line((P.size[0] / 2, 0, P.size[0] / 2, P.size[1]), fill=(255, 255, 255), width=7)

P.save(fn)

# clear temp files
os.remove(fn_hebb)
os.remove(fn_ahebb)
os.remove(fn_ltp)
os.remove(fn_ltd)
