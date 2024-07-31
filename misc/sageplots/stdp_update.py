from PIL import Image
from sage.all import line, plot, exp, text, var, e

# Global Setting
dispres = 300
saveres = 300
hspace = 0.1
vspace = 0.1

# STDP Hyperparams Viz
dt_min = -80
dt_max = 80
tc_pre = 20
tc_post = 30
eta_pos = 1.0
eta_neg = -0.5
y_over = 0.25

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
