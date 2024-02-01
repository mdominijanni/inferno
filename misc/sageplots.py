# QIF Slope Field
vrest = -60
vcrit = -50
vmin = vrest - 5
vmax = vcrit + 5
labelspace = 0.625
labelh = 0.03

qif_slopefield = plot_slope_field(lambda t, v: (v - vrest) * (v - vcrit), (0, 1), (vmin, vmax), color='black')
vrest_line = line([(0, vrest), (1, vrest)], color='blue')
vrest_label = text(r"$V_R$", (1 - labelh, vrest - labelspace), color='blue', vertical_alignment='top', fontsize=18, background_color='white')
vcrit_line = line([(0, vcrit), (1, vcrit)], color='red')
vcrit_label = text(r"$V_C$", (1 - labelh, vcrit + labelspace), color='red', vertical_alignment='bottom', fontsize=18, background_color='white')
(qif_slopefield + vrest_line + vrest_label + vcrit_line + vcrit_label).show(dpi=300)
