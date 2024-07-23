# read config generator file
with open("docs/conf.gen.py", "r") as file:
    conf = file.readlines()

# write config file with autosummary generation
with open("docs/conf.py", "w") as file:
    file.writelines(conf + ["", "autosummary_generate = True", ""])
