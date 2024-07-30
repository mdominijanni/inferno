import os
import shutil

# delete generated documentation
shutil.rmtree(os.path.join("reference", "generated"), ignore_errors=True)

# read config generator file
with open("conf.gen.py", "r") as file:
    conf = file.readlines()

# write config file with autosummary generation
with open("conf.py", "w") as file:
    file.writelines(conf + ["", "autosummary_generate = True", ""])

# run to generate .rst files
os.system("make clean html")


# replace generated files with overrides
osrsplit = lambda S, L: (  # noqa:E731;
    L
    if not S
    else osrsplit(
        *(sstr if not idx else [sstr] + L for idx, sstr in enumerate(os.path.split(S)))
    )
)
for root, dirs, files in os.walk(os.path.join("..", "docs-override")):
    if files:
        for file in files:
            print(f"overwriting with: {os.path.join(root, file)}")
            shutil.copyfile(
                os.path.join(root, file),
                os.path.join(".", *osrsplit(root, [])[2:], "generated", file),
            )

# overwrite config file without autosummary generation
with open("conf.py", "w") as file:
    file.writelines(conf + ["", "autosummary_generate = False", ""])

# run to generate html files
os.system("make clean html")


# copy image files
shutil.copy("images/logo-darkmode.svg", "_build/html/_static/logo-darkmode.svg")
shutil.copy("images/logo-lightmode.svg", "_build/html/_static/logo-lightmode.svg")

# delete config file
os.remove("conf.py")
