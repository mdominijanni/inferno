import os
import shutil


# read config generator file
with open("docs/conf.gen.py", "r") as file:
    conf = file.readlines()

# write config file with autosummary generation
with open("docs/conf.py", "w") as file:
    file.writelines(conf + ["", "autosummary_generate = True", ""])

# change relative directory to docs
os.chdir(os.path.join(".", "docs"))

# run to generate .rst files
os.system("python -m sphinx -T -b html -d _build/doctrees -D language=en . $READTHEDOCS_OUTPUT/html")

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
                os.path.join(*osrsplit(root, [])[2:], "generated", file),
            )

# change relative directory to parent
os.chdir(os.path.join(".."))

# overwrite config file without autosummary generation
with open("docs/conf.py", "w") as file:
    file.writelines(conf + ["", "autosummary_generate = False", ""])
