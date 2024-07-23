import shutil

# copy image files
shutil.copy(
    "docs/images/logo-darkmode.svg", "docs/_build/html/_static/logo-darkmode.svg"
)
shutil.copy(
    "docs/images/logo-lightmode.svg", "docs/_build/html/_static/logo-lightmode.svg"
)
