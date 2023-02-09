from configparser import ConfigParser, ExtendedInterpolation
from itertools import chain
from jinja2 import Environment, FileSystemLoader, select_autoescape
from urllib.request import urlopen


class Table:
    def __init__(self, data, cols, transpose=False):
        self.data = [*data]
        if isinstance(self.data[0], str):
            self.width = max(map(len, self.data))
            self.blank = ""
        else:
            self.width = max(map(len, [d[0] for d in self.data]))
            self.blank = tuple([""]*len(self.data[0]))
        self.ncols = cols
        rows, r = divmod(len(data), cols)
        self.nrows = rows + 1 if r else rows
        self.transpose = transpose

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if self.transpose:
            idx = key[0] * self.nrows + key[1]
        else:
            idx = key[0] * self.ncols + key[1]
        return self.data[idx] if idx < len(self) else self.blank

    @property
    def cols(self):
        return range(self.ncols)

    @property
    def rows(self):
        return range(self.nrows)


def cache_web_image(name, url):
    img_name = name.split()[0].lower().encode("punycode").decode()
    img_name = img_name[:-1] if img_name[-1] == "-" else img_name
    with urlopen(url) as response:
        filetype = response.getheader("Content-Type")
        ext = filetype.split("/")[1]
        if ext == "jpeg":
            ext = "jpg"
        with open("images/" + img_name + "." + ext, "wb") as fh:
            fh.write(response.read())


#
team = ConfigParser(interpolation=ExtendedInterpolation())
team.optionxform = lambda x: x
team.read("team.ini")

#
for name, links in team["active-team"].items():
    parts = links.split(",")
    if parts[1:]:
        website = parts[1]
        cache_web_image(name, website)

#
env = Environment(
    loader=FileSystemLoader("."),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True
)

#
extra = [*team["contributing-individual"].items()]
exclude = [*team["active-team"].keys()] + [*team["inactive-team"].keys()]
extra = [e for e in extra if e[0] not in exclude]

#
team_rst = env.get_template("team.rst_t")
with open("team.rst", "w") as fh:
    fh.write(team_rst.render(
        team=team,
        atable=Table(team["active-team"].items(), 4),
        itable=Table(team["inactive-team"].items(), 4),
        ctable=Table(extra, 3, transpose=True)
    ))

#
authors_rst = env.get_template("authors.rst_t")

institution_set = set(chain(
    team["dev-institution"].items(), team["contributing-institution"].items()
))
institution_set = list(institution_set)
institution_set.sort(key=lambda x: x[0])

individual_set = dict(team["contributing-individual"].items())
individual_set.update(team["active-team"].items())
individual_set.update(team["inactive-team"].items())
individual_set = list(individual_set.items())
individual_set.sort(key=lambda x: x[0].split()[-1])
with open("AUTHORS.rst", "w") as fh:
    fh.write(authors_rst.render(
        institution_set=institution_set,
        individual_set=individual_set
    ))
