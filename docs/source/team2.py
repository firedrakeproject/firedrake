from configparser import ConfigParser, ExtendedInterpolation
from itertools import chain
from jinja2 import Environment, FileSystemLoader, select_autoescape
from urllib.request import urlopen


class Table:
    """ This class makes it easier to generate tables in jinja templates
    """
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


# Read the current team information from configuration file
team = ConfigParser(interpolation=ExtendedInterpolation())
team.optionxform = lambda x: x
team.read("team.ini")

# Environment for applying templates
env = Environment(
    loader=FileSystemLoader("."),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True
)

# Collect names for the team page
extra = [*team["contributing-individual"].items()]
exclude = [*team["active-team"].keys()] + [*team["inactive-team"].keys()]
extra = [e for e in extra if e[0] not in exclude]

# Create team webpage from template
team_rst = env.get_template("team.rst_t")
with open("team.rst", "w") as fh:
    fh.write(team_rst.render(
        team=team,
        atable=Table(team["active-team"].items(), 4),
        itable=Table(team["inactive-team"].items(), 4),
        ctable=Table(extra, 3, transpose=True)
    ))

# Create authors file for the Github repository
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

# Create citations file for the Github repository
citation_rst = env.get_template("citation.rst_t")
institution_list = [inst[0] for inst in team["dev-institution"].items()]
institution = ' and '.join(institution_list)
author_list = list(team['active-team'].keys())
author_list += list(team['inactive-team'].keys())
author = ' and '.join(author_list)
with open("CITATION.rst", "w") as fh:
    fh.write(citation_rst.render(
        author=author,
        institution=institution
    ))
