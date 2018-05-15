#!/usr/bin/python
"""This script forms part of the internal workings of Firedrake-Zenodo. It is not intended for direct user interaction."""
from __future__ import absolute_import, print_function, division
import cgi
import cgitb
cgitb.enable()
import urllib2
import json
import sys


def fail(code, reason):
    sys.stdout.write('Status: %s\r\n\r\n%s\r\n\r\n' % (code, reason))
    sys.exit(1)


def find_entries(response, firedrake_tag):
    entries = []
    for record in response:
        try:
            for id in record['metadata']['related_identifiers']:
                if id['identifier'].endswith(firedrake_tag):
                    entries.append(record)
        except KeyError:
            pass
    return entries


def get_bibtex(entries):
    bibtex = []
    months = {"01": "jan",
              "02": "feb",
              "03": "mar",
              "04": "apr",
              "05": "may",
              "06": "jun",
              "07": "jul",
              "08": "aug",
              "09": "sep",
              "10": "oct",
              "11": "nov",
              "12": "dec"}
    for entry in entries:
        bib = """@misc{%(key)s,
  author = {%(author)s},
  title = {%(title)s},
  year  = {%(year)s},
  month = {%(month)s},
  doi  = {%(doi)s},
  url = {https://doi.org/%(doi)s},
}"""
        vals = {}
        title = entry["title"]
        title = title[title.find("/")+1:]
        vals["title"] = title
        vals["doi"] = entry["doi"]
        time = entry["created"]
        year = time[:4]
        month = months[time[5:7]]
        vals["year"] = year
        vals["month"] = month
        key = "zenodo/%s" % title[:title.find(":")]
        vals["key"] = "%s:%s" % (key, year)
        vals["author"] = key
        bibtex.append(bib % vals)
    return "\n\n".join(bibtex)


form = cgi.FieldStorage()
firedrake_tag = form.getfirst("tag")
if not firedrake_tag:
    fail("400 Bad Request", "Request must specify the tag.")
# Do some primitive parsing to avoid more blatant security issues.
try:
    assert firedrake_tag.startswith("Firedrake_")
    float(firedrake_tag[10:])
    assert firedrake_tag[18] == "."
    digits = map(str, range(10))
    assert all([a in digits for a in (firedrake_tag[10:18] + firedrake_tag[19:])])
except:                         # noqa: E722
    fail("400 Bad Request", "%s is not a legal Firedrake release tag" % firedrake_tag)

# Use sed to insert OAUTH token on next line before uploading to web server.
try:
    response = urllib2.urlopen("https://zenodo.org/api/deposit/depositions?access_token=ZENODO_OAUTH&size=9999")
except:                         # noqa: E722
    fail("400 Bad Request", "Unable to open deposit records")

try:
    entries = find_entries(json.load(response), firedrake_tag)
except:                         # noqa: E722
    fail("400 Bad Request", "Unable to create entries")

try:
    bibtex = get_bibtex(entries)
except:                         # noqa: E722
    fail("400 Bad Request", "Unable to generate bibtex")

print("Content-Type: text/plain; charset=utf-8")
print()
print(bibtex.encode("utf-8"))
