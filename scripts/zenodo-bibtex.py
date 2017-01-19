#!/usr/bin/python
"""This script forms part of the internal workings of Firedrake-Zenodo. It is not intended for direct user interaction."""
from __future__ import absolute_import, print_function, division
import cgi
import cgitb
cgitb.enable()
import urllib2
import json
from HTMLParser import HTMLParser
import sys


def fail(code, reason):

    sys.stdout.write('Status: %s\r\n\r\n%s\r\n\r\n' % (code, reason))
    sys.exit(1)


def find_ids(response, firedrake_tag):

    ids = []
    for record in response:
        try:
            for id in record['metadata']['related_identifiers']:
                if id['identifier'].endswith(firedrake_tag):
                    ids.append(record['record_id'])
        except KeyError:
            pass

    return ids


class BibtexFinder(HTMLParser):
    """The BibTeX data is kept inside tag <pre id="clipboard_text">."""
    def __init__(self):
        HTMLParser.__init__(self)

        self.in_bibtex = False
        self.bibtexdata = None

    def handle_starttag(self, tag, attrs):
        if tag == "pre":
            a = dict(attrs)
            if a.get("style") == "white-space: pre-wrap;":
                self.in_bibtex = True

    def handle_data(self, data):
        if self.in_bibtex:
            self.bibtexdata = data

    def handle_endtag(self, tag):
        self.in_bibtex = False


def get_bibtex(ids):

    bibtex = []
    for id in ids:
        html = urllib2.urlopen("https://zenodo.org/record/%s/export/hx" % id)
        parser = BibtexFinder()
        charset = html.headers.getparam("charset")
        parser.feed(html.read().decode(charset))
        if parser.bibtexdata:
            bibtex.append(parser.bibtexdata)

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
except:
    fail("400 Bad Request", "%s is not a legal Firedrake release tag" % firedrake_tag)

# Use sed to insert OAUTH token on next line before uploading to web server.
response = urllib2.urlopen("https://zenodo.org/api/deposit/depositions/?access_token=ZENODO_OAUTH")

ids = find_ids(json.load(response), firedrake_tag)

print("Content-Type: text/plain; charset=utf-8")
print()
print(get_bibtex(ids).encode("utf-8"))
