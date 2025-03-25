#!/usr/bin/env python3
import io
from argparse import ArgumentParser, RawDescriptionHelpFormatter
try:
    from bibtexparser.bwriter import BibTexWriter
    import bibtexparser
except ImportError:
    raise ImportError("Failed to import bibtexparser. Run:\n firedrake-update --documentation-dependencies")


def main():
    parser = ArgumentParser(description="""Ensure BibTeX entries for inclusion in the Firedrake website have a
URL or DOI, and impose clean formatting.""",
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("bibtex_file", help="The BibTeX file to process")
    parser.add_argument("--validate", action='store_true',
                        help="Instead of rewriting the bibtex file, raise an exception if anything would have changed.")

    args = parser.parse_args()

    filename = args.bibtex_file

    parser = bibtexparser.bparser.BibTexParser()
    parser.common_strings = True
    parser.ignore_nonstandard_types = False

    with open(filename) as bibtex_file:
        bib_database = parser.parse_file(bibtex_file)

    for entry in bib_database.entries:
        if "url" not in entry and \
           "doi" not in entry:
            if entry.get("archiveprefix", None) == "arXiv":
                entry["url"] = "https://arxiv.org/abs/" + entry["eprint"]
            else:
                raise ValueError("%s in bibliograpy %s\n has no url and no DOI.\n" % (entry["ID"], filename))

    writer = BibTexWriter()
    writer.indent = '  '     # indent entries with 2 spaces instead of one
    writer.align_values = True

    if args.validate:
        with io.StringIO() as outbuffer:
            outbuffer.write(writer.write(bib_database))
            processed = outbuffer.getvalue()
            with open(filename) as bibtex_file:
                inbuffer = bibtex_file.read()
            if processed != inbuffer:
                raise ValueError("%s would be changed by firedrake-preprocess-bibtex. Please preprocess it and commit the result" % filename)

    else:
        with open(filename, 'w') as bibfile:
            bibfile.write(writer.write(bib_database))


if __name__ == "__main__":
    main()
