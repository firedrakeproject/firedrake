#!/usr/bin/env python3
from argparse import ArgumentParser, RawDescriptionHelpFormatter
try:
    import bibtexparser
    from bibtexparser.middlewares import NormalizeFieldKeys, SortFieldsAlphabeticallyMiddleware
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

    library = bibtexparser.parse_file(filename, append_middleware=[NormalizeFieldKeys()])

    for entry in library.entries:
        if "url" not in entry and \
           "doi" not in entry:
            if "archiveprefix" in entry and entry["archiveprefix"] == "arXiv":
                entry["url"] = "https://arxiv.org/abs/" + entry["eprint"]
            else:
                raise ValueError("%s in bibliograpy %s\n has no url and no DOI.\n" % (entry["ID"], filename))

    bibtex_format = bibtexparser.BibtexFormat()
    bibtex_format.indent = '  '     # indent entries with 2 spaces instead of one
    bibtex_format.value_column = 'auto'
    bibtex_format.block_separator = '\n'     # one blank line between entries, not two

    processed = bibtexparser.write_string(library,
                                          prepend_middleware=[SortFieldsAlphabeticallyMiddleware()],
                                          bibtex_format=bibtex_format)

    if args.validate:
        with open(filename) as bibtex_file:
            if processed != bibtex_file.read():
                raise ValueError("%s would be changed by firedrake-preprocess-bibtex. Please preprocess it and commit the result" % filename)

    else:
        with open(filename, 'w') as bibfile:
            bibfile.write(processed)


if __name__ == "__main__":
    main()
