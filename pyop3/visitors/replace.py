from __future__ import annotations

import functools
from typing import Any

from immutabledict import immutabledict as idict

import pyop3.obj

from .identity import IdentityVisitor


class Replacer(IdentityVisitor):

    @functools.singledispatchmethod
    def process(self, obj: Any, /, **kwargs):
        return super().process(obj, **kwargs)

    @process.register
    def _(
        self,
        obj: pyop3.obj.Object,
        /,
        # replace_map: collections.abc.Mapping[pyop3.obj.Object, pyop3.obj.Object],
        replace_map,
    ):
        try:
            return replace_map[obj]
        except KeyError:
            return super().process(obj, replace_map=replace_map)


def replace(obj, replace_map, *, assert_modified: bool = False) -> pyop3.obj.Object:
    replaced = Replacer()(obj, replace_map=idict(replace_map))
    if assert_modified:
        # TODO: could be another exception type
        assert replaced != obj
    return replaced
