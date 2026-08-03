# from __future__ import annotations
#
# import collections
# import functools
# from typing import Any
#
# from immutabledict import immutabledict as idict
#
# import pyop3.obj
# from pyop3 import utils
#
# from .identity import ExpressionIdentityVisitor
#
#
# class ExpressionReplacer(ExpressionIdentityVisitor):
#
#     @functools.singledispatchmethod
#     def process(self, obj: Any, /, **kwargs):
#         return super().process(obj, **kwargs)
#
#     @process.register
#     def _(
#         self,
#         expr: pyop3.expr.Expression,
#         /,
#         # replace_map: collections.abc.Mapping[pyop3.obj.Object, pyop3.obj.Object],
#         replace_map,
#     ):
#         try:
#             return replace_map[expr]
#         except KeyError:
#             return super().process(expr, replace_map=replace_map)
#
#
# def replace(expr, replace_map, *, assert_modified: bool = False):
#     replaced = ExpressionReplacer()(expr, replace_map=idict(replace_map))
#     if assert_modified:
#         # TODO: could be another exception type
#         assert replaced != expr
#     return replaced
