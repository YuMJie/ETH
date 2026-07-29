"""Compatibility shims loaded automatically by Python runtime startup."""

import collections.abc
import sys
import types


if "torch._six" not in sys.modules:
    torch_six = types.ModuleType("torch._six")
    torch_six.container_abcs = collections.abc
    torch_six.int_classes = (int,)
    torch_six.string_classes = (str,)
    sys.modules["torch._six"] = torch_six
