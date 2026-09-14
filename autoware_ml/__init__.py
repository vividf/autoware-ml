# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Autoware-ML top-level package.

This file must exist: without it the package is a namespace package, pytest's
rootdir logic inserts ``autoware_ml/`` itself onto ``sys.path``, and the
top-level ``autoware_ml.dataclasses`` package then shadows the stdlib
``dataclasses`` module in freshly spawned worker processes (breaking
``import torch`` under ``multiprocessing`` spawn).
"""
