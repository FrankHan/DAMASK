# -*- coding: UTF-8 no BOM -*-
# Copyright 2011-16 Max-Planck-Institut für Eisenforschung GmbH
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Main aggregator"""
import os

with open(os.path.join(os.path.dirname(__file__),'../../VERSION')) as f:
  version = f.readline()[:-1]

from .environment import Environment      # noqa
from .asciitable  import ASCIItable       # noqa
from .config      import Material         # noqa
from .colormaps   import Colormap, Color  # noqa
from .orientation import Quaternion, Rodrigues, Symmetry, Orientation # noqa

#from .block       import Block           # only one class
from .result      import Result           # noqa
from .geometry    import Geometry         # noqa
from .solver      import Solver           # noqa
from .test        import Test             # noqa
from .util        import extendableOption # noqa
