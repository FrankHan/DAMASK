#!/usr/bin/env bash
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

for geom in "$@"
do
  geom_toTable \
  < $geom \
  | \
  vtk_rectilinearGrid > ${geom%.*}.vtk

  geom_toTable \
  < $geom \
  | \
  vtk_addRectilinearGridData \
    --scalar microstructure \
    --inplace \
    --vtk ${geom%.*}.vtk
  rm ${geom%.*}.vtk
done
