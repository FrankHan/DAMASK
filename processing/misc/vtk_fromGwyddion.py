#!/usr/bin/env python2.7
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

import os,string,vtk
import numpy as np
import damask
from optparse import OptionParser

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

scalingFactor = { \
                 'm':  {
                        'm':  1e0,
                        'mm': 1e-3,
                        'µm': 1e-6,
                       },
                 'mm':  {
                        'm':  1e+3,
                        'mm': 1e0,
                        'µm': 1e-3,
                       },
                 'µm':  {
                        'm':  1e+6,
                        'mm': 1e+3,
                        'µm': 1e0,
                       },
                }

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [file[s]]', description = """
Produce VTK rectilinear grid from Gwyddion dataset exported as text.
""" + string.replace(scriptID,'\n','\\n')
)

parser.add_option('-s', '--scaling',   dest='scaling', type='float',
                  help = 'scaling factor for elevation data [auto]')

parser.set_defaults(scaling = 0.0)

(options, filenames) = parser.parse_args()


# ------------------------------------------ read Gwyddion data ---------------------------------------  

for file in filenames:
  with open(file,'r') as f:
    for line in f:
      pieces = line.split()
      if pieces[0] != '#': break
      if len(pieces) < 2: continue
      if pieces[1] == 'Width:':
        width  = float(pieces[2])
        lateralunit = pieces[3]
      if pieces[1] == 'Height:':
        height = float(pieces[2])
        lateralunit = pieces[3]
      if pieces[1] == 'Value' and pieces[2] == 'units:':
        elevationunit = pieces[3]

    if options.scaling == 0.0:
      options.scaling = scalingFactor[lateralunit][elevationunit]
  
    elevation = np.loadtxt(file)*options.scaling
    
    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(elevation.shape[1],elevation.shape[0],1)

    xCoords = vtk.vtkDoubleArray()
    for x in np.arange(0.0,width,width/elevation.shape[1],'d'):
      xCoords.InsertNextValue(x)
    yCoords = vtk.vtkDoubleArray()
    for y in np.arange(0.0,height,height/elevation.shape[0],'d'):
      yCoords.InsertNextValue(y)
    zCoords = vtk.vtkDoubleArray()
    zCoords.InsertNextValue(0.0)

    grid.SetXCoordinates(xCoords)
    grid.SetYCoordinates(yCoords)
    grid.SetZCoordinates(zCoords)

    vector = vtk.vtkFloatArray()
    vector.SetName("elevation");
    vector.SetNumberOfComponents(3);
    vector.SetNumberOfTuples(np.prod(elevation.shape));
    for i,z in enumerate(np.ravel(elevation)):
      vector.SetTuple3(i,0,0,z)

    grid.GetPointData().AddArray(vector)

    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToZLib()
    writer.SetFileName(os.path.splitext(file)[0]+'.vtr')
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(grid)
    else:
        writer.SetInputData(grid)
    writer.Write()
