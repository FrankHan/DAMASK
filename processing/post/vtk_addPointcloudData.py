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

import os,vtk
import damask
from collections import defaultdict
from optparse import OptionParser

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [file[s]]', description = """
Add scalar and RGB tuples from ASCIItable to existing VTK point cloud (.vtp).

""", version = scriptID)

parser.add_option(      '--vtk',
                  dest = 'vtk',
                  type = 'string', metavar = 'string',
                  help = 'VTK file name')
parser.add_option(      '--inplace',
                  dest = 'inplace',
                  action = 'store_true',
                  help = 'modify VTK file in-place')
parser.add_option('-r', '--render',
                  dest = 'render',
                  action = 'store_true',
                  help = 'open output in VTK render window')
parser.add_option('-s', '--scalar',   dest='scalar', action='extend',
                  metavar ='<string LIST>',
                  help = 'scalar values')
parser.add_option('-v', '--vector',
                  dest = 'vector',
                  action = 'extend', metavar = '<string LIST>',
                  help = 'vector value label(s)')
parser.add_option('-c', '--color',   dest='color', action='extend',
                  metavar ='<string LIST>',
                  help = 'RGB color tuples')

parser.set_defaults(scalar = [],
                    vector = [],
                    color = [],
                    inplace = False,
                    render = False,
)

(options, filenames) = parser.parse_args()

if not options.vtk:                 parser.error('No VTK file specified.')
if not os.path.exists(options.vtk): parser.error('VTK file does not exist.')

if os.path.splitext(options.vtk)[1] == '.vtp':
  reader = vtk.vtkXMLPolyDataReader()
  reader.SetFileName(options.vtk)
  reader.Update()
  Polydata = reader.GetOutput()
elif os.path.splitext(options.vtk)[1] == '.vtk':
  reader = vtk.vtkGenericDataObjectReader()
  reader.SetFileName(options.vtk)
  reader.Update()
  Polydata = reader.GetPolyDataOutput()
else:
  parser.error('Unsupported VTK file type extension.')

Npoints   = Polydata.GetNumberOfPoints()
Ncells    = Polydata.GetNumberOfCells()
Nvertices = Polydata.GetNumberOfVerts()

if Npoints != Ncells or Npoints != Nvertices:
  parser.error('Number of points, cells, and vertices in VTK differ from each other.')

damask.util.croak('{}: {} points, {} vertices, and {} cells...'.format(options.vtk,Npoints,Nvertices,Ncells))

# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:    table = damask.ASCIItable(name = name,
                                    buffered = False,
                                    readonly = True)
  except: continue
  damask.util.report(scriptName, name)

# --- interpret header ----------------------------------------------------------------------------

  table.head_read()

  remarks = []
  errors  = []
  VTKarray = {}
  active = defaultdict(list)
  
  for datatype,dimension,label in [['scalar',1,options.scalar],
                                   ['vector',3,options.vector],
                                   ['color',3,options.color],
                                   ]:
    for i,dim in enumerate(table.label_dimension(label)):
      me = label[i]
      if dim == -1: remarks.append('{} "{}" not found...'.format(datatype,me))
      elif dim > dimension: remarks.append('"{}" not of dimension {}...'.format(me,dimension))
      else:
        remarks.append('adding {} "{}"...'.format(datatype,me))
        active[datatype].append(me)

        if   datatype in ['scalar','vector']: VTKarray[me] = vtk.vtkDoubleArray()
        elif datatype == 'color':             VTKarray[me] = vtk.vtkUnsignedCharArray()

        VTKarray[me].SetNumberOfComponents(dimension)
        VTKarray[me].SetName(label[i])

  if remarks != []: damask.util.croak(remarks)
  if errors  != []:
    damask.util.croak(errors)
    table.close(dismiss = True)
    continue

# ------------------------------------------ process data ---------------------------------------  

  while table.data_read():                                                                          # read next data line of ASCII table
    
    for datatype,labels in active.items():                                                          # loop over scalar,color
      for me in labels:                                                                             # loop over all requested items
        theData = [table.data[i] for i in table.label_indexrange(me)]                               # read strings
        if   datatype == 'color':  VTKarray[me].InsertNextTuple3(*map(lambda x: int(255.*float(x)),theData))
        elif datatype == 'vector': VTKarray[me].InsertNextTuple3(*map(float,theData))
        elif datatype == 'scalar': VTKarray[me].InsertNextValue(float(theData[0]))

  table.input_close()                                                     # close input ASCII table

# ------------------------------------------ add data ---------------------------------------  

  for datatype,labels in active.items():                                                            # loop over scalar,color
    if datatype == 'color':
      Polydata.GetPointData().SetScalars(VTKarray[active['color'][0]])
      Polydata.GetCellData().SetScalars(VTKarray[active['color'][0]])
    for me in labels:                                                                               # loop over all requested items
      Polydata.GetPointData().AddArray(VTKarray[me])
      Polydata.GetCellData().AddArray(VTKarray[me])

  Polydata.Modified()
  if vtk.VTK_MAJOR_VERSION <= 5: Polydata.Update()

# ------------------------------------------ output result ---------------------------------------  

  writer = vtk.vtkXMLPolyDataWriter()
  writer.SetDataModeToBinary()
  writer.SetCompressorTypeToZLib()
  writer.SetFileName(os.path.splitext(options.vtk)[0]+('.vtp' if options.inplace else '_added.vtp'))
  if vtk.VTK_MAJOR_VERSION <= 5: writer.SetInput(Polydata)
  else:                          writer.SetInputData(Polydata)
  writer.Write()

# ------------------------------------------ render result ---------------------------------------  

if options.render:
  mapper = vtk.vtkDataSetMapper()
  mapper.SetInputData(Polydata)
  actor = vtk.vtkActor()
  actor.SetMapper(mapper)

# Create the graphics structure. The renderer renders into the
# render window. The render window interactor captures mouse events
# and will perform appropriate camera or actor manipulation
# depending on the nature of the events.

  ren = vtk.vtkRenderer()

  renWin = vtk.vtkRenderWindow()
  renWin.AddRenderer(ren)

  ren.AddActor(actor)
  ren.SetBackground(1, 1, 1)
  renWin.SetSize(200, 200)

  iren = vtk.vtkRenderWindowInteractor()
  iren.SetRenderWindow(renWin)
 
  iren.Initialize()
  renWin.Render()
  iren.Start()
