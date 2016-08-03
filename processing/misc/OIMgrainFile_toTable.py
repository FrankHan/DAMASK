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

import os,sys
from optparse import OptionParser
import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

#--------------------------------------------------------------------------------------------------
#                                MAIN
#--------------------------------------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [file[s]]', description = """
Adds header to OIM grain file to make it accesible as ASCII table

""", version = scriptID)

parser.add_option('-l', '--labels',
                  dest   = 'labels',
                  action = 'extend', metavar = '<string LIST>',
                  help   = 'lables of requested columns')

parser.set_defaults(labels = ['1_euler','2_euler','3_euler',
                              '1_pos','2_pos', 'IQ', 'CI', 'Fit', 'GrainID',],
                   )

(options, filenames) = parser.parse_args()

# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:
    table = damask.ASCIItable(name = name,
                              buffered = False,
                              labeled = False)
  except: continue
  damask.util.report(scriptName,name)
  table.head_read()
  data = []
  while  table.data_read():
    data.append(table.data[0:len(options.labels)])

  table.info_append(scriptID + '\t' + ' '.join(sys.argv[1:]))
  table.labels_append(options.labels)
  table.head_write()
  for i in data:
    table.data = i
    table.data_write()

# --- output finalization --------------------------------------------------------------------------

  table.close()                                                                                     # close ASCII table
