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
import numpy as np
from optparse import OptionParser
import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [file[s]]', description = """
Replace all rows for which column 'label' has identical values by a single row containing their average.
Output table will contain as many rows as there are different (unique) values in the grouping column.

Examples:
For grain averaged values, replace all rows of particular 'texture' with a single row containing their average.
""", version = scriptID)

parser.add_option('-l','--label',
                  dest = 'label',
                  type = 'string', metavar = 'string',
                  help = 'column label for grouping rows')

(options,filenames) = parser.parse_args()

if options.label is None:
  parser.error('no grouping column specified.')


# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:    table = damask.ASCIItable(name = name,
                                    outname = os.path.join(
                                              os.path.split(name)[0],
                                              options.label+'_averaged_'+os.path.split(name)[1]
                                              ) if name else name,
                                    buffered = False)
  except: continue
  damask.util.report(scriptName,name)

# ------------------------------------------ sanity checks ---------------------------------------  

  table.head_read()
  if table.label_dimension(options.label) != 1:
    damask.util.croak('column {} is not of scalar dimension.'.format(options.label))
    table.close(dismiss = True)                                                                     # close ASCIItable and remove empty file
    continue

# ------------------------------------------ assemble info ---------------------------------------  

  table.info_append(scriptID + '\t' + ' '.join(sys.argv[1:]))
  table.head_write()

# ------------------------------------------ process data -------------------------------- 

  table.data_readArray()
  rows,cols  = table.data.shape

  table.data = table.data[np.lexsort([table.data[:,table.label_index(options.label)]])]
  
  values,index = np.unique(table.data[:,table.label_index(options.label)], return_index = True)
  index = np.append(index,rows)
  avgTable = np.empty((len(values), cols))
  
  for j in xrange(cols) :
    for i in xrange(len(values)) :
      avgTable[i,j] = np.average(table.data[index[i]:index[i+1],j])
  
  table.data = avgTable

# ------------------------------------------ output result -------------------------------  

  table.data_writeArray()
  table.close()                                                                                     # close ASCII table
