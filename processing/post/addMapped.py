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

# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [file[s]]', description = """
Add data in column(s) of second ASCIItable selected from row that is given by the value in a mapping column.

""", version = scriptID)

parser.add_option('-c','--map',
                  dest = 'map',
                  type = 'string', metavar = 'string',
                  help = 'heading of column containing row mapping')
parser.add_option('-o','--offset',
                  dest = 'offset',
                  type = 'int', metavar = 'int',
                  help = 'offset between mapping column value and actual row in mapped table [%default]')
parser.add_option('-l','--label',
                  dest = 'label',
                  action = 'extend', metavar = '<string LIST>',
                  help='heading of column(s) to be mapped')
parser.add_option('-a','--asciitable',
                  dest = 'asciitable',
                  type = 'string', metavar = 'string',
                  help = 'mapped ASCIItable')

parser.set_defaults(offset = 0,
                   )

(options,filenames) = parser.parse_args()

if options.label is None:
  parser.error('no data columns specified.')
if options.map is None:
  parser.error('no mapping column given.')

# ------------------------------------------ process mapping ASCIItable ---------------------------

if options.asciitable is not None and os.path.isfile(options.asciitable):

  mappedTable = damask.ASCIItable(name = options.asciitable,
                                  buffered = False, readonly = True) 
  mappedTable.head_read()                                                                           # read ASCII header info of mapped table
  missing_labels = mappedTable.data_readArray(options.label)

  if len(missing_labels) > 0:
    mappedTable.croak('column{} {} not found...'.format('s' if len(missing_labels) > 1 else '',', '.join(missing_labels)))

else:
  parser.error('no mapped ASCIItable given.')

# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:
    table = damask.ASCIItable(name = name,
                              buffered = False)
  except: continue
  damask.util.report(scriptName,name)

# ------------------------------------------ read header ------------------------------------------

  table.head_read()

# ------------------------------------------ sanity checks ----------------------------------------

  errors = []

  mappedColumn = table.label_index(options.map)  
  if mappedColumn <  0: errors.append('mapping column {} not found.'.format(options.map))

  if errors != []:
    damask.util.croak(errors)
    table.close(dismiss = True)
    continue

# ------------------------------------------ assemble header --------------------------------------

  table.info_append(scriptID + '\t' + ' '.join(sys.argv[1:]))
  table.labels_append(mappedTable.labels(raw = True))                                              # extend ASCII header with new labels
  table.head_write()

# ------------------------------------------ process data ------------------------------------------

  outputAlive = True
  while outputAlive and table.data_read():                                                          # read next data line of ASCII table
    table.data_append(mappedTable.data[int(round(float(table.data[mappedColumn])))+options.offset-1]) # add all mapped data types
    outputAlive = table.data_write()                                                                # output processed line

# ------------------------------------------ output finalization -----------------------------------  

  table.close()                                                                                     # close ASCII tables

mappedTable.close()                                                                                 # close mapped input ASCII table
