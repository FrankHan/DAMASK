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

import os,sys,re
import damask
from optparse import OptionParser

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog [options] dfile[s]', description = """
Rename scalar, vectorial, and/or tensorial data header labels.

""", version = scriptID)

parser.add_option('-l','--label',
                  dest = 'label',
                  action = 'extend', metavar='<string LIST>',
                  help = 'column(s) to rename')
parser.add_option('-s','--substitute',
                  dest = 'substitute',
                  action = 'extend', metavar='<string LIST>',
                  help = 'new column label(s)')

parser.set_defaults(label = [],
                    substitute = [],
                   )

(options,filenames) = parser.parse_args()

pattern = [re.compile('^()(.+)$'),                                                                # label pattern for scalar
           re.compile('^(\d+_)?(.+)$'),                                                           # label pattern for multidimension
          ]

# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:    table = damask.ASCIItable(name = name,
                                    buffered = False)
  except: continue
  damask.util.report(scriptName,name)

# ------------------------------------------ read header ------------------------------------------  

  table.head_read()

# ------------------------------------------ process labels ---------------------------------------  

  errors  = []
  remarks = []

  if len(options.label) == 0:
    errors.append('no labels specified.')
  elif len(options.label) != len(options.substitute):
    errors.append('mismatch between number of labels ({}) and substitutes ({}).'.format(len(options.label),
                                                                                        len(options.substitute)))
  else:
    indices    = table.label_index    (options.label)
    dimensions = table.label_dimension(options.label)
    for i,index in enumerate(indices):
      if index == -1: remarks.append('label "{}" not present...'.format(options.label[i]))
      else:
        m = pattern[dimensions[i]>1].match(table.tags[index])                                       # isolate label name
        for j in xrange(dimensions[i]):
          table.tags[index+j] = table.tags[index+j].replace(m.group(2),options.substitute[i])       # replace name with substitute

  if remarks != []: damask.util.croak(remarks)
  if errors  != []:
    damask.util.croak(errors)
    table.close(dismiss = True)
    continue

# ------------------------------------------ assemble header ---------------------------------------  

  table.info_append(scriptID + '\t' + ' '.join(sys.argv[1:]))
  table.head_write()

# ------------------------------------------ process data ------------------------------------------  

  outputAlive = True
  while outputAlive and table.data_read():                                                          # read next data line of ASCII table
    outputAlive = table.data_write()                                                                # output processed line

# ------------------------------------------ output finalization -----------------------------------  

  table.close()                                                                                     # close ASCII tables
