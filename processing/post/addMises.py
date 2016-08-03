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

import os,sys,math
import numpy as np
from optparse import OptionParser
import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

def Mises(what,tensor):

  dev = tensor - np.trace(tensor)/3.0*np.eye(3)
  symdev = 0.5*(dev+dev.T)
  return math.sqrt(np.sum(symdev*symdev.T)*
        {
         'stress': 3.0/2.0,
         'strain': 2.0/3.0,
         }[what.lower()])

# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [file[s]]', description = """
Add vonMises equivalent values for symmetric part of requested strains and/or stresses.

""", version = scriptID)

parser.add_option('-e','--strain',
                  dest = 'strain',
                  action = 'extend', metavar = '<string LIST>',
                  help = 'heading(s) of columns containing strain tensors')
parser.add_option('-s','--stress',
                  dest = 'stress',
                  action = 'extend', metavar = '<string LIST>',
                  help = 'heading(s) of columns containing stress tensors')

parser.set_defaults(strain = [],
                    stress = [],
                   )

(options,filenames) = parser.parse_args()

if len(options.stress+options.strain) == 0:
  parser.error('no data column specified...')

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

  items = {
            'strain': {'dim': 9, 'shape': [3,3], 'labels':options.strain, 'active':[], 'column': []},
            'stress': {'dim': 9, 'shape': [3,3], 'labels':options.stress, 'active':[], 'column': []},
          }
  errors  = []
  remarks = []
  
  for type, data in items.iteritems():
    for what in data['labels']:
      dim = table.label_dimension(what)
      if dim != data['dim']: remarks.append('column {} is not a {}...'.format(what,type))
      else:
        items[type]['active'].append(what)
        items[type]['column'].append(table.label_index(what))
        table.labels_append('Mises({})'.format(what))                                               # extend ASCII header with new labels

  if remarks != []: damask.util.croak(remarks)
  if errors  != []:
    damask.util.croak(errors)
    table.close(dismiss = True)
    continue

# ------------------------------------------ assemble header --------------------------------------

  table.info_append(scriptID + '\t' + ' '.join(sys.argv[1:]))
  table.head_write()

# ------------------------------------------ process data ------------------------------------------

  outputAlive = True
  while outputAlive and table.data_read():                                                          # read next data line of ASCII table
    for type, data in items.iteritems():
      for column in data['column']:
        table.data_append(Mises(type,
                                np.array(table.data[column:column+data['dim']],'d').reshape(data['shape'])))
    outputAlive = table.data_write()                                                                # output processed line

# ------------------------------------------ output finalization -----------------------------------

  table.close()                                                                                     # close input ASCII table (works for stdin)
