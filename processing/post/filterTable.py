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

import os,re,sys,fnmatch
import math                                                                                         # noqa
import numpy as np
from optparse import OptionParser
import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

def sortingList(labels,whitelistitems):

  indices = []
  names   = []

  for label in labels:
    if re.match('^\d+_',label):
      indices.append(int(label.split('_',1)[0]))
      names.append(label.split('_',1)[1])
    else:
      indices.append(0)
      names.append(label)
      
  return [indices,names,whitelistitems]


# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [file[s]]', description = """
Filter rows according to condition and columns by either white or black listing.

Examples:
Every odd row if x coordinate is positive -- " #ip.x# >= 0.0 and #_row_#%2 == 1 ).
All rows where label 'foo' equals 'bar' -- " #s#foo# == 'bar' "

""", version = scriptID)

parser.add_option('-w','--white',
                  dest   = 'whitelist',
                  action = 'extend', metavar = '<string LIST>',
                  help   = 'whitelist of column labels (a,b,c,...)')
parser.add_option('-b','--black',
                  dest   = 'blacklist',
                  action = 'extend', metavar='<string LIST>',
                  help   = 'blacklist of column labels (a,b,c,...)')
parser.add_option('-c','--condition',
                  dest   = 'condition', metavar='string',
                  help   = 'condition to filter rows')

parser.set_defaults(condition = '',
                   )

(options,filenames) = parser.parse_args()

# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:
    table = damask.ASCIItable(name = name,
                              buffered = False)
  except: continue
  damask.util.report(scriptName,name)

# ------------------------------------------ assemble info ---------------------------------------  

  table.head_read()

# ------------------------------------------ process data ---------------------------------------  

  specials = { \
               '_row_': 0,
             }
  labels = []
  positions = []

  for position,label in enumerate(table.labels(raw = True)):
    if    (options.whitelist is None or     any([   position in table.label_indexrange(needle) \
                                                 or fnmatch.fnmatch(label,needle) for needle in options.whitelist])) \
      and (options.blacklist is None or not any([   position in table.label_indexrange(needle) \
                                                 or fnmatch.fnmatch(label,needle) for needle in options.blacklist])):  # a label to keep?
      labels.append(label)                                                                          # remember name...
      positions.append(position)                                                                    # ...and position

  if len(labels) > 0 and options.whitelist is not None and options.blacklist is None:               # check whether reordering is possible
    whitelistitem = np.zeros(len(labels),dtype=int)
    for i,label in enumerate(labels):                                                               # check each selected label
      match = [   positions[i] in table.label_indexrange(needle) \
               or fnmatch.fnmatch(label,needle) for needle in options.whitelist]                    # which whitelist items do match it
      whitelistitem[i] = match.index(True) if np.sum(match) == 1 else -1                            # unique match to a whitelist item --> store which

    sorted = np.lexsort(sortingList(labels,whitelistitem))
    order = range(len(labels)) if sorted[0] < 0 else sorted                                         # skip reordering if non-unique, i.e. first sorted is "-1"
  else:
    order = range(len(labels))                                                                      # maintain original order of labels
  
  interpolator = []
  condition = options.condition                                                                     # copy per file, might be altered
  for position,operand in enumerate(set(re.findall(r'#(([s]#)?(.+?))#',condition))):                # find three groups
    condition = condition.replace('#'+operand[0]+'#',
                                          {  '': '{{{}}}' .format(position),
                                           's#':'"{{{}}}"'.format(position)}[operand[1]])
    if operand[2] in specials:                                                                      # special label ?
      interpolator += ['specials["{}"]'.format(operand[2])]
    else:
      try:
        interpolator += ['{}(table.data[{}])'.format({  '':'float',
                                                      's#':'str'}[operand[1]],
                                                     table.label_index(operand[2]))]
      except:
        parser.error('column "{}" not found...\n'.format(operand[2]))

  evaluator = "'" + condition + "'.format(" + ','.join(interpolator) + ")"
  
# ------------------------------------------ assemble header ---------------------------------------

  table.info_append(scriptID + '\t' + ' '.join(sys.argv[1:]))
  table.labels_clear()
  table.labels_append(np.array(labels)[order])                                                      # update with new label set
  table.head_write()

# ------------------------------------------ process and output data ------------------------------------------

  positions = np.array(positions)[order]
  outputAlive = True
  while outputAlive and table.data_read():                                                          # read next data line of ASCII table
    specials['_row_'] += 1                                                                          # count row
    if condition == '' or eval(eval(evaluator)):                                                    # valid row ?
      table.data = [table.data[position] for position in positions]                                 # retain filtered columns
      outputAlive = table.data_write()                                                              # output processed line

# ------------------------------------------ finalize output -----------------------------------------

  table.close()                                                                                     # close input ASCII table (works for stdin)
