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

"""
Writes meaningful labels to the marc input file (*.dat) 

output is based on the files 
<modelname_jobname>.output<Homogenization/Crystallite/Constitutive>
that are written during the first run of the model.
"""
import sys,os,re
from optparse import OptionParser
import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

# -----------------------------
def ParseOutputFormat(filename,what,me):
  format = {'outputs':{},'specials':{'brothers':[]}}

  outputmetafile = filename+'.output'+what
  try:
    file = open(outputmetafile)
  except:
    print('Could not open file %s'%outputmetafile)
    raise
  else: 
    content = file.readlines()
    file.close()

  tag = ''
  tagID = 0
  for line in content:
    if re.match("\s*$",line) or re.match("#",line):     # skip blank lines and comments
      continue
    m = re.match("\[(.+)\]",line)             # look for block indicator
    if m:                         # next section
      tag = m.group(1)
      tagID += 1
      format['specials']['brothers'].append(tag)
      if tag == me or (me.isdigit() and tagID == int(me)):
        format['specials']['_id'] = tagID
        format['outputs'] = []
        tag = me
    else:                         # data from section
      if tag == me:
        (output,length) = line.split()
        output.lower()
        if length.isdigit():
          length = int(length)
        if re.match("\((.+)\)",output):         # special data, (e.g. (Ngrains)
          format['specials'][output] = length
        elif length > 0:
          format['outputs'].append([output,length])
  return format

parser = OptionParser(option_class=damask.extendableOption, usage='%prog [option(s)] Marc.Inputfile(s)', description="""
Transfer the output variables requested in the material.config to
properly labelled user defined variables within the Marc input file (*.dat).

Requires the files 
<modelname_jobname>.output<Homogenization/Crystallite/Constitutive>
that are written during the first run of the model.

Specify which user block format you want to apply by stating the homogenization, crystallite, and phase identifiers.
Or have an existing set of user variables copied over from another *.dat file.

""", version = scriptID)


parser.add_option('-n','--number', dest='number', type='int', \
          metavar='int',
          help='maximum requested User Defined Variable [%default]')
parser.add_option('--homogenization', dest='homog', \
          metavar='string',
          help='homogenization identifier (as string or integer [%default])')
parser.add_option('--crystallite', dest='cryst', \
          metavar='string',
          help='crystallite identifier (as string or integer [%default])')
parser.add_option('--phase', dest='phase', \
          metavar='string',
          help='phase identifier (as string or integer [%default])')
parser.add_option('--use', dest='useFile', \
          metavar='string',
          help='Optionally parse output descriptors from '+
               'different <model_job>.outputZZZ file. Saves the effort '+
               'to start a calculation for each job)')
parser.add_option('--option', dest='damaskOption', \
          metavar='string',
          help='Add damask option to input file '+
               'for example: "periodic x z"')
parser.set_defaults(number = 0)
parser.set_defaults(homog = '1')
parser.set_defaults(cryst = '1')
parser.set_defaults(phase = '1')
parser.set_defaults(useFile = '')
parser.set_defaults(damaskOption = '')

(options, files) = parser.parse_args()

if not files:
  parser.print_help()
  parser.error('no file(s) specified...')

me = {  'Homogenization':   options.homog,
        'Crystallite':      options.cryst,
        'Constitutive':     options.phase,
     }


for file in files:
  print '\033[1m'+scriptName+'\033[0m: '+file+'\n'
  if options.useFile != '':
    formatFile = os.path.splitext(options.useFile)[0]
  else:
    formatFile = os.path.splitext(file)[0]
  file = os.path.splitext(file)[0]+'.dat'
  if not os.path.lexists(file):
    print file,'not found'
    continue
    
  print('Scanning format files of: %s'%formatFile)

  if options.number < 1:
    outputFormat = {}

    for what in me:
      outputFormat[what] = ParseOutputFormat(formatFile,what,me[what])
      if '_id' not in outputFormat[what]['specials']:
        print "'%s' not found in <%s>"%(me[what],what)
        print '\n'.join(map(lambda x:'  '+x,outputFormat[what]['specials']['brothers']))
        sys.exit(1)

    UserVars = ['HomogenizationCount']
    for var in outputFormat['Homogenization']['outputs']:
      if var[1] > 1:
        UserVars += ['%i_%s'%(i+1,var[0]) for i in range(var[1])]
      else:
        UserVars += ['%s'%(var[0]) for i in range(var[1])]

    UserVars += ['GrainCount']

    for grain in range(outputFormat['Homogenization']['specials']['(ngrains)']):
      UserVars += ['%i_CrystalliteCount'%(grain+1)]
      for var in outputFormat['Crystallite']['outputs']:
        if var[1] > 1:
          UserVars += ['%i_%i_%s'%(grain+1,i+1,var[0]) for i in range(var[1])]
        else:
          UserVars += ['%i_%s'%(grain+1,var[0]) for i in range(var[1])]

      UserVars += ['%i_ConstitutiveCount'%(grain+1)]
      for var in outputFormat['Constitutive']['outputs']:
        if var[1] > 1:
          UserVars += ['%i_%i_%s'%(grain+1,i+1,var[0]) for i in range(var[1])]
        else:
          UserVars += ['%i_%s'%(grain+1,var[0]) for i in range(var[1])]

# Now change *.dat file(s)        
  print('Adding labels to:         %s'%file)
  inFile = open(file)
  input = inFile.readlines()
  inFile.close()
  output = open(file,'w')
  thisSection = ''
  if options.damaskOption:
    output.write('$damask {0}\n'.format(options.damaskOption))
  for line in input:
    m = re.match('(\w+)\s',line)
    if m:
      lastSection = thisSection
      thisSection = m.group(1)
      if (lastSection == 'post' and thisSection == 'parameters'):
        if options.number > 0:
          for i in range(options.number): 
             output.write('%10i%10i\n'%(-i-1,0))
        else:
          for i in range(len(UserVars)): 
             output.write('%10i%10i%s\n'%(-i-1,0,UserVars[i]))
    if (thisSection != 'post' or not re.match('\s*\-',line)):
      output.write(line)
  output.close()
