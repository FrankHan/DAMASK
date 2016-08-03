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
import damask

bin_link = { \
            '.' : [
                    'DAMASK_spectral.exe',
                    'DAMASK_FEM.exe'
                  ],
           }

MarcReleases =[2011,2012,2013,2013.1,2014,2014.2,2015]

damaskEnv = damask.Environment()
baseDir = damaskEnv.relPath('code/')
binDir  = damaskEnv.options['DAMASK_BIN']

if not os.path.isdir(binDir):
  os.mkdir(binDir)

for dir in bin_link:
  for file in bin_link[dir]:
    src = os.path.abspath(os.path.join(baseDir,dir,file))
    if os.path.exists(src): 
      sym_link = os.path.abspath(os.path.join(binDir,\
                                              {True: dir,
                                               False:os.path.splitext(file)[0]}[file == '']))
      if os.path.lexists(sym_link): os.remove(sym_link)
      os.symlink(src,sym_link)
      sys.stdout.write(sym_link+' -> '+src+'\n')


for version in MarcReleases:
  src = os.path.abspath(os.path.join(baseDir,'DAMASK_marc.f90'))
  if os.path.exists(src): 
    sym_link = os.path.abspath(os.path.join(baseDir,'DAMASK_marc'+str(version)+'.f90'))                    
    if os.path.lexists(sym_link):
      os.remove(sym_link)
      sys.stdout.write(sym_link)
    else:
      sys.stdout.write(damask.util.emph(sym_link))

    os.symlink(src,sym_link)
    sys.stdout.write(' -> '+src+'\n')
