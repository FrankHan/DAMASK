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

# Makes postprocessing routines acessible from everywhere.
import os,sys
import damask

damaskEnv = damask.Environment()
baseDir = damaskEnv.relPath('processing/')
binDir = damaskEnv.options['DAMASK_BIN']

if not os.path.isdir(binDir):
  os.mkdir(binDir)

#define ToDo list
processing_subDirs = ['pre','post','misc',]
processing_extensions = ['.py','.sh',]
            
for subDir in processing_subDirs:
  theDir = os.path.abspath(os.path.join(baseDir,subDir))

  for theFile in os.listdir(theDir):
    if os.path.splitext(theFile)[1] in processing_extensions:                           # only consider files with proper extensions

      src      = os.path.abspath(os.path.join(theDir,theFile))
      sym_link = os.path.abspath(os.path.join(binDir,os.path.splitext(theFile)[0]))

      if os.path.lexists(sym_link):
        os.remove(sym_link)
        sys.stdout.write(sym_link)
      else:
        sys.stdout.write(damask.util.emph(sym_link))

      os.symlink(src,sym_link)
      sys.stdout.write(' -> '+src+'\n')
