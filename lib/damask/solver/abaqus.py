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

from .solver import Solver
import damask

class Abaqus(Solver):

  def __init__(self,version='',solver=''):                                                          # example version string: 6.12-2, solver: std or exp
    self.solver='Abaqus'
    if version =='':
      version = damask.Environment().options['ABAQUS_VERSION']
    else:
      self.version = version
    
    if solver.lower() in ['','std','standard']:
      self.solver = 'std'
    elif solver.lower() in ['exp','explicit']:
      self.solver = 'exp'
    else:
      raise Exception('unknown Abaqus solver %'%solver)

  def return_run_command(self,model):
    import subprocess
    import re
    env=damask.Environment()
    shortVersion = re.sub('[\.,-]', '',self.version)
    try:
      cmd='abq'+shortVersion
      subprocess.check_output(['abq'+shortVersion,'information=release'])
    except OSError:                                                                                 # link to abqXXX not existing
      cmd='abaqus'
      process = subprocess.Popen(['abaqus','information=release'],stdout = subprocess.PIPE,stderr = subprocess.PIPE)
      detectedVersion = process.stdout.readlines()[1].split()[1]
      if self.version != detectedVersion:
        raise Exception('found Abaqus version %s, but requested %s'%(detectedVersion,self.version))
    return '%s -job %s -user %s/code/DAMASK_abaqus_%s interactive'%(cmd,model,env.rootDir(),self.solver)
