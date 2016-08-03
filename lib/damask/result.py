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


import numpy as np
#import sys

try:
  import h5py
except (ImportError) as e:
  pass #  sys.stderr.write('\nREMARK: h5py module not available \n\n')

class Result():
  """
  General class for result parsing.

  Needs h5py to be installed
  """
  
  def __init__(self,resultsFile):
    self.data=h5py.File(resultsFile,"r")
    self.Npoints=self.data.attrs['Number of Materialpoints']
    print("Opened "+resultsFile+" with %i points"%self.Npoints)

  def getCrystallite(self,labels,inc,constituent=1,points=None):
    if points is None: points = np.array(np.array(xrange(self.Npoints)))
    results = {}
    mapping=self.data['mapping/crystallite']
    for instance in self.data['increments/%s/crystallite/'%inc]:
      dsets = self.data['increments/%s/crystallite/%s'%(inc,instance)].keys()
      for label in labels:
        if label in dsets and label not in results:
          shape = np.shape(self.data['increments/%s/crystallite/%s/%s'%(inc,instance,label)])[1:]
          results[label] = np.nan*np.ones(np.array((self.Npoints,)+shape))
    for myPoint in xrange(len(points)):
      matPoint = points[myPoint]
      pos = mapping[matPoint,constituent-1]
      if pos[0] != 0:
        try:
          for label in labels:
            results[label][matPoint,...] = self.data['increments/%s/crystallite/%s/%s'%(inc,pos[0],label)][pos[1],...]
        except:
          pass
    return results


