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
# sets up an environment for DAMASK on bash
# usage:  source DAMASK_env.sh


if [ "$OSTYPE" == "linux-gnu" ] || [ "$OSTYPE" == 'linux' ]; then
  DAMASK_ROOT=$(python -c "import os,sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))" "$(dirname $BASH_SOURCE)")
else
  [[ "${BASH_SOURCE::1}" == "/" ]] && BASE="" || BASE="`pwd`/"
  STAT=$(stat "`dirname $BASE$BASH_SOURCE`")
  DAMASK_ROOT=${STAT##* }
fi

# defining set() allows to source the same file for tcsh and bash, with and without space around =
set() {
    export $1$2$3
 }
source $DAMASK_ROOT/CONFIG
unset -f set

# if DAMASK_BIN is present and not in $PATH, add it
if [[ "x$DAMASK_BIN" != "x" && ! `echo ":$PATH:" | grep $DAMASK_BIN:` ]]; then
  export PATH=$DAMASK_BIN:$PATH
fi

SOLVER=`which DAMASK_spectral 2>/dev/null`
if [ "x$SOLVER" == "x" ]; then
  SOLVER='Not found!'
fi
PROCESSING=`which postResults 2>/dev/null`
if [ "x$PROCESSING" == "x" ]; then
  PROCESSING='Not found!'
fi
if [ "x$DAMASK_NUM_THREADS" == "x" ]; then
  DAMASK_NUM_THREADS=1
fi

# according to http://software.intel.com/en-us/forums/topic/501500
# this seems to make sense for the stack size
FREE=`which free 2>/dev/null`
if [ "x$FREE" != "x" ]; then
  freeMem=`free -k | grep -E '(Mem|Speicher):' | awk '{print $4;}'`
  heap=`expr $freeMem / 2`
  stack=`expr $freeMem / $DAMASK_NUM_THREADS / 2`

  # http://superuser.com/questions/220059/what-parameters-has-ulimit             
  ulimit -s $stack      2>/dev/null # maximum stack size (kB)
  ulimit -d $heap       2>/dev/null # maximum heap size (kB)
fi
ulimit -c 2000        2>/dev/null # core  file size (512-byte blocks)
ulimit -v unlimited   2>/dev/null # maximum virtual memory size
ulimit -m unlimited   2>/dev/null # maximum physical memory size

# disable output in case of scp
if [ ! -z "$PS1" ]; then
  echo
  echo Düsseldorf Advanced Materials Simulation Kit - DAMASK
  echo Max-Planck-Institut für Eisenforschung, Düsseldorf
  echo https://damask.mpie.de
  echo
  echo Using environment with ...
  echo "DAMASK             $DAMASK_ROOT"
  echo "Spectral Solver    $SOLVER" 
  echo "Post Processing    $PROCESSING"
  echo "Multithreading     DAMASK_NUM_THREADS=$DAMASK_NUM_THREADS"
  if [ "x$PETSC_DIR"   != "x" ]; then
    echo "PETSc location     $PETSC_DIR"
    [[ `python -c "import os,sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))" "$PETSC_DIR"` == $PETSC_DIR ]] \
    || echo "               ~~> "`python -c "import os,sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))" "$PETSC_DIR"`
  fi
  [[ "x$PETSC_ARCH"  != "x" ]] && echo "PETSc architecture $PETSC_ARCH"
  echo "MSC.Marc/Mentat    $MSC_ROOT"
  echo
  echo -n "heap  size/MiB     "; echo "`ulimit -d`/1024" | bc
  echo -n "stack size/MiB     "; echo "`ulimit -s`/1024" | bc
fi

export DAMASK_NUM_THREADS
export PYTHONPATH=$DAMASK_ROOT/lib:$PYTHONPATH

for var in BASE STAT SOLVER PROCESSING FREE DAMASK_BIN; do
  unset "${var}"
done
for var in DAMASK MSC; do
  unset "${var}_ROOT"
done
for var in ABAQUS MARC; do
  unset "${var}_VERSION"
done
