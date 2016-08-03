# sets up an environment for DAMASK on tcsh
# usage:  source DAMASK_env.csh

set CALLED=($_)
set DIRNAME=`dirname $CALLED[2]`
set DAMASK_ROOT=`python -c "import os,sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))" $DIRNAME`

source $DAMASK_ROOT/CONFIG

# if DAMASK_BIN is present and not in $PATH, add it
if ( $?DAMASK_BIN) then
  set MATCH=`echo :${PATH}: | grep ${DAMASK_BIN}:`
  if ( "x$MATCH" == "x" ) then
    set PATH=${DAMASK_BIN}:${PATH}
  endif
endif

set SOLVER=`which DAMASK_spectral`                                                          
set PROCESSING=`which postResults`                                                          
if ( "x$DAMASK_NUM_THREADS" == "x" ) then                                                                  
  set DAMASK_NUM_THREADS=1
endif

# according to http://software.intel.com/en-us/forums/topic/501500
# this seems to make sense for the stack size
set FREE=`which free`
set FREE=''
if ( "x$FREE" != "x" ) then
  set freeMem=`free -k | grep -E '(Mem|Speicher):' | awk '{print $4;}'`
  set heap=`expr $freeMem / 2`
  set stack=`expr $freeMem / $DAMASK_NUM_THREADS / 2`
  # http://superuser.com/questions/220059/what-parameters-has-ulimit             
  limit stacksize $stack # maximum stack size (kB)
  limit datasize  $heap  # maximum heap size (kB)
endif
limit coredumpsize 2000     # core  file size (512-byte blocks)
limit vmemoryuse unlimited  # maximum virtual memory size
limit memoryuse  unlimited  # maximum physical memory size

# disable output in case of scp
if($?prompt) then
  echo ''
  echo Düsseldorf Advanced Materials Simulation Kit - DAMASK
  echo Max-Planck-Institut für Eisenforschung, Düsseldorf
  echo https://damask.mpie.de
  echo
  echo Using environment with ...
  echo "DAMASK             $DAMASK_ROOT"
  echo "Spectral Solver    $SOLVER" 
  echo "Post Processing    $PROCESSING"
  echo "Multithreading     DAMASK_NUM_THREADS=$DAMASK_NUM_THREADS"
  if ( $?PETSC_DIR) then
    echo "PETSc location     $PETSC_DIR"
  endif
  if ( $?PETSC_ARCH) then
    echo "PETSc architecture $PETSC_ARCH"
  endif
  if ( $?MSC_ROOT) then
    echo "MSC.Marc/Mentat    $MSC_ROOT"
  endif
  echo
  echo `limit stacksize`
  echo `limit datasize`
endif

setenv DAMASK_NUM_THREADS $DAMASK_NUM_THREADS
setenv PYTHONPATH $DAMASK_ROOT/lib:$PYTHONPATH
