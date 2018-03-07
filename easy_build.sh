#!/bin/sh

tesse_root=$1
shift
b_type=$1
shift

source all_modules.sh

install_root=$tesse_root/install
mkdir -p $install_root

module_path=$tesse_root/modules
for x in $module_path; do
    case ":$MODULEPATH:" in
        *":$x:"*) :;; # already there
        *) MODULEPATH="$x:$MODULEPATH";;
    esac
done

export MODULEPATH=$MODULEPATH

module_root=$module_path/gnu/${b_type}
mkdir -p $module_root

module purge
module load mpi/openmpi/2.1.1-thread-multiple
module load plasma/2.8.0g
module load blas/mkl/2017.4
module load gcc/7.1.0
module load cmake/3.9.0
module load cuda/8.0
module load papi/5.5.1
module load hwloc/1.11.7

# Eigen
module load gnu/release/eigen

parsec_root=$tesse_root/parsec
parsec_build=$parsec_root/build_gnu_${b_type}
parsec_install=$install_root/gnu/${b_type}/parsec
parsec_module=$module_root/parsec
echo "parsec_root        = $parsec_root"
echo "parsec_build       = $parsec_build"
echo "parsec_install     = $parsec_install"
echo "parsec_module      = $parsec_module"

madness_root=$tesse_root/madness
madness_build=$madness_root/build_gnu_${b_type}
madness_install=$install_root/gnu/${b_type}/madness
madness_module=$module_root/madness
echo "madness_root       = $madness_root"
echo "madness_build      = $madness_build"
echo "madness_install    = $madness_install"
echo "madness_module     = $madness_module"

tiledarray_root=$tesse_root/tiledarray
tiledarray_build=$tiledarray_root/build_gnu_${b_type}
tiledarray_install=$install_root/gnu/${b_type}/tiledarray
tiledarray_module=$module_root/tiledarray
echo "tiledarray_root    = $tiledarray_root"
echo "tiledarray_build   = $tiledarray_build"
echo "tiledarray_install = $tiledarray_install"
echo "tiledarray_module  = $tiledarray_module"

parsec='0'
madness='0'
tiledarray='1'

# PaRSEC
if [ "x$parsec" = "x1" ]; then
    cd $parsec_root
    rm -rf $parsec_build
    mkdir -p $parsec_build
    cd $parsec_build

    cmake -DCMAKE_INSTALL_PREFIX=$parsec_install \
          -DCMAKE_BUILD_TYPE=$btype \
          -DMPI_C_COMPILER=$(which mpicc 2>/dev/null) \
          -DDPLASMA_PRECISIONS="d;s;c;z" \
          -DPARSEC_WITH_DEVEL_HEADERS=ON \
          -DPYTHON_EXECUTABLE=$(which python 2>/dev/null) \
          $parsec_root

    make parsec_ptgpp
    VERBOSE=1 make -j 12
    rm -rf $parsec_install
    make install

    parsec_module
fi
module load gnu/${b_type}/parsec

# Building MADNESS

export MADNESS_ROOT=$tesse_root/madness
export MADNESS_INSTALL_DIR=$madness_install

if [ "x$madness" = "x1" ]; then
    old_cc=$CC
    old_fc=$FC
    old_cxx=$CXX
    export CC=mpicc
    export CXX=mpicxx
    export FC=mpif90

    cd $madness_root
    rm -rf $madness_build
    mkdir -p $madness_build
    cd $madness_build

    cmake -DCMAKE_INSTALL_PREFIX=$madness_install \
          -DCMAKE_BUILD_TYPE=$b_type \
          -DCMAKE_TOOLCHAIN_FILE=$tesse_root/madness/cmake/toolchains/gnu-mkl.cmake \
          -DLAPACK_LIBRARIES="-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl" \
          -DDISABLEPIE_LINKER_FLAG="" \
          -DENABLE_PARSEC=ON \
          -DENABLE_TBB=OFF \
          -DCUDA_TOOLKIT_ROOT_DIR=/ \
          -DCUDA_SDK_ROOT_DIR=/ \
          -DCUDA_HOST_COMPILER=false \
          $madness_root

    make -j 8
    make install

    export CC=$old_cc
    export CXX=$old_cxx
    export FC=$old_fc

    madness_module
fi
module load gnu/${b_type}/madness

# Building TiledArray
if [ "x$tiledarray" = "x1" ]; then

    cd $tiledarray_root
    rm -rf $tiledarray_build
    mkdir -p $tiledarray_build
    cd $tiledarray_build

    cmake -DCMAKE_INSTALL_PREFIX=$tiledarray_install \
          -DCMAKE_BUILD_TYPE=$b_type \
          -DCMAKE_TOOLCHAIN_FILE=$tesse_root/madness/cmake/toolchains/gnu-mkl.cmake \
          -DLAPACK_LIBRARIES=" -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl" \
          -DENABLE_TBB=OFF \
          -DCMAKE_PREFIX_PATH=$madness_install \
          -DMADNESS_ROOT_DIR=$tesse_root/madness \
          -DEIGEN3_INCLUDE_DIR=$EIGEN_INSTALL_DIR/include \
          -DPAPI_DIR=/sw/papi/5.5.1 \
          $tiledarray_root

    make -j 4

    # if configure step yells at you for a missing parsec_wrapper.h
    # go into $tesse_root/tiledarray/examples/dgemm/CMakelists.txt,
    # look for the file name, remove it
    cd examples/dgemm
    make ta_cc_abcd

    tiledarray_module
fi
module load gnu/${b_type}/tiledarray
