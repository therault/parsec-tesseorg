#!/bin/sh

parsec_module()
{
    cd $tesse_root

    if [ -f $parsec_module ]; then
        rm -f $parsec_module
    fi

    echo "#%Module1.0#####################################################################
##
## modules PaRSEC 1.0.0 for GNU compilers
##
## modulefiles/gnu/${b_type}/parsec
##
proc ModulesHelp { } {
        global version modroot

        puts stderr \"PaRSEC 1.0.0\"
}

module-whatis   \"Sets the environment for using PaRSEC compiled with GCC\"

# for Tcl script use only
set     topdir          $parsec_install
set     version         1.0.0
set     sys             linux86

setenv          PARSEC_PREFIX   \$topdir
prepend-path    INCLUDE_DIR     \$topdir/include
prepend-path    PATH            \$topdir/include
prepend-path    LD_LIBRARY_PATH \$topdir/lib
prepend-path    PKG_CONFIG_PATH \$topdir/lib/pkgconfig
prepend-path    PKG_CONFIG_PATH \$topdir/dplasma/lib/pkgconfig
" > $parsec_module
}

madness_module()
{
    cd $tesse_root

    if [ -e $madness_module ]; then
        rm -f $madness_module
    fi

    echo "#%Module1.0#####################################################################
##
## modules MADNESS 1.0.0 for GNU compilers
##
## modulefiles/gnu/${b_type}/madness
##
proc ModulesHelp { } {
        global version modroot

        puts stderr \"MADNESS 1.0.0\"
}

module-whatis   \"Sets the environment for using MADNESS compiled with GCC\"

# for Tcl script use only
set     topdir          $MADNESS_INSTALL_DIR
set     version         1.0.0
set     sys             linux86

prepend-path    INCLUDE_DIR     \$topdir/include
prepend-path    PATH            \$topdir/include
prepend-path    LD_LIBRARY_PATH \$topdir/lib
prepend-path    PKG_CONFIG_PATH \$topdir/lib/pkgconfig
prepend-path    CPLUS_INCLUDE_PATH \$topdir/include/madness/world
" > $madness_module
}

tiledarray_module()
{
    cd $tesse_root

    if [ -e $tiledarray_module ]; then
        rm -f $tiledarray_module
    fi

    echo "#%Module1.0#####################################################################
##
## modules TiledArray 1.0.0 for GNU compilers
##
## modulefiles/gnu/${b_type}/tiledarray
##
proc ModulesHelp { } {
        global version modroot

        puts stderr \"TiledArray 1.0.0\"
}

module-whatis   \"Sets the environment for using TILEDARRAY compiled with GCC\"

# for Tcl script use only
set     topdir          $tiledarray_install
set     version         1.0.0
set     sys             linux86

prepend-path    INCLUDE_DIR     \$topdir/include
prepend-path    PATH            \$topdir/include
prepend-path    LD_LIBRARY_PATH \$topdir/lib
prepend-path    PKG_CONFIG_PATH \$topdir/lib/pkgconfig
" > $tiledarray_module
}
