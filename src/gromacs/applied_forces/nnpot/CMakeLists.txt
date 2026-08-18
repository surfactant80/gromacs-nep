#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2024- The GROMACS Authors
# and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
# Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# https://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at https://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out https://www.gromacs.org.

include(CheckLinkerFlag)

gmx_add_libgromacs_sources(
    binary_information.cpp
    nnpotoptions.cpp
    nnpot.cpp
)

if (GMX_TORCH)
    gmx_add_libgromacs_sources(
        nnpotforceprovider.cpp
        torchmodel.cpp
    )
    
    # find torch and python to explicitly link against: required for some torch extensions
    find_package(Python REQUIRED COMPONENTS Development)

    # -Wl,--no-as-needed needed to prevent the linker from removing the library if it is not directly used,
    # which is the case for torch extension libraries
    # we need to check if it is supported first
    check_linker_flag(CXX "-Wl,--no-as-needed" LINKER_SUPPORTS_NO_AS_NEEDED)
    if(LINKER_SUPPORTS_NO_AS_NEEDED)
        # the link interface should ideally be declared as PRIVATE here, but it can't be used until applied_forces is turned into a proper OBJECT module
        target_link_libraries(applied_forces INTERFACE -Wl,--no-as-needed ${Python_LIBRARIES})
    elseif(NOT FIND_TORCH_QUIETLY)
        message(WARNING "Linker does not support --no-as-needed flag. This may cause torch extension libraries to not be loaded corerctly.")
    endif()

    # The user is required to provide the path to any torch extension library which might be required by the NNP model they want to use
    set(TORCH_EXTENSION_PATH "" CACHE PATH "Path to PyTorch extension library")
    mark_as_advanced(TORCH_EXTENSION_PATH)

    if (TORCH_EXTENSION_PATH)
        if (NOT EXISTS ${TORCH_EXTENSION_PATH})
            message(FATAL_ERROR "TORCH_EXTENSION_PATH does not exist: ${TORCH_EXTENSION_PATH}")
        endif()

        # define target for the extension library and set properties
        find_library(TORCH_EXTENSION_LIB NAMES torch_extension PATHS ${TORCH_EXTENSION_PATH} REQUIRED NO_DEFAULT_PATH)
        message(STATUS "Found Pytorch extension library: ${TORCH_EXTENSION_LIB}")

        # link against the extension library
        target_link_libraries(applied_forces INTERFACE ${TORCH_EXTENSION_LIB})

    endif()
else()
    if (GMX_NNPOT STREQUAL "NEP")
        if (NOT GMX_GPU_CUDA)
            message(FATAL_ERROR "GMX_NNPOT=NEP requires a CUDA GROMACS build.")
        endif()

        # GPUMD is vendored as an unmodified source tree. Updating GPUMD only
        # requires replacing src/external/gpumd; all integration code stays in
        # this GROMACS module.
        set(GPUMD_VENDOR_ROOT "${PROJECT_SOURCE_DIR}/src/external/gpumd")
        set(GPUMD_SOURCE_ROOT "${GPUMD_VENDOR_ROOT}/src")
        if (NOT EXISTS "${GPUMD_VENDOR_ROOT}/LICENCE"
            OR NOT EXISTS "${GPUMD_SOURCE_ROOT}/force/nep.cu")
            message(FATAL_ERROR
                "GMX_NNPOT=NEP requires an unmodified GPUMD source tree at "
                "${GPUMD_VENDOR_ROOT}.")
        endif()

        set(GPUMD_NEP_VENDOR_CUDA_SOURCES
            ${GPUMD_SOURCE_ROOT}/force/dftd3.cu
            ${GPUMD_SOURCE_ROOT}/force/ewald.cu
            ${GPUMD_SOURCE_ROOT}/force/nep.cu
            ${GPUMD_SOURCE_ROOT}/force/nep_charge.cu
            ${GPUMD_SOURCE_ROOT}/force/neighbor.cu
            ${GPUMD_SOURCE_ROOT}/force/potential.cu
            ${GPUMD_SOURCE_ROOT}/force/pppm.cu
            ${GPUMD_SOURCE_ROOT}/model/box.cu
            ${GPUMD_SOURCE_ROOT}/utilities/error.cu
            ${GPUMD_SOURCE_ROOT}/utilities/read_file.cu
        )
        set_property(GLOBAL APPEND PROPERTY
            GMX_GPUMD_NEP_VENDOR_CUDA_SOURCES
            ${GPUMD_NEP_VENDOR_CUDA_SOURCES})

        gmx_add_libgromacs_sources(
            nnpotforceprovider.cpp
            nepmodel.cu
            nepmodel_qnep_gromacs.cu
            ${GPUMD_NEP_VENDOR_CUDA_SOURCES}
        )
        target_compile_definitions(applied_forces INTERFACE GMX_NEP=1)
        target_include_directories(applied_forces INTERFACE
            ${GPUMD_SOURCE_ROOT})
        target_link_libraries(applied_forces INTERFACE CUDA::cudart)
    else()
        gmx_add_libgromacs_sources(
            nnpotforceprovider_stub.cpp
        )
    endif()
endif()

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
