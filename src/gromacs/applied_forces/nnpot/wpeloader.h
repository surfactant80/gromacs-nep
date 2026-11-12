/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * This file integrates the Wisevolve Potential Engine (WPE). Only the public
 * WPE C interface (src/external/wpe/wisevolve_potential_api.h, BSD-2-Clause)
 * is used here. The proprietary engine is loaded at run time with dlopen();
 * GROMACS itself is never linked against libwisevolve_potential.so, so a
 * build without WPE installed keeps working and the normal NEP backends stay
 * available.
 */
#ifndef GMX_APPLIED_FORCES_NNPOT_WPELOADER_H
#define GMX_APPLIED_FORCES_NNPOT_WPELOADER_H

#include <string>

#include "wisevolve_potential_api.h"

namespace gmx
{

/*! \brief Metadata reported by wpe_get_info(). */
struct WpeEngineInfo
{
    std::string name;
    std::string version;
    std::string provider;
    std::string soname;
    std::string citation;
    std::string citationDoi;
};

/*! \brief Resolved entry points of the WPE shared library. */
struct WpeApi
{
    WpeStatus (*getInfo)(WpeInfo* info)                                 = nullptr;
    WpeStatus (*create)(const WpeCreateInfo*, WpeCreateResult*, WpeContext**) = nullptr;
    WpeStatus (*activateStage)(WpeContext*, const WpeStageInfo*)         = nullptr;
    WpeStatus (*bind)(WpeContext*, const WpeBinding*)                    = nullptr;
    WpeStatus (*compute)(WpeContext*)                                    = nullptr;
    void (*destroy)(WpeContext*)                                         = nullptr;
};

/*! \brief Process-wide loader for libwisevolve_potential.so.
 *
 * The library is opened lazily on first use. Loading never initializes CUDA:
 * wpe_create() is the only call that allocates device resources, so grompp
 * can inspect availability without touching the GPU runtime.
 */
class WpeLoader
{
public:
    //! Hard-coded library names tried when no explicit path is given.
    static const char* const defaultLibraryName;

    /*! \brief Locate the WPE shared library on disk without loading it.
     *
     * Search order: \p hint (mdp option), $GMX_WPE_LIBRARY, the default soname
     * through the dynamic linker search path, and finally a small set of
     * common installation directories. Returns an empty string when nothing is
     * found. This function has no CUDA side effects and is safe during grompp.
     */
    static std::string findLibrary(const std::string& hint);

    /*! \brief Return the loaded library, or nullptr with \p error filled in.
     *
     * The first successful call wins; later calls with different paths return
     * the already loaded library (a process cannot switch CUDA engines
     * safely). Only initialized contexts are engine-specific.
     */
    static WpeLoader* instance(const std::string& hint, std::string* error);

    //! Whether the library has been loaded successfully.
    static bool isLoaded();

    //! Whether GROMACS was compiled with WPE backend support.
    static bool isCompiledIn();

    //! Entry points; valid after a successful instance() call.
    const WpeApi& api() const { return api_; }

    //! Engine metadata, read once at load time.
    const WpeEngineInfo& info() const { return info_; }

    //! Path the library was actually loaded from.
    const std::string& path() const { return path_; }

    //! Translate a WpeStatus into a human readable string.
    static const char* statusName(WpeStatus status);

    //! Releases the dlopen() handle (kept for the whole process lifetime).
    ~WpeLoader();

private:
    WpeLoader() = default;
    WpeLoader(const WpeLoader&)            = delete;
    WpeLoader& operator=(const WpeLoader&) = delete;

    bool load(const std::string& hint, std::string* error);

    void*    handle_ = nullptr;
    WpeApi   api_;
    WpeEngineInfo info_;
    std::string   path_;
};

} // namespace gmx

#endif
