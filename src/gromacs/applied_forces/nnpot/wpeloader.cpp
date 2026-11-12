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
 */
#include "gmxpre.h"

#include "wpeloader.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include <dlfcn.h>

#include <filesystem>

#include "gromacs/utility/exceptions.h"

#include "config.h"

namespace gmx
{

const char* const WpeLoader::defaultLibraryName = "libwisevolve_potential.so";

namespace
{

std::mutex& loaderMutex()
{
    static std::mutex mutex;
    return mutex;
}

WpeLoader*& loaderSingleton()
{
    static WpeLoader* loader = nullptr;
    return loader;
}

void appendSearchDirectories(std::vector<std::filesystem::path>* directories)
{
    const auto appendPathList = [directories](const char* list)
    {
        if (list == nullptr)
        {
            return;
        }
        std::string paths(list);
        std::size_t begin = 0;
        while (begin <= paths.size())
        {
            const std::size_t end = paths.find(':', begin);
            const std::string entry = paths.substr(
                    begin, end == std::string::npos ? std::string::npos : end - begin);
            if (!entry.empty())
            {
                directories->emplace_back(entry);
            }
            if (end == std::string::npos)
            {
                break;
            }
            begin = end + 1;
        }
    };

    appendPathList(std::getenv("GMX_WPE_LIBRARY_PATH"));
    appendPathList(std::getenv("LD_LIBRARY_PATH"));

    std::error_code error;
    const auto      executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (!error)
    {
        directories->push_back(executable.parent_path());
        directories->push_back(executable.parent_path() / ".." / "lib");
        directories->push_back(executable.parent_path() / ".." / "lib64");
    }

    directories->emplace_back("/usr/local/lib");
    directories->emplace_back("/usr/local/lib64");
    directories->emplace_back("/usr/lib/x86_64-linux-gnu");
    directories->emplace_back("/usr/lib64");
    directories->emplace_back("/usr/lib");
    directories->emplace_back("/lib/x86_64-linux-gnu");
}

bool isRegularFile(const std::filesystem::path& path)
{
    std::error_code error;
    return std::filesystem::is_regular_file(path, error);
}

} // namespace

bool WpeLoader::isCompiledIn()
{
#ifdef GMX_NEP_WPE
    return true;
#else
    return false;
#endif
}

std::string WpeLoader::findLibrary(const std::string& hint)
{
    std::error_code error;

    const auto tryCandidate = [](const std::filesystem::path& candidate) -> std::string
    {
        if (isRegularFile(candidate))
        {
            return std::filesystem::absolute(candidate).string();
        }
        return {};
    };

    if (!hint.empty())
    {
        const std::filesystem::path hintPath(hint);
        if (std::filesystem::is_directory(hintPath, error))
        {
            const std::string found = tryCandidate(hintPath / defaultLibraryName);
            if (!found.empty())
            {
                return found;
            }
            for (const auto& entry : std::filesystem::directory_iterator(hintPath, error))
            {
                const std::string name = entry.path().filename().string();
                if (name.rfind("libwisevolve_potential.so", 0) == 0 && isRegularFile(entry.path()))
                {
                    return std::filesystem::absolute(entry.path()).string();
                }
            }
        }
        else
        {
            const std::string found = tryCandidate(hintPath);
            if (!found.empty())
            {
                return found;
            }
        }
    }

    if (const char* environmentPath = std::getenv("GMX_WPE_LIBRARY"))
    {
        if (environmentPath[0] != '\0')
        {
            const std::filesystem::path envPath(environmentPath);
            if (std::filesystem::is_directory(envPath, error))
            {
                const std::string found = tryCandidate(envPath / defaultLibraryName);
                if (!found.empty())
                {
                    return found;
                }
            }
            else
            {
                const std::string found = tryCandidate(envPath);
                if (!found.empty())
                {
                    return found;
                }
            }
        }
    }

    std::vector<std::filesystem::path> directories;
    appendSearchDirectories(&directories);
    for (const auto& directory : directories)
    {
        const std::string found = tryCandidate(directory / defaultLibraryName);
        if (!found.empty())
        {
            return found;
        }
        const std::string versioned = tryCandidate(directory / "libwisevolve_potential.so.1");
        if (!versioned.empty())
        {
            return versioned;
        }
    }

    // No dlopen() probe here: grompp calls findLibrary() and must not load the
    // proprietary runtime. A plain soname is still attempted by load() in
    // mdrun, which keeps ldconfig-managed installations working.
    return {};
}

WpeLoader* WpeLoader::instance(const std::string& hint, std::string* error)
{
    std::lock_guard<std::mutex> lock(loaderMutex());
    if (loaderSingleton() != nullptr)
    {
        return loaderSingleton();
    }

    auto loader = std::unique_ptr<WpeLoader>(new WpeLoader());
    std::string loadError;
    if (!loader->load(hint, &loadError))
    {
        if (error != nullptr)
        {
            *error = loadError;
        }
        return nullptr;
    }
    loaderSingleton() = loader.release();
    return loaderSingleton();
}

bool WpeLoader::isLoaded()
{
    std::lock_guard<std::mutex> lock(loaderMutex());
    return loaderSingleton() != nullptr;
}

WpeLoader::~WpeLoader()
{
    if (handle_ != nullptr)
    {
        dlclose(handle_);
        handle_ = nullptr;
    }
}

bool WpeLoader::load(const std::string& hint, std::string* error)
{
    if (!isCompiledIn())
    {
        *error = "this GROMACS build was compiled without WPE support "
                 "(configure with -DGMX_NEP_WPE=ON)";
        return false;
    }

    std::string path = findLibrary(hint);
    if (path.empty())
    {
        // mdrun may still find a soname that only the dynamic linker knows
        // (for example an ldconfig-managed installation).
        path = defaultLibraryName;
    }

    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr)
    {
        const char* dlError = dlerror();
        *error = "failed to load '" + path + "': " + (dlError != nullptr ? dlError : "unknown error");
        return false;
    }

    const auto resolve = [this, error, &path](const char* name) -> void*
    {
        dlerror();
        void* symbol = dlsym(handle_, name);
        if (symbol == nullptr)
        {
            *error = std::string("WPE library '") + path + "' does not export '" + name
                     + "' (unsupported WPE version?)";
        }
        return symbol;
    };

    void* getInfo = resolve("wpe_get_info");
    void* create = resolve("wpe_create");
    void* activateStage = resolve("wpe_activate_stage");
    void* bind = resolve("wpe_bind");
    void* compute = resolve("wpe_compute");
    void* destroy = resolve("wpe_destroy");
    if (getInfo == nullptr || create == nullptr || activateStage == nullptr || bind == nullptr
        || compute == nullptr || destroy == nullptr)
    {
        return false;
    }

    api_.getInfo       = reinterpret_cast<WpeStatus (*)(WpeInfo*)>(getInfo);
    api_.create        = reinterpret_cast<WpeStatus (*)(const WpeCreateInfo*, WpeCreateResult*,
                                                       WpeContext**)>(create);
    api_.activateStage = reinterpret_cast<WpeStatus (*)(WpeContext*, const WpeStageInfo*)>(
            activateStage);
    api_.bind   = reinterpret_cast<WpeStatus (*)(WpeContext*, const WpeBinding*)>(bind);
    api_.compute = reinterpret_cast<WpeStatus (*)(WpeContext*)>(compute);
    api_.destroy = reinterpret_cast<void (*)(WpeContext*)>(destroy);

    WpeInfo engineInfo{};
    engineInfo.struct_size = sizeof(engineInfo);
    const WpeStatus status = api_.getInfo(&engineInfo);
    if (status != WPE_OK)
    {
        *error = std::string("wpe_get_info failed: ") + statusName(status);
        return false;
    }
    if (engineInfo.struct_size < WPE_INFO_V1_MIN_SIZE)
    {
        *error = "WPE returned incompatible metadata (struct_size too small); "
                 "the library is older than the required ABI 1 interface";
        return false;
    }

    info_.name     = engineInfo.name != nullptr ? engineInfo.name : "";
    info_.version  = engineInfo.version != nullptr ? engineInfo.version : "";
    info_.provider = engineInfo.provider != nullptr ? engineInfo.provider : "";
    info_.soname   = engineInfo.library_soname != nullptr ? engineInfo.library_soname : "";
    info_.citation = engineInfo.citation != nullptr ? engineInfo.citation : "";
    info_.citationDoi = engineInfo.citation_doi != nullptr ? engineInfo.citation_doi : "";
    path_             = path;
    return true;
}

const char* WpeLoader::statusName(WpeStatus status)
{
    switch (status)
    {
        case WPE_OK: return "WPE_OK";
        case WPE_ERR_INVALID_ARGUMENT: return "WPE_ERR_INVALID_ARGUMENT";
        case WPE_ERR_UNSUPPORTED_MODEL: return "WPE_ERR_UNSUPPORTED_MODEL";
        case WPE_ERR_CUDA: return "WPE_ERR_CUDA";
        case WPE_ERR_NEIGHBOR_CAPACITY: return "WPE_ERR_NEIGHBOR_CAPACITY";
        case WPE_ERR_INTERNAL: return "WPE_ERR_INTERNAL";
        default: return "WPE_ERR_UNKNOWN";
    }
}

} // namespace gmx
