#include <filesystem>

#include "shared.h"

using namespace std::filesystem;
using namespace shared;
using namespace std;

unique_ptr<globals> globals::_instance = nullptr;

globals::globals() {}

const unique_ptr<globals> &globals::get_instance()
{
    if (!_instance)
    {
        globals::_instance = unique_ptr<globals>(new globals());
        globals::_instance->init();
    }

    return globals::_instance;
}

const path &globals::get_cwd() const
{
    return _cwd;
}

const path &shared::globals::get_res() const
{
    return _res;
}

void globals::init()
{
    _cwd = canonical("/proc/self/exe");
    _res = _cwd.parent_path() / "resources";
}
