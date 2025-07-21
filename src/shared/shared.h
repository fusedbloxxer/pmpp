#pragma once

#include <filesystem>
#include <memory>

namespace shared
{
    using namespace std;
    using namespace std::filesystem;

    class globals final
    {
      private:
        globals();

      public:
        static const unique_ptr<globals> &get_instance();
        const path &get_cwd() const;
        const path &get_res() const;

      public:
        void operator=(const globals &) = delete;
        globals(globals &other) = delete;

      private:
        static unique_ptr<globals> _instance;
        void init();
        path _res;
        path _cwd;
    };
}
