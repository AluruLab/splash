#pragma once

#include "CLI/CLI.hpp"

class parameters_base {
    public:
        virtual void config(CLI::App& app) = 0;
        virtual void print(const char* prefix) = 0;

        parameters_base() {};
        virtual ~parameters_base() {};

        // TO PARSE, USE
        // CLI11_PARSE(app, argc, argv)
};
