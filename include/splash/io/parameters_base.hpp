/*
 * Copyright 2020 Georgia Tech Research Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Author(s): Tony C. Pan
 */

#pragma once

#include "CLI/CLI.hpp"

class parameters_base {
    public:
        virtual void config(CLI::App& app) = 0;
        virtual void print(const char* prefix) const = 0;

        parameters_base() {};
        virtual ~parameters_base() {};

        // TO PARSE, USE
        // CLI11_PARSE(app, argc, argv)
};
