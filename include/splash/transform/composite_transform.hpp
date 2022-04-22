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


#include "splash/kernel/kernel_base.hpp"
#include "splash/ds/buffer.hpp"

namespace splash { namespace kernel { 

/* TODO:
 * [ ] make buffer thread safe
 */

template<typename Op1, typename Op2>
class CompositeTransformKernel : public splash::kernel::transform<
    typename Op1::InputType, typename Op2::OutputType,
    splash::kernel::DEGREE::VECTOR> {

    protected:
        mutable splash::ds::buffer<typename Op2::InputType> __buffer;
        using MT = typename Op2::InputType; 

        Op1 op1;
        Op2 op2; 

	public:
        using InputType = typename Op1::InputType;
        using OutputType = typename Op2::OutputType;

        CompositeTransformKernel() {}        
        CompositeTransformKernel(Op1 const & _op1, Op2 const & _op2) : op1(_op1), op2(_op2) {} 
        virtual ~CompositeTransformKernel() { }

        void copy_parameters(CompositeTransformKernel const & other) {
            op1.copy_parameters(other.op1);
            op2.copy_parameters(other.op2);
        }

		inline virtual void operator()(InputType const * in, size_t const & count, OutputType * out) const  {
            __buffer.resize(count);  // ensure sufficient space.
            op1(in, count, __buffer.data);
            op2(__buffer.data, count, out);
		};
};




}}



