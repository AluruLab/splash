#pragma once


#include "kernel/kernel_base.hpp"
#include "utils/memory.hpp"

namespace splash { namespace kernel { 


template<typename Op1, typename Op2>
class CompositeTransformKernel : public splash::kernel::V2VOp<
    typename Op1::InputType, typename Op2::OutputType> {

    protected:
        using MT = typename Op2::InputType; 

        Op1  const & op1;
        Op2  const & op2; 
        mutable MT * buffer; 
        mutable size_t vecSize;

	public:
        using InputType = typename Op1::InputType;
        using OutputType = typename Op2::OutputType;

        CompositeTransformKernel(Op1 const & _op1, Op2 const & _op2, size_t const & _vecSize) : 
            op1(_op1), op2(_op2), vecSize(_vecSize) {
            buffer = reinterpret_cast<MT* >(splash::utils::aalloc(_vecSize * sizeof(MT)));
        }
        ~CompositeTransformKernel() {
            if (buffer) splash::utils::afree(buffer);
        }
        void resize(size_t const & count) const {
            if (count > vecSize) {
                if (buffer) splash::utils::afree(buffer);
                buffer = reinterpret_cast<MT* >(splash::utils::aalloc(count * sizeof(MT)));
                vecSize = count;
            }
        }


		inline void operator()(InputType const * in, size_t const & count, OutputType * out) const  {
            this->resize(count);
            op1(in, count,this->buffer);
            op2(this->buffer, count, out);
		};
};




}}



