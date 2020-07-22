#pragma once


#include "kernel/kernel_base.hpp"
#include "utils/memory.hpp"

namespace splash { namespace kernel { 

/* TODO:
 * [ ] make buffer thread safe
 */

template<typename Op1, typename Op2>
class CompositeTransformKernel : public splash::kernel::transform<
    typename Op1::InputType, typename Op2::OutputType,
    splash::kernel::DEGREE::VECTOR>, 
    public splash::kernel::buffered_kernel<typename Op2::InputType> {

    protected:
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
            this->resize(count);  // ensure sufficient space.
            op1(in, count, this->buffer);
            op2(this->buffer, count, out);
		};
};




}}



