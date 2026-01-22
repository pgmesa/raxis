#pragma once

#include <raxis/array.hpp>
#include <raxis/array_types.hpp>


namespace raxis {

    struct Operand {
        void* data;
        Int* stride_bytes;
        bool is_output;
    };
    
    
    class ArrayIterator;
    class ArrayIteratorConfig;


    class ArrayIteratorConfig {
    public:
        friend struct ArrayIterator;
        
        ArrayIteratorConfig& add_output(const NDArray& output);
        ArrayIteratorConfig& add_input(const NDArray& input);

        ArrayIterator build();

    private:
        std::vector<const NDArray*> inputs_;
        std::vector<const NDArray*> outputs_;
        int num_inputs_ = 0;
        int num_outputs_ = 0;
    };


    struct ArrayIterator {
    public:
        size_t ndim() const;
        size_t numel() const;
        IntArray& shape() const;
        int num_inputs() const;
        int num_outputs() const;
        int num_operands() const;

        void build(const ArrayIteratorConfig& config);

        // template <typename Fn>
        // void ArrayIterator::for_each(Fn&& fn) {
        //     const int64_t N = numel_;
        //     char* ptrs[noperands_];

        //     for (int64_t i = 0; i < N; ++i) {
        //         compute_ptrs(i, ptrs);
        //         fn(ptrs);
        //     }
        // }

    private:
        std::vector<Operand> operands_;
        IntArray shape_;
        int num_outputs_ = 0;

        // void compute_ptrs(int op_idx, ptrs);
    };

}