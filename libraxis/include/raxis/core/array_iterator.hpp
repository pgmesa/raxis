#pragma once

#include <ostream>
#include <iostream>

#include <raxis/array.hpp>
#include <raxis/array_types.hpp>


namespace raxis {

    struct Operand {

        IntArray stride_bytes;
        bool is_output = false;
        bool is_read_write = false;
        bool will_resize = false;
        // std::optional<Device>
        // Dtype target_dtype;
        // Dtype current_dtype;

        explicit Operand(const NDArray* arr) {
            // if (arr->defined()) {
            //     // TODO
            // }
            array_ = arr;
            //target_dtype = arr->dtype();
            //current_dtype = target_dtype;
        }

        const NDArray& array() const {
            return *array_;
        }

    private:
        const NDArray* array_;
    };
    
    
    class ArrayIterator;
    class ArrayIteratorConfig;


    class ArrayIteratorConfig {
    public:
        friend struct ArrayIterator;
        
        ArrayIteratorConfig& add_output(const NDArray& output);
        ArrayIteratorConfig& add_input(const NDArray& input);
        ArrayIteratorConfig& is_reduction(bool is_reduction);

        ArrayIterator build();

    private:
        std::vector<const NDArray*> arrays_;
        int num_outputs_ = 0;
        int num_inputs_ = 0;
        bool is_reduction_ = false;
    };


    struct ArrayIterator {
    public:
        int ndim() const;
        int64_t numel() const;
        IntArray& shape() const;
        int num_inputs() const;
        int num_outputs() const;
        int num_operands() const;

        void build(const ArrayIteratorConfig& config);
        void print_state(std::ostream& os = std::cout) const;
        
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
        int num_outputs_ = 0;
        
        IntArray shape_;
        // IntArray perm_;

        bool all_ops_same_shape_ = false;
        bool all_ops_are_scalars_ = false;

        // Dtype common_dtype_ = Dtype::Undefined;
        // common_device_

        bool is_reduction_ = false;

        // void compute_ptrs(int op_idx, ptrs);

        void populate_operands(const ArrayIteratorConfig&);
        void compute_shape();
        void compute_types();
        void compute_strides();
        void reorder_dimensions();
        void allocate_or_resize_outputs();
        void coalesce_dimensions();
    };

}