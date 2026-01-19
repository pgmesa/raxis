#pragma once

#include <vector>
#include <cstdint>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <raxis/array_types.hpp>


namespace pybind11::detail {

template <>
struct type_caster<raxis::IntArray> {
public:
    PYBIND11_TYPE_CASTER(raxis::IntArray, _("IntArray"));

    // Python -> C++
    bool load(handle src, bool) {
        value.clear();

        if (src.is_none()) {
            // None -> scalar tensor (empty shape)
            return true;
        }

        if (pybind11::isinstance<pybind11::int_>(src)) {
            value.push_back(pybind11::cast<raxis::Int>(src));
            return true;
        }

        if (pybind11::isinstance<pybind11::sequence>(src)) {
            for (auto item : pybind11::reinterpret_borrow<pybind11::sequence>(src)) {
                value.push_back(pybind11::cast<raxis::Int>(item));
            }
            return true;
        }

        return false;
    }

    // C++ -> Python
    static handle cast(
        const raxis::IntArray& src,
        return_value_policy,
        handle
    ) {
        pybind11::list lst;
        for (auto v : src) {
            lst.append(pybind11::int_(v));
        }
        return lst.release();
    }
};

}
