#pragma once

#include <functional>
#include <iostream>
#include <sstream>
#include <string>

#include <libalgebra/libalgebra.h>

using namespace alg;

template<DEG WIDTH, DEG DEPTH>
struct Environment {
    using scalar_field = coefficients::rational_field;
    using S = typename scalar_field::S;

    static constexpr DEG WIDTH = WIDTH;
    static constexpr DEG DEPTH = DEPTH;
    static constexpr DEG poly_width = hall_basis<WIDTH, DEPTH>::start_of_degree(DEPTH + 1);

    using poly_t = alg::poly<scalar_field>;
    using poly_coeffs = coefficients::coefficient_ring<poly_t, typename scalar_field::Q>;
    static_assert(std::is_same<poly_coeffs::S, poly_t>::value, "the trait class of a coefficient ring must report the type of the coefficients");

    template<DEG DEPTH>
    using LIE_ = alg::lie<poly_coeffs, WIDTH, DEPTH, vectors::dense_vector>;
    using LIE = LIE_<DEPTH>;

    template<DEG DEPTH>
    using lie_basis_t_ = lie_basis<WIDTH, DEPTH>;

    using lie_basis_t = lie_basis_t_<WIDTH>;
    lie_basis_t lbasis;

    template<DEG DEPTH>
    using TENSOR_ = alg::free_tensor<poly_coeffs, WIDTH, DEPTH, vectors::dense_vector>;
    using TENSOR = TENSOR_<DEPTH>;

    template<DEG DEPTH>
    using tensor_basis_t_ = alg::tensor_basis<WIDTH, DEPTH>;

    using tensor_basis_t = tensor_basis_t_< DEPTH>;
    tensor_basis_t tbasis;

    //using SHUFFLE_TENSOR = alg::shuffle_tensor<scalar_field, WIDTH, DEPTH>;
    template<DEG DEPTH>
    using SHUFFLE_TENSOR_ = alg::shuffle_tensor<poly_coeffs, WIDTH, DEPTH>;
    using SHUFFLE_TENSOR = SHUFFLE_TENSOR_<DEPTH>;
    
    template<DEG DEPTH>
    using shuffle_tensor_basis_t_ = alg::tensor_basis<WIDTH, DEPTH>;
    using shuffle_tensor_basis_t = shuffle_tensor_basis_t_<DEPTH>;
    shuffle_tensor_basis_t sbasis;

    using SHUFFLE_TENSOR_OVER_POLYS = alg::shuffle_tensor<poly_coeffs, WIDTH, DEPTH>;
    using inner_product = alg::operators::shuffle_tensor_functional<TENSOR, SHUFFLE_TENSOR_OVER_POLYS>;

    using MAPS = maps<poly_coeffs, WIDTH, DEPTH, TENSOR, LIE>;
    using CBH = cbh<poly_coeffs, WIDTH, DEPTH, TENSOR, LIE>;

    MAPS maps_;
    CBH cbh_;

    // make a LIE element whose hall coefficients are monomials
    LIE generic_lie(const int offset = 0) const
    {
        LIE result;
        for (auto lie_key : lbasis.iterate_keys()) {
            result.add_scal_prod(lie_key, poly_t(lie_key + offset, S(1)));
        }
        return result;
    }

    // The bilinear function K takes a shuffle and contracts it with a tensor to produce a scalar.
    // In this case the scalar is itself a polynomial
    static typename poly_coeffs::S K(const SHUFFLE_TENSOR& functional, const TENSOR& sig_data)
    {
        SHUFFLE_TENSOR_OVER_POLYS functional_p;
        for (auto& key_value : functional)
            functional_p.add_scal_prod(key_value.key(), typename poly_coeffs::S(key_value.value()));
        inner_product f(sig_data);//todo: left and right switched here?
        return f(functional_p);
    }

    // creates a generic vector with monomial coefficients
    template<class VECTOR, const int offset0 = 0>
    static VECTOR generic_vector(const int offset = offset0)
    {
        const typename VECTOR::BASIS& basis = VECTOR::basis;
        // LIE basis starts at 1 which is confusing

        int count;
        if constexpr (std::is_integral<decltype(basis.begin())>::value) {
            // invalid code if the premise is false - constexpr essential to avoid compilation
            count = basis.begin();
        }
        else {
            count = 0;
        }

        std::map<int, std::pair<typename VECTOR::KEY, std::string>> legend;

        VECTOR result;
        //for range type for loop approach use ": basis.iterate_keys()"
        for (auto key = basis.begin(), end = basis.end(); key != end; key = basis.nextkey(key)) {
            result[key] = poly_t(count + offset, 1);

            // record the mapping from keys to monomials
            auto basis_key_pair = std::pair<typename VECTOR::BASIS*, typename VECTOR::KEY>(&VECTOR::basis, key);
            std::stringstream buffer;
            buffer << basis_key_pair;
            legend[count + offset] = std::pair(key, buffer.str());
            std::cout << " monomial index:" << count + offset << " basis index:" << key << " basis value:" << buffer.str() << "\n";

            ++count;
        }
        return result;
    }

};

template<class FULL_TENSOR, class SHORT_TENSOR>
FULL_TENSOR& add_equals_short(FULL_TENSOR& out, const SHORT_TENSOR& in)
{
    const static typename FULL_TENSOR::BASIS fbasis;
    const static typename SHORT_TENSOR::BASIS sbasis;
    auto key = sbasis.begin(), end = sbasis.end();
    auto fkey = fbasis.begin(), fend = fbasis.end();
    for (; key != end && fkey != fend; key = sbasis.nextkey(key), fkey = fbasis.nextkey(fkey))
        out[fkey] += in[key];
    return out;
}

template<class EnvironmentOUT, class EnvironmentIN>
typename EnvironmentOUT::TENSOR apply1(const std::map<typename EnvironmentOUT::TENSOR::BASIS::KEY, typename EnvironmentIN::SHUFFLE_TENSOR>& result, const typename EnvironmentIN::TENSOR& in)
{
    typename EnvironmentOUT::TENSOR out;
    for (const auto& x : result) {
        auto& key = x.first;
        auto& tvalue = x.second;
        // both environments must have compatible scalar types
        out[key] = typename EnvironmentIN::K(tvalue, in);
    };

    return out;
}