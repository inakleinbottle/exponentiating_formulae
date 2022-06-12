#pragma once

#include <string>
#include <iostream>
#include <functional>
#include <sstream>

#include <libalgebra/libalgebra.h>

using namespace alg;

constexpr DEG WIDTH = 3;
constexpr DEG DEPTH = 3; // For testing, to keep things small enough to run at home


struct Environment {
	using scalar_field = coefficients::rational_field;
	using S = typename scalar_field::S;

	static constexpr DEG poly_width = hall_basis<WIDTH, DEPTH>::start_of_degree(DEPTH + 1);

	using poly_t = alg::poly<scalar_field>;
	using poly_coeffs = coefficients::coefficient_ring<poly_t, typename scalar_field::Q>;
	static_assert(std::is_same<poly_coeffs::S, poly_t>::value, "the trait class of a coefficient ring must report the type of the coefficients");

	using LIE = alg::lie<poly_coeffs, WIDTH, DEPTH, vectors::dense_vector>;
	using lie_basis_t = lie_basis<WIDTH, DEPTH>;
	lie_basis_t lbasis;

	using TENSOR = alg::free_tensor<poly_coeffs, WIDTH, DEPTH, vectors::dense_vector>;
	using tensor_basis_t = alg::tensor_basis<WIDTH, DEPTH>;
	tensor_basis_t tbasis;
	//using SHUFFLE_TENSOR = alg::shuffle_tensor<scalar_field, WIDTH, DEPTH>;
	using SHUFFLE_TENSOR = alg::shuffle_tensor<poly_coeffs, WIDTH, DEPTH>;
	using shuffle_tensor_basis_t = alg::tensor_basis<WIDTH, DEPTH>;
	shuffle_tensor_basis_t sbasis;

	using SHUFFLE_TENSOR_OVER_POLYS = alg::shuffle_tensor<poly_coeffs, WIDTH, DEPTH>;
	using inner_product = alg::operators::shuffle_tensor_functional<TENSOR, SHUFFLE_TENSOR_OVER_POLYS>;

	using MAPS = maps<poly_coeffs, WIDTH, DEPTH, TENSOR, LIE>;
	using CBH = cbh<poly_coeffs, WIDTH, DEPTH, TENSOR, LIE>;

	MAPS maps_;
	CBH cbh_;

	LIE generic_lie(const int offset = 0) const
	{
		LIE result;
		for (auto lie_key : lbasis.iterate_keys()) {
			result.add_scal_prod(lie_key, poly_t(lie_key + offset, S(1)));
		}

		return result;
	}

	/// The bilinear function K takes a shuffle and contracts it with a tensor to produce a scalar.
	/// In this case the scalar is itself a polynomial 
	static typename poly_coeffs::S K(const SHUFFLE_TENSOR& functional, const TENSOR& sig_data) {
		SHUFFLE_TENSOR_OVER_POLYS functional_p;
		for (auto& key_value : functional)
			functional_p.add_scal_prod(key_value.key(), typename poly_coeffs::S(key_value.value()));
		inner_product f(sig_data);//todo: left and right switched here?
		return f(functional_p);
	}

	template<class VECTOR, const int offset0 = 0>
    VECTOR generic_vector(const int offset = offset0)
    {
        typename VECTOR::BASIS& basis = VECTOR::basis;
		// LIE basis starts at 1 which is confusing
        int count = (std::is_integral<decltype(basis.begin())>::value) ? basis.begin() : 0;

		std::map < int, std::pair<typename VECTOR::KEY, std::string> > legend;

        VECTOR result;
        //for range type for loop approach use ": basis.iterate_keys()"
        for (auto key = basis.begin(), end = basis.end(); key != end; key = basis.nextkey(key)) 
		{
            result[key]= poly_t(count + offset, 1);

            auto basis_key_pair = std::pair<typename VECTOR::BASIS*, typename VECTOR::KEY>(&VECTOR::basis, key);
            std::stringstream buffer;
            buffer << basis_key_pair;
            legend[count + offset] = std::pair(key, buffer.str());
            std::cout << count + offset << " " << key << " " << buffer.str() << "\n";

            ++count;
        }
        return result;
    }
};

