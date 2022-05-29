#pragma once

#include <iostream>
#include <functional>

#include <libalgebra/coefficients/rational_coefficients.h>
#include <libalgebra/libalgebra.h>
#include <libalgebra/coefficients/coefficients.h>
#include <libalgebra/operators/functionals.h>

namespace alg {
#include "libalgebra/half_shuffle_tensor_basis.h"
#include "libalgebra/half_shuffle_tensor_multiplication.h"
#include "libalgebra/area_tensor_basis.h"
#include "libalgebra/area_tensor_multiplication.h"
#include "libalgebra/alternative_multiplications.h"
};

using namespace alg;

constexpr DEG WIDTH = 2;
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

	using SHUFFLE_TENSOR = alg::shuffle_tensor<scalar_field, WIDTH, DEPTH>;
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
};

