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

	static typename poly_coeffs::S K(const SHUFFLE_TENSOR& functional, const TENSOR& sig_data) {
		SHUFFLE_TENSOR_OVER_POLYS functional_p;
		for (auto key_value : functional)
			functional_p.add_scal_prod(key_value.key(), typename poly_coeffs::S(key_value.value()));
		inner_product f(sig_data);
		return f(functional_p);
	}
};

int main()
{
	// give simplified names and access to the objects defined in Environment
	using LIE = Environment::LIE;
	using STENSOR = Environment::SHUFFLE_TENSOR;
	using TENSOR = Environment::TENSOR;
	using S = Environment::S;

	auto k = Environment::K;

	Environment env;
	const auto& sbasis = env.sbasis;
	//////////////////////////////////////////////////////////////////////////////////

	auto glie = env.generic_lie();
	std::cout << "The generic lie element:\n" << glie << "\n\n";

	TENSOR L = env.maps_.l2t(glie);

	std::cout << "The generic lie element in tensor co-ordinates:\n";
	for (auto& x : sbasis.iterate_keys())
		//for (auto x = sbasis.begin(), e = sbasis.end(); x != e; x = sbasis.nextkey(x))
		std::cout << STENSOR(x) << "\t"
		<< k(STENSOR(x), L) << "\n";

	std::cout << "\n\n";
	std::cout << "The exponential of the generic lie element in tensor co-ordinates:\n";
	for (auto& x : sbasis.iterate_keys())
		std::cout << STENSOR(x) << "\t"
		<< k(STENSOR(x), exp(L)) << "\n";

	std::cout << "\n\n";

	std::cout << "The exponential of the generic lie element contracted with each nonzero half shuffles of pairs of basis elements:\n";
	for (auto& x : sbasis.iterate_keys())
		for (auto& y : sbasis.iterate_keys()) {
			STENSOR z = alg::half_shuffle_multiply(STENSOR(x), STENSOR(y));
			if (z != STENSOR(STENSOR::zero))
				std::cout << z << "\t"
				<< k(z, exp(L)) << "\n";
		}
	std::cout << "\n\n";

	auto glie1 = env.generic_lie(100);
	auto glie2 = env.generic_lie(200);
	auto glie3 = env.generic_lie(300);
	auto L1 = env.maps_.l2t(glie1);
	auto L2 = env.maps_.l2t(glie2);
	auto L3 = env.maps_.l2t(glie3);
	auto sig = exp(L1) * exp(L2) * exp(L3);

	std::cout << "\n\n";
	std::cout << "The exponential composition of three generic lie elements contracted with each element of the dual shuffle basis:\n";
	for (auto& x : sbasis.iterate_keys())
		std::cout << STENSOR(x) << "\t"
		<< k(STENSOR(x), sig) << "\n";

	std::cout << "\n\n";
}
