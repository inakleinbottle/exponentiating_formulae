#include "environment.h"

int cbh_formula();
int main()
{
	// give simplified names and access to the objects defined in Environment
	using LIE = Environment::LIE;
	using STENSOR = Environment::SHUFFLE_TENSOR;
	using TENSOR = Environment::TENSOR;
	using S = Environment::S;
	using poly_t = Environment::poly_t;

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

	 cbh_formula();
}
