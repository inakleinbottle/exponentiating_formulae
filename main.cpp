#include <iostream>

#include <libalgebra/libalgebra.h>
#include <libalgebra/coefficients/coefficients.h>
#include <libalgebra/coefficients/rational_coefficients.h>

using namespace alg;

constexpr DEG WIDTH = 2;
constexpr DEG DEPTH = 2; // For testing, to keep things small enough to run at home
constexpr DEG POLY_DEPTH = 2;


struct Environment {
    using scalar_field = coefficients::rational_field;
    using S = typename scalar_field::S;

    using lie_basis_t = lie_basis<WIDTH, DEPTH>;
    lie_basis_t lbasis;
    static constexpr DEG poly_width = hall_basis<WIDTH, DEPTH>::start_of_degree(DEPTH+1);

    using poly_t = alg::multi_polynomial<scalar_field, poly_width, POLY_DEPTH>;
    using poly_coeffs = coefficients::coefficient_ring<poly_t, poly_t>;

    using LIE = alg::lie<poly_coeffs, WIDTH, DEPTH, vectors::dense_vector>;
    using TENSOR = alg::free_tensor<poly_coeffs, WIDTH, DEPTH, vectors::dense_vector>;

    using MAPS = maps<poly_coeffs, WIDTH, DEPTH, TENSOR, LIE>;
    using CBH = cbh<poly_coeffs, WIDTH, DEPTH, TENSOR, LIE>;


    LIE generic_lie() const
    {
        LIE result;
        for (auto lie_key : lbasis.iterate_keys()) {
            result.add_scal_prod(lie_key, poly_t(lie_key, S(1)));
        }

        return result;
    }

};




int main()
{
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
