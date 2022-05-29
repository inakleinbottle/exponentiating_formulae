#include "environment.h"
//************************************
// Method:    cbh_formula
// FullName:  cbh_formula
// Access:    public 
// Returns:   int
// Qualifier:
//************************************
int cbh_formula()
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


    auto glie1 = env.generic_lie(100);
    auto glie2 = env.generic_lie(200);
    auto L1 = env.maps_.l2t(glie1);
    auto L2 = env.maps_.l2t(glie2);
    auto logsig = env.maps_.t2l(log(exp(L1) * exp(L2)));
    
    std::cout << "\n\n";
    std::cout << "The generic lie elements:\n";
    for (auto& x : env.lbasis.iterate_keys())
        std::cout << LIE(x) << "\t"
                  << glie1[x] << "\t" << glie2[x] << "\n";
    std::cout << "\n\n";

    std::cout << "\n\n";
    std::cout << "The composition of two generic lie elements via cbh:\n";
    for (auto& x : env.lbasis.iterate_keys())
        std::cout << LIE(x) * logsig[x] << "\n";

    std::cout << "\n\n";
    return 0;
}