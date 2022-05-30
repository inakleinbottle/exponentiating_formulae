#include "environment.h"

int danyu()
{
    /*
    w1,w2,...wk, and a signature S_[0,t] 
    then we can define a new path 
         (<w1,S_[0,t] >,<w2,S_[0,t] >,...<wk,S_[0,t] >)
    and compute the signature of this path over an 
    interval [s,t]. Crucially, it involves only the 
    signature of S on the intervals [0,s] and [s,t]. Its
    complexity depends on the depth of the shuffles wi.

    Danyu's observation that is that our level one output
    of the transformer y can be rewritten in explicit form 
                          <w,S_[0,t] >
    and although there is a linear transformation from S_[0,1]
    this can be reinterpreted as different more complex w. So
    we can put these all together. 

    Ultimately the goal is to go from log signatures to 
    log signatures.
    
    */
    // give simplified names and access to the objects defined in Environment
    using LIE = Environment::LIE;
    using STENSOR = Environment::SHUFFLE_TENSOR;
    using TENSOR = Environment::TENSOR;
    using S = Environment::S;
    using poly_t = Environment::poly_t;

    auto k = Environment::K;

    Environment env;
    const auto& sbasis = env.sbasis;

    LIE glie1 = env.generic_lie(100);
    LIE glie2 = env.generic_lie(200);
    TENSOR L1 = env.maps_.l2t(glie1);
    TENSOR L2 = env.maps_.l2t(glie2);
    LIE logsig = env.maps_.t2l(log(exp(L1) * exp(L2)));

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
