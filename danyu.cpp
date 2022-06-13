#include "environment.h"
#include <array>
constexpr DEG WIDTH = 3;
constexpr DEG DEPTH = 3;

    int danyu2()
{
    using Environment = Environment<WIDTH, DEPTH>;
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

    LIE logsig_before = env.generic_lie(100);
    TENSOR tensor_logsig_before = env.maps_.l2t(logsig_before);
    TENSOR sig_before = exp(tensor_logsig_before);

    LIE logsig_during = env.generic_lie(200);
    TENSOR tensor_logsig_during = env.maps_.l2t(logsig_during);
    TENSOR sig_during = exp(tensor_logsig_during);

    LIE logsig_after = env.generic_lie(300);
    TENSOR tensor_logsig_after = env.maps_.l2t(logsig_after);
    TENSOR sig_after = exp(tensor_logsig_after);

    TENSOR sig = antipode(sig_before * sig_during * sig_after) * sig_before * sig_during * sig_after;
    TENSOR tensor_logsig = log(sig);
    LIE logsig = env.maps_.t2l(tensor_logsig);

    std::cout << "\n\n";
    std::cout << "The generic logsig of the triple:\n";
    for (auto& x : env.lbasis.iterate_keys())
        std::cout << "(" << std::pair<Environment::lie_basis_t*, alg::LET>(&env.lbasis, x) << ")"
                  << ":\t   "
                  << logsig[x] << "\n \n";
    std::cout << "\n\n";

    return 0;
}

    template <class Environment>
typename Environment::TENSOR apply(const std::map<typename Environment::TENSOR::BASIS::KEY, typename Environment::SHUFFLE_TENSOR>& result, const typename Environment::TENSOR& in)
{
    typename Environment::TENSOR out;
    for (const auto & x: result) {
        auto& key = x.first;
        auto& tvalue = x.second;
        out[key] = typename  Environment::K(tvalue, in);
    };

    return out;
}
//constexpr int N = 3;
//const auto one = Environment::SHUFFLE_TENSOR::SCALAR(1);
//const Environment::SHUFFLE_TENSOR x1(typename Environment::SHUFFLE_TENSOR::SCALAR(1));
//
//std::array<Environment::SHUFFLE_TENSOR, N> getInShuffles()
//{
//    std::array<int, N> a{};// now initialized!
//    for (auto idx = 0; idx < a.size(); ++idx) {
//        a[idx] = idx * idx;
//    }
//    return a;
//}

int danyu()
{
        using Environment = Environment<WIDTH, DEPTH>;
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
    LIE logsig_before = env.generic_vector<typename Environment::LIE, 2>(100);

    //    LIE logsig_before = env.generic_lie(100);
    TENSOR tensor_logsig_before = env.maps_.l2t(logsig_before);
    TENSOR sig_before = exp(tensor_logsig_before);

    LIE logsig_during = env.generic_lie(200);
    TENSOR tensor_logsig_during = env.maps_.l2t(logsig_during);
    TENSOR sig_during = exp(tensor_logsig_during);

    LIE logsig_after = env.generic_lie(300);
    TENSOR tensor_logsig_after = env.maps_.l2t(logsig_after);
    TENSOR sig_after = exp(tensor_logsig_after);

    TENSOR sig = sig_before * sig_during * sig_after;
    // TENSOR sig = antipode(sig_before * sig_during * sig_after) * sig_before * sig_during * sig_after;

    TENSOR tensor_logsig = log(sig);
    LIE logsig = env.maps_.t2l(tensor_logsig);

    ////////////////
    //STENSOR v_basic_shuffles[WIDTH]{STENSOR(1, poly_t(1, S(1))), STENSOR(2, poly_t(2, S(1))), STENSOR(3, poly_t(3, S(1)))};
    STENSOR v_basic_shuffles[WIDTH]{STENSOR(1, poly_t(S(1))), STENSOR(2, poly_t(S(1))), STENSOR(3, poly_t(S(1)))};
    //STENSOR v_basic_shuffles[WIDTH]{STENSOR(1, S(1)), STENSOR(2, S(1)), STENSOR(3, S(1))};

    std::map<TENSOR::BASIS::KEY, STENSOR> result;
    auto& out_tbasis = env.tbasis;

    auto tkey = out_tbasis.begin();
    if (tkey != out_tbasis.end()) {
        result[tkey] = STENSOR(poly_t(1));
        tkey = out_tbasis.nextkey(tkey);

        if (tkey != out_tbasis.end())
            for (auto v_basic_shuffle : v_basic_shuffles) {
                result[tkey] = v_basic_shuffle;
                tkey = out_tbasis.nextkey(tkey);
            }

        for (; tkey != out_tbasis.end(); tkey = out_tbasis.nextkey(tkey)) {
            auto letter = tkey.lparent();// first letter of key as a key
            auto rest = tkey.rparent();  // remainder of key as a key
            result[tkey] = half_shuffle_multiply(result[letter], result[rest]);
        }
    }

    std::cout << "\n\nBasic tests:\n " ;

    for (auto& u : result) std::cout << STENSOR(u.first, poly_t(S(1))) << " " << u.second << "\n";
    std::cout << "Basic tests finished\n\n";
    TENSOR ans = apply<Environment>(result, sig); 
    std::cout << ans << "\n\n";

    std::cout << sig << "\n\n";

    return 0;
}