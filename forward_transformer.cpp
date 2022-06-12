#include "environment.h"
#include <array>

constexpr DEG WIDTHIN = 3;
constexpr DEG DEPTHIN = 3;

constexpr DEG WIDTHOUT = 3;
constexpr DEG DEPTHOUT = 3;

using IN = Environment<WIDTHIN,DEPTHIN>;
using OUT = Environment<WIDTHOUT,DEPTHOUT>;

template<class EnvironmentIN,class EnvironmentOUT>
typename EnvironmentOUT::TENSOR apply(const std::map<typename EnvironmentOUT::TENSOR::BASIS::KEY, typename EnvironmentIN::SHUFFLE_TENSOR>& result, const typename EnvironmentIN::TENSOR& in)
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

int forward_transformer()
{
    auto k = IN::K;

    IN in;
    const auto& sbasis = in.sbasis;
    IN::LIE logsig_before = in.generic_vector<typename IN::LIE>(100);

    //    IN::LIE logsig_before = in.generic_lie(100);
    IN::TENSOR tensor_logsig_before = in.maps_.l2t(logsig_before);
    IN::TENSOR sig_before = exp(tensor_logsig_before);

    IN::LIE logsig_during = in.generic_lie(200);
    IN::TENSOR tensor_logsig_during = in.maps_.l2t(logsig_during);
    IN::TENSOR sig_during = exp(tensor_logsig_during);

    IN::LIE logsig_after = in.generic_lie(300);
    IN::TENSOR tensor_logsig_after = in.maps_.l2t(logsig_after);
    IN::TENSOR sig_after = exp(tensor_logsig_after);

    IN::TENSOR sig = sig_before * sig_during * sig_after;
    // IN::TENSOR sig = antipode(sig_before * sig_during * sig_after) * sig_before * sig_during * sig_after;

    IN::TENSOR tensor_logsig = log(sig);
    IN::LIE logsig = in.maps_.t2l(tensor_logsig);

    OUT out;
    
    IN::SHUFFLE_TENSOR v_basic_shuffles[OUT::WIDTH]{IN::SHUFFLE_TENSOR(1, IN::poly_t(IN::S(1))), IN::SHUFFLE_TENSOR(2, IN::poly_t(IN::S(1))), IN::SHUFFLE_TENSOR(3, IN::poly_t(IN::S(1)))};

    std::map<OUT::TENSOR::BASIS::KEY, IN::SHUFFLE_TENSOR> result;
    auto& out_tbasis = out.tbasis;

    auto tkey = out_tbasis.begin();
    if (tkey != out_tbasis.end()) {
        result[tkey] = IN::SHUFFLE_TENSOR(IN::poly_t(1));
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

    std::cout << "\n\nBasic tests:\n ";

    for (auto& u : result) std::cout << IN::SHUFFLE_TENSOR(u.first, IN::poly_t(IN::S(1))) << " " << u.second << "\n";
    std::cout << "Basic tests finished\n\n";
    OUT::TENSOR ans = apply<OUT,IN>(result, sig);
    std::cout << ans << "\n\n";

    std::cout << sig << "\n\n";

    return 0;
}