#include "SHOW.h"
#include "environment.h"
// the receiving environment
constexpr DEG WIDTHOUT = 2;
constexpr DEG DEPTHOUT = 2;

// the depth of the shuffles producing the channels
constexpr DEG INOUTDEPTH = 2;

// the incoming stream with enough accuracy to determine the
// required integrals of the projections
constexpr DEG WIDTHIN = 2;
constexpr DEG DEPTHIN = (DEPTHOUT * INOUTDEPTH);

// the input and output environments
using IN = Environment<WIDTHIN, DEPTHIN>;
using OUT = Environment<WIDTHOUT, DEPTHOUT>;

// steady state requires inputs limited to the same depth as the output
using SHORT_LIE = IN::LIE_<DEPTHOUT>;

// limiting the degree of nonlinearity in the functions on paths
using SHORT_SHUFFLE = IN::SHUFFLE_TENSOR_<INOUTDEPTH>;

int test()
{
    auto k = IN::K;

    IN in;
    const auto& sbasis = in.sbasis;
    IN::LIE logsig;
    add_equals_short(logsig, SHOW(in.generic_vector<SHORT_LIE>(1000)));
    SHOW(logsig);
    IN::TENSOR sig = exp(in.maps_.l2t(logsig));
    SHOW(sig);
    SHOW(antipode(sig)*sig);


    OUT out;

    IN::SHUFFLE_TENSOR g_basic_shuffles[OUT::WIDTH];
    const DEG in_shuffle_tensor_width = SHORT_SHUFFLE::BASIS::start_of_degree(INOUTDEPTH + 1) - SHORT_SHUFFLE::BASIS::start_of_degree(0);

    // now populate a vector of shuffles that gives the OUT path
    {
        int count = 0;
        for (auto& sh : g_basic_shuffles) {
            SHOW(count);
            add_equals_short(sh, in.generic_vector<SHORT_SHUFFLE>(count));
            count += in_shuffle_tensor_width;
            SHOW(sh);
        }
        std::cout << "Created " << count << " generic shuffle weight polynomials \n";
    }
     // now populate the tensor over the vector of shuffle coordinates
     // to get the signature of the OUT path (a grouplike element).
    
    std::map<OUT::TENSOR::BASIS::KEY, IN::SHUFFLE_TENSOR> result;
     

    const OUT::TENSOR::BASIS obasis;
    auto& obegin = obasis.begin();
    auto& oend = obasis.end();

    const IN::SHUFFLE_TENSOR::BASIS ibasis;
    auto& ibegin = ibasis.begin();
    auto& iend = ibasis.end();

    auto tkey = obegin;
    // the first entry in the tensor is the polynomial that is the constant 1
    if (tkey != oend) {
        result[tkey] = IN::SHUFFLE_TENSOR(IN::poly_t(1));
        tkey = obasis.nextkey(tkey);
    }

    // the increment of the path (so constant terms get set to zero!!!)
    if (tkey != oend) {
        for (auto basic_shuffle : g_basic_shuffles) {
            basic_shuffle[ibegin] -= basic_shuffle[ibegin];
            result[tkey] = basic_shuffle;
            tkey = obasis.nextkey(tkey);
        }      
    }
        for (  ; tkey != oend; tkey = obasis.nextkey(tkey)) {
        auto letter = tkey.lparent();// first letter of key as a key
        auto rest = tkey.rparent();  // remainder of key as a key
        result[tkey] = half_shuffle_multiply(result[letter], result[rest]);
    }

    OUT::TENSOR ans = apply1<OUT, IN>(result, sig);
    SHOW(antipode(ans) * ans);

    return 0;
}