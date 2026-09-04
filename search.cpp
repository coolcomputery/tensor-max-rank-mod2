#include "utils.cpp"
#include <map>

class Tensor {
    private:
        vi32 shape;
        vu8 data;  // row-major order of elements
    public:
        ~Tensor() {}
        Tensor(const vi32 &shape, const vu8 &data) {
            for (i32 n:shape)
                assert(n>=0, "Tensor: negative length on an axis");

            assert(data.size()==prod(shape), "Tensor: wrong size");
            for (i32 v:data)
                assert(0<=v && v<MOD, "Tensor: value not in mod");

            this->shape=vi32(shape);
            this->data=vu8(data);
        }
        bool operator==(const Tensor &o) const {
            return shape==o.shape && data==o.data;
        }
        bool operator!=(const Tensor &o) const {
            return !(*this==o);
        }
        vi32 get_shape() const {
            return vi32(shape);
        }
        i32 ndim() const {
            return shape.size();
        }
        i32 len(i32 ax) const {
            return shape.at(ax);
        }
        i32 size() const {
            return data.size();
        }
        i32 at(i32 i) const {
            return data.at(i);
        }
        vu8 flattened() const {
            return vu8(data);
        }
        i32 mat_at(i32 i, i32 j) const {
            assert(shape.size()==2, "Tensor::mat_at: not 2D");
            assert(0<=i && i<shape.at(0), "Tensor::mat_at: row index out of range");
            assert(0<=j && j<shape.at(1), "Tensor::mat_at: clm index out of range");
            return data.at(i*shape.at(1)+j);
        }
        Tensor mat_clm(i32 j) const {
            assert(shape.size()==2, "Tensor::mat_clm: not 2D");
            assert(0<=j && j<shape.at(1), "Tensor::mat_clm: index out of range");
            vu8 out;
            for (i32 i=0; i<shape.at(0); i++)
                out.push_back(data.at(i*shape.at(1)+j));
            return Tensor({shape.at(0)},out);
        }
        Tensor copy() const {
            return Tensor(shape,data);
        }
};
typedef std::vector<Tensor> vTensor;
typedef std::vector<vTensor> vvTensor;
std::ostream& operator << (std::ostream &os, const Tensor &T) {
    os<<"Tensor("<<T.get_shape()<<", [";
    for (i32 i=0; i<T.size(); i++)
        os<<(i>0?", ":"")<<(i32)T.at(i);
    os<<"])";
    return os;
}

Tensor axis_op(const Tensor &M, const Tensor &T, i32 ax) {
    assert(M.ndim()==2, "axis_op: operator not matrix");
    assert(0<=ax && ax<T.ndim(), "axis_op: ais out of range");
    assert(M.len(1)==T.len(ax), "axis_op: mismatched inner dimension");
    i32 n_pre=1;
    for (i32 d=0; d<ax; d++)
        n_pre*=T.len(d);
    i32 n_suf=1;
    for (i32 d=ax+1; d<T.ndim(); d++)
        n_suf*=T.len(d);
    i32 n_mid=T.len(ax), n_new=M.len(0);
    vu8 data;
    for (i32 i_pre=0; i_pre<n_pre; i_pre++)
        for (i32 i_new=0; i_new<n_new; i_new++)
            for (i32 i_suf=0; i_suf<n_suf; i_suf++) {
                u8 tot=0;
                for (i32 i_mid=0; i_mid<n_mid; i_mid++)
                    tot=(tot+M.at(i_new*n_mid+i_mid)*T.at((i_pre*n_mid+i_mid)*n_suf+i_suf))%MOD;
                data.push_back(tot);
            }
    vi32 nshape;
    for (i32 d=0; d<ax; d++)
        nshape.push_back(T.len(d));
    nshape.push_back(n_new);
    for (i32 d=ax+1; d<T.ndim(); d++)
        nshape.push_back(T.len(d));
    return Tensor(nshape,data);
}
Tensor axis0_op(const Tensor &M, const Tensor &T) {
    return axis_op(M,T,0);
}

Tensor stack_vecs(i32 nrows, i32 nclms, const vTensor &vecs) {
    assert(vecs.size()==nrows, "stack_vecs: wrong # of vecs");
    for (const Tensor &v:vecs)
        assert(v.ndim()==1 && v.len(0)==nclms, "stack_vecs: vec has wrong shape");
    vu8 data;
    for (const Tensor &v:vecs)
        for (u8 e:v.flattened())
            data.push_back(e);
    return Tensor({nrows,nclms},data);
}

typedef struct {
    i32 rank;
    Tensor reducer;
} RowReduce;
RowReduce mat_row_reduce(const Tensor &mat) {
    /*
    return r, Q s.t.
    * r=rank(mat)
    * Q invertible
    * Q@M=rref(M)
    */
    assert(mat.ndim()==2, "mat_row_reduce: not 2D");
    i32 m=mat.len(0), n=mat.len(1);
    vvu8 elems;
    for (i32 i=0; i<m; i++) {
        elems.push_back({});
        for (i32 j=0; j<n; j++)
            elems.at(i).push_back(mat.mat_at(i,j));
    }
    vvu8 reducer;
    for (i32 i=0; i<m; i++) {
        reducer.push_back(vu8(m,0));
        reducer.at(i)[i]=1;
    }
    i32 rank=0;
    for (i32 j=0; j<n; j++) {
        i32 i0=-1;
        for (i32 i=rank; i<m; i++)
            if (elems.at(i).at(j)!=0) {
                i0=i;
                break;
            }
        if (i0<0)
            continue;
        // swap rows rank, i0
        for (i32 k=0; k<n; k++) {
            i32 tmp=elems.at(rank).at(k);
            elems.at(rank)[k]=elems.at(i0).at(k);
            elems.at(i0)[k]=tmp;
        }
        for (i32 k=0; k<m; k++) {
            i32 tmp=reducer.at(rank).at(k);
            reducer.at(rank)[k]=reducer.at(i0).at(k);
            reducer.at(i0)[k]=tmp;
        }
        // all nonzero rows are already normalized over mod 2
        // subtract row i0 from all other rows
        for (i32 i=0; i<m; i++)
            if (i!=rank && elems.at(i).at(j)!=0) {
                for (i32 k=0; k<n; k++)
                    elems.at(i)[k]^=elems.at(rank)[k];
                for (i32 k=0; k<m; k++)
                    reducer.at(i)[k]^=reducer.at(rank)[k];
            }
        rank++;
    }
    return {
        .rank=rank,
        .reducer=Tensor({m,m},concat(reducer)),
    };
}
i32 mat_rank(const Tensor &mat) {
    return mat_row_reduce(mat).rank;
}
Tensor mat_inv(const Tensor &mat) {
    assert(mat.ndim()==2, "mat_inv: not 2D");
    assert(mat.len(0)==mat.len(1), "mat_inv: not square");
    RowReduce ret=mat_row_reduce(mat);
    assert(ret.rank==mat.len(0), "mat_inv: matrix not invertible");
    return ret.reducer;
}
Tensor unfold(const Tensor &T, i32 ax) {
    assert(0<=ax && ax<T.ndim(), "unfold: axis out of range");
    i32 suffix_prod=1;
    for (i32 d=ax+1; d<T.ndim(); d++)
        suffix_prod*=T.len(d);
    i32 prefix_prod=1;
    for (i32 d=0; d<ax; d++)
        prefix_prod*=T.len(d);
    i32 n_ax=T.len(ax);

    vvu8 slices;
    for (i32 i_ax=0; i_ax<n_ax; i_ax++) {
        vu8 slice;
        for (i32 i_pre=0; i_pre<prefix_prod; i_pre++)
            for (i32 i_suf=0; i_suf<suffix_prod; i_suf++)
                slice.push_back(T.at(
                    (i_pre*n_ax+i_ax)*suffix_prod+i_suf
                ));
        slices.push_back(slice);
    }
    return Tensor({n_ax,prefix_prod*suffix_prod},concat(slices));
}
i32 axis_rank(const Tensor &T, i32 ax) {
    return mat_rank(unfold(T,ax));
}
typedef struct {
    Tensor concise;
    std::vector<Tensor> expanders;
} ConciseTensor;
ConciseTensor concise(const Tensor &T) {
    Tensor concise=T;
    std::vector<Tensor> expanders;
    for (i32 ax=0; ax<T.ndim(); ax++) {
        i32 n=T.len(ax);
        RowReduce ret=mat_row_reduce(unfold(T,ax));
        i32 r=ret.rank;
        Tensor expander=mat_inv(ret.reducer);
        vu8 truncated_expander_data;
        for (i32 i=0; i<n; i++)
            for (i32 j=0; j<r; j++)
                truncated_expander_data.push_back(expander.at(i*n+j));
        Tensor trunc_expander({n,r},truncated_expander_data);
        expanders.push_back(trunc_expander);
        vu8 reducer_data=ret.reducer.flattened();
        Tensor trunc_reducer({r,n},vu8(reducer_data.begin(),reducer_data.begin()+r*n));
        concise=axis_op(trunc_reducer,concise,ax);
    }
    return {
        .concise=concise,
        .expanders=expanders,
    };
}
bool is_concise(const Tensor &T) {
    assert(T.ndim()>0, "is_concise: 0-dim not supported");
    for (i32 d=0; d<T.ndim(); d++)
        if (axis_rank(T,d)!=T.len(d))
            return false;
    return true;
}

vTensor all_normalized_vecs(i32 n) {
    vTensor out;
    for (const vu8 &elems:all_lists(n)) {
        i32 first_nonzero_val=0;
        for (i32 i=0; i<n; i++)
            if (elems.at(i)!=0) {
                first_nonzero_val=elems.at(i);
                break;
            }
        if (first_nonzero_val==1)
            out.push_back(Tensor({n},elems));
    }
    return out;
}
vTensor all_tensors(const vi32 &shape) {
    vTensor out;
    for (const vu8 &elems:all_lists(prod(shape)))
        out.push_back(Tensor(shape,elems));
    return out;
}

typedef u16 pack_t;  // int type to pack a n1 x n2 tensor slice into
const int PACK_SIZE=8*sizeof(pack_t);
typedef std::vector<pack_t> vpack_t;
pack_t packed(const Tensor &T) {
    assert(T.size()<=PACK_SIZE, "packed: too large for packing");
    pack_t out=0;
    for (i32 i=0; i<T.size(); i++)
        out=out|(((pack_t)T.at(i))<<i);
    return out;
}
Tensor unpacked_vec(pack_t w, i32 n) {
    assert(0<=w && w<(1L<<n), "unpacked_vec: word out of domain");
    vu8 data;
    for (i32 i=0; i<n; i++)
        data.push_back(bit_at<pack_t,u8>(w,i));
    return Tensor({n},data);
}

Tensor outer_prod(const vTensor &vecs) {
    for (const Tensor &v:vecs)
        assert(v.ndim()==1, "outer_prod: input not 1D");
    vu8 data;
    if (vecs.size()==2) {
        const Tensor &v0=vecs.at(0);
        const Tensor &v1=vecs.at(1);
        for (i32 i=0; i<v0.size(); i++)
            for (i32 j=0; j<v1.size(); j++)
                data.push_back((v0.at(i)*v1.at(j))%MOD);
    }
    else if (vecs.size()==3) {
        const Tensor &v0=vecs.at(0);
        const Tensor &v1=vecs.at(1);
        const Tensor &v2=vecs.at(2);
        for (i32 i=0; i<v0.size(); i++)
            for (i32 j=0; j<v1.size(); j++)
                for (i32 k=0; k<v2.size(); k++)
                    data.push_back((v0.at(i)*v1.at(j)*v2.at(k))%MOD);
    }
    else
        throw std::runtime_error("outer_prod: dim>3 not implemented");
    vi32 shape;
    for (const Tensor &v:vecs)
        shape.push_back(v.size());
    return Tensor(shape,data);
}

// set of lists of {0,...,n-1}^d
class Trie {
    private:
        static constexpr i32 DUMMY=-1;
        i32 d;
        i32 n;
        vvi32 children;  // children[i][v] = index of child of node i by appending v to corresponding list
    public:
        ~Trie() {}
        Trie(i32 d, i32 n) {
            this->d=d;
            this->n=n;
            // root node represents empty list
            children={vi32(n,DUMMY)};
        }
        void add(const vi32 &list) {
            assert(list.size()==d, "Trie::add: wrong length");
            for (i32 v:list)
                assert(0<=v && v<n, "Trie::add: element out of range");
            i32 prefix_node=0;
            i32 depth=0;
            while (depth<d) {
                i32 next=children.at(prefix_node).at(list.at(depth));
                if (next==DUMMY)
                    break;
                prefix_node=next;
                depth++;
            }
            for (i32 i=depth; i<d; i++) {
                children.push_back(vi32(n,DUMMY));
                i32 cur=children.size()-1, parent=(i==depth?prefix_node:(cur-1));
                children.at(parent)[list.at(i)]=cur;
            }
        }
        bool has_ptwise_ge(const vi32 &list) {
            assert(list.size()==d, "Trie::has_ptwise_ge: wrong length");
            for (i32 v:list)
                assert(0<=v && v<n, "Trie::has_ptwise_ge: element out of range");
            std::function<bool(i32,i32)> dfs=[&](i32 depth, i32 node) {
                assert(0<=node && node<children.size(), "has_ptwise_ge dfs: node out of range");
                if (depth==d)
                    return true;
                for (i32 v=list.at(depth); v<n; v++) {
                    i32 ch=children.at(node).at(v);
                    if (ch!=DUMMY && dfs(depth+1,ch))
                        return true;
                }
                return false;
            };
            return dfs(0,0);
        }
};
class CubeSubset {
    private:
        typedef u64 word_t;
        const static i32 WORD_SIZE=64;
        i32 n, d;
        i64 tot;
        std::vector<word_t> bitset;
        i64 code(const vi32 &A) {
            assert(A.size()==d, "CubeSubset::code: wrong length");
            for (i32 i=0; i<d; i++)
                assert(0<=A.at(i) && A.at(i)<n, "CubeSubset::code: elements out of range");
            i64 out=0, pow=1;
            for (i32 i=0; i<d; i++) {
                out+=pow*A.at(i);
                pow*=n;
            }
            return out;
        }
        void add_bit(i64 i) {
            assert(0<=i && i<tot, "CubeSubset::add_bit: out of range");
            bitset[i/WORD_SIZE]|=((word_t)1)<<(i%WORD_SIZE);
        }
        void add(const vi32 &A) {
            add_bit(code(A));
        }
        bool contains_bit(i64 i) {
            assert(0<=i && i<tot, "CubeSubset::contains_bit: out of range");
            return bit_at<word_t,word_t>(bitset.at(i/WORD_SIZE),i%WORD_SIZE)!=0;
        }
        CubeSubset(i32 n, i32 d) {
            this->n=n;
            this->d=d;
            this->tot=1;
            for (i32 i=0; i<d; i++)
                tot*=n;
            bitset=std::vector<word_t>((tot-1)/WORD_SIZE+1,0);
        }
    public:
        ~CubeSubset() {}
        static CubeSubset all_ptwise_le(i32 n, i32 d, const vvi32 &pts) {
            for (const vi32 &pt:pts) {
                assert(pt.size()==d, "CubeSubset::all_ptwise_le: wrong ndim");
                for (i32 i=0; i<d; i++)
                    // NOTE pt[i] is allowed to be >=n
                    assert(0<=pt.at(i), "CubeSubset::all_ptwise_le: negative coordinates");
            }
            CubeSubset out(n,d); {
                vi32 buffer(d,-1);
                for (const vi32 &pt:pts) {
                    for (i32 i=0; i<d; i++)
                        buffer[i]=std::min(n-1,pt.at(i));
                    out.add(buffer);
                }
            }
            // suffix boolean OR
            for (i32 ax=d-1; ax>=0; ax--) {
                i64 pow=1;
                for (i32 rep=0; rep<ax; rep++)
                    pow*=n;
                for (i64 block=out.bitset.size()-1; block>=0; block--) {
                    if (out.bitset.at(block)!=0) {
                        for (i64 pt=std::min(out.tot-1,(block+1)*WORD_SIZE-1); pt>=block*WORD_SIZE; pt--) {
                            i64 ax_coord=(pt/pow)%(i64)n;
                            if (ax_coord>0 && out.contains_bit(pt)) {
                                i64 next_pt=pt-pow;
                                out.add_bit(next_pt);
                            }
                        }
                    }
                }
            }
            return out;
        }
        bool contains(const vi32 &A) {
            return contains_bit(code(A));
        }
};

vvi32 all_compressed_signatures(i32 n, i32 R, i32 elem_bound) {
    assert(elem_bound>0, "all_compressed_signatures: elem_bound<=0");
    vTensor vecs=all_tensors({n});
    vvi32 out;
    for_all_multisets(vecs.size(),R,[&](const vi32 &clm_idxs) {
        vi32 signature((1<<n)-1,-1);
        for_all_words<pack_t>(n, [&](pack_t q) {
            if (q==0)
                return false;
            i32 nnz=0;
            for (i32 idx:clm_idxs) {
                const Tensor &clm=vecs.at(idx);
                u8 dot=0;
                for (i32 i=0; i<n; i++)
                    // TODO replace with xor and popcount
                    dot=(dot+bit_at<pack_t,u8>(q,i)*clm.at(i))%MOD;
                if (dot!=0)
                    nnz++;
            }
            signature[q-1]=std::min(elem_bound-1,nnz-1);
            return false;
        });
        bool full_row_rank=true;
        for (i32 v:signature)
            if (v<0) {
                full_row_rank=false;
                break;
            }
        if (full_row_rank)
            out.push_back(signature);
        return false;
    });
    return out;
}
i64 nCr(i32 n, i32 k) {
    assert(n>=0, "nCr: n<0");
    assert(k>=0, "nCr: k<0");
    if (k>n)
        return 0;
    if (n-k<k)
        k=n-k;
    // nCr(n,k) = nCr(n,k-1) * (n-k+1)/k
    i64 out=1;
    for (i32 i=1; i<=k; i++) {
        assert(out<INT64_MAX/(n-i+1), "nCr: output possibly too large");
        out=(out*(n-i+1))/i;
    }
    return out;
}
double log_nCr(i32 n, i32 k) {
    assert(n>=0, "log_nCr: n<0");
    assert(k>=0, "log_nCr: k<0");
    if (k>n)
        return -INFINITY;
    double out=0;
    for (i32 i=1; i<=k; i++)
        out+=std::log(n-i+1)-std::log(i);
    return out;
}
class VectorSpan {
    private:
        i32 n;
        i32 dim, size;
        std::vector<bool> in_span;
        vpack_t span_buffer;  // span_buffer[:size] is the set of all vectors contained in this object
    public:
        ~VectorSpan() {}
        VectorSpan(i32 n) {
            this->n=n;
            assert(0<n && n<=PACK_SIZE, "VectorSpan: ambient dimension out of allowed range");
            in_span=std::vector<bool>(1L<<n,false);
            span_buffer=vpack_t(1L<<n,0);
            reset();
        }
        void reset() {
            dim=0;
            size=1;
            std::fill(in_span.begin(),in_span.end(),false);
            in_span[0]=true;
            std::fill(span_buffer.begin(),span_buffer.end(),0);
        }
        void add(pack_t v) {
            assert(0<=v && v<(1L<<n), "VectorSpan::add: packed vector out of domain");
            if (in_span.at(v))
                return;
            for (i32 i=0; i<size; i++) {
                pack_t new_elem=span_buffer.at(i)^v;
                in_span[new_elem]=true;
                span_buffer[i+size]=new_elem;
            }
            size*=MOD;
            dim++;
        }
        bool contains(pack_t v) const {
            assert(0<=v && v<(1L<<n), "VectorSpan::add: packed vector out of domain");
            return in_span.at(v);
        }
        i32 get_dim() const {
            return dim;
        }
};

enum class PrunerMode {
    SUFFIX_ARRAY,
    PREFIX_TREE,
    LOSSY,
};
// given a map f:(int mod 2)^n -> int,
// return "no" or "maybe" on whether there exists A n x R
// of full row-rank such that f(v) <= nnz(v@A) for all v
class Pruner {
    private:
        static constexpr double MAX_MEM=2000'000'000;
        i32 n, R, elem_bound;
        PrunerMode mode;
        std::optional<CubeSubset> suffix_array;
        std::optional<Trie> trie;
        vi32 _trimmed_sig;
        VectorSpan _span;
        void assert_is_valid_signature(const vi32 &sig) {
            assert(sig.size()==(1<<n), "Pruner::is_valid_signature: wrong size (signature should not be trimmed)");
            assert(sig.at(0)==0, "Pruner::is_valid_signature: should be 0 at origin");
            for (i32 v:sig)
                assert(v>=0, "Pruner::is_valid_signature: negative elements");
        }
        bool _allows_rref(const vi32 &sig) {
            assert_is_valid_signature(sig);
            _span.reset();
            for_all_words<pack_t>(n, [&](pack_t q) {
                if (sig.at(q)<=R-n+1)
                    _span.add(q);
                return false;
            });
            return _span.get_dim()==n;
        }
        bool _allows_rref2(const vi32 &sig) {
            assert_is_valid_signature(sig);
            if (R<=n+1)
                return true;
            _span.reset();
            for_all_words<pack_t>(n, [&](pack_t q) {
                if (sig.at(q)<=R-n)
                    _span.add(q);
                return false;
            });
            return _span.get_dim()>=n-1;
        }
        bool _allows_lask(const vi32 &sig, i32 k) {
            assert_is_valid_signature(sig);
            u64 tot=0;
            for_all_words<pack_t>(n, [&](pack_t q) {
                tot+=nCr(R-sig.at(q),k);
                return false;
            });
            return tot>=nCr(R,k)*(1LL<<(n-k));
        }
        bool allows_lossy(const vi32 &sig) {
            assert_is_valid_signature(sig);
            if (!_allows_rref(sig))
                return false;
            if (!_allows_rref2(sig))
                return false;
            for (i32 k=1; k<n; k++)
                if (!_allows_lask(sig,k))
                    return false;
            return true;
        }
    public:
        ~Pruner() {}
        Pruner(i32 n, i32 R, i32 elem_bound): _span(n) {
            assert(elem_bound>0, "Pruner: elem_bound<=0");
            this->n=n;
            this->R=R;
            this->elem_bound=elem_bound;
            _trimmed_sig=vi32((1<<n)-1,0);

            i32 ndim=(1<<n)-1;
            double log_signature_mem=log_nCr((1<<n)-1 + R-1, R)+std::log(ndim);
            if (log_signature_mem>std::log(MAX_MEM)) {
                mode=PrunerMode::LOSSY;
                return;
            }

            vvi32 compressed_sigs=all_compressed_signatures(n,R,elem_bound);
            double log_suffix_array_mem=std::log(elem_bound)*ndim;
            if (log_suffix_array_mem<std::log(MAX_MEM)) {
                mode=PrunerMode::SUFFIX_ARRAY;
                suffix_array=CubeSubset::all_ptwise_le(elem_bound,ndim,compressed_sigs);
            }
            else {
                // NOTE prefix tree memory is roughly proportional to that of compressed_sigs
                mode=PrunerMode::PREFIX_TREE;
                Trie _trie(ndim,elem_bound);
                for (const vi32 &sig:compressed_sigs)
                    _trie.add(sig);
                trie=_trie;
            }
        }
        bool allows(const vi32 &sig) {
            assert_is_valid_signature(sig);
            if (mode==PrunerMode::LOSSY)
                return allows_lossy(sig);

            for (i32 i=1; i<sig.size(); i++)
                _trimmed_sig[i-1]=std::max(0,sig.at(i)-1);
            if (mode==PrunerMode::SUFFIX_ARRAY)
                return suffix_array.value().contains(_trimmed_sig);
            if (mode==PrunerMode::PREFIX_TREE)
                return trie.value().has_ptwise_ge(_trimmed_sig);
            throw std::runtime_error("Pruner::allows: unrecognized compute mode");
        }
        std::string compute_mode() {
            if (mode==PrunerMode::SUFFIX_ARRAY)
                return "suffix array";
            if (mode==PrunerMode::PREFIX_TREE)
                return "prefix tree";
            if (mode==PrunerMode::LOSSY)
                return "lossy";
            throw std::runtime_error("Pruner::compute_mode: unrecognized compute mode");
        }
};

class MatrixLookup {
    private:
        i32 m, n;
        vi32 mat_ranks;
        std::map<pack_t, vTensor> rank1_cpds;
    public:
        ~MatrixLookup() {}
        MatrixLookup(i32 m, i32 n) {
            this->m=m;
            this->n=n;
            mat_ranks=vi32(1L<<(m*n),-1);
            for (const Tensor &mat:all_tensors({m,n}))
                mat_ranks[packed(mat)]=mat_rank(mat);
            rank1_cpds={};
            for (const Tensor &v0:all_tensors({m}))
                for (const Tensor &v1:all_tensors({n})) {
                    Tensor mat=outer_prod({v0,v1});
                    pack_t p=packed(mat);
                    if (rank1_cpds.find(p)==rank1_cpds.end())
                        rank1_cpds.insert({p,vTensor{v0,v1}});
                }
        }
        vTensor rank1_cpd(pack_t p) {
            return rank1_cpds.at(p);
        }
        i32 rank(pack_t p) {
            return mat_ranks.at(p);
        }
};
class Search {
    private:
        i32 n0, n1, n2;
        MatrixLookup lookup;
        vvTensor all_tup1s;
        vpack_t all_tup1_packed_outer_prods;
        std::map<i32,Pruner> pruners;
        vi32 _rank_signature;  // used in Search::is_admissible()

        // fields that get updated during DFS
        VectorSpan axis0_span;
        vi32 tup1_idxs_buffer;  // should not be used directly but via Tup1_Subset
        typedef struct {
            i32 depth;  // represents subarray tup1_idxs_buffer[:depth]
        } Tup1_Subset;
        std::vector<u64> nvisited, npruned;
        i32 tup1_subset_size(const Tup1_Subset &subset) {
            return subset.depth;
        }
        pack_t get_tup1_packed_outer_prod(const Tup1_Subset &subset, i32 p) {
            assert(0<=p && p<subset.depth, "Search::get_tup1_packed_outer_prod: invalid index for DFS stack");
            return all_tup1_packed_outer_prods.at(tup1_idxs_buffer.at(p));
        }
        vTensor get_tup1(const Tup1_Subset &subset, i32 p) {
            assert(0<=p && p<subset.depth, "Search::get_tup1_packed_outer_prod: invalid index for DFS stack");
            return all_tup1s.at(tup1_idxs_buffer.at(p));
        }
        pack_t get_contraction(const vpack_t &T_slices, const Tup1_Subset &subset, pack_t q, pack_t u) {
            i32 P=tup1_subset_size(subset);
            assert(0<=q && q<(1L<<n0), "Search::get_contraction: q out of domain");
            assert(0<=u && u<(1L<<P), "Search::get_contraction: u out of domain");
            pack_t contraction=0;
            for (i32 i=0; i<n0; i++)
                contraction^=bit_at<pack_t,pack_t>(q,i)*T_slices.at(i);
            for (i32 p=0; p<P; p++)
                contraction^=bit_at<pack_t,pack_t>(u,p)*get_tup1_packed_outer_prod(subset,p);
            return contraction;
        }
        std::optional<vvTensor> base_case(const vpack_t &T_slices, const Tup1_Subset &subset) {
            assert(T_slices.size()==n0, "Search::base_case: wrong number of slices");
            i32 P=tup1_subset_size(subset);

            axis0_span.reset();
            for_all_words<pack_t>(n0, [&](pack_t q) {
                if (!axis0_span.contains(q)) {
                    for_all_words<pack_t>(P, [&](pack_t u) {
                        pack_t contraction=get_contraction(T_slices,subset,q,u);
                        if (lookup.rank(contraction)<=1) {
                            axis0_span.add(q);
                            return true;
                        }
                        return false;
                    });
                }
                return false;
            });
            if (axis0_span.get_dim()!=n0)
                return {};

            vTensor q_basis, used_us;
            vpack_t used_contractions;
            axis0_span.reset();
            for_all_words<pack_t>(n0, [&](pack_t q) {
                if (!axis0_span.contains(q)) {
                    for_all_words<pack_t>(P, [&](pack_t u) {
                        pack_t contraction=get_contraction(T_slices,subset,q,u);
                        if (lookup.rank(contraction)<=1) {
                            q_basis.push_back(unpacked_vec(q,n0));
                            used_us.push_back(unpacked_vec(u,P));
                            used_contractions.push_back(contraction);
                            axis0_span.add(q);
                            return true;
                        }
                        return false;
                    });
                }
                return false;
            });
            assert(q_basis.size()==n0, "Search::base_case: base case unexpectedly not a solution");

            Tensor Q=stack_vecs(n0,n0,q_basis);
            Tensor U=stack_vecs(n0,P,used_us);
            // Q @_0 T + cpd(U, B1, B2) = cpd(I, A1, A2)
            // ==> T = cpd(Q^{-1} @ [I | -U], [A1 | B1], [A2, B2])
            // over mod 2, -1 = 1
            Tensor iQ=mat_inv(Q);
            Tensor iQ_U=axis0_op(iQ,U);
            vvTensor cpd;
            for (i32 i=0; i<n0; i++)
                cpd.push_back(concat(vvTensor{
                    {iQ.mat_clm(i)},
                    lookup.rank1_cpd(used_contractions.at(i)),
                }));
            for (i32 p=0; p<P; p++)
                cpd.push_back(concat(vvTensor{
                    {iQ_U.mat_clm(p)},
                    get_tup1(subset,p),
                }));
            return cpd;
        }
        bool is_admissible(const vpack_t &T_slices, const Tup1_Subset &subset, i32 R) {
            assert(T_slices.size()==n0, "Search::is_admissible: wrong number of slices");
            i32 P=tup1_subset_size(subset);
            i32 rem_R=R-P;
            for_all_words<pack_t>(n0, [&](pack_t q) {
                i32 min_rank;
                if (q==0)
                    min_rank=0;
                else {
                    min_rank=INT32_MAX;
                    for_all_words<pack_t>(P, [&](pack_t u) {
                        pack_t contraction=get_contraction(T_slices,subset,q,u);
                        min_rank=std::min(min_rank,lookup.rank(contraction));
                        return false;
                    });
                }
                _rank_signature[q]=min_rank;
                return false;
            });
            return pruners.at(rem_R).allows(_rank_signature);
        }
        std::optional<vvTensor> dfs(const vpack_t &T_slices, i32 R, i32 P) {
            assert(T_slices.size()==n0, "Search::dfs: wrong number of slices");
            assert(n0+P<=R, "Search::dfs: too many tup1s");
            nvisited[P]++;
            Tup1_Subset subset{.depth=P};
            std::optional<vvTensor> base_ret=base_case(T_slices,subset);
            if (base_ret.has_value())
                return base_ret;
            if (n0+P==R)
                return {};
            if (!is_admissible(T_slices,subset,R)) {
                npruned[P]++;
                return {};
            }
            for (i32 idx=(P>0?tup1_idxs_buffer.at(P-1)+1:0); idx<all_tup1s.size(); idx++) {
                tup1_idxs_buffer[P]=idx;
                std::optional<vvTensor> ret=dfs(T_slices,R,P+1);
                if (ret.has_value())
                    return ret;
            }
            return {};
        }
    public:
        ~Search() {}
        Search(i32 n0, i32 n1, i32 n2): lookup(n1,n2), axis0_span(n0) {
            assert(n0>0 && n1>0 && n2>0, "Search: size-0 tensor shapes not supported");
            this->n0=n0;
            this->n1=n1;
            this->n2=n2;
            _rank_signature=vi32(1<<n0,-1);
            all_tup1s={};
            all_tup1_packed_outer_prods={};
            for (const Tensor &v1:all_normalized_vecs(n1))
                for (const Tensor &v2:all_normalized_vecs(n2)) {
                    all_tup1s.push_back({v1,v2});
                    all_tup1_packed_outer_prods.push_back(packed(outer_prod({v1,v2})));
                }
        }
        typedef struct {
            Tensor T;
            i32 R;
            std::optional<vvTensor> cpd;
            std::vector<u64> nvisited;
            std::vector<u64> npruned;
        } Result;
        Result search(const Tensor &T, i32 R) {
            assert(T.ndim()==3 && T.len(0)==n0 && T.len(1)==n1 && T.len(2)==n2, "Search::search: wrong shape");
            assert(is_concise(T), "Search::search: tensor not concise");
            if (R<n0)
                return {
                    .T=T,
                    .R=R,
                    .cpd=std::nullopt,
                    .nvisited={},
                    .npruned={},
                };
            for (i32 r=n0; r<=R; r++)
                if (pruners.find(r)==pruners.end()) {
                    timept st=time();
                    // NOTE bound is min(n1,n2) (exclusive) because we subtract by 1 when calculating signatures
                    Pruner pruner(n0,r,std::min(n1,n2));
                    pruners.insert({r,pruner});
                    std::cout<<"pruner init:"
                        <<" n0="<<n0<<" n1="<<n1<<" n2="<<n2<<" rank="<<r
                        <<" ("<<pruner.compute_mode()<<")"
                        <<" time="<<seconds_since(st)<<" sec"
                        <<std::endl;
                }
            i32 slice_size=n1*n2;
            vpack_t packed_slices;
            for (i32 i=0; i<n0; i++) {
                vu8 data;
                for (i32 j=0; j<slice_size; j++)
                    data.push_back(T.at(i*slice_size+j));
                packed_slices.push_back(packed(Tensor({n1,n2},data)));
            }
            i32 depth_bound=R-n0+1;
            tup1_idxs_buffer=vi32(depth_bound,-1);
            nvisited=std::vector<u64>(depth_bound,0);
            npruned=std::vector<u64>(depth_bound,0);
            std::optional<vvTensor> cpd=dfs(packed_slices,R,0);
            return {
                .T=T,
                .R=R,
                .cpd=cpd,
                .nvisited=nvisited,
                .npruned=npruned,
            };
        }
};