#include "utils.cpp"
#include <functional>
#include <map>
#include <unordered_map>

#pragma pack(1)
class Tensor {
    public:
        typedef u64 chunk_t;
        static const i32 N_CHUNKS=2;
        static const i32 CHUNK_SIZE=8*sizeof(chunk_t);
        static const i32 MAX_SIZE=CHUNK_SIZE*N_CHUNKS;
        static const i32 MAX_NDIM=3;
    private:
        // row-major order of elements
        // everything beyond the lowest ``size`` many bits is 0
        chunk_t bits[N_CHUNKS]={0,0};
        u8 shape[MAX_NDIM];
        u8 _ndim;
        Tensor() {}
        u8 _at(i32 i) const {
            // omit bounds check for speed
            return bit_at<chunk_t,u8>(bits[i/CHUNK_SIZE],i%CHUNK_SIZE);
        }
        void _set(i32 i, u8 v) {
            // mutates internal data
            // omit bounds check for speed
            bits[i/CHUNK_SIZE]=bit_set(bits[i/CHUNK_SIZE],i%CHUNK_SIZE,v);
        }
    public:
        ~Tensor() {}
        Tensor(const vi32 &shape, const vu8 &data) {
            assert(shape.size()<=MAX_NDIM, "Tensor: ndim too large");
            assert(data.size()<=MAX_SIZE, "Tensor: size too large");
            assert(prod(shape)==data.size(), "Tensor: wrong number of elements");
            this->_ndim=shape.size();
            for (i32 d=0; d<shape.size(); d++) {
                assert(shape.at(d)>=0, "Tensor: negative length");
                this->shape[d]=shape.at(d);
            }
            for (i32 i=0; i<data.size(); i++)
                _set(i,data.at(i));
        }
        static Tensor zeros(const vi32 &shape) {
            assert(shape.size()<=MAX_NDIM, "Tensor::zeros: ndim too large");
            assert(prod(shape)<=MAX_SIZE, "Tensor::zeros: size too large");
            Tensor out;
            out._ndim=shape.size();
            for (i32 d=0; d<shape.size(); d++) {
                assert(shape.at(d)>=0, "Tensor::zeros: negative length");
                out.shape[d]=shape.at(d);
            }
            return out;
        }
        bool operator==(const Tensor &o) const {
            for (i32 c=0; c<N_CHUNKS; c++)
                if (bits[c]!=o.bits[c])
                    return false;
            if (_ndim!=o._ndim)
                return false;
            for (i32 d=0; d<_ndim; d++)
                if (shape[d]!=o.shape[d])
                    return false;
            return true;
        }
        bool operator!=(const Tensor &o) const {
            return !(*this==o);
        }
        static std::size_t hash(const Tensor &T) {
            std::size_t out=0;
            for (i32 c=0; c<N_CHUNKS; c++)
                out=out*3+std::hash<chunk_t>{}(T.bits[c]);
            for (i32 d=0; d<T.ndim(); d++)
                out=out*3+std::hash<i32>{}(T.len(d));
            return out;
        }

        i32 ndim() const {
            return _ndim;
        }
        i32 len(i32 ax) const {
            assert(0<=ax && ax<_ndim, "Tensor::len: axis out of range");
            return shape[ax];
        }
        vi32 get_shape() const {
            vi32 out;
            for (i32 d=0; d<_ndim; d++)
                out.push_back(shape[d]);
            return out;
        }
        i32 size() const {
            i32 out=1;
            for (i32 d=0; d<_ndim; d++)
                out*=shape[d];
            return out;
        }
        u8 at(i32 i) const {
            assert(0<=i && i<size(), "Tensor::at: index out of range");
            return _at(i);
        }
        Tensor slice0(i32 i) const {
            assert(0<=i && i<len(0), "Tensor::slice0: index out of range");
            // if assert passes, we know len(0) >= 1
            // so the size of the slice must be <= MAX_SIZE
            Tensor out;
            out._ndim=ndim()-1;
            for (i32 d=1; d<ndim(); d++)
                out.shape[d-1]=len(d);
            i32 slice_size=size()/len(0);
            for (i32 j=0; j<slice_size; j++)
                out._set(j,_at(i*slice_size+j));
            return out;
        }
        static Tensor augment(const Tensor &T, const Tensor &M) {
            assert(M.ndim()==T.ndim()-1, "Tensor::augment: wrong ndim");
            for (i32 d=1; d<T.ndim(); d++)
                assert(M.len(d-1)==T.len(d), "Tensor::augment: wrong shape");
            i32 t_sz=T.size(), m_sz=M.size();
            assert(t_sz+m_sz<=MAX_SIZE, "Tensor::augment: too large");
            Tensor out=T;
            out.shape[0]++;
            for (i32 j=0; j<m_sz; j++)
                out._set(t_sz+j,M._at(j));
            return out;
        }
        vu8 flattened() const {
            vu8 out;
            for (i32 i=0; i<size(); i++)
                out.push_back(at(i));
            return out;
        }

        // enumerate tensors *in lex increasing order*
        static void for_all_tensors_lex(const vi32 &shape, std::function<bool(const Tensor &)> func) {
            i32 size=prod(shape);
            Tensor T_zero=Tensor::zeros(shape);
            for_all_lists(size,[&](const vu8 &bits) {
                Tensor T=T_zero;
                for (i32 i=0; i<size; i++)
                    T._set(i,bits.at(i));
                return func(T);
            });
        }
        static std::vector<Tensor> all_tensors_lex(const vi32 &shape) {
            std::vector<Tensor> out;
            for_all_tensors_lex(shape,[&](const Tensor &T) {
                out.push_back(T);
                return false;
            });
            return out;
        }
        static std::vector<Tensor> all_augments_lex(const Tensor &T) {
            vi32 shape=T.get_shape();
            vi32 slice_shape(shape.begin()+1,shape.end());
            std::vector<Tensor> out;
            for (const Tensor &M:Tensor::all_tensors_lex(slice_shape))
                out.push_back(Tensor::augment(T,M));
            return out;
        }
        static Tensor eye(i32 n) {
            assert(n*n<=MAX_SIZE, "Tensor::eye: too large");
            Tensor out;
            out._ndim=2;
            out.shape[0]=n;
            out.shape[1]=n;
            for (i32 i=0; i<n; i++)
                out._set(i*n+i,1);
            return out;
        }

        // cannot define directly because struct sizes must be known at compile time
        template <class T>
        struct _RowReduceRet {
            T rref;
            T reducer;
            i32 rank;
        };
        typedef _RowReduceRet<Tensor> RowReduceRet;
        static RowReduceRet rref(const Tensor &mat) {
            assert(mat.ndim()==2, "Tensor::rref: input not 2D");
            i32 m=mat.len(0), n=mat.len(1);
            Tensor rref=mat;  // copy by value
            Tensor reducer=Tensor::eye(m);
            i32 rank=0;
            for (i32 j=0; j<n; j++) {
                i32 i0=-1;
                for (i32 i=rank; i<m; i++)
                    if (rref._at(i*n+j)!=0) {
                        i0=i;
                        break;
                    }
                if (i0<0)
                    continue;
                // swap rows rank, i0
                for (i32 k=0; k<n; k++) {
                    i32 idx0=rank*n+k, idx1=i0*n+k;
                    u8 tmp=rref._at(idx0);
                    rref._set(idx0,rref._at(idx1));
                    rref._set(idx1,tmp);
                }
                for (i32 k=0; k<m; k++) {
                    i32 idx0=rank*m+k, idx1=i0*m+k;
                    u8 tmp=reducer._at(idx0);
                    reducer._set(idx0,reducer._at(idx1));
                    reducer._set(idx1,tmp);
                }
                // all nonzero rows are already normalized over mod 2
                // subtract row i0 from all other rows
                for (i32 i=0; i<m; i++)
                    if (i!=rank && rref._at(i*n+j)!=0) {
                        for (i32 k=0; k<n; k++)
                            rref._set(i*n+k,rref._at(i*n+k)^rref._at(rank*n+k));
                        for (i32 k=0; k<m; k++)
                            reducer._set(i*m+k,reducer._at(i*m+k)^reducer._at(rank*m+k));
                    }
                rank++;
            }
            return {
                .rref=rref,
                .reducer=reducer,
                .rank=rank,
            };
        }
        static Tensor axis_op(const Tensor &M, const Tensor &T, i32 ax) {
            assert(0<=ax && ax<T.ndim(), "Tensor::axis_op: axis out of range");
            assert(M.ndim()==2, "Tensor::axis_op: M not 2D");
            assert(M.len(1)==T.len(ax), "Tensor::axis_op: inner dimensions do not match");

            Tensor out=T;
            out.shape[ax]=M.len(0);
            assert(out.size()<=MAX_SIZE, "Tensor::axis_op: new tensor too large");

            i32 npre=1;
            for (i32 d=0; d<ax; d++)
                npre*=T.len(d);
            i32 nsuf=1;
            for (i32 d=ax+1; d<T.ndim(); d++)
                nsuf*=T.len(d);
            i32 nmid=T.len(ax);
            i32 nnew=M.len(0);
            // treat T as npre x nmid x nsuf
            for (i32 ipre=0; ipre<npre; ipre++)
                for (i32 inew=0; inew<nnew; inew++)
                    for (i32 isuf=0; isuf<nsuf; isuf++) {
                        u8 tot=0;
                        for (i32 imid=0; imid<nmid; imid++)
                            tot=(tot+M._at(inew*nmid+imid)*T._at((ipre*nmid+imid)*nsuf+isuf))%MOD;
                        out._set((ipre*nnew+inew)*nsuf+isuf,tot);
                    }
            return out;
        }
        static Tensor contract0(const Tensor &v, const Tensor &T) {
            assert(v.ndim()==1, "Tensor::contract0: v not 1D");
            assert(v.len(0)==T.len(0), "Tensor::contract0: inner dimensions do not match");

            Tensor out;
            out._ndim=T.ndim()-1;
            for (i32 d=1; d<T.ndim(); d++)
                out.shape[d-1]=T.len(d);

            i32 nsuf=out.size();
            assert(nsuf<=MAX_SIZE, "Tensor::contract0: new tensor too large");

            i32 n0=T.len(0);
            for (i32 j=0; j<nsuf; j++) {
                u8 tot=0;
                for (i32 i=0; i<n0; i++)
                    tot^=v._at(i)*T._at(i*nsuf+j);
                out._set(j,tot);
            }
            return out;
        }
        static Tensor unfold(const Tensor &T, i32 ax) {
            assert(0<=ax && ax<T.ndim(), "Tensor::unfold: axis out of range");
            i32 npre=1;
            for (i32 d=0; d<ax; d++)
                npre*=T.len(d);
            i32 nsuf=1;
            for (i32 d=ax+1; d<T.ndim(); d++)
                nsuf*=T.len(d);
            i32 nmid=T.len(ax);

            // total number of elements is the same as that of T,
            // so it is already <= MAX_SIZE
            Tensor out;
            out._ndim=2;
            out.shape[0]=nmid;
            out.shape[1]=npre*nsuf;

            // treat T as having shape (npre,nmid,nnew)
            // permute axes to (nmid,npre,nnew)
            for (i32 ipre=0; ipre<npre; ipre++)
                for (i32 imid=0; imid<nmid; imid++)
                    for (i32 isuf=0; isuf<nsuf; isuf++)
                        out._set((imid*npre+ipre)*nsuf+isuf,T._at((ipre*nmid+imid)*nsuf+isuf));
            return out;
        }
        static Tensor truncate0(const Tensor &T, i32 nnew) {
            assert(T.ndim()>0, "Tensor::truncate0: ndim too small");
            assert(0<=nnew && nnew<=T.len(0), "Tensor::truncate0: target length too long");
            // NOTE branch on nnew==0 so that the other branch guarantees T.shape[0]!=0
            i32 nsize=nnew==0?0:T.size()/T.shape[0]*nnew;
            Tensor out;
            out._ndim=T.ndim();
            out.shape[0]=nnew;
            for (i32 d=1; d<T.ndim(); d++)
                out.shape[d]=T.len(d);
            for (i32 i=0; i<nsize; i++)
                out._set(i,T._at(i));
            return out;
        }
};
#pragma pack()

struct TensorHash {
    std::size_t operator()(const Tensor& T) const {
        return Tensor::hash(T);
    }
};

typedef std::pair<Tensor,Tensor> pTensor;
typedef std::vector<Tensor> vTensor;
typedef std::vector<pTensor> vpTensor;
typedef std::vector<vTensor> vvTensor;
std::ostream& operator << (std::ostream &os, const Tensor &T) {
    os<<"Tensor("<<T.get_shape()<<", [";
    for (i32 i=0; i<T.size(); i++)
        os<<(i>0?", ":"")<<(i32)T.at(i);
    os<<"])";
    return os;
}

Tensor mat_mul(const Tensor &A, const Tensor &B) {
    assert(A.ndim()==2 && B.ndim()==2, "mat_mul: not 2D");
    return Tensor::axis_op(A,B,0);
}
i32 mat_rank(const Tensor &mat) {
    assert(mat.ndim()==2, "mat_rank: not 2D");
    return Tensor::rref(mat).rank;
}
bool mat_is_invle(const Tensor &M) {
    assert(M.ndim()==2, "mat_is_invle: not 2D");
    if (M.len(0)!=M.len(1))
        return false;
    return mat_rank(M)==M.len(0);
}
std::optional<Tensor> mat_inv(const Tensor &M) {
    assert(M.ndim()==2, "mat_inv: not 2D");
    if (M.len(0)!=M.len(1))
        return {};
    i32 n=M.len(0);
    Tensor::RowReduceRet ret=Tensor::rref(M);
    if (ret.rank!=n)
        return {};
    return ret.reducer;
}
vTensor all_invles(i32 n) {
    vTensor out;
    for (const Tensor &M:Tensor::all_tensors_lex({n,n})) {
        if (mat_is_invle(M))
            out.push_back(M);
    }
    return out;
}

// enumerate all tensors mod 2 of given shape, in lexicographic order
vTensor all_tensors_lex(const vi32 &shape) {
    vTensor out;
    for (const vu8 &elems:all_lists(prod(shape)))
        out.push_back(Tensor(shape,elems));
    return out;
}

// list all invertible Q s.t. Q@A=B
vTensor all_left_invles(const Tensor &A, const Tensor &B) {
    assert(A.ndim()==2 && B.ndim()==2, "all_left_invles: wrong ndim");
    assert(A.len(0)==B.len(0) && A.len(1)==B.len(1), "all_left_invles: mismatched shapes");
    i32 n0=A.len(0);
    vvTensor allowed_rowss;
    for (i32 i=0; i<n0; i++) {
        Tensor B0=B.slice0(i);
        vTensor rows;
        Tensor::for_all_tensors_lex({n0},[&](const Tensor &v0) {
            if (Tensor::contract0(v0,A)==B0)
                rows.push_back(v0);
            return false;
        });
        allowed_rowss.push_back(rows);
    }
    vi32 counts;
    for (const vTensor &rows:allowed_rowss)
        counts.push_back(rows.size());
    vTensor out;
    for_all_idxs(counts,[&](const vi32 &idxs) {
        Tensor Q=Tensor::zeros({0,n0});
        for (i32 i=0; i<n0; i++)
            Q=Tensor::augment(Q,allowed_rowss.at(i).at(idxs.at(i)));
        if (mat_is_invle(Q))
            out.push_back(Q);
        return false;
    });
    return out;
}

class Suffix_Iso12 {
    private:
        i32 n1, n2;
        vTensor op1s, op2s;
        std::unordered_map<Tensor,Tensor,TensorHash> op1_inv_mem, op2_inv_mem;
        typedef struct {
            Tensor orbit_rep;
            Tensor P;
            Tensor Q;
            Tensor iP;
            Tensor iQ;
        } iso12_act;
        typedef std::unordered_map<Tensor,iso12_act,TensorHash> iso12_table;
        std::unordered_map<Tensor,iso12_table,TensorHash> iso12_mem;
        std::unordered_map<Tensor,vpTensor,TensorHash> auto12_mem;
        // return list of all (P, Q) s.t. P @1 Q @2 T = T
        vpTensor make_auto12(const Tensor &T) {
            if (auto12_mem.find(T)!=auto12_mem.end())
                return auto12_mem.at(T);
            assert(T.len(0)>0, "Suffix_Iso12::make_auto12: zero-length tensor too costly");
            vpTensor out;
            if (T.len(0)==1) {
                Tensor M=T.slice0(0);
                for (const Tensor &Q:op2s) {
                    Tensor M1=Tensor::axis_op(Q,M,1);
                    for (const Tensor &P:all_left_invles(M1,M)) {
                        // assert(Tensor::axis_op(P,Tensor::axis_op(Q,M,1),0)==M, "Suffix_Iso12::make_auto12: incorrect pair for auto-group");
                        out.push_back({P,Q});
                    }
                }
            }
            else {
                Tensor pre=Tensor::truncate0(T,T.len(0)-1);
                assert(auto12_mem.find(pre)!=auto12_mem.end(), "Suffix_Iso12::make_auto12: expected prefix to already be calculated");
                Tensor last_slice=T.slice0(T.len(0)-1);
                for (const pTensor &p:auto12_mem.at(pre)) {
                    if (Tensor::axis_op(p.first,Tensor::axis_op(p.second,last_slice,1),0)==last_slice)
                        out.push_back(p);
                }
            }
            auto12_mem.insert({T,out});
            return out;
        }
        iso12_table make_iso12_no_prefix() {
            // canonicalize along axis 1, then along axis 2
            typedef struct {
                Tensor orbit_rep;
                Tensor op;
            } iso_single_act;
            std::unordered_map<Tensor,iso_single_act,TensorHash> left_table;
            {
                for (const Tensor &M:Tensor::all_tensors_lex({n1,n2})) {
                    if (left_table.find(M)!=left_table.end())
                        continue;
                    for (const Tensor &P:op1s) {
                        Tensor nM=Tensor::axis_op(P,M,0);
                        left_table.insert({
                            nM,
                            {
                                .orbit_rep=M,
                                .op=op1_inv_mem.at(P),
                            }
                        });
                    }
                }
            }
            std::unordered_map<Tensor,iso_single_act,TensorHash> right_table;
            {
                for (const Tensor &M:Tensor::all_tensors_lex({n1,n2})) {
                    if (right_table.find(M)!=right_table.end())
                        continue;
                    for (const Tensor &Q:op2s) {
                        Tensor nM=Tensor::axis_op(Q,M,1);
                        right_table.insert({
                            nM,
                            {
                                .orbit_rep=M,
                                .op=op2_inv_mem.at(Q),
                            }
                        });
                    }
                }
            }
            iso12_table out;
            for (const Tensor &M:Tensor::all_tensors_lex({n1,n2})) {
                const iso_single_act &left=left_table.at(M);
                const iso_single_act &right=right_table.at(left.orbit_rep);
                // assert(Tensor::axis_op(left.op,Tensor::axis_op(right.op,M,1),0)==right.orbit_rep, "iso12 incorrect");
                out.insert({
                    M,
                    {
                        .orbit_rep=right.orbit_rep,
                        .P=left.op,
                        .Q=right.op,
                        .iP=op1_inv_mem.at(left.op),
                        .iQ=op2_inv_mem.at(right.op),
                    }
                });
            }
            return out;
        }
        iso12_table make_iso12(const Tensor &T) {
            assert(T.ndim()==3, "Suffix_Iso12::make_iso12: wrong ndim");
            assert(T.len(1)==n1 && T.len(2)==n2, "Suffix_Iso12::make_iso12: wrong shape");
            if (T.len(0)==0)
                return make_iso12_no_prefix();
            vpTensor auto_group=make_auto12(T);
            iso12_table out;
            for (const Tensor &M:Tensor::all_tensors_lex({n1,n2})) {
                if (out.find(M)!=out.end())
                    continue;
                for (const pTensor &act:auto_group) {
                    Tensor nM=Tensor::axis_op(act.first,Tensor::axis_op(act.second,M,1),0);
                    out.insert({
                        nM,
                        {
                            .orbit_rep=M,
                            .P=op1_inv_mem.at(act.first),
                            .Q=op2_inv_mem.at(act.second),
                            .iP=act.first,
                            .iQ=act.second,
                        }
                    });
                }
            }
            return out;
        }
    public:
        ~Suffix_Iso12() {}
        Suffix_Iso12(i32 n1, i32 n2) {
            this->n1=n1;
            this->n2=n2;
            iso12_mem={};
            op1s=all_invles(n1);
            op2s=all_invles(n2);
            op1_inv_mem={};
            for (const Tensor &P:op1s)
                op1_inv_mem.insert({P,mat_inv(P).value()});
            op2_inv_mem={};
            for (const Tensor &Q:op2s)
                op2_inv_mem.insert({Q,mat_inv(Q).value()});
        }
        void prepare(const Tensor &T) {
            assert(T.ndim()==3, "Suffix_Iso12::prepare: wrong ndim");
            assert(T.len(1)==n1 && T.len(2)==n2, "Suffix_Iso12::prepare: T wrong shape");
            assert(iso12_mem.find(T)==iso12_mem.end(), "Suffix_Iso12::prepare: already exists");
            iso12_mem.insert({T,make_iso12(T)});
        }

        // given T n0 x n1 x n2, M n1 x n2, M' n1 x n2:
        // return P n1 x n1, Q n2 x n2
        // such that P @1 Q @2 stack([*T,M], axis=0) = stack([*T,M'], axis=0),
        // or detect if such (P,Q) do not exist
    private:
        Tensor _prev_tensor=Tensor::zeros({0});
        iso12_table *_cur_table=NULL;
    public:
        std::optional<pTensor> iso12(const Tensor &T, const Tensor &M0, const Tensor &M1) {
            assert(T.ndim()==3, "Suffix_Iso12::iso12: wrong ndim");
            assert(T.len(1)==n1 && T.len(2)==n2, "Suffix_Iso12::iso12: T wrong shape");
            assert(M0.len(0)==n1 && M0.len(1)==n2, "Suffix_Iso12::iso12: M0 wrong shape");
            assert(M1.len(0)==n1 && M1.len(1)==n2, "Suffix_Iso12::iso12: M1 wrong shape");
            if (_prev_tensor!=T || _cur_table==NULL) {
                _prev_tensor=T;
                _cur_table=&iso12_mem.at(T);
            }
            const iso12_act &act0=_cur_table->at(M0);
            const iso12_act &act1=_cur_table->at(M1);
            if (act0.orbit_rep!=act1.orbit_rep)
                return {};
            return pTensor{
                mat_mul(act1.iP,act0.P),
                mat_mul(act1.iQ,act0.Q),
            };
        }
};

// return lexically minimal 1 x m x n tensor whose axis-0 slice has rank r
Tensor lex_min_singleton_tensor(i32 m, i32 n, i32 r) {
    assert(0<=r && r<=std::min(m,n), "lex_min_singleton_tensor: rank out of range");
    vu8 out(m*n,0);
    for (i32 i=0; i<r; i++)
        out[(m-1-i)*n+(n-r+i)]=1;
    return Tensor({1,m,n},out);
}
bool lex_lt(const Tensor &A, const Tensor &B) {
    assert(A.ndim()==B.ndim(), "lex_lt: different ndims");
    for (i32 d=0; d<A.ndim(); d++)
        assert(A.len(d)==B.len(d), "lex_lt: different shapes");
    for (i32 i=0; i<A.size(); i++) {
        i32 a=A.at(i), b=B.at(i);
        if (a<b)
            return true;
        if (a>b)
            return false;
    }
    return false;
}

class VectorSpan {
    public:
        typedef u16 pack_t;
        static const int PACK_SIZE=8*sizeof(pack_t);
        typedef std::vector<pack_t> vpack_t;
    private:
        i32 n;
        i32 size;
        std::vector<bool> in_span;
        vpack_t span_buffer;  // span_buffer[:size] is the set of all vectors contained in this object
    public:
        ~VectorSpan() {}
        VectorSpan(i32 n) {
            this->n=n;
            assert(0<=n && n<=PACK_SIZE, "VectorSpan: ambient dimension out of allowed range");
            in_span=std::vector<bool>(1L<<n,false);
            span_buffer=vpack_t(1L<<n,0);
            reset();
        }
        void reset() {
            size=1;
            std::fill(in_span.begin(),in_span.end(),false);
            in_span[0]=true;
            std::fill(span_buffer.begin(),span_buffer.end(),0);
        }
        void add(pack_t packed_v) {
            assert(0<=packed_v && packed_v<(1L<<n), "VectorSpan::add: out of domain");
            if (in_span.at(packed_v))
                return;
            for (i32 i=0; i<size; i++) {
                pack_t new_elem=span_buffer.at(i)^packed_v;
                in_span[new_elem]=true;
                span_buffer[i+size]=new_elem;
            }
            size*=MOD;
        }
        bool contains(pack_t packed_v) const {
            assert(0<=packed_v && packed_v<(1L<<n), "VectorSpan::contains: out of domain");
            return in_span.at(packed_v);
        }
};

typedef u16 lex_pack_t;
const int LEX_PACK_SIZE=8*sizeof(lex_pack_t);
typedef std::vector<lex_pack_t> vlex_pack_t;
typedef struct {
    vlex_pack_t T_slices;
    std::vector<std::pair<lex_pack_t,VectorSpan::pack_t>> pairs;
    VectorSpan span;
} is_canonical0_mem_t;
std::map<i32,is_canonical0_mem_t> _IS_CANONICAL0_BUFFERS;
bool is_canonical0(const Tensor &T) {
    i32 n0=T.len(0);
    if (_IS_CANONICAL0_BUFFERS.find(n0)==_IS_CANONICAL0_BUFFERS.end()) {
        _IS_CANONICAL0_BUFFERS.insert({n0,{
            .T_slices=vlex_pack_t(n0,0),
            .pairs=std::vector<std::pair<lex_pack_t,VectorSpan::pack_t>>(1L<<n0,{0,0}),
            .span=VectorSpan(n0),
        }});
    }
    is_canonical0_mem_t &mem=_IS_CANONICAL0_BUFFERS.at(n0);
    if (n0>0) {
        // bitpack axis-0 slices in lexicographic order (big-endian)
        i32 slice_size=T.size()/n0;
        assert(slice_size<=LEX_PACK_SIZE, "is_canonical0: slice size too large");
        for (i32 i=0; i<n0; i++) {
            mem.T_slices[i]=0;
            for (i32 j=0; j<slice_size; j++)
                mem.T_slices[i]|=T.at(i*slice_size+j)<<(slice_size-1-j);
        }
    }
    for_all_words<VectorSpan::pack_t>(n0,[&](VectorSpan::pack_t v) {
        lex_pack_t w=0;
        for (i32 i=0; i<n0; i++)
            w^=bit_at<VectorSpan::pack_t,lex_pack_t>(v,i)*mem.T_slices.at(i);
        mem.pairs[v]={w,v};
        return false;
    });
    sort(mem.pairs);
    mem.span.reset();
    i32 i0=0;
    for (const std::pair<lex_pack_t,VectorSpan::pack_t> &pair:mem.pairs) {
        VectorSpan::pack_t v=pair.second;
        if (!mem.span.contains(v)) {
            mem.span.add(v);
            lex_pack_t can_slice=pair.first;
            lex_pack_t T_slice=mem.T_slices.at(i0);
            assert(T_slice>=can_slice, "is_canonical0: tensor lex smaller than axis-0 canonical");
            if (T_slice>can_slice)
                return false;
            i0++;
        }
    }
    assert(i0==n0, "is_canonical0: failed to recover tensor shape");
    return true;
}
vTensor all_canonicals(const vi32 &shape) {
    assert(shape.size()==3, "all_canonicals: ndim!=3 not implemented");
    i32 n0=shape.at(0), n1=shape.at(1), n2=shape.at(2);
    std::cout<<"shape="<<shape<<std::endl;
    Suffix_Iso12 iso12_calc(n1,n2);
    vvTensor concise_fronts; {
        concise_fronts.push_back({Tensor::zeros({0,n1,n2})});
        vTensor fronts1;
        for (i32 r=1; r<=std::min(n1,n2); r++)
            fronts1.push_back(lex_min_singleton_tensor(n1,n2,r));
        concise_fronts.push_back(fronts1);

        timept st=time();
        for (i32 l0=0; l0<2; l0++)
            for (const Tensor &T:concise_fronts.at(l0))
            iso12_calc.prepare(T);
        std::cout<<"\tiso12 memo for l0<2: "<<seconds_since(st)<<std::endl;
    }
    std::unordered_map<Tensor,vTensor,TensorHash> mem_augments;
    std::function<vTensor(const Tensor &)> all_surjective_augments=[&](const Tensor &A) {
        assert(A.ndim()==2, "all_canonicals::all_surjective_augments: not 2D");
        if (mem_augments.find(A)!=mem_augments.end())
            return mem_augments.at(A);
        i32 m=A.len(0), n=A.len(1);
        vTensor out;
        for (const Tensor &v:Tensor::all_tensors_lex({n})) {
            Tensor B=Tensor::augment(A,v);
            if (mat_rank(B)>m)
                out.push_back(B);
        }
        mem_augments.insert({A,out});
        return out;
    };
    for (i32 l0=2; l0<=n0; l0++) {
        std::cout<<"\tl0="<<l0<<std::endl;
        timept st=time();
        double time_mark=10;
        vvTensor nfront_chunks;
        u64 n_prefixes=0, n_tried=0;
        std::vector<u64> n_sur0_iso12(l0,0), n_iso12(l0,0);
        std::function<void(bool)> _log_time=[&](bool force) {
            double elapsed=seconds_since(st);
            if (force || elapsed>time_mark) {
                std::cout<<"\t\tn_prefixes="<<n_prefixes
                    <<" n_tried="<<n_tried
                    <<" n_sur0_iso12="<<n_sur0_iso12
                    <<" n_iso12="<<n_iso12
                    <<" search_time="<<elapsed
                    <<std::endl;
                while (elapsed>time_mark)
                    time_mark*=2;
            }
        };
        mem_augments.clear();
        for (const Tensor &Tpre:concise_fronts.at(l0-1)) {
            n_prefixes++;
            vTensor pre_nfront;
            vvTensor to_check;
            for (i32 k=0; k<l0-1; k++) {
                to_check.push_back({});
                for (const Tensor &F:concise_fronts.at(k+1)) {
                    if (Tensor::truncate0(F,k)==Tensor::truncate0(Tpre,k) && lex_lt(F,Tensor::truncate0(Tpre,k+1)))
                        to_check.at(k).push_back(F);
                }
            }
            std::function<bool(const Tensor &)> is_good=[&](const Tensor &T) {
                // T l0 x n1 x n2
                // maintain (P p x l0, S l0 x n1 x n2) pairs for p<l0
                // s.t. P @0 S = Tpre[:p,...]
                typedef struct {
                    Tensor P;
                    Tensor S;
                } State;
                std::vector<State> states{{.P=Tensor::zeros({0,l0}),.S=T}};
                for (i32 k=0; k<l0; k++) {
                    Tensor pre=Tensor::truncate0(T,k);
                    // for (const State &state:states)
                    //     assert(Tensor::axis_op(state.P,state.S,0)==pre, "all_canonicals::is_good: invariant violated during DFS");
                    n_sur0_iso12[k]+=states.size();
                    for (const State &state:states) {
                        bool match=false;
                        for (const Tensor &nP:all_surjective_augments(state.P)) {
                            Tensor M=Tensor::contract0(nP.slice0(k),state.S);
                            for (const Tensor &F:(k==l0-1?pre_nfront:to_check.at(k))) {
                                n_iso12[k]++;
                                if (iso12_calc.iso12(pre,M,F.slice0(k)).has_value()) {
                                    match=true;
                                    break;
                                }
                            }
                            if (match)
                                break;
                        }
                        if (match)
                            return false;
                    }
                    if (k==l0-1)
                        break;
                    std::vector<State> nstates;
                    for (const State &state:states) {
                        for (const Tensor &nP:all_surjective_augments(state.P)) {
                            Tensor M=Tensor::contract0(nP.slice0(k),state.S);
                            n_iso12[k]++;
                            std::optional<pTensor> _ret=iso12_calc.iso12(pre,M,T.slice0(k));
                            if (_ret.has_value()) {
                                const pTensor &ret=_ret.value();
                                const Tensor &Q1=ret.first, &Q2=ret.second;
                                nstates.push_back({.P=nP,.S=Tensor::axis_op(Q1,Tensor::axis_op(Q2,state.S,2),1)});
                            }
                        }
                    }
                    states=nstates;
                }
                return true;
            };
            for (const Tensor &T:Tensor::all_augments_lex(Tpre)) {
                if (mat_rank(Tensor::unfold(T,0))!=l0)
                    continue;
                if (!is_canonical0(T))
                    continue;
                n_tried++;
                if (is_good(T)) {
                    pre_nfront.push_back(T);
                }
                _log_time(false);
            }
            nfront_chunks.push_back(pre_nfront);
        }
        concise_fronts.push_back(concat(nfront_chunks));
        _log_time(true);
        std::cout<<"\t\tn_concise_canonicals="<<concise_fronts.at(l0).size()
            <<std::endl;

        if (l0<n0) {
            st=time();
            for (const Tensor &T:concise_fronts.at(l0))
                iso12_calc.prepare(T);
            std::cout<<"\t\tiso12 memo: "<<seconds_since(st)<<std::endl;
        }
    }
    vTensor out;
    for (const Tensor &T:concat(concise_fronts)) {
        out.push_back(Tensor({n0,n1,n2},concat(vvu8{vu8((n0-T.len(0))*n1*n2,0),T.flattened()})));
    }
    std::cout<<"tot_canonicals="<<out.size()<<std::endl;
    std::cout<<"--------------------------------"<<std::endl;
    return out;
}