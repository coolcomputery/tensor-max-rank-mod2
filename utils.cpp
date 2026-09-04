#ifndef UTILS
#define UTILS

#include <cstdint>

using std::uint8_t;
using std::uint16_t;
using std::uint32_t;
using std::uint64_t;
using std::int8_t;
using std::int16_t;
using std::int32_t;
using std::int64_t;
typedef std::uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

#include <stdexcept>
void assert(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template<typename in_t, typename out_t>
out_t bit_at(in_t v, i32 i) {
    return (out_t)((v>>i)&((in_t)1));
}

template<typename word_t>
word_t bit_set(word_t w, i32 i, u8 v) {
    assert(v==0 || v==1, "bit_set: invalid value");
    word_t cleared=w&(~(((word_t)1)<<i));
    return cleared|(((word_t)v)<<i);
}

#include <vector>
typedef std::vector<bool> vbool;
typedef std::vector<i32> vi32;
typedef std::vector<vi32> vvi32;

template<typename T>
std::vector<T> concat(const std::vector<std::vector<T>> &lists) {
    std::vector<T> out;
    for (const std::vector<T> &list:lists)
        for (const T &v:list)
            out.push_back(v);
    return out;
}

// use u8 to store unpacked tensor elements
typedef std::vector<u8> vu8;
typedef std::vector<vu8> vvu8;
const u8 MOD=2;

i32 prod(const vi32 &shape) {
    i32 prod=1;
    for (i32 n:shape)
        prod*=n;
        return prod;
}
vvu8 all_lists(i32 n) {
    if (n==0)
        return {{}};
    vvu8 out;
    vu8 elems(n,0);
    while (elems.at(0)<MOD) {
        out.push_back(vu8(elems));
        
        elems[n-1]++;
        for (i32 i=n-1; i>0 && elems.at(i)>=MOD; i--) {
            elems[i]=0;
            elems[i-1]++;
        }
    }
    return out;
}

#include <functional>
// precision-safe method to iterate over all binary words up to given number of bits
// func: (word_t,) -> bool
template<typename word_t>
void for_all_words(i32 size, std::function<bool(word_t)> func) {
    assert(size<64, "for_all_words: size>=64 not supported");
    assert(size<=8*sizeof(word_t), "for_all_words: size larger than width of word type");
    for (u64 _w=0; _w<(1L<<size); _w++) {
        bool stop=func((word_t)_w);
        if (stop)
            break;
    }
}

void for_all_lists(i32 n, std::function<bool(const vu8 &)> func) {
    if (n==0) {
        func({});
        return;
    }
    vu8 elems(n,0);
    while (elems.at(0)<MOD) {
        bool stop=func(elems);
        if (stop)
            return;
        elems[n-1]++;
        for (i32 i=n-1; i>0 && elems.at(i)>=MOD; i--) {
            elems[i]=0;
            elems[i-1]++;
        }
    }
}

void for_all_multisets(i32 n, i32 k, std::function<bool(const vi32 &)> func) {
    assert(n>=0, "for_all_multisets: n negative");
    assert(k>=0, "for_all_multisets: k negative");
    vi32 elems(k,-1);
    bool stop_dfs=false;
    std::function<void(i32,i32)> dfs=[&](i32 depth, i32 min_val) {
        if (stop_dfs)
            return;
        if (depth==k) {
            bool stop=func(elems);
            if (stop)
                stop_dfs=true;
            return;
        }
        for (i32 v=min_val; v<n; v++) {
            elems[depth]=v;
            dfs(depth+1,v);
        }
    };
    dfs(0,0);
}
void for_all_idxs(const vi32 &shape, std::function<bool(const vi32 &)> func) {
    for (i32 n:shape) {
        assert(n>=0, "for_all_idxs: negative length");
        if (n==0)
            return;
    }
    i32 ndim=shape.size();
    vi32 idxs(ndim,0);
    while (idxs.at(0)<shape.at(0)) {
        bool ret=func(idxs);
        if (ret)
            break;
        idxs[ndim-1]++;
        for (i32 i=ndim-1; i>0 && idxs.at(i)>=shape.at(i); i--) {
            idxs[i]=0;
            idxs[i-1]++;
        }
    }
}

template<typename T>
void sort(std::vector<T> &A) {
    std::sort(A.begin(),A.end());
}

template<typename T>
std::vector<T> sorted(const std::vector<T> &A) {
    std::vector<T> out;
    for (const T &v:A)
        out.push_back(v);
    sort(out);
    return out;
}

template<typename T, typename C>
void sort(std::vector<T> &A, C comp) {
    std::sort(A.begin(),A.end(),comp);
}

template<typename T, typename C>
std::vector<T> sorted(const std::vector<T> &A, C comp) {
    std::vector<T> out;
    for (const T &v:A)
        out.push_back(v);
    sort(out,comp);
    return out;
}

#include <iostream>
template <typename T>
std::ostream& operator << (std::ostream &os, const std::vector<T> &V) {
    os<<"[";
    for (i32 i=0; i<V.size(); i++)
        os<<(i==0?"":", ")<<V.at(i);
    os<<"]";
    return os;
}

template <typename A, typename B>
std::ostream& operator << (std::ostream &os, const std::pair<A,B> &p) {
    os<<"("<<p.first<<", "<<p.second<<")";
    return os;
}

template <typename T>
std::ostream& operator << (std::ostream &os, const std::optional<T> &v) {
    if (v.has_value())
        os<<v.value();
    else
        os<<"NO_VALUE";
    return os;
}

#include <chrono>
typedef std::chrono::steady_clock::time_point timept;
double seconds_since(const timept &time_st) {
    std::chrono::nanoseconds duration = std::chrono::steady_clock::now() - time_st;
    return duration.count() / 1000'000'000.0;
}
timept time() {
    return std::chrono::steady_clock::now();
}

#endif