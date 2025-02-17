#ifndef PS_BASE_MASK_SET_H
#define PS_BASE_MASK_SET_H

#include <emmintrin.h>
#include <boost/container/small_vector.hpp>

namespace ps{


#if 1

struct mask_set{
        enum{ UseBoostSmallVector = false };
        using vector_type = std::conditional_t<
                UseBoostSmallVector,
                boost::container::small_vector<size_t, 1>,
                std::vector<size_t>
        >;

        void add(size_t mask){
                masks_.push_back(mask);
                union_ |= mask;
        }
        template<class IntegerType>
        IntegerType count_disjoint_impl_(size_t that)const noexcept{
                IntegerType count = 0;
                for(auto mask : masks_ ){
                        count += !( mask & that );
                }
                return count;
        }
        size_t count_disjoint(size_t that)const noexcept{
                return count_disjoint_impl_<size_t>(that);
        }
        template<class IntegerType>
        IntegerType count_disjoint_with(size_t that)const noexcept{
                return count_disjoint_impl_<IntegerType>(that);
        }
        size_t size()const{ return masks_.size(); }
        friend bool operator<(mask_set const& l, mask_set const& r)noexcept{
                return l.masks_ < r.masks_;
        }
        size_t get_union()const{ return union_; }

        auto begin()const {
            return masks_.begin();
        }
        auto end()const {
            return masks_.end();
        }
private:
        vector_type masks_;
        size_t union_{0};
};

#elif 1

struct mask_set{
        void add(size_t mask){
                masks_.emplace_back(mask, 1);
        }
        size_t count_disjoint_impl_(size_t that)const noexcept{
                size_t count = 0;
                for(auto p : masks_ ){
                        count += p.second * (!!( p.first & that ) );
                }
                return count;
        }
        size_t count_disjoint(size_t that)const noexcept{
                return count_disjoint_impl_(that);
        }
        size_t size()const{ return masks_.size(); }
        friend bool operator<(mask_set const& l, mask_set const& r)noexcept{
                return l.masks_ < r.masks_;
        }
private:
        std::vector<std::pair<size_t, unsigned> > masks_;
};

#elif 0
struct mask_set{
        void add(size_t mask){
                masks_.push_back(mask);
        }
        size_t count_disjoint_impl_(size_t that)const noexcept{
                size_t count = 0;
                for(auto mask : masks_ ){
                        if( ( mask & that ) == 0 ){
                                ++count;
                        }
                }
                return count;
        }
        size_t count_disjoint_sse2_impl_(size_t that)const noexcept{
                __m128i _that = _mm_set1_epi64x(that);
                size_t offset=0;
                size_t sse2_count = 0;
                __m128i const* v = reinterpret_cast<__m128i const*>(&masks_[0]);
                for(;offset + 2 < masks_.size(); offset += 2, ++v ){
                        __m128i c = _mm_load_si128(v);
                        __m128i a = _mm_and_si128(_that, c);
                        __m128i b = _mm_cmpeq_epi64(a, _mm_setzero_si128());
                        auto popcnt = __builtin_popcount(_mm_movemask_epi8(b));
                        //std::cout << "popcnt => " << popcnt << "\n"; // __CandyPrint__(cxx-print-scalar,popcnt)
                        sse2_count += popcnt;
                }
                sse2_count /= 8;
                for(;offset != masks_.size(); ++offset ){
                        if( ( masks_[offset] & that ) == 0 ){
                                ++sse2_count;
                        }
                }
                return sse2_count;
        }
        size_t count_disjoint_sse2_checked_(size_t that)const noexcept{
                auto result = count_disjoint_sse2_impl_(that);

                PS_ASSERT( result == count_disjoint_impl_(that),
                           "result=" << std::bitset<64>(result).to_string() 
                           << ", count_disjoint_impl_(that)="
                           << std::bitset<64>(count_disjoint_impl_(that)).to_string() );

                return result;
        }
        size_t count_disjoint(size_t that)const noexcept{
                return count_disjoint_sse2_impl_(that);
        }
        size_t size()const{ return masks_.size(); }
        friend bool operator<(mask_set const& l, mask_set const& r)noexcept{
                return l.masks_ < r.masks_;
        }
private:
        std::vector<size_t> masks_ __attribute__((aligned (16)));
};


#else

/*
        weirdly this is slower
 */

struct mask_set{
        void add(size_t mask){
                masks_.push_back(mask);
                union_ |= mask;
        }
        size_t count_disjoint(size_t that)const noexcept{
                return 1;
                #if 1
                #if 0
                static size_t hit = 0;
                static size_t total = 0;
                ++total;
                #endif
                if( ( that & union_ ) == 0 ){
                        #if 0
                        ++hit;
                        std::cout << "hit => " << hit << "\n"; // __CandyPrint__(cxx-print-scalar,hit)
                        std::cout << "total => " << total << "\n"; // __CandyPrint__(cxx-print-scalar,total)
                        #endif
                        return size();
                }
                #if 0
                if( size() == 1 )
                        return 0;
                #endif
                #endif
                size_t count = 0;
                for(auto mask : masks_ ){
                        if( ( mask & that ) == 0 ){
                                ++count;
                        }
                }
                return count;
        }
        size_t size()const{ return masks_.size(); }
        friend bool operator<(mask_set const& l, mask_set const& r)noexcept{
                return l.masks_ < r.masks_;
        }
private:
        std::vector<size_t> masks_;
        size_t union_{0};
};
#endif



} // end namespace ps

#endif // PS_BASE_MASK_SET_H
