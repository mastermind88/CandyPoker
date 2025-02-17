/*

CandyPoker
https://github.com/sweeterthancandy/CandyPoker

MIT License

Copyright (c) 2019 Gerry Candy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
#ifndef PS_SUPPORT_INDEX_SEQUENCE_H
#define PS_SUPPORT_INDEX_SEQUENCE_H

#include <cstddef>
#include <vector>

namespace ps{

/*
                00
                01
                10
                11
 */
struct range_policy{
        template<class Vec>
        static void init(Vec& vec, size_t n, size_t m){
                vec.resize(n);
                for(size_t i=0;i!=vec.size();++i)
                        vec[i] = 0;
        }
        template<class Vec>
        static bool next(Vec& vec, size_t m){
                size_t cursor = vec.size() - 1;
                for(;cursor!=-1;){
                        if( vec[cursor] + 1 == m ){
                                --cursor;
                                continue;
                        }

                        break; // <----------------------
                }
                if( cursor == -1 ){
                        // at end
                        return false; 
                }
                ++vec[cursor];
                ++cursor;
                for(;cursor != vec.size();++cursor){
                        vec[cursor] = 0;
                }
                return true;
        }
};

/*
                00
                01
                10
                11
 */
struct ordered_policy{
        template<class Vec>
        static void init(Vec& vec, size_t n, size_t m){
                vec.resize(n);
                for(size_t i=0;i!=vec.size();++i)
                        vec[i] = 0;
        }
        template<class Vec>
        static bool next(Vec& vec, size_t m){
                size_t cursor = vec.size() - 1;
                for(;cursor!=-1;){

                        // First see if we can't decrement the board
                        // at the cursor
                        if( cursor == vec.size() -1 ){
                                if( vec[cursor] + 1 == m ){
                                        // case XXXX0
                                        --cursor;
                                        continue;
                                }
                        } else {
                                if( vec[cursor] == vec[cursor+1] )
                                {
                                        // case XXX10
                                        --cursor;
                                        continue;
                                }
                        }
                        break;
                }
                if( cursor == -1 ){
                        // at end
                        return false; 
                }
                ++vec[cursor];
                ++cursor;
                for(;cursor != vec.size();++cursor){
                        vec[cursor] = vec[cursor-1];
                }
                return true;
        }
};

struct strict_lower_triangle_policy{
        template<class Vec>
        static void init(Vec& vec, size_t n, size_t m){
                vec.resize(n);
                using value_ty = std::decay<decltype(vec[0])>::type;
                for (size_t i = 0; i != vec.size(); ++i)
                {
                    vec[i] = static_cast<value_ty>(i);
                }
        }
        template<class Vec>
        static bool next(Vec& vec, size_t m){
                size_t cursor = vec.size() - 1;
                for(;cursor!=-1;){

                        // First see if we can't decrement the board
                        // at the cursor
                        if( cursor == vec.size() -1 ){
                                if( vec[cursor] + 1 == m ){
                                        // case XXXX0
                                        --cursor;
                                        continue;
                                }
                        } else {
                                if( vec[cursor] + 1 == vec[cursor+1] ){
                                        // case XXX10
                                        --cursor;
                                        continue;
                                }
                        }
                        break;
                }
                if( cursor == -1 ){
                        // at end
                        return false; 
                }
                ++vec[cursor];
                ++cursor;
                for(;cursor != vec.size();++cursor){
                        vec[cursor] = vec[cursor-1] + 1;
                }
                return true;
        }
};

struct lower_triangle_policy{
        template<class Vec>
        static void init(Vec& vec, size_t n, size_t m){
                vec.resize(n);
                for(size_t i=0;i!=vec.size();++i)
                        vec[i] = i;
        }
        template<class Vec>
        static bool next(Vec& vec, size_t m){
                size_t cursor = vec.size() - 1;
                for(;cursor!=-1;){

                        // First see if we can't decrement the board
                        // at the cursor
                        if( cursor == vec.size() -1 ){
                                if( vec[cursor] + 1 == m ){
                                        // case XXXX0
                                        --cursor;
                                        continue;
                                }
                        } else {
                                if( vec[cursor] == vec[cursor+1] )
                                {
                                        // case XXX10
                                        --cursor;
                                        continue;
                                }
                        }
                        break;
                }
                if( cursor == -1 ){
                        // at end
                        return false; 
                }
                ++vec[cursor];
                ++cursor;
                for(;cursor != vec.size();++cursor){
                        vec[cursor] = vec[cursor-1];
                }
                return true;
        }
};

struct strict_upper_triangle_policy{
        template<class Vec>
        static void init(Vec& vec, size_t n, size_t m){
                vec.resize(n);
                for(size_t i=0;i!=vec.size();++i)
                        vec[i] = m-1-i;
        }
        template<class Vec>
        static bool next(Vec& vec, size_t m){
                size_t cursor = vec.size() - 1;
                for(;cursor!=-1;){

                        // First see if we can't decrement the board
                        // at the cursor
                        if( cursor == vec.size() -1 ){
                                if( vec[cursor] == 0 ){
                                        // case XXXX0
                                        --cursor;
                                        continue;
                                }
                        } else {
                                if( vec[cursor] - 1 == vec[cursor+1] ){
                                        // case XXX10
                                        --cursor;
                                        continue;
                                }
                        }
                        break;
                }
                if( cursor == -1 ){
                        // at end
                        return false; 
                }
                --vec[cursor];
                ++cursor;
                for(;cursor != vec.size();++cursor){
                        vec[cursor] = vec[cursor-1] - 1;
                }
                return true;
        }
};

template<class T, class Policy, class Vec = std::vector<T> >
struct basic_index_iterator{
        using integer_t = T;
        using vector_t  = Vec;
        using policy_t  = Policy;

        // construct psuedo end iterator
        basic_index_iterator():end_flag_{true}{}

        explicit basic_index_iterator(size_t n, size_t m)
                :n_{n}
                ,m_{m}
        {
                policy_t::init(vec_, n_, m_);
        }
        auto const& operator*()const{ return vec_; }
        auto operator->()const{ return &this->operator*(); }
        basic_index_iterator& operator++(){
                end_flag_ = (  ! policy_t::next(vec_, m_) );
                return *this;
        }

        bool operator==(basic_index_iterator const& that)const{
                return this->end_flag_ && that.end_flag_;
        }
        bool operator!=(basic_index_iterator const& that)const{
                return ! ( *this == that);
        }
private:
        size_t n_;
        size_t m_;
        vector_t vec_;
        // flag to indicate at end
        bool end_flag_{false};
};

} // ps

#endif // PS_SUPPORT_INDEX_SEQUENCE_H
