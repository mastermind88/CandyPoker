#ifndef PS_EVAL_H
#define PS_EVAL_H

#include <array>


#include "ps/base/cards.h"
#include "ps/support/singleton_factory.h"

namespace ps{

struct ranking{
        ranking():
                rank_{0},
                name_{"__invalid__"}
        {}
        explicit ranking(int r, std::string n):
                rank_{r},
                name_{std::move(n)}
        {}
        ranking& assign(int r, std::string n){
                rank_ = r;
                name_ = std::move(n);
                return *this;
        }
        auto rank()const{ return rank_; }
        auto const& name()const{ return name_; }

        #if 0
        bool operator<(ranking const& l_param,
                       ranking const& r_param){
                return l_param.rank() < r_param.rank();
        }
        #endif

        operator int()const{ return this->rank(); }
private:
        int rank_;
        std::string name_;
};

struct evaluater{
        virtual ~evaluater()=default;
        //virtual ranking const& rank(std::vector<long> const& cards)const;
        virtual ranking const& rank(long a, long b, long c, long d, long e)const;
        virtual ranking const& rank(long a, long b, long c, long d, long e, long f)const;
        virtual ranking const& rank(long a, long b, long c, long d, long e, long f, long g)const;
};

using evaluater_factory = support::singleton_factory<evaluater>;

struct detail_eval;

struct eval{
        eval();
        std::uint32_t eval_5(std::vector<long> const& cards)const;
        std::uint32_t operator()(long a, long b, long c, long d, long e)const;
        std::uint32_t operator()(long a, long b, long c, long d, long e, long f)const;
        std::uint32_t operator()(long a, long b, long c, long d, long e, long f, long g)const;
};


} // namespace ps

#endif // #ifndef PS_EVAL_H
