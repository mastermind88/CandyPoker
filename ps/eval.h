#ifndef PS_EVAL_H
#define PS_EVAL_H

#include <array>

#include <boost/range/algorithm.hpp>

#include "generate.h"

#include "ps/cards.h"

namespace ps{

struct detail_eval_impl;

struct eval{
        eval();
        std::uint32_t eval_5(std::vector<long> const& cards)const;
        std::uint32_t operator()(long a, long b, long c, long d, long e)const;
        std::uint32_t operator()(long a, long b, long c, long d, long e, long f)const;
        std::uint32_t operator()(long a, long b, long c, long d, long e, long f, long g)const;
private:
        std::shared_ptr<detail_eval_impl> impl_;
};

} // namespace ps

#endif // #ifndef PS_EVAL_H
