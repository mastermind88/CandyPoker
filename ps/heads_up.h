#ifndef PS_HEADS_UP_H
#define PS_HEADS_UP_H

#include "ps/cards.h"

#include <vector>
#include <iostream>
#include <map>
#include <numeric>

#include <boost/format.hpp>

namespace ps{
        
template<class T>
struct basic_hu_result_t{
        basic_hu_result_t()=default;
        template<class U>
        basic_hu_result_t(basic_hu_result_t<U> const& that)
                : win(static_cast<T>(that.win))
                , lose(static_cast<T>(that.lose))
                , draw(static_cast<T>(that.draw))
        {}
        basic_hu_result_t permutate(std::vector<int> const& perm)const{
                basic_hu_result_t ret(*this);
                if( perm[0] == 1 )
                        std::swap(ret.win, ret.lose);
                return ret;
        }
        basic_hu_result_t const& append(basic_hu_result_t const& that){
                win  += that.win;
                lose += that.lose;
                draw += that.draw;
                return *this;
        }
        template<class U>
        basic_hu_result_t& operator*=(U val){
                win  *= val;
                lose *= val;
                draw *= val;
                return *this;
        }
        template<class Archive>
        void serialize(Archive& ar, unsigned int){
                ar & win;
                ar & lose;
                ar & draw;
        }

        friend std::ostream& operator<<(std::ostream& ostr, basic_hu_result_t const& self){
                return ostr << boost::format("%2.2f%% (%d,%d,%d)") % self.equity() % self.win % self.draw % self.lose;
        }
        auto sigma()const{
                return win + lose + draw;
        }
        auto equity()const{
                return ( win + draw / 2.0 ) / sigma();
        }
        T win  = T();
        T lose = T();
        T draw = T();
};
using hu_result_t = basic_hu_result_t<size_t>;
using hu_fresult_t = basic_hu_result_t<double>;

        

/*
        This call is take in a sequence of hands, and then return
        the preflop equity
 */
struct equity_calc_detail;

struct equity_cacher{
        equity_cacher();
        ~equity_cacher();
        hu_result_t visit_boards( std::vector<ps::holdem_id> const& players);
        hu_result_t do_visit_boards( std::vector<int> const& perm, std::vector<ps::holdem_id> const& players);
        template<class Archive>
        void serialize(Archive& ar, unsigned int){
                ar & cache_;          
        }
        bool load(std::string const& name);
        bool save(std::string const& name)const;
        void append( equity_cacher const& that);
        auto cache_size()const{ return cache_.size(); }
private:
        std::map<std::vector<ps::holdem_id>, hu_result_t> cache_;
        std::shared_ptr<equity_calc_detail> ec_;
};

struct class_equity_cacher{
        explicit class_equity_cacher(equity_cacher& ec)
                :ec_{&ec}
        {}

        bool load(std::string const& name);
        bool save(std::string const& name)const;
        template<class Archive>
        void serialize(Archive& ar, unsigned int){
                ar & cache_;          
        }

        hu_result_t const& visit_boards( std::vector<ps::holdem_class_id> const& players);
private:
        std::map<std::vector<ps::holdem_class_id>, hu_result_t> cache_;
        equity_cacher* ec_;
};



void generate_cache();

struct hu_strategy{
        hu_strategy(double fill){
                for(size_t i{0};i!=169;++i){
                        vec_[i] = fill;
                }
        }
        auto begin()const{ return vec_.begin(); }
        auto end()  const{ return vec_.end();   }
        double&       operator[](size_t idx)     {return vec_[idx]; }
        double const& operator[](size_t idx)const{return vec_[idx]; }
        void check(){
                for(size_t i{0};i!=169;++i){
                        if ( !( 0 <= vec_[i] && vec_[i] <= 1.0 && "not a strat") ){
                                std::cerr << "FAILED\n";
                                return;
                        }
                }
        }
        hu_strategy& operator*=(double val){
                for(size_t i{0};i!=169;++i){
                        vec_[i] *= val;
                }
                return *this;
        }
        hu_strategy& operator+=(hu_strategy const& that){
                for(size_t i{0};i!=169;++i){
                        vec_[i] += that.vec_[i];
                }
                return *this;
        }
        hu_strategy& operator-=(hu_strategy const& that){
                for(size_t i{0};i!=169;++i){
                        vec_[i] -= that.vec_[i];
                }
                return *this;
        }

        hu_strategy operator*(double val){
                hu_strategy result{*this};
                result *= val;
                return std::move(result);
        }
        hu_strategy operator+(hu_strategy const& that){
                hu_strategy result{*this};
                result += that;
                return std::move(result);
        }
        hu_strategy operator-(hu_strategy const& that){
                hu_strategy result{*this};
                result -= that;
                return std::move(result);
        }
        double sigma(){
                return std::accumulate(vec_.begin(), vec_.end(), 0.0);
        }

        double norm()const{
                double result{0.0};
                for(size_t i{0};i!=169;++i){
                        result = std::max(result, std::fabs(vec_[i]));
                }
                return result;
        }
        friend std::ostream& operator<<(std::ostream& ostr, hu_strategy const& strat){
                return ostr << ps::detail::to_string(strat.vec_);
        }
private:
        std::array<double, 169> vec_;
};

} // ps

#endif // PS_HEADS_UP_H
