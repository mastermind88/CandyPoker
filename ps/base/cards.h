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
#ifndef PS_CORE_CARDS_H
#define PS_CORE_CARDS_H

#include <vector>
#include <set>
#include <map>
#include <bitset>
#include <unordered_set>


#include "ps/base/cards_fwd.h"
#include "ps/detail/void_t.h"
#include "ps/detail/print.h"
#include "ps/detail/popcount.h"
#include "ps/support/index_sequence.h"


#define PS_ASSERT_VALID_SUIT_ID(x) PS_ASSERT( x < suit_decl::max_id, "suit_id = " << static_cast<int>(x))
#define PS_ASSERT_VALID_RANK_ID(x) PS_ASSERT( x < rank_decl::max_id, "rank_id = " << static_cast<int>(x))
#define PS_ASSERT_VALID_CARD_ID(x) PS_ASSERT( x < card_decl::max_id, "card_id = " << static_cast<int>(x))
#define PS_ASSERT_VALID_HOLDEM_ID(x) PS_ASSERT( x < holdem_hand_decl::max_id, "holdem_hand_id = " <<  static_cast<int>(x))

namespace ps{
        
        struct card_vector : std::vector<card_id>{
                template<class... Args>
                card_vector(Args&&... args):std::vector<card_id>{std::forward<Args>(args)...}{}

                size_t mask()const;
                static card_vector from_bitmask(size_t mask);
                static card_vector parse(std::string const& s);

                friend inline std::ostream& operator<<(std::ostream& ostr, card_vector const& self);

                std::unordered_set<card_id> to_set()const {
                    return std::unordered_set<card_id>{begin(), end()};
                }
        };

        struct suit_decl{
                using index_ty = suit_id;
                static constexpr suit_id max_id = 4;
                suit_decl(suit_id id, char sym, std::string const& name);
                auto id()const{ return id_; }
                const char symbol()const { return sym_; }
                std::string to_string()const{ return std::string{sym_}; }
                friend std::ostream& operator<<(std::ostream& ostr, suit_decl const& self){
                        return ostr << self.to_string();
                }
                bool operator<(suit_decl const& that)const{
                        return id_ < that.id_;
                }
                static suit_decl const& get(suit_id id);
                static suit_decl const& parse(std::string const& s);
                static suit_decl const& parse(char c){ return parse(std::string{c}); }
                operator suit_id()const{ return id_; }

                // to help with unit testing
                template<class V>
                static void visit_suit_id_combinations(V&& v)
                {
                    for (suit_id a = 0; (a + 1) != suit_decl::max_id; ++a)
                    {
                        for (suit_id b = a + 1; b != suit_decl::max_id; ++b)
                        {
                            v(a, b);
                        }
                    }
                }
                template<class V>
                static void visit_suit_combinations(V&& v)
                {
                    auto outer = [&](suit_id a, suit_id b)
                    {
                        v(suit_decl::get(a), suit_decl::get(b));
                    };
                    suit_decl::visit_suit_id_combinations(outer);
                }


        private:
                suit_id id_;
                char sym_;
                std::string name_;
        };

        struct rank_decl{
                using index_ty = rank_id;
                static constexpr rank_id max_id = 13;
                rank_decl(rank_id id, char sym);
                auto id()const{ return id_; }
                const char symbol()const { return sym_; }
                std::string to_string()const{ return std::string{sym_}; }
                friend std::ostream& operator<<(std::ostream& ostr, rank_decl const& self){
                        return ostr << self.to_string();
                }
                bool operator<(rank_decl const& that)const{
                        return id_ < that.id_;
                }
                static rank_decl const& get(rank_id id);
                static rank_decl const& parse(std::string const& s);
                static rank_decl const& parse(char c){ return parse(std::string{c}); }
                operator rank_id()const{ return id_; }

                
        private:
                rank_id id_;
                char sym_;
        };


        struct card_decl{
                using index_ty = card_id;
                static constexpr const card_id max_id = 52;
                card_decl(suit_decl const& s, rank_decl const& r);
                auto id()const{ return id_; }
                size_t mask()const{ return static_cast<size_t>(1) << id(); }
                std::string to_string()const;
                // id & 0x3
                suit_decl const& suit()const{ return suit_; }
                // id >> 2
                rank_decl const& rank()const{ return rank_; }
                friend std::ostream& operator<<(std::ostream& ostr, card_decl const& self){
                        return ostr << self.to_string();
                }
                bool operator<(card_decl const& that)const{
                        return id_ < that.id_;
                }
                static card_decl const& get(card_id id);
                static card_decl const& parse(std::string const& s);
                operator card_id()const{ return id_; }
                static card_id make_id( suit_id s, rank_id r){
                        return s + r * 4;
                }
        private:
                card_id id_;
                suit_decl suit_;
                rank_decl rank_;
        };

        struct holdem_hand_decl{
                using index_ty = holdem_id;
                static constexpr const holdem_id max_id = 52 * 51 / 2;
                // a must be the biggest
                holdem_hand_decl(card_decl const& a, card_decl const& b);
                auto id()const{ return id_; }
                size_t mask()const{
                        return first().mask() | second().mask();
                }
                std::string to_string()const{
                        return first_.to_string() + 
                               second_.to_string();
                }
                friend std::ostream& operator<<(std::ostream& ostr, holdem_hand_decl const& self){
                        return ostr << self.to_string();
                }
                bool operator<(holdem_hand_decl const& that)const{
                        return id_ < that.id_;
                }
                card_decl const& first()const{ return first_; }
                card_decl const& second()const{ return second_; }
                static holdem_hand_decl const& get(holdem_id id);
                static holdem_hand_decl const& get(card_id x, card_id y){
                        return get( make_id(x,y) );
                }
                static holdem_hand_decl const& parse(std::string const& s);
                static holdem_id make_id( rank_id r0, suit_id s0, rank_id r1, suit_id s1);
                static holdem_id make_id(card_id x, card_id y);
                operator holdem_id()const{ return id_; }
                holdem_class_id class_()const;

                template<class... Args,
                        class _ = detail::void_t<
                                        std::enable_if_t<
                                                std::is_same<std::decay_t<Args>, holdem_hand_decl>::value>...
                        >
                >
                static bool disjoint(Args&&... args){
                        size_t mask{0};
                        int aux[] = {0, ( mask |= args.mask(), 0)...};
                        return detail::popcount(mask)*2 == sizeof...(args);
                }
                template<class... Args,
                        class _ = detail::void_t<
                                        std::enable_if_t<
                                                std::is_integral<std::decay_t<Args> >::value>...
                        >
                >
                static bool disjoint_id(Args&&... args){
                        return disjoint( holdem_hand_decl::get(args)... );
                }

                bool is_suited()const noexcept{ return ( card_suit_from_id(first_.id()) == card_suit_from_id(second_.id()) ); }
                bool is_offsuit()const noexcept{ return ! is_suited(); }
                                                               
        private:
                holdem_id id_;
                card_decl first_;
                card_decl second_;

        };
        
        /*
                Hand vector is a vector of hands
         */
        struct holdem_hand_vector : std::vector<ps::holdem_id>{
                template<class... Args>
                holdem_hand_vector(Args&&... args)
                : std::vector<ps::holdem_id>{std::forward<Args>(args)...}
                {}
                // for unit testing
                template<class... Args>
                static holdem_hand_vector union_(Args&&... args)
                {
                    std::unordered_set< holdem_id> s;
                    const int aux[] = {
                        (std::for_each(std::cbegin(args), std::cend(args), [&](holdem_id id) { s.insert(id); }), 0)...
                    };
                    return holdem_hand_vector(std::cbegin(s), std::cend(s));
                }


                holdem_hand_decl const& decl_at(size_t i)const;
                friend std::ostream& operator<<(std::ostream& ostr, holdem_hand_vector const& self);
                auto find_injective_permutation()const;
                bool disjoint()const;
                bool is_standard_form()const;
                size_t mask()const;
                card_vector to_card_vector()const;
                std::string to_string()const{
                        std::stringstream sstr;
                        sstr << *this;
                        return sstr.str();
                }
                // treat every 4 chars as a card, ie
                //          parse("2c2hAsKs");
                // -> holdem_hand_vector({holdem_hand_decl::parse("2c2h"), holdem_hand_decl::parse("AsKs")})
                static holdem_hand_vector parse(std::string const& s);

                std::unordered_set<holdem_id> to_set()const {
                    return std::unordered_set<holdem_id>{begin(), end()};
                }
        };

        struct holdem_hand_iterator :
                basic_index_iterator<
                        holdem_id,
                        strict_lower_triangle_policy,
                        holdem_hand_vector
                >
        {
                using impl_t = 
                        basic_index_iterator<
                                holdem_id,
                                strict_lower_triangle_policy,
                                holdem_hand_vector
                        >
                ;
                holdem_hand_iterator():impl_t{}{}
                holdem_hand_iterator(size_t n):
                        impl_t(n, 52 * 51 / 2)
                {}
        };

        struct holdem_class_decl{
                using index_ty = holdem_class_id;
                static constexpr const holdem_class_id max_id = 13 * 13;
                holdem_class_decl(holdem_class_type cat,
                                  rank_decl const& a,
                                  rank_decl const& b);
                auto const& get_hand_set()const{ return hand_set_; }
                holdem_hand_vector const& get_hand_vector()const{ return hand_id_set_; }
                auto id()const{ return id_; }
                holdem_class_type category()const{ return cat_; }
                std::string to_string()const;
                friend std::ostream& operator<<(std::ostream& ostr, holdem_class_decl const& self){
                        return ostr << self.to_string();
                }
                bool operator<(holdem_class_decl const& that)const{
                        return id_ < that.id_;
                }
                size_t weight()const{
                        switch(cat_){
                        case holdem_class_type::suited:
                                return 4;
                        case holdem_class_type::offsuit:
                                return 12;
                        default: // don't care
                        case holdem_class_type::pocket_pair:
                                return 6;
                        }
                }
                double prob()const{
                        constexpr size_t half{ ( 13 * 13 - 13 ) / 2 };
                        constexpr size_t sigma{ 
                                13   *  6 +
                                half * 12 +
                                half * 4
                        };
                        return static_cast<double>(weight()) / sigma;
                }
                decltype(auto) first()const{ return first_; }
                decltype(auto) second()const{ return second_; }
                static holdem_class_decl const& get(holdem_id id);

                /*
                parse("XX") or parse("XYs") or parse("XYo")
                */
                static holdem_class_decl const& parse(std::string const& s);
                // any bijection will do, nice to keep the mapping within a char
                static holdem_class_id make_id(holdem_class_type cat, rank_id x, rank_id y);
                operator holdem_class_id()const{ return id_; }

                // TODO generlaize these
                static size_t weight(holdem_class_id c0, holdem_class_id c1);
                static double prob(holdem_class_id c0, holdem_class_id c1);

        private:
                holdem_class_id id_;
                holdem_class_type cat_;
                rank_decl first_;
                rank_decl second_;
                std::vector<holdem_hand_decl> hand_set_;
                std::vector<holdem_id> hand_id_vec_;
                holdem_hand_vector hand_id_set_;
        };


        struct rank_vector : std::vector<rank_id>{
                template<class... Args>
                rank_vector(Args&&... args):std::vector<rank_id>{std::forward<Args>(args)...}{}

                friend std::ostream& operator<<(std::ostream& ostr, rank_vector const& self);
        };
        struct holdem_class_range : std::vector<ps::holdem_class_id>{
                template<
                        class... Args,
                        class = std::enable_if_t< ! std::is_constructible<std::string, Args...>::value  >
                >
                holdem_class_range(Args&&... args)
                : std::vector<ps::holdem_id>{std::forward<Args>(args)...}
                {}
                holdem_class_range(std::string const& item);
                friend std::ostream& operator<<(std::ostream& ostr, holdem_class_range const& self);
                void parse(std::string const& item);
        };
        
        
        struct holdem_class_vector : std::vector<ps::holdem_class_id>{
                template<class... Args>
                holdem_class_vector(Args&&... args)
                : std::vector<ps::holdem_class_id>{std::forward<Args>(args)...}
                {}
                
                // parse("AA,KK");
                static holdem_class_vector parse(std::string const& s);
                friend std::ostream& operator<<(std::ostream& ostr, holdem_class_vector const& self);
                holdem_class_decl const& decl_at(size_t i)const;
                std::vector< holdem_hand_vector > get_hand_vectors()const;

                std::string to_string()const{
                        std::stringstream sstr;
                        sstr << *this;
                        return sstr.str();
                }

                template<
                        class... Args,
                        class = std::enable_if_t< ! std::is_constructible<std::string, Args...>::value  >
                >
                void push_back(Args&&... args){
                        this->std::vector<ps::holdem_class_id>::push_back(std::forward<Args...>(args)...);
                }
                void push_back(std::string const& item){
                        this->push_back( holdem_class_decl::parse(item).id() );
                }
                template<class Archive>
                void serialize(Archive& ar, unsigned int){
                        ar & (*reinterpret_cast<std::vector<ps::holdem_class_id>*>(this));
                }


                std::tuple<
                        std::vector<int>,
                        holdem_class_vector
                > 
                to_standard_form()const;
                
                std::vector<
                       std::tuple< std::vector<int>, holdem_hand_vector >
                > to_standard_form_hands()const;
                
                bool is_standard_form()const;

                auto prob()const{
                        BOOST_ASSERT(size() == 2 );
                        return holdem_class_decl::prob(at(0), at(1));
                }
        };
        
        struct holdem_class_iterator :
                basic_index_iterator<
                holdem_class_id,
                ordered_policy,
                holdem_class_vector
                >
        {
                using impl_t = 
                        basic_index_iterator<
                                holdem_class_id,
                                ordered_policy,
                                holdem_class_vector
                        >
                        ;
                holdem_class_iterator():impl_t{}{}
                holdem_class_iterator(size_t n):
                        impl_t(n, holdem_class_decl::max_id)
                {}
        };
        struct holdem_class_perm_iterator :
                basic_index_iterator<
                holdem_class_id,
                range_policy,
                holdem_class_vector
                >
        {
                using impl_t = 
                        basic_index_iterator<
                                holdem_class_id,
                                range_policy,
                                holdem_class_vector
                        >
                        ;
                holdem_class_perm_iterator():impl_t{}{}
                holdem_class_perm_iterator(size_t n):
                        impl_t(n, holdem_class_decl::max_id)
                {}
        };
        typedef std::tuple< std::vector<int>, holdem_class_vector> standard_form_result;
        inline std::ostream& operator<<(std::ostream& ostr, standard_form_result const& self){
                return ostr << detail::to_string(std::get<0>(self))
                        << " x "
                        << std::get<1>(self);
        }
        
        typedef std::tuple< std::vector<int>, holdem_hand_vector> standard_form_hands_result;
        inline std::ostream& operator<<(std::ostream& ostr, standard_form_hands_result const& self){
                return ostr << detail::to_string(std::get<0>(self))
                        << " x "
                        << std::get<1>(self);
        }
        
        struct holdem_class_range_vector : std::vector<holdem_class_range>{
                template<class... Args>
                holdem_class_range_vector(Args&&... args)
                : std::vector<holdem_class_range>{std::forward<Args>(args)...}
                {}
                friend std::ostream& operator<<(std::ostream& ostr, holdem_class_range_vector const& self);

                void push_back(std::string const& s);

                // Return this expand, ie 
                //        {{AA,KK},{22}} => {AA,22}, {KK,22}
                std::vector<holdem_class_vector> get_cross_product()const;
                // Returns this as a vector of
                //        (matrix, standard-form-hand-vector)
                std::vector<
                       std::tuple< std::vector<int>, holdem_hand_vector >
                > to_standard_form()const;
                // Returns this as a vector of
                //        (matrix, standard-form-class-vector)
                std::vector<
                       std::tuple< std::vector<int>, holdem_class_vector >
                > to_class_standard_form()const;

        };

        struct equity_view : std::vector<double>{
                equity_view(matrix_t const& breakdown);
                unsigned long long sigma()const{ return sigma_; }
                bool valid()const;
        private:
                unsigned long long sigma_;
        };
        
        
        std::ostream& operator<<(std::ostream& ostr, card_vector const& self){
                return ostr << detail::to_string(self, [](auto id){
                        return card_decl::get(id).to_string();
                });
        }
        

        enum hand_rank_category{
                HR_RoyalFlush,
                HR_StraightFlush,
                HR_Quads,
                HR_FullHouse,
                HR_Flush,
                HR_Straight,
                HR_Trips,
                HR_TwoPair,
                HR_OnePair,
                HR_HighCard,

                HR_NotAHandRank,
        };
        inline
        std::ostream& operator<<(std::ostream& ostr, hand_rank_category cat){
                switch(cat){
                case HR_RoyalFlush: return ostr << "RoyalFlush";
                case HR_StraightFlush: return ostr << "StraightFlush";
                case HR_Quads: return ostr << "Quads";
                case HR_FullHouse: return ostr << "FullHouse";
                case HR_Flush: return ostr << "Flush";
                case HR_Straight: return ostr << "Straight";
                case HR_Trips: return ostr << "Trips";
                case HR_TwoPair: return ostr << "TwoPair";
                case HR_OnePair: return ostr << "OnePair";
                case HR_HighCard: return ostr << "HighCard";
                case HR_NotAHandRank: return ostr << "NotAHandRank";
                default: return ostr << "(invalid)";
                }
        }
                

} // end namespace cards

#include "ps/base/decl.h"




#endif // PS_CORE_CARDS_H
