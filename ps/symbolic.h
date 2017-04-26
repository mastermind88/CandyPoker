#ifndef PS_SYMBOLIC_H
#define PS_SYMBOLIC_H

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/optional.hpp>

#include "ps/cards.h"
#include "ps/frontend.h"
#include "ps/equity_calc.h"
                
namespace bnu = boost::numeric::ublas;

        /*
        
        The idea here is what when computing range
        vs range evaluation, there are a number of
        results which are linearly related, because
        the way the suits are distribution evently
        when can essentially creat a bijective
        mapping (or permutation)
                f: {h,d,s,c} -> {h',d',s',c'},
        such such that there are dupliucate 
        computations. 
                Consider the computeration

                        AKs vs 98s
                =>
                        AhKh  9h8h 
                        AcKc  9c8c
                        AsKs  9s8s
                        AdKd  9d8d

                =>
                        AhKh   9h8h
                        AhKh   9c8c
                        AhKh   9s8s
                        AhKh   9d8d
                        AcKc   9h8h
                        AcKc   9c8c
                        AcKc   9s8s
                        AcKc   9d8d
                        AsKs   9h8h
                        AsKs   9c8c
                        AsKs   9s8s
                        AsKs   9d8d
                        AdKd   9h8h
                        AdKd   9c8c
                        AdKd   9s8s
                        AdKd   9d8d

        For there 16 computations there exists a
        bijective mapping that the above results in
                
                +--+------------+
                |N |Computation |
                +--+------------+
                |12|AhKh vs 9c8c|
                |4 |AhKh vs 9h8h|
                +--+------------+
                   figure 1

                Another optimization to consider is
        the computation set
               
                  +----------------------+-------------+
                  |     Computation      | Permutation |
                  +----------------------+-------------+
                  | AhKh vs TcTh vs 2s2c |    0 1 2    |
                  | AhKh vs 2s2c vs TcTh |    0 2 1    |
                  | TcTh vs AhKh vs 2s2c |    1 0 2    |
                  | 2s2c vs AhKh vs TcTh |    1 2 0    |
                  | TcTh vs 2s2c vs AhKh |    2 0 1    |
                  | 2s2c vs TcTh vs AhKh |    2 1 0    |
                  +----------------------+-------------+
                            figure 2

                This again can be represented by creating an
        auxillary computation set, and then the result will
        be a linear combintation of this.
                To compute fig 2, we have the below
                
                Let Comp = AhKh vs TcTh vs 2s2c 

                        |S_AhKh|   | w_0 d_0 s_0 e_0 |
                Let S = |S_TcTh| = | w_1 d_1 s_1 e_1 |
                        |S_2s2C|   | w_2 d_2 s_2 e_2 |

                where
                    = Eval(AhKh vs TcTh vs 2s2c)
                      Eval(Comp) 
                    = S

                This implies that
                                                     | 1 0 0 |
                        S_{AhKh vs 2s2c vs TcTh} = S | 0 0 1 |
                                                     | 0 1 0 |

                And thus a computeration set can be reduced
        to a linear combintations of permutation matrixis.
                        
                
                Another optimization that will be applied is
        cached values. For this, consider the computation
                        AA vs TT,
        if I allow this to be represented in the computation
        symbolically, rather than expand to suits, and using
        the fact that
                        log(52^4) + 15.804975 + \epsilon,
        i can precalculate all the heads up probilities
        for heads up calculations

                                                     



        there will be lots of redundant computtaion 


                For this I need to abstract the way computation
        results are stored, to allow the freedom of how the 
        result is calculated

         */

namespace ps{


        namespace detail{
                struct print_context{
                        std::ostream* stream;
                        std::vector<size_t> indent;

                        void push(){
                                indent.push_back( indent.back()+1);
                        }
                        void pop(){
                                indent.pop_back();
                        }

                        std::ostream& put(){ return *stream << std::string(indent.back()*4, ' '); }
                };
        }

        struct calculation_context{
                equity_calc ec;
        };

        struct symbolic_computation{

                struct transform_schedular;
                
                enum kind{
                        Begin_Terminals,
                                Kind_Symbolic_Primitive,
                        End_Terminals,
                        Begin_NonTerminals,
                                Kind_Symbolic_Range,
                                Kind_Symbolic_Primitive_Range,
                                Kind_Symbolic_Player_Perm,
                                Kind_Symbolic_Suit_Perm,
                                Kind_Symbolic_Non_Terminal,
                        End_NonTerminals
                };

                explicit symbolic_computation(kind k):kind_{k}{}

                using handle = std::shared_ptr<symbolic_computation>;



                virtual ~symbolic_computation()=default;

                void print(std::ostream& ostr = std::cout)const{
                        detail::print_context ctx;
                        ctx.indent.emplace_back(0);
                        ctx.stream = &ostr;
                        print_impl(ctx);
                }
                virtual void print_impl(detail::print_context& ctx)const=0;

                virtual bnu::matrix<size_t> calculate(calculation_context&)=0;

                auto get_kind()const{ return kind_; }
                auto is_terminal()const{
                        return 
                                Begin_Terminals < get_kind() && get_kind() < End_Terminals;
                }
                auto is_non_terminal()const{
                        return ! is_terminal();
                }

                virtual std::string to_string()const{
                        switch(get_kind()){
                        case Kind_Symbolic_Primitive:
                                return "primitve";
                        case Kind_Symbolic_Range:
                                return "range";
                        case Kind_Symbolic_Primitive_Range:
                                return "primitive_range";
                        case Kind_Symbolic_Player_Perm:
                                return "player_perm";
                        case Kind_Symbolic_Suit_Perm:
                                return "suit_perm";
                        case Kind_Symbolic_Non_Terminal:
                                return "non_terminal";
                        default:
                                return "unknown";
                        }
                }

        private:
                kind kind_;
        };

        /*
         * Idea here is to have non-termianls derive from this, to allow
         * graph enumeration for transform and optimizations
         *
         */
        struct symbolic_non_terminal : symbolic_computation{

                using iterator= std::list<handle>::iterator;

                explicit symbolic_non_terminal(kind k = Kind_Symbolic_Non_Terminal)
                        :symbolic_computation{k}
                {}

                auto begin(){ return children_.begin(); }
                auto end(){ return children_.end(); }
                auto begin()const{ return children_.begin(); }
                auto end()const{ return children_.end(); }
                auto size()const{ return children_.size(); }
                std::list<handle> const& get_children()const{ return children_; }
                std::list<handle>& get_children(){ return children_; }
                void push_child( handle ptr){ children_.push_back(ptr); }

                void erase_child(iterator iter){
                        children_.erase(iter);
                }
                handle& get_only_child(){ 
                        assert( children_.size() ==1 && "preconditon failed");
                        return children_.back();
                }
                handle const& get_only_child()const{ 
                        assert( children_.size() ==1 && "preconditon failed");
                        return children_.back();
                }
                void print_impl(detail::print_context& ctx)const override{
                        for( auto const& c : get_children() ){
                                c->print_impl(ctx);
                        }
                }
                bnu::matrix<size_t> calculate(calculation_context& cache)override{
                        auto iter{ begin() };
                        auto last{ end() };
                        bnu::matrix<size_t> result { (*iter)->calculate(cache) };
                        ++iter;
                        for(;iter!=last;++iter){
                                result += (*iter)->calculate(cache);
                        }
                        return result;
                }


        private:
                std::list<handle> children_;
        };


        struct symbolic_suit_perm : symbolic_non_terminal{
                explicit symbolic_suit_perm( std::vector<int> const& suit_perm,
                                              handle child)
                        :symbolic_non_terminal(Kind_Symbolic_Suit_Perm),
                                suit_perm_{suit_perm}
                {
                        push_child(child);
                }
                void print_impl(detail::print_context& ctx)const override{
                        auto to_string = [](auto vec){
                                std::stringstream sstr;
                                sstr << "{";
                                boost::copy( vec, std::ostream_iterator<int>(sstr,","));
                                std::string s{sstr.str()};
                                s.pop_back();
                                s += "}";
                                return std::move(s);
                        };
                        ctx.put() << "symbolic_suit_perm(" << to_string(suit_perm_) << ")\n";

                        ctx.push();
                        for( auto const& c : get_children())
                                c->print_impl(ctx);
                        ctx.pop();
                }
                bnu::matrix<size_t> calculate(calculation_context& cache)override{
                        return get_only_child()->calculate(cache);
                }
                decltype(auto) get_suit_perm()const{ return suit_perm_; }
        private:
                std::vector<int> suit_perm_;
        };

        struct symbolic_player_perm : symbolic_non_terminal{
                explicit symbolic_player_perm(std::vector<int> const& player_perm,
                                              handle child)
                        :symbolic_non_terminal(Kind_Symbolic_Player_Perm),
                                player_perm_{player_perm}
                {
                        push_child(child);
                }
                void print_impl(detail::print_context& ctx)const override{
                        auto to_string = [](auto vec){
                                std::stringstream sstr;
                                sstr << "{";
                                boost::copy( vec, std::ostream_iterator<int>(sstr,","));
                                std::string s{sstr.str()};
                                s.pop_back();
                                s += "}";
                                return std::move(s);
                        };
                        ctx.put() << "symbolic_player_perm(" << to_string(player_perm_) << ")\n";

                        ctx.push();
                        for( auto const& c : get_children())
                                c->print_impl(ctx);
                        ctx.pop();
                }
                decltype(auto) get_player_perm()const{ return player_perm_; }
                
                bnu::matrix<size_t> calculate(calculation_context& cache)override{
                        bnu::matrix<size_t> child_result{ get_only_child()->calculate(cache) };
                        bnu::matrix<size_t> result{ child_result }; // just for size

                        for(size_t j=0;j!= result.size2();++j){
                                for(size_t i=0;i!= result.size1(); ++i){
                                        result(player_perm_[i], j) = child_result( i, j );
                                }
                        }

                        return result;
                }
        private:
                std::vector<int> player_perm_;
        };

        
        struct symbolic_primitive : symbolic_computation{
                symbolic_primitive(std::vector<frontend::hand> const& hands,
                                   std::vector<id_type> const& board = std::vector<id_type>{}) 
                        :symbolic_computation(Kind_Symbolic_Primitive),
                        hands_{hands},
                        board_{board},
                        hash_{make_hash(hands_, board_)}
                {
                }
                
                void print_impl(detail::print_context& ctx)const override{
                        std::stringstream sstr;
                        for(size_t i{0};i!=hands_.size();++i){
                                if( i != 0 ) sstr << " vs ";
                                sstr << hands_[i];
                        }
                        ctx.put() << sstr.str() << "\n";
                                //<<  "  (" << hash_ << ")\n";
                }
                decltype(auto) get_hands()const{ return hands_; }
                decltype(auto) get_board()const{ return board_; }

                static std::string make_hash( std::vector<frontend::hand> const& hands,
                                              std::vector<id_type> const& board)
                {
                        std::string hash;
                        for( auto h : hands ){
                                // XXX hash :(
                                /* nned to avoid 5c5dAhKc
                                                 5c5dAhKd
                                */
                                std::string atom{holdem_hand_decl::get(h.get()).to_string()};
                                std::string alt_atom{ atom.substr(2,2) + atom.substr(0,2)};
                                assert( atom.size() == alt_atom.size() && "expected");
                                if( atom < alt_atom )
                                        hash += atom;
                                else
                                        hash += alt_atom;
                        }
                        hash += ":";
                        for( auto b : board ){
                                hash += card_decl::get(b).to_string();
                        }
                        return std::move(hash);
                }
                std::string const& get_hash()const{ return hash_; }
                
                bnu::matrix<size_t> calculate(calculation_context& cache)override{
                        if( ! cache_ ){
                                std::vector<id_type> players;

                                for( auto h : hands_ ){
                                        players.push_back(h.get());
                                }
                                bnu::matrix<size_t> ret;
                                cache.ec.run_ordered( ret, players, board_);
                                cache_ = std::move(ret);
                        }
                        return *cache_;
                }
                std::string to_string()const override{
                        std::stringstream sstr;
                        for(size_t i{0};i!=get_hands().size();++i){
                                if( i != 0 ) sstr << " vs ";
                                sstr << get_hands()[i];
                        }
                        if( board_.size()){
                                sstr << " on board ";
                                for(size_t i{0};i!=board_.size();++i){
                                        sstr << card_decl::get(board_[i]);
                                }
                        }
                        return sstr.str();
                }
        private:
                std::vector<frontend::hand> hands_;
                std::vector<id_type> board_;
                std::string hash_;
                boost::optional< bnu::matrix<size_t> > cache_;
        };

        struct symbolic_primitive_range : symbolic_non_terminal{
                symbolic_primitive_range(std::vector<frontend::primitive_t> const& prims)
                        :symbolic_non_terminal(Kind_Symbolic_Primitive_Range),
                        prims_{prims}
                {
                        std::vector<size_t> size_vec;
                        std::vector<std::vector<holdem_id> > aux;

                        for( auto const& p : prims_){
                                aux.emplace_back( to_hand_vector(p));
                                size_vec.push_back( aux.back().size()-1);
                        }
                        
                        switch(prims_.size()){
                        case 2:
                                detail::visit_exclusive_combinations<2>(
                                        [&](auto a, auto b){
                                        
                                        // make sure disjoint

                                        if( disjoint( holdem_hand_decl::get(aux[0][a]),
                                                      holdem_hand_decl::get(aux[1][b]) ) )
                                        {
                                                push_child(
                                                       std::make_shared<symbolic_primitive>(
                                                               std::vector<frontend::hand>{
                                                                        frontend::hand{aux[0][a]},
                                                                        frontend::hand{aux[1][b]}}));
                                        }
                                }, detail::true_, size_vec);
                                break;
                        case 3:
                                detail::visit_exclusive_combinations<3>(
                                        [&](auto a, auto b, auto c){
                                        
                                        // make sure disjoint

                                        if( disjoint( holdem_hand_decl::get(aux[0][a]),
                                                      holdem_hand_decl::get(aux[1][b]),
                                                      holdem_hand_decl::get(aux[2][c]) ) )
                                        {
                                                push_child(
                                                       std::make_shared<symbolic_primitive>(
                                                               std::vector<frontend::hand>{
                                                                        frontend::hand{aux[0][a]},
                                                                        frontend::hand{aux[1][b]},
                                                                        frontend::hand{aux[2][c]}}));
                                        }
                                }, detail::true_, size_vec);
                                break;
                        case 4:
                                detail::visit_exclusive_combinations<4>(
                                        [&](auto a, auto b, auto c, auto d){
                                        
                                        // make sure disjoint

                                        if( disjoint( holdem_hand_decl::get(aux[0][a]),
                                                      holdem_hand_decl::get(aux[1][b]),
                                                      holdem_hand_decl::get(aux[2][c]),
                                                      holdem_hand_decl::get(aux[3][d]) ) )
                                        {
                                                push_child(
                                                       std::make_shared<symbolic_primitive>(
                                                               std::vector<frontend::hand>{
                                                                        frontend::hand{aux[0][a]},
                                                                        frontend::hand{aux[1][b]},
                                                                        frontend::hand{aux[2][c]},
                                                                        frontend::hand{aux[3][d]} }));
                                        }
                                }, detail::true_, size_vec);
                                break;
                        case 5:
                                detail::visit_exclusive_combinations<5>(
                                        [&](auto a, auto b, auto c, auto d, auto e){
                                        
                                        // make sure disjoint

                                        if( disjoint( holdem_hand_decl::get(aux[0][a]),
                                                      holdem_hand_decl::get(aux[1][b]),
                                                      holdem_hand_decl::get(aux[2][c]),
                                                      holdem_hand_decl::get(aux[3][d]),
                                                      holdem_hand_decl::get(aux[4][e]) ) )
                                        {
                                                push_child(
                                                       std::make_shared<symbolic_primitive>(
                                                               std::vector<frontend::hand>{
                                                                        frontend::hand{aux[0][a]},
                                                                        frontend::hand{aux[1][b]},
                                                                        frontend::hand{aux[2][c]},
                                                                        frontend::hand{aux[3][d]},
                                                                        frontend::hand{aux[4][e]} }));
                                        }
                                }, detail::true_, size_vec);
                                break;
                        default:
                                assert( 0 && " not implemented");
                        }


                }
                #if 0
                void print_impl(detail::print_context& ctx)const override{
                        ctx.put() << "children.size() = " << get_children().size() << "\n";
                        for(size_t i{0};i!=prims_.size();++i){
                                ctx.put() << "prim " << i << " : " << prims_[i] << "\n";
                        }
                        ctx.push();
                        for( auto const& c : get_children() ){
                                c->print_impl(ctx);
                        }
                        ctx.pop();
                }
                #endif
                
                bnu::matrix<size_t> calculate(calculation_context& cache)override{
                        auto iter{ begin() };
                        auto last{ end() };
                        bnu::matrix<size_t> result { (*iter)->calculate(cache) };
                        ++iter;
                        for(;iter!=last;++iter){
                                result += (*iter)->calculate(cache);
                        }
                        return result;
                }
                std::vector<frontend::primitive_t> const& get_players()const{ return prims_; }
        private:
                std::vector<frontend::primitive_t> prims_;
        };

        

        struct symbolic_range : symbolic_non_terminal{
                symbolic_range(std::vector<frontend::range> const& players)
                        :symbolic_non_terminal(Kind_Symbolic_Range),
                        players_{players}
                {
                        std::vector<size_t> size_vec;
                        std::vector<frontend::primitive_range> prims;
                        for(auto const& rng : players_){
                                prims.emplace_back( expand(rng).to_primitive_range());
                                assert( prims.back().size() != 0 && "precondition failed");
                                size_vec.emplace_back(prims.back().size()-1);
                        }

                        // XXX dispatch for N
                        switch(players.size()){
                        case 2:
                                detail::visit_exclusive_combinations<2>(
                                        [&](auto a, auto b){
                                        push_child(
                                               std::make_shared<symbolic_primitive_range>( std::vector<frontend::primitive_t>{prims[0][a], prims[1][b] }));
                                }, detail::true_, size_vec);
                                break;
                        case 3:
                                detail::visit_exclusive_combinations<3>(
                                        [&](auto a, auto b, auto c){
                                        push_child(
                                               std::make_shared<symbolic_primitive_range>( std::vector<frontend::primitive_t>{prims[0][a], prims[1][b], prims[2][c] }));
                                }, detail::true_, size_vec);
                                break;
                        case 4:
                                detail::visit_exclusive_combinations<4>(
                                        [&](auto a, auto b, auto c, auto d){
                                        push_child(
                                               std::make_shared<symbolic_primitive_range>( std::vector<frontend::primitive_t>{prims[0][a], prims[1][b], prims[2][c], prims[3][d] }));
                                }, detail::true_, size_vec);
                                break;
                        case 5:
                                detail::visit_exclusive_combinations<5>(
                                        [&](auto a, auto b, auto c, auto d, auto e){
                                        push_child(
                                               std::make_shared<symbolic_primitive_range>( std::vector<frontend::primitive_t>{prims[0][a], prims[1][b], prims[2][c], prims[3][d], prims[4][e] }));
                                }, detail::true_, size_vec);
                                break;
                        default:
                                assert( 0 && " not implemented");
                        }
                }
                #if 0
                void print_impl(detail::print_context& ctx)const override{
                        ctx.put() << "children.size() = " << get_children().size() << "\n";
                        for(size_t i{0};i!=players_.size();++i){
                                ctx.put() << "player " << i << " : " << players_[i] << "\n";
                        }
                        ctx.push();
                        for( auto const& c : get_children() ){
                                c->print_impl(ctx);
                        }
                        ctx.pop();
                }
                #endif
                bnu::matrix<size_t> calculate(calculation_context& cache)override{
                        auto iter{ begin() };
                        auto last{ end() };
                        bnu::matrix<size_t> result { (*iter)->calculate(cache) };
                        ++iter;
                        for(;iter!=last;++iter){
                                result += (*iter)->calculate(cache);
                        }
                        return result;
                }
                std::vector<frontend::range> const& get_players()const{ return players_; }
        private:
                std::vector<frontend::range> players_;

        };




} // ps
#endif // PS_SYMBOLIC_H
