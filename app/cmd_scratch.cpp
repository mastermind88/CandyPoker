#include <thread>
#include <numeric>
#include <atomic>
#include <fstream>

#include <Eigen/Dense>

#include <boost/format.hpp>
#include <boost/intrusive/list.hpp>
#include <boost/iterator/indirect_iterator.hpp>
#include <boost/timer/timer.hpp>
#include <boost/log/trivial.hpp>

#include "ps/support/config.h"
#include "ps/base/cards.h"
#include "ps/base/board_combination_iterator.h"
#include "ps/support/index_sequence.h"
#include "app/pretty_printer.h"
#include "ps/base/algorithm.h"
#include "ps/base/rank_hasher.h"
#include "ps/base/suit_hasher.h"
#include "ps/base/holdem_board_decl.h"
#include "ps/support/command.h"
#include "ps/support/persistent.h"

using namespace ps;
/*
        The game tree can utlimatley be transformed into a linear list of terminals, and then we have
                EV(S) = \sigma P(T_i) * Value_i

        for example, for hu push/fold we have
                EV : [0,1]^169  X 2 -> R

                EV(S) = P( pp ) * Value(pp, S)
                      + P( pf ) * Value(pf, S)
                      + P( f  ) * Value(f , S)

        thus
                EV(S)   -> \sigma deals d \in D P(d) *EV(S,d)
                EV(S,d) ->       ( P(pp,d,S) * Value(pp,d)  +
                                   P(pf,d,S) * Value(pf,d)  +
                                   P(f ,d,S) * Value(f ,d) )
                                

        Where
                Value(pp,d) -> all in equity for d[0] vs d[1]
                value(pf,d) -> [+bb, -sb]
                Value(f ,d) -> [-sb, +sb]

        As an implementation detail
                P(pp,S) -> S[0] * S[1]
                P(pf,S) -> S[0] * ( 1- S[1] )
                P(f, S) -> ( 1- S[0] )
        =>
                \sigma P = S[0] * S[1]        +
                           S[0] * ( 1- S[1] ) +
                           ( 1- S[0] )
                         = S[0]*S[1] + (S[0]-S[0]*S[1]) +  (1-S[0])
                         =           + (S[0]          ) +  (1-S[0])
                         = 1

        We also have one condition distrbution
                P(X=X|p) 
        where
                P(pp|p,S) = S[1]
                P(pf|p,S) = 1 - S[1]

        We also have the distribution
                P(p) -> S[0]
                P(f) -> 1 - S[0]


        This when we solve this, we can just create one huge expression tree


        For the solution, we look at each players deal individually, and choose the 
        one with the maximum ev

                \forall player p \in P
                        \forall cid \in C
                                let a = sup { EV(S'|d) : a \in A, S' is S modified so that action a is taken always}

                                                        Computation
                                                        ===========
        For HU, we have 169^2 diffetent deals, and each deal d \in D has a certain probabiltiy function
                                \sigma_{ e\in E} P(p,d) = 1,
        and thus the ev for the game is
                                EV(S) = \sigma d \in D EV(d,S)
                                      = \sigma d \in D \sigma e \in E EV(e,d,S)
                                      = \sigma d \in D \sigma e \in E P_d(e,d,S) * Value(e,d).
        Firstly, we can express the EV of the strategy S as
                                EV(S)   = EV(d[0], S) + EV(d[1], S) + ... + EV(d[169^2-1])
                                EV(d,s) =      EV(pp, d, S)     +      EV(pf, d, S)     +     EV(f, d, S)
                                        = P(pp,d,S)*Value(pp,d) + P(pf,d,S)*Value(pf,d) +  P(f,d,S)*Value(f,d)
        where in the above, Value() is independant of S, and thus can be precomputed. The function P() is a non-linear
        function of S
                EV(S) = P_d(e,d[0],S) * P(pp,d[0],S)*Value(pp,d[0]) +
                        P_d(e,d[0],S) * P(pf,d[0],S)*Value(pf,d[0]) +
                        P_d(e,d[0],S) * P(f ,d[0],S)*Value(f ,d[0]) +

                        P_d(e,d[1],S) * P(pp,d[1],S)*Value(pp,d[1]) +
                        P_d(e,d[1],S) * P(pf,d[1],S)*Value(pf,d[1]) +
                        P_d(e,d[1],S) * P(f ,d[1],S)*Value(f ,d[1]) +
                        
                        ...                                         +
 
        As an optmization rather an operate on the set D of all deals, we operate on D_l \union D_d, which only has deals in
        standard form. where D_l = { (c0,c1) : c0 < c1 }, D_d = { (c0,c0) }. And thus for HU, we have
                        EV(S) = EV((0,1),D_l,S) + EV((1,0),D_l,S) + EV((0,1), D_d, S),
        If we split up the computation of Value() into those indpeent of the deal, we have
                        Value(f, d) -> [-sb, +sb],
        which is indpendent of the deal, however for ev caluclations we have
                        Value(pp,d) ->  (2*eff - eff ) * ev(d),
                                                         ^-- vector, ie [0.80, 0.2] for AA vs KK
        however if we have the permutation (1,0), the value should be 
                        Value(pp,d) ->  (2*eff - eff ) * ev(d) * P.

        However for three player we have
                        Value(ppf,d) ->  [0,0,-bb] + (2*eff + bb - eff ) * ev((d[0],d[1])) * P
                                                                                 ^ expend this to [a,b,0] etc

        This can be implemented by having EV -> [A,B], and this EV * P -> A + B * P
                        

                                                        Solving
                                                        =======

        Solving will be the most computationall intensive. For solving, given a strategy S, we need to find a strategy
        S' which maximizes EV(S'). We do this elementwise, ie for, 
                
        A

        
 */







struct event_tree{

        struct visitor{
                virtual ~visitor()=default;
                virtual void path_decl(size_t n, double eff){}
                virtual void player_fold(event_tree const*, size_t player_idx){}
                virtual void player_raise(event_tree const*, size_t player_idx, double amt, bool allin){}
                virtual void post_sb(size_t player_idx, double amt, bool allin){}
                virtual void post_bb(size_t player_idx, double amt, bool allin){}
        };
        struct to_string_visitor : visitor{
                virtual void path_decl(size_t n, double eff)override{
                        sstr_ << "path_decl(" << n << "," << eff << ");";
                }
                virtual void player_fold(event_tree const* EV, size_t player_idx)override{
                        sstr_ << "player_fold(" << map_(EV) << "," << player_idx << ");";
                }
                virtual void player_raise(event_tree const* EV, size_t player_idx, double amt, bool allin)override{
                        sstr_ << "player_raise(" << map_(EV)  << ","<< player_idx << "," << amt << "," << allin << ");";
                }
                virtual void post_sb(size_t player_idx, double amt, bool allin)override{
                        sstr_ << "post_sb(" << player_idx << "," << amt << "," << allin << ");";
                }
                virtual void post_bb(size_t player_idx, double amt, bool allin)override{
                        sstr_ << "post_bb(" << player_idx << "," << amt << "," << allin << ");";
                }
                std::string to_string()const{ return sstr_.str(); }
                void clear(){ sstr_.str(""); }
        private:
                std::string map_(event_tree const* ev)const{
                        if( m_.count(ev) == 0 ){
                                m_[ev] = std::string(1,'A' + m_.size());
                        }
                        return m_[ev];
                }
                std::stringstream sstr_;
                mutable std::map<event_tree const*, std::string> m_;
        };

        using strategy_impl_t = std::vector<Eigen::VectorXd>;
        using terminals_vector_type = std::vector<std::shared_ptr<event_tree > >;

        virtual std::string to_string()const noexcept{ return "<>"; }

        void display(){
                std::vector<std::vector<event_tree const*> > stack;
                stack.emplace_back();
                stack.back().push_back(this);
                
                for(;stack.size();){
                        if( stack.back().empty()){
                                // A'
                                stack.pop_back();
                                continue;
                        }
                        auto head = stack.back().back();
                        stack.back().pop_back();
                        
                        std::cout << std::string(stack.size()*2, ' ') << head->to_string() << "\n";
                        if( ! head->is_terminal()){
                                stack.emplace_back();
                                for(size_t i=head->next_.size();i!=0;){
                                        --i;
                                        stack.back().push_back(head->next_[i].get());
                                }
                        }
                }
        }

        
        virtual std::string event()const noexcept{ return ""; }
        
        static std::shared_ptr<event_tree> build(size_t n, double sb, double bb, double eff);
        static std::shared_ptr<event_tree> build_raise_fold(size_t n, double sb, double bb, double eff);
        bool is_terminal()const{
                return next_.empty();
        }
        using terminal_iterator = boost::indirect_iterator<terminals_vector_type::const_iterator>;
        terminal_iterator terminal_begin()const{ return terminals_.begin(); }
        terminal_iterator terminal_end()const{ return terminals_.end(); }

        terminal_iterator non_terminal_begin()const{ return non_terminals_.begin(); }
        terminal_iterator non_terminal_end()const{ return non_terminals_.end(); }
        
        terminal_iterator children_begin()const{ return next_.begin(); }
        terminal_iterator children_end()const{ return next_.end(); }


        virtual void apply(visitor& v)const{
                std::vector<event_tree const*> path{this};
                for(;;){
                        if( path.back()->parent_ == nullptr)
                                break;
                        path.push_back(path.back()->parent_);
                }
                for(size_t idx=path.size();idx!=0;){
                        --idx;
                        path[idx]->apply_impl(v);
                }
        }

        virtual void apply_impl(visitor& v)const{}

        virtual size_t player_idx()const { return player_idx_; }

        explicit event_tree(size_t player_idx = 0)
                : player_idx_{player_idx}
        {}

        std::string pretty()const{
                to_string_visitor v;
                apply(v);
                return v.to_string();
        }
private:

        void add_child(std::shared_ptr<event_tree > child){
                next_.push_back(child);
                child->parent_ = this;
        }
        void finish(){
                //std::vector<std::shared_ptr<event_tree>> stack{std::shared_ptr<event_tree>(this, [](auto&&_){})};
                std::vector<std::shared_ptr<event_tree>> stack = next_;
                for(;stack.size();){
                        auto head = stack.back();
                        stack.pop_back();
                        if( head->is_terminal() ){
                                terminals_.push_back(head);
                        } else {
                                non_terminals_.push_back(head);
                        }
                        std::copy( head->next_.begin(), head->next_.end(), std::back_inserter(stack));
                }
        }
        void make_parent(){
                finish();
                for(auto ptr : non_terminals_ ){
                        ptr->finish();
                }
        }

protected:
        event_tree* parent_{nullptr};
        std::vector< std::shared_ptr<event_tree> > next_;

        terminals_vector_type terminals_;
        terminals_vector_type non_terminals_;

        size_t player_idx_{static_cast<size_t>(-1)};

};

struct event_tree_hu_sb_bb : event_tree{
        event_tree_hu_sb_bb(size_t n, double sb, double bb, double eff)
                : n_{n}, sb_{sb}, bb_{bb}, eff_{eff}
        {}
private:
        virtual void apply_impl(visitor& v)const override{
                v.path_decl(n_, eff_);
                v.post_bb(0, bb_, false);
                v.post_sb(1, sb_, false);
        }
        size_t n_;
        double sb_;
        double bb_;
        double eff_;
};
struct event_tree_sb_bb : event_tree{
        event_tree_sb_bb(size_t n, double sb, double bb, double eff)
                : n_{n}, sb_{sb}, bb_{bb}, eff_{eff}
        {}
private:
        virtual void apply_impl(visitor& v)const override{
                v.path_decl(n_, eff_);
                v.post_sb(0, sb_, false);
                v.post_bb(1, bb_, false);
        }
        size_t n_;
        double sb_;
        double bb_;
        double eff_;
};

struct event_tree_push : event_tree{
        explicit event_tree_push(size_t player_idx, double amt):event_tree{player_idx}, amt_{amt}{}
        virtual std::string event()const noexcept override{ return "p"; }
        virtual std::string to_string()const noexcept override{ return "push"; }
private:
        virtual void apply_impl(visitor& v)const override{
                v.player_raise(this, parent_->player_idx(), amt_, true);
        }
        double amt_;
};
struct event_tree_raise : event_tree{
        explicit event_tree_raise(size_t player_idx, double amt):event_tree{player_idx}, amt_{amt} {}
        virtual std::string event()const noexcept override{ return "r"; }
        virtual std::string to_string()const noexcept override{ return "raise"; }
private:
        virtual void apply_impl(visitor& v)const override{
                v.player_raise(this, parent_->player_idx(), amt_, false);
        }
        double amt_;
};
struct event_tree_fold : event_tree{
        explicit event_tree_fold(size_t player_idx):event_tree{player_idx}{}
        virtual std::string event()const noexcept override{ return "f"; }
        virtual std::string to_string()const noexcept override{ return "fold"; }
private:
        virtual void apply_impl(visitor& v)const override{
                v.player_fold(this, parent_->player_idx());
        }
};


std::shared_ptr<event_tree> event_tree::build_raise_fold(size_t n, double sb, double bb, double eff){
        std::shared_ptr<event_tree> root;

        root = std::make_shared<event_tree_hu_sb_bb>(2, sb, bb, eff);
        auto p = std::make_shared<event_tree_push>(1, eff);
        auto r = std::make_shared<event_tree_raise>(1, 2);
        auto f = std::make_shared<event_tree_fold>(1);
        root->add_child(p);
        root->add_child(r);
        root->add_child(f);


        auto pp = std::make_shared<event_tree_push>(0, eff);
        auto pf = std::make_shared<event_tree_fold>(0);
        p->add_child(pp);
        p->add_child(pf);

        auto rp = std::make_shared<event_tree_push>(0, eff);
        auto rf = std::make_shared<event_tree_fold>(0);
        
        r->add_child(rp);
        r->add_child(rf);
        
        auto rpp = std::make_shared<event_tree_push>(1, eff);
        auto rpf = std::make_shared<event_tree_fold>(1);
        rp->add_child(rpp);
        rp->add_child(rpf);

        root->make_parent();
        return root;
}
std::shared_ptr<event_tree> event_tree::build(size_t n, double sb, double bb, double eff){
        std::shared_ptr<event_tree> root;

        switch(n){
                case 2:
                {

                        root = std::make_shared<event_tree_hu_sb_bb>(2, sb, bb, eff);
                        auto p = std::make_shared<event_tree_push>(1, eff);
                        auto f = std::make_shared<event_tree_fold>(1);
                        root->add_child(p);
                        root->add_child(f);



                        auto pp = std::make_shared<event_tree_push>(0, eff);
                        auto pf = std::make_shared<event_tree_fold>(0);
                        p->add_child(pp);
                        p->add_child(pf);
                        


                        break;

                }
                #if 0
                case 3:
                {
                        /*
                                             <0>
                                           /     \
                                         P         F
                                         |         |
                                        <1>       <2>
                                       /   \      /  \
                                      P    F     P    F
                                      |    |     |
                                     <3>  <4>   <5>
                                     / \  / \   / \
                                     P F  P F   P F  

                         */
                        root = std::make_shared<event_tree_sb_bb>(3, sb, bb, eff);

                        auto p = std::make_shared<event_tree_push>(2, 0, eff);
                        auto f = std::make_shared<event_tree_fold>(2, 0);
                        root->add_child(p);
                        root->add_child(f);

                        auto pp = std::make_shared<event_tree_push>(0, 1, eff);
                        auto pf = std::make_shared<event_tree_fold>(0, 1);
                        p->add_child(pp);
                        p->add_child(pf);

                        auto fp = std::make_shared<event_tree_push>(0, 2, eff);
                        auto ff = std::make_shared<event_tree_fold>(0, 2);
                        f->add_child(fp);
                        f->add_child(ff);
                        
                        auto ppp = std::make_shared<event_tree_push>(1, 3, eff);
                        auto ppf = std::make_shared<event_tree_fold>(1, 3);
                        pp->add_child(ppp);
                        pp->add_child(ppf);
                        
                        auto pfp = std::make_shared<event_tree_push>(1, 4, eff);
                        auto pff = std::make_shared<event_tree_fold>(1, 4);
                        pf->add_child(pfp);
                        pf->add_child(pff);

                        auto fpp = std::make_shared<event_tree_push>(1, 5, eff);
                        auto fpf = std::make_shared<event_tree_fold>(1, 5);
                        fp->add_child(fpp);
                        fp->add_child(fpf);

                        break;

                }
                #endif

        }
        root->make_parent();
        return root;

}

struct strategy_decl{
        

        struct strategy_choice_decl{
                strategy_choice_decl(event_tree const* ev, size_t idx, size_t player_idx, std::vector<size_t> const& alloc)
                        :ev_{ev}
                        ,idx_(idx)
                        ,player_idx_(player_idx),
                        alloc_(alloc)
                {}
                size_t index()const{ return idx_; }
                size_t num_choies()const{ return alloc_.size(); }
                size_t player_index()const{ return player_idx_; }
                size_t at(size_t idx)const{ return alloc_[idx]; }
                friend std::ostream& operator<<(std::ostream& ostr, strategy_choice_decl const& self){
                        ostr << "idx_ = " << self.idx_ << ",";
                        ostr << "pretty_ = " << self.ev_->pretty() << ",";
                        ostr << "player_idx_ = " << self.player_idx_ << ",";
                        typedef std::vector<size_t>::const_iterator CI0;
                        const char* comma = "";
                        ostr << "\nalloc_" << " = {";
                        for(CI0 iter= self.alloc_.begin(), end=self.alloc_.end();iter!=end;++iter){
                                ostr << comma << *iter;
                                comma = ", ";
                        }
                        ostr << "}\n";
                        return ostr;
                }
        private:
                event_tree const* ev_;
                size_t idx_;
                size_t player_idx_;
                std::vector<size_t> alloc_;
        };
        
        size_t dimensions()const{ return dims_; }
        auto begin()const{ return v_.begin(); }
        auto end()const{ return v_.end(); }

        friend std::ostream& operator<<(std::ostream& ostr, strategy_decl const& self){
                ostr << "dims_ = " << self.dims_;
                typedef std::vector<strategy_choice_decl>::const_iterator CI0;
                const char* comma = "";
                ostr << "v_" << " = {";
                for(CI0 iter= self.v_.begin(), end=self.v_.end();iter!=end;++iter){
                        ostr << comma << *iter;
                        comma = ", ";
                }
                ostr << "}\n";
                return ostr;
        }
private:
        size_t dims_;
        std::vector<strategy_choice_decl> v_;

public:

        static strategy_decl generate(event_tree const* root){

                strategy_decl result;

                std::vector< event_tree const*> stack;
                stack.push_back(root);
                std::map<event_tree const*, std::vector<event_tree const*> > G;
                for(;stack.size();){
                        auto head = stack.back();
                        stack.pop_back();
                        if( head->is_terminal()){
                                continue;
                        }
                        for(auto iter=head->children_begin(), end=head->children_end();iter!=end;++iter){
                                G[head].push_back(&*iter);
                                stack.push_back(&*iter);
                        }
                }
                typedef std::map<event_tree const*, std::vector<event_tree const*> >::const_iterator VI;
                for(VI iter(G.begin()), end(G.end());iter!=end;++iter){

                        std::stringstream sstr;
                        sstr << iter->first << "->" << "{";
                        for(size_t j=0;j!=iter->second.size();++j){
                                if( j != 0 )
                                        sstr << ", ";
                                sstr << iter->second[j];
                        }
                        sstr << "}";
                        std::cout << sstr.str() << "\n";
                }
                
                size_t choice_idx = 0;
                size_t strat_idx = 0;
                for(VI iter(G.begin()), end(G.end());iter!=end;++iter){
                        std::vector<size_t> alloc;
                        for(;alloc.size() < iter->second.size();){
                                alloc.push_back(strat_idx++);
                        }
                        result.v_.emplace_back(iter->first, choice_idx, iter->first->player_idx(), std::move(alloc));
                        ++choice_idx;
                }
                result.dims_ = strat_idx;
                return result;
        }
};


struct Scratch : Command{
        explicit
        Scratch(std::vector<std::string> const& args):args_{args}{}
        virtual int Execute()override{
                //auto gt = event_tree::build_raise_fold(2, 0.5, 1.0, 10.0);
                auto gt = event_tree::build(2, 0.5, 1.0, 10.0);
                gt->display();

                event_tree::to_string_visitor v;
                for(auto iter=gt->terminal_begin(),end=gt->terminal_end();iter!=end;++iter){
                        iter->apply(v);
                        std::cout << v.to_string() << "\n";
                        v.clear();
                }

                std::vector<event_tree const*> non_terminals;
                std::map<event_tree const*, std::vector<event_tree const*> > G;

                strategy_decl sd = strategy_decl::generate(gt.get());
                std::cout << sd << "\n";

                return EXIT_SUCCESS;
        }
private:
        std::vector<std::string> const& args_;
};
static TrivialCommandDecl<Scratch> ScratchDecl{"scratch"};
