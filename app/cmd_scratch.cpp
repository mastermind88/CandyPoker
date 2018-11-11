#include <thread>
#include <numeric>
#include <atomic>
#include <boost/format.hpp>
#include "ps/support/config.h"
#include "ps/base/frontend.h"
#include "ps/base/cards.h"
#include "ps/base/tree.h"
#include "ps/base/board_combination_iterator.h"
#include "ps/support/index_sequence.h"
#include "ps/eval/evaluator_5_card_map.h"
#include "ps/eval/evaluator_6_card_map.h"
#include "app/pretty_printer.h"
#include "ps/base/algorithm.h"
#include "ps/eval/instruction.h"
#include "ps/base/rank_hasher.h"
#include "ps/detail/tree_printer.h"
#include "ps/base/suit_hasher.h"
#include "ps/base/holdem_board_decl.h"
#include "ps/eval/pass_mask_eval.h"
#include "ps/eval/class_cache.h"

#include <boost/timer/timer.hpp>

#include <boost/log/trivial.hpp>

#include <Eigen/Dense>
#include <fstream>

#include "ps/support/command.h"
#include "ps/support/persistent.h"
#include <boost/intrusive/list.hpp>
//#include <boost/program_options.hpp>
#include "ps/eval/holdem_class_vector_cache.h"
#include <boost/iterator/indirect_iterator.hpp>

using namespace ps;
struct event_tree{

        struct visitor{
                virtual ~visitor()=default;
                virtual void begin_path(){}
                virtual void end_path(){}
                virtual void player_fold(size_t player_idx){}
                virtual void player_raise(size_t player_idx, double amt, bool allin){}
                virtual void post_sb(size_t player_idx, double amt, bool allin){}
                virtual void post_bb(size_t player_idx, double amt, bool allin){}
        };
        struct to_string_visitor : visitor{
                virtual void begin_path(){
                        sstr_ << "begin_path();";
                }
                virtual void end_path(){
                        sstr_ << "end_path();";
                }
                virtual void player_fold(size_t player_idx){
                        sstr_ << "player_fold(" << player_idx << ");";
                }
                virtual void player_raise(size_t player_idx, double amt, bool allin){
                        sstr_ << "player_raise(" << player_idx << "," << amt << "," << allin << ");";
                }
                virtual void post_sb(size_t player_idx, double amt, bool allin){
                        sstr_ << "post_sb(" << player_idx << "," << amt << "," << allin << ");";
                }
                virtual void post_bb(size_t player_idx, double amt, bool allin){
                        sstr_ << "post_bb(" << player_idx << "," << amt << "," << allin << ");";
                }
                std::string to_string()const{ return sstr_.str(); }
        private:
                std::stringstream sstr_;
        };

        using strategy_impl_t = std::vector<Eigen::VectorXd>;
        using terminals_vector_type = std::vector<std::shared_ptr<event_tree > >;

        double probability_of_event(strategy_impl_t const& S, holdem_class_vector const& cv)const noexcept{
                auto p = probability_of_event_given(S, cv);
                if( parent_ )
                        p *= parent_->probability_of_event(S, cv);
                return p;
        }
        std::string event_list()const noexcept{
                if( parent_ )
                        return parent_->event_list() + "p";
                return "";
        }
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

        
        virtual double probability_of_event_given(strategy_impl_t const& S, holdem_class_vector const& cv)const noexcept{ return 1.0; }
        virtual std::string event()const noexcept{ return ""; }
        
        static std::shared_ptr<event_tree> build(size_t n, double sb, double bb, double eff);
        static std::shared_ptr<event_tree> build_raise_fold(size_t n, double sb, double bb, double eff);
        bool is_terminal()const{
                return next_.empty();
        }
        using terminal_iterator = boost::indirect_iterator<terminals_vector_type::const_iterator>;
        terminal_iterator terminal_begin()const{ return terminals_.begin(); }
        terminal_iterator terminal_end()const{ return terminals_.end(); }


        virtual void apply(visitor& v)const{
                std::vector<event_tree const*> path{this};
                for(;;){
                        if( path.back()->parent_ == nullptr)
                                break;
                        path.push_back(path.back()->parent_);
                }
                v.begin_path();
                for(size_t idx=path.size();idx!=0;){
                        --idx;
                        path[idx]->apply_impl(v);
                }
                v.end_path();
        }
private:
        virtual void apply_impl(visitor& v)const{}

        void add_child(std::shared_ptr<event_tree > child){
                next_.push_back(child);
                child->parent_ = this;
        }
        void make_parent(){
                std::vector<std::shared_ptr<event_tree>> stack = next_;
                for(;stack.size();){
                        auto head = stack.back();
                        stack.pop_back();
                        if( head->is_terminal() ){
                                terminals_.push_back(head);
                        }
                        std::copy( head->next_.begin(), head->next_.end(), std::back_inserter(stack));
                }
        }

        event_tree* parent_{nullptr};
        std::vector< std::shared_ptr<event_tree> > next_;

        terminals_vector_type terminals_;
};

struct event_tree_hu_sb_bb : event_tree{
        event_tree_hu_sb_bb(double sb, double bb)
                : sb_{sb}, bb_{bb}
        {}
private:
        virtual void apply_impl(visitor& v)const override{
                v.post_bb(0, bb_, false);
                v.post_sb(1, sb_, false);
        }
        double sb_;
        double bb_;
};
struct event_tree_sb_bb : event_tree{
        event_tree_sb_bb(double sb, double bb)
                : sb_{sb}, bb_{bb}
        {}
private:
        virtual void apply_impl(visitor& v)const override{
                v.post_sb(0, sb_, false);
                v.post_bb(1, bb_, false);
        }
        double sb_;
        double bb_;
};

struct event_tree_push : event_tree{
        explicit event_tree_push(size_t player_idx, size_t index, double amt):player_idx_{player_idx}, index_{index}, amt_{amt}{}
        virtual double probability_of_event_given(strategy_impl_t const& S, holdem_class_vector const& cv)const noexcept override{
                return S[index_][cv[player_idx_]];
        }
        virtual std::string event()const noexcept override{ return "p"; }
        virtual std::string to_string()const noexcept override{ return "push"; }
private:
        virtual void apply_impl(visitor& v)const override{
                v.player_raise(player_idx_, amt_, true);
        }
        size_t player_idx_;
        size_t index_;
        double amt_;
};
struct event_tree_raise : event_tree{
        explicit event_tree_raise(size_t player_idx, size_t index, double amt):player_idx_{player_idx}, index_{index}, amt_{amt} {}
        virtual double probability_of_event_given(strategy_impl_t const& S, holdem_class_vector const& cv)const noexcept override{
                return S[index_][cv[player_idx_]];
        }
        virtual std::string event()const noexcept override{ return "r"; }
        virtual std::string to_string()const noexcept override{ return "raise"; }
private:
        virtual void apply_impl(visitor& v)const override{
                v.player_raise(player_idx_, amt_, false);
        }
        size_t player_idx_;
        size_t index_;
        double amt_;
};
struct event_tree_fold : event_tree{
        explicit event_tree_fold(size_t player_idx, size_t index):player_idx_{player_idx}, index_{index}{}
        virtual double probability_of_event_given(strategy_impl_t const& S, holdem_class_vector const& cv)const noexcept override{
                return 1.0 - S[index_][cv[player_idx_]];
        }
        virtual std::string event()const noexcept override{ return "f"; }
        virtual std::string to_string()const noexcept override{ return "fold"; }
private:
        virtual void apply_impl(visitor& v)const override{
                v.player_fold(player_idx_);
        }
        size_t player_idx_;
        size_t index_;
};


std::shared_ptr<event_tree> event_tree::build_raise_fold(size_t n, double sb, double bb, double eff){
        std::shared_ptr<event_tree> root;

        root = std::make_shared<event_tree_hu_sb_bb>(sb, bb);
        auto p = std::make_shared<event_tree_push>(1, 0, eff);
        auto r = std::make_shared<event_tree_raise>(1, 0, 2);
        auto f = std::make_shared<event_tree_fold>(1, 0);
        root->add_child(p);
        root->add_child(r);
        root->add_child(f);


        auto pp = std::make_shared<event_tree_push>(0, 1, eff);
        auto pf = std::make_shared<event_tree_fold>(0, 1);
        p->add_child(pp);
        p->add_child(pf);

        auto rp = std::make_shared<event_tree_push>(0, 1, eff);
        auto rf = std::make_shared<event_tree_fold>(0, 1);
        r->add_child(rp);
        r->add_child(rf);
        
        auto rpp = std::make_shared<event_tree_push>(1, 2, eff);
        auto rpf = std::make_shared<event_tree_fold>(1, 2);
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
                        root = std::make_shared<event_tree_hu_sb_bb>(sb, bb);
                        auto p = std::make_shared<event_tree_push>(1, 0, eff);
                        auto f = std::make_shared<event_tree_fold>(1, 0);
                        root->add_child(p);
                        root->add_child(f);

                        auto pp = std::make_shared<event_tree_push>(0, 1, eff);
                        auto pf = std::make_shared<event_tree_fold>(0, 1);
                        p->add_child(pp);
                        p->add_child(pf);


                        break;

                }
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
                        root = std::make_shared<event_tree_sb_bb>(sb, bb);

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

        }
        root->make_parent();
        return root;

}

struct Scratch : Command{
        explicit
        Scratch(std::vector<std::string> const& args):args_{args}{}
        virtual int Execute()override{
                auto gt = event_tree::build_raise_fold(2, 0.5, 1.0, 10.0);
                gt->display();

                for(auto iter=gt->terminal_begin(),end=gt->terminal_end();iter!=end;++iter){
                        event_tree::to_string_visitor v;
                        iter->apply(v);
                        std::cout << v.to_string() << "\n";
                }

                return EXIT_SUCCESS;
        }
private:
        std::vector<std::string> const& args_;
};
static TrivialCommandDecl<Scratch> ScratchDecl{"scatch"};
