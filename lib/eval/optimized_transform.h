#ifndef LIB_EVAL_OPTIMIZED_TRANSFORM_H
#define LIB_EVAL_OPTIMIZED_TRANSFORM_H

#include "lib/eval/rank_opt_device.h"

#include <valgrind/callgrind.h>

namespace ps{


template<class Sub,
         class Schedular,
         class Factory,
         class Eval>
struct optimized_transform : optimized_transform_base
{
        using factory_type   = typename Factory::template bind<Sub>;
        using sub_ptr_type   = typename factory_type::sub_ptr_type;
        using schedular_type = typename Schedular::template bind<sub_ptr_type>;
        using eval_type      = Eval;

        virtual void apply(optimized_transform_context& otc, computation_context* ctx, instruction_list* instr_list, computation_result* result,
                   std::vector<typename instruction_list::iterator> const& target_list)noexcept override
        {
                CALLGRIND_START_INSTRUMENTATION;

                // this needs to outlive the lifetime of every object
                factory_type factory;

                std::vector<sub_ptr_type> subs;

                for(auto& target : target_list){
                        auto instr = reinterpret_cast<any_card_eval_vector_instruction*>((*target).get());
                        subs.push_back( factory(target, instr) );
                }

                // Here we create a list of all the evaluations that need to be done.
                // Because each evalution is done card wise, doing multiple evaluations
                // are the same time, some of the cards are shared
                // 
                std::unordered_set<holdem_id> S;
                for(auto& _ : subs){
                        _->declare(S);
                }

                // this is the maximually speed up the compution, by preocompyting some stuff
                rank_opt_device rod = rank_opt_device::create(S);
                std::unordered_map<holdem_id, size_t> allocation_table;
                for(size_t idx=0;idx!=rod.size();++idx){
                        allocation_table[rod[idx].hid] = idx;
                }
                for(auto& _ : subs){
                        _->allocate( [&](auto hid){ return allocation_table.find(hid)->second; });
                }

                PS_LOG(trace) << "Have " << subs.size() << " subs";

                std::vector<ranking_t> R;
                R.resize(rod.size());
                
                
                schedular_type shed{ rod.size(), subs};

                auto apply_any_board = [&](auto const& b)noexcept{
                        suit_id flush_suit = b.flush_suit();
                        auto flush_mask    = b.flush_mask();
                        
                        
                        for(size_t idx=0;idx!=rod.size();++idx){
                                auto const& _ = rod[idx];

                                ranking_t rr = b.no_flush_rank(_.r0, _.r1);

                                bool s0m = ( _.s0 == flush_suit );
                                bool s1m = ( _.s1 == flush_suit );
                                
                                auto fm = flush_mask;

                                if( s0m ){
                                        fm |= 1ull << _.r0;
                                }
                                if( s1m ){
                                        fm |= 1ull << _.r1;
                                }

                                ranking_t sr = otc.fme(fm);
                                ranking_t tr = std::min(sr, rr);

                                shed.put(idx, tr);

                        }

                };

                {
                        boost::timer::auto_cpu_timer at("no    flush took %w seconds\n");
                        for(auto const& b : otc.w.weighted_aggregate_rng() ){
                                apply_any_board(b);
                                shed.end_eval_no_flush(&b.masks, 0ull);
                        }
                }
                {
                        boost::timer::auto_cpu_timer at("maybe flush took %w seconds\n");
                        for(auto const& b : otc.w.weighted_singleton_rng() ){
                                apply_any_board(b);
                                shed.end_eval_maybe_flush(&b.masks, 0ull);
                        }
                }

                shed.regroup();

                
                for(auto& _ : subs){
                        _->finish();
                }

                CALLGRIND_STOP_INSTRUMENTATION;
                CALLGRIND_DUMP_STATS;
        }
};
} // end namespace ps

#endif // LIB_EVAL_OPTIMIZED_TRANSFORM_H
