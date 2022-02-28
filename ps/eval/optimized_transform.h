#ifndef LIB_EVAL_OPTIMIZED_TRANSFORM_H
#define LIB_EVAL_OPTIMIZED_TRANSFORM_H

#include "ps/eval/rank_opt_device.h"

// #define DO_VAlGRIND


#ifdef DO_VAlGRIND
#include <valgrind/callgrind.h>
#endif // DO_VAlGRIND

namespace ps{

    template<class T>
    struct supports_single_mask : std::false_type {};


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
                boost::timer::cpu_timer tmr;

                constexpr bool WithLogging = true;

#ifdef DO_VAlGRIND
                CALLGRIND_START_INSTRUMENTATION;
#endif // DO_VAlGRIND

                // this needs to outlive the lifetime of every object
                factory_type factory;

                std::vector<sub_ptr_type> subs;

                for(auto& target : target_list){
                        auto instr = reinterpret_cast<card_eval_instruction*>((*target).get());
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

                if(WithLogging) PS_LOG(trace) << "Have " << S.size() << " unique holdem hands";

                // this is the maximually speed up the compution, by preocompyting some stuff
                rank_opt_device rod = rank_opt_device::create(S);
                std::unordered_map<holdem_id, size_t> allocation_table;
                for(size_t idx=0;idx!=rod.size();++idx){
                        allocation_table[rod[idx].hid] = idx;
                }
                for(auto& _ : subs){
                        _->allocate( [&](auto hid){ return allocation_table.find(hid)->second; });
                }

                if (WithLogging) PS_LOG(trace) << "Have " << subs.size() << " subs";

                std::vector<ranking_t> R, R_proto;
                R.resize(rod.size());
                R_proto.resize(rod.size());

                using weights_ty = std::vector< eval_counter_type>;
                weights_ty weights;
                weights.resize(subs.size());
                
                
                schedular_type shed{ rod.size(), subs};

                

                auto apply_any_board = [&](auto const& b)noexcept{
                        suit_id flush_suit = b.flush_suit();
                        auto flush_mask    = b.flush_mask();
                        
                        
                        for(size_t idx=0;idx!=rod.size();++idx){
                                auto const& _ = rod[idx];

                              

                                ranking_t rr = b.no_flush_rank(_.r0, _.r1);

                                bool s0m = ( _.s0 == flush_suit );
                                bool s1m = ( _.s1  == flush_suit );
                                
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

                if (WithLogging) PS_LOG(trace) << tmr.format(4, "init took %w seconds");
                tmr.start();
                size_t count = 0;
                //std::unordered_map<int, int> non_zero_m;

                constexpr bool enable_weight_branch{ false };

#if 0
                for(auto const& b : otc.w.weighted_aggregate_rng() ){
                        

                        apply_any_board(b);
                        shed.end_eval(&b.masks, 0ull);
                       
                        ++count;
                }
#endif
                for (auto const& g : otc.w.grouping)
                {
                    for (size_t idx = 0; idx != rod.size(); ++idx) {
                        auto const& hand_decl = rod[idx];
                        ranking_t rr = g.no_flush_rank(hand_decl.r0, hand_decl.r1);
                        R[idx] = rr;
                        shed.put(idx, rr);
                    }
                    shed.end_eval(&g.get_no_flush_masks(), 0ull);

                    R_proto = R;

                    for (auto f : g.suit_symmetry_vec())
                    {
                        for (suit_id sid = 0; sid != 4; ++sid)
                        {
                            R = R_proto;

                            for (size_t idx = 0; idx != rod.size(); ++idx) {
                                auto const& hand_decl = rod[idx];

                                bool s0m = (hand_decl.s0 == flush_suit);
                                bool s1m = (hand_decl.s1 == flush_suit);

                                auto fm = flush_mask;

                                if (s0m) {
                                    fm |= 1ull << _.r0;
                                }
                                if (s1m) {
                                    fm |= 1ull << _.r1;
                                }


                        }
                    }

                    
                    ++count;
                }








                if (WithLogging) PS_LOG(trace) << tmr.format(4, "no    flush boards took %w seconds") << " to do " << count << " boards";
                tmr.start();
                count = 0;
                for(auto const& b : otc.w.weighted_singleton_rng() ){
                        
                        apply_any_board(b);

                        if constexpr (supports_single_mask< Schedular>{})
                        {
                            shed.end_eval_single(b.single_rank_mask());
                        }
                        else
                        {
                            shed.end_eval(&b.masks, 0ull);
                        }
                        ++count;
                }
                if (WithLogging) PS_LOG(trace) << tmr.format(4, "maybe flush boards took %w seconds") << " to do " << count << " boards";

                //for (const auto& p : non_zero_m)
                //{
                //    PS_LOG(trace) << "non_zero_m : " << p.first << " => " << p.second;
                //}

                shed.regroup();

                
                for(auto& _ : subs){
                        _->finish();
                }

#ifdef DO_VAlGRIND
                CALLGRIND_STOP_INSTRUMENTATION;
                CALLGRIND_DUMP_STATS;
#endif
        }
};
} // end namespace ps

#endif // LIB_EVAL_OPTIMIZED_TRANSFORM_H
