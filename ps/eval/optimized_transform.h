#ifndef LIB_EVAL_OPTIMIZED_TRANSFORM_H
#define LIB_EVAL_OPTIMIZED_TRANSFORM_H

#include "ps/eval/rank_opt_device.h"

// #define DO_VAlGRIND


#ifdef DO_VAlGRIND
#include <valgrind/callgrind.h>
#endif // DO_VAlGRIND

namespace ps{




template<class Sub,
         class Schedular,
         class Factory>
struct optimized_transform : optimized_transform_base
{
        using factory_type   = typename Factory::template bind<Sub>;
        using sub_ptr_type   = typename factory_type::sub_ptr_type;
        using schedular_type = typename Schedular::template bind<sub_ptr_type>;

        virtual void apply(optimized_transform_context& otc, computation_context* ctx, instruction_list* instr_list, computation_result* result,
                   std::vector<typename instruction_list::iterator> const& target_list)noexcept override
        {
                boost::timer::cpu_timer tmr;

                const bool WithLogging = ( ctx->Verboseicity() > 0 );
                const bool WithStepping = ( ctx->Verboseicity() > 1 );

#ifdef DO_VAlGRIND
                CALLGRIND_START_INSTRUMENTATION;
#endif // DO_VAlGRIND

                // this needs to outlive the lifetime of every object
                factory_type factory;

                std::vector<sub_ptr_type> subs;

                subs.reserve(target_list.size());
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
                        _.declare(S);
                }


                if(WithLogging) PS_LOG(trace) << "Have " << S.size() << " unique holdem hands";

                // this is the maximually speed up the compution, by preocompyting some stuff

                holdem_hand_vector hhv;
                for(holdem_id hid=0;hid!=holdem_hand_decl::max_id;++hid)
                {
                        hhv.push_back(hid);
                }
                //rank_opt_device rod = rank_opt_device::create(S);
                rank_opt_device rod = rank_opt_device::create(hhv);
                

                if (WithLogging) PS_LOG(trace) << "Have " << subs.size() << " subs";

                std::vector<ranking_t> R;
                R.resize(rod.size());

                
                auto dispatch = [&](std::vector<ranking_t> const& evals, mask_set const* ms)mutable noexcept{
                        for(auto& ptr : subs){
                                auto weight = generic_weight_policy{}
                                        .calculate(ptr.hand_mask(), *ms);
                                if( weight == 0 )
                                        continue;
                                ptr.accept_weight(weight, evals);
                        }
                };


                if (WithLogging) PS_LOG(trace) << tmr.format(4, "init took %u seconds");
                tmr.start();
                size_t count = 0;



                std::vector<ranking_t> ranking_proto;
                ranking_proto.resize(rod.size());

                std::array< std::vector<ranking_t>, 4> suit_batch;
                suit_batch[0].resize(rod.size());
                suit_batch[1].resize(rod.size());
                suit_batch[2].resize(rod.size());
                suit_batch[3].resize(rod.size());

                boost::timer::cpu_timer shed_timer;
                shed_timer.stop();

                //for (auto const& g : otc.w.grouping)
                
                size_t head_pct = 0;
                for(size_t gidx= 0; gidx != otc.w.grouping.size();++gidx)
                {
                    auto const& g = otc.w.grouping[gidx];

                    for (size_t idx = 0; idx != rod.size(); ++idx) {
                        auto const& hand_decl = rod[idx];
                        ranking_proto[idx] = g.no_flush_rank(hand_decl.r0, hand_decl.r1);
                    }

                    shed_timer.resume();
                    //shed.end_eval_from_mem(ranking_proto, &g.get_no_flush_masks(), 0ull);
                    dispatch(ranking_proto, &g.get_no_flush_masks());
                    shed_timer.stop();


                    


                    for (auto const& f : g.suit_symmetry_vec())
                    {

#if 1
                        const size_t fm_common = f.flush_mask();
                        
                        suit_batch[0] = ranking_proto;

                        if (detail::popcount(fm_common) >= 5)
                        {
                            ranking_t sr_minimum = otc.fme(fm_common);
                            for (auto& value : suit_batch[0])
                            {
                                value = std::min(value, sr_minimum);
                            }
                        }
                        
                        suit_batch[1] = suit_batch[0];
                        suit_batch[2] = suit_batch[0];
                        suit_batch[3] = suit_batch[0];

                        for (size_t idx = 0; idx != rod.size(); ++idx) {
                            auto const& hand_decl = rod[idx];

                            auto commit_flush = [&](suit_id sid, size_t flush_mask)
                            {
                                ranking_t sr = otc.fme(flush_mask);
                                suit_batch[sid][idx] = std::min(sr, suit_batch[sid][idx]);
                            };

                            if (hand_decl.s0 == hand_decl.s1)
                            {
                                auto fm = fm_common | (1ull << hand_decl.r0) | (1ull << hand_decl.r1);
                                commit_flush(hand_decl.s0, fm);
                            }
                            else
                            {
                                auto first_fm = fm_common | (1ull << hand_decl.r0);
                                auto second_fm = fm_common | (1ull << hand_decl.r1);
                                commit_flush(hand_decl.s0, first_fm);
                                commit_flush(hand_decl.s1, second_fm);
                            }
                        }

                        for (suit_id sid = 0; sid != 4; ++sid)
                        {
                           
                            mask_set const& suit_mask_set = f.board_card_masks()[sid];

                            shed_timer.resume();
                            dispatch(suit_batch[sid], &suit_mask_set);
                            shed_timer.stop();

                        }
#else
                        for (suit_id sid = 0; sid != 4; ++sid)
                        {

                            for (size_t idx = 0; idx != rod.size(); ++idx) {
                                auto const& hand_decl = rod[idx];
                                ranking_t rr = g.no_flush_rank(hand_decl.r0, hand_decl.r1);
                                bool s0m = (hand_decl.s0 == sid);
                                bool s1m = (hand_decl.s1 == sid);

                                auto fm = f.flush_mask();

                                if (s0m) {
                                    fm |= 1ull << hand_decl.r0;
                                }
                                if (s1m) {
                                    fm |= 1ull << hand_decl.r1;
                                }

                                ranking_t sr = otc.fme(fm);
                                ranking_t tr = std::min(sr, rr);

                                shed.put(idx, tr);
                            }
                            shed.end_eval(&f.board_card_masks()[sid], 0ull);

                        }   
#endif
                    }
            


                    
                    ++count;
                    if( WithStepping )
                    {
                             const size_t implied_pct = (gidx+1) * 100 / otc.w.grouping.size();
                            if( implied_pct > head_pct)
                            {
                                    head_pct = implied_pct;
                                    PS_LOG(debug) << " Done " << gidx << "/" << otc.w.grouping.size() << " [" << head_pct << "%]";
                            }
                    }
                   
                }





                if (WithLogging) PS_LOG(trace) << shed_timer.format(4, "shed took %u seconds");

                

                for (size_t idx = 0; idx != target_list.size(); ++idx)
                {
                        auto target = target_list[idx];
                        auto instr = reinterpret_cast<card_eval_instruction*>((*target).get());
                        subs[idx].finish( target, instr);
                }

#ifdef DO_VAlGRIND
                CALLGRIND_STOP_INSTRUMENTATION;
                CALLGRIND_DUMP_STATS;
#endif
        }
};
} // end namespace ps

#endif // LIB_EVAL_OPTIMIZED_TRANSFORM_H
