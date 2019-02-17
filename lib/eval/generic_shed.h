#ifndef LIB_EVAL_GENERIC_SHED_H
#define LIB_EVAL_GENERIC_SHED_H


namespace ps{

        //using eval_counter_type = std::uint_fast32_t;
        using eval_counter_type = size_t;

        /*
                This was originally meant to be a small interface, and then
                from the small interface be able to change the order of evaluations,
                thinking I could do something in terns or reordering for CPU-cache
                improvment. This didn't work. but this is still helpfull
         */

        struct generic_weight_policy{
                // this is 5% faster (in my limited testing)
                enum{ CheckUnion = true };
                eval_counter_type calculate(size_t hv_mask, mask_set const* ms, size_t single_mask)const noexcept{
                        if( !! ms ){
                                if( CheckUnion ){
                                        if( ( ms->get_union() & hv_mask) == 0 ){
                                                return ms->size();
                                        }
                                }
                                return ms->count_disjoint_with<eval_counter_type>(hv_mask);
                        } else {
                                return (( hv_mask & single_mask )==0?1:0);
                        }
                }
        };

        struct generic_shed{
                template<class SubPtrType>
                struct bind{
                        // this is 5% faster (in my limited testing)
                        enum{ CheckZeroWeight = true };
                        explicit bind(size_t batch_size, std::vector<SubPtrType>& subs)
                        {
                                evals_.resize(batch_size);
                                
                                for(auto const& _ : subs ){
                                        _->display();
                                        auto flush_bit = !( _->get_type() | instruction::T_CardMaybeFlushEval );
                                        if( flush_bit ){
                                                maybe_flush_subs_.push_back(_);
                                        } else {
                                                no_flush_subs_.push_back(_);
                                        }
                                }

                                PS_LOG(trace) << "maybe_flush_subs_.size() => " << maybe_flush_subs_.size();
                                PS_LOG(trace) << "no_flush_subs_.size() => " << no_flush_subs_.size();
                        }
                        void put(size_t index, ranking_t rank)noexcept{
                                evals_[index] = rank;
                        }
                        void end_eval_no_flush(mask_set const* ms, size_t single_mask)noexcept{
                                do_end_eval_flush(no_flush_subs_, ms, single_mask);
                        }
                        void end_eval_maybe_flush(mask_set const* ms, size_t single_mask)noexcept{
                                do_end_eval_flush(maybe_flush_subs_, ms, single_mask);
                        }
                        void do_end_eval_flush(std::vector<SubPtrType>& subs, mask_set const* ms, size_t single_mask)noexcept{
                                for(auto& _ : subs){
                                        #if 0
                                        size_t weight = [&]()->size_t{
                                                auto hv_mask = _->hand_mask();
                                                if( !! ms ){
                                                        if( CheckUnion ){
                                                                if( ( ms->get_union() & hv_mask) == 0 ){
                                                                        return ms->size();
                                                                }
                                                        }
                                                        return ms->count_disjoint(hv_mask);
                                                } else {
                                                        return (( hv_mask & single_mask )==0?1:0);
                                                }
                                        }();
                                        #endif
                                        auto weight = generic_weight_policy{}
                                                .calculate(_->hand_mask(), ms, single_mask);
                                        if( CheckZeroWeight ){
                                                if( weight == 0 )
                                                        continue;
                                        }
                                        _->accept_weight(weight, evals_);
                                }
                        }
                        void regroup()noexcept{
                                // nop
                        }
                private:
                        std::vector<ranking_t> evals_;
                        std::vector<SubPtrType> no_flush_subs_;
                        std::vector<SubPtrType> maybe_flush_subs_;
                };
        };

} // end namespace ps

#endif // LIB_EVAL_GENERIC_SHED_H
