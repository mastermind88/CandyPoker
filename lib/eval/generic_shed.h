#ifndef LIB_EVAL_GENERIC_SHED_H
#define LIB_EVAL_GENERIC_SHED_H


namespace ps{

        struct generic_shed{
                template<class SubPtrType>
                struct bind{
                        explicit bind(size_t batch_size, std::vector<SubPtrType>& subs)
                                :batch_size_{batch_size}
                                ,subs_{subs}
                        {
                                evals_.resize(batch_size_);
                        }
                        void put(size_t index, ranking_t rank)noexcept{
                                evals_[index] = rank;
                        }
                        #if 0
                        template<class... Args>
                        void dispatch_eval(Args&&... args)noexcept{
                                for(auto& _ : subs_){
                                        _->accept(std::forward<Args>(args)..., evals_);
                                }
                        }
                        #endif
                        void dispatch_eval(tag_aggregate const&, mask_set const& ms)noexcept{
                                for(auto& _ : subs_){
                                        _->accept(tag_aggregate{}, ms, evals_);
                                }
                        }
                        void dispatch_eval(tag_singleton const&, size_t single_mask)noexcept{
                                for(auto& _ : subs_){
                                        _->accept(tag_singleton{}, single_mask, evals_);
                                }
                        }
                        #if 0
                        void begin_eval(mask_set const* ms, size_t single_mask)noexcept{
                                ms_          = ms;
                                single_mask_ = single_mask;
                                out_  = 0;
                        }
                        void end_eval()noexcept{
                                BOOST_ASSERT( out_ == batch_size_ );
                        }
                        void regroup()noexcept{
                                // nop
                        }
                        #endif
                private:
                        size_t batch_size_;
                        std::vector<ranking_t> evals_;
                        std::vector<SubPtrType>& subs_;
                        mask_set const* ms_;
                        size_t single_mask_;
                        size_t out_{0};
                };
        };

} // end namespace ps

#endif // LIB_EVAL_GENERIC_SHED_H
