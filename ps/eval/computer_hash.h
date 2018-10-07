#ifndef PS_EVAL_COMPUTER_HASH_H
#define PS_EVAL_COMPUTER_HASH_H

#include "ps/base/computer.h"
#include <future>

namespace ps{

namespace hash_computer_detail{

struct hash_eval : evaluator
{
        hash_eval(){
                impl_ = &evaluator_factory::get("6_card_map");
                card_map_7_.resize(rhasher_.max());

                for(size_t i=0;i!=52;++i){
                        card_rank_device_[i] = card_decl::get(i).rank().id();
                }

                using iter_t = basic_index_iterator<
                        int, ordered_policy, rank_vector
                >;

                for(iter_t iter(7,13),end;iter!=end;++iter){
                        maybe_add_(*iter);
                }
        }
        ranking_t rank(long a, long b, long c, long d, long e)const override{
                return impl_->rank(a,b,c,d,e);
        }
        ranking_t rank(long a, long b, long c, long d, long e, long f)const override{
                return impl_->rank(a,b,c,d,e,f);
        }
        ranking_t rank(long a, long b, long c, long d, long e, long f, long g)const override{

                auto shash =  shasher_.create_from_cards(a,b,c,d,e,f,g);

                if( shasher_.has_flush(shash)){
                        //++miss;
                        return impl_->rank(a,b,c,d,e,f,g);
                }

                auto rhash = rhasher_.create_from_cards(a,b,c,d,e,f,g);
                auto ret = card_map_7_[rhash];

                return ret;
        }
        ranking_t rank(card_vector const& cv, size_t suit_hash, size_t rank_hash, long a, long b)const noexcept{


                if( shasher_.has_flush_unsafe(suit_hash) ){
                        //return 0; // XXX
                        return impl_->rank(a,b,cv[0], cv[1], cv[2], cv[3], cv[4]);
                }
                auto ret = card_map_7_[rank_hash];
                return ret;
        }
private:
        ranking_t rank_from_rank_impl_(long a, long b, long c, long d, long e, long f, long g)const{
                return impl_->rank( card_decl::make_id(0,a),
                                    card_decl::make_id(0,b),
                                    card_decl::make_id(0,c),
                                    card_decl::make_id(0,d),
                                    card_decl::make_id(1,e),
                                    card_decl::make_id(1,f),
                                    card_decl::make_id(1,g) );
        }
        ranking_t rank_from_rank_(long a, long b, long c, long d, long e, long f, long g)const{
                return this->rank( card_decl::make_id(0,a),
                                   card_decl::make_id(0,b),
                                   card_decl::make_id(0,c),
                                   card_decl::make_id(0,d),
                                   card_decl::make_id(1,e),
                                   card_decl::make_id(1,f),
                                   card_decl::make_id(1,g) );
        }
        void maybe_add_(rank_vector const& b){
                // first check we don't have more than 4 of each card
                std::array<int, 13> aux = {0};
                for(size_t i=0;i!=7;++i){
                        ++aux[b[i]];
                }
                for(size_t i=0;i!=aux.size();++i){
                        if( aux[i] > 4 )
                                return;
                }
                auto hash = rhasher_.create( b[0], b[1], b[2], b[3], b[4], b[5], b[6] );

                auto val  = rank_from_rank_impl_( b[0], b[1], b[2], b[3], b[4], b[5], b[6] );

                //std::cout << detail::to_string(aux) << " - " << detail::to_string(b) << " => " << std::bitset<30>(static_cast<unsigned long long>(hash)).to_string() << "\n";
                //

                card_map_7_[hash] = val;
        }
        rank_hasher rhasher_;
        suit_hasher shasher_;
        evaluator* impl_;
        std::array<size_t, 52> card_rank_device_;
        std::vector<ranking_t> card_map_7_;
};


} // hash_computer_detail



struct hash_computer : card_eval_computer{
        compute_single_result_t compute_single(computation_context const& ctx, card_eval_instruction const& instr)const noexcept override{
                auto const& hv   = instr.get_vector();
                auto hv_mask = hv.mask();
                        
                // put this here

                // cache stuff

                size_t n = hv.size();
                std::array<ranking_t, 9> ranked;
                std::array<card_id, 9> hv_first;
                std::array<card_id, 9> hv_second;
                std::array<rank_id, 9> hv_first_rank;
                std::array<rank_id, 9> hv_second_rank;
                std::array<suit_id, 9> hv_first_suit;
                std::array<suit_id, 9> hv_second_suit;
                        
                for(size_t i=0;i!=hv.size();++i){
                        auto const& hand{holdem_hand_decl::get(hv[i])};

                        hv_first[i]       = hand.first().id();
                        hv_first_rank[i]  = hand.first().rank().id();
                        hv_first_suit[i]  = hand.first().suit().id();
                        hv_second[i]      = hand.second().id();
                        hv_second_rank[i] = hand.second().rank().id();
                        hv_second_suit[i] = hand.second().suit().id();
                }

                auto sub = std::make_shared<equity_breakdown_matrix_aggregator>(ctx.NumPlayers());
                for(auto const& b : w ){

                        bool cond = (b.mask() & hv_mask ) == 0;
                        if(!cond){
                                continue;
                        }
                        auto rank_proto = b.rank_hash();
                        auto suit_proto = b.suit_hash();


                        for(size_t i=0;i!=n;++i){

                                auto rank_hash = rank_proto;
                                auto suit_hash = suit_proto;

                                rank_hash = rh.append(rank_hash, hv_first_rank[i]);
                                rank_hash = rh.append(rank_hash, hv_second_rank[i]);

                                suit_hash = sh.append(suit_hash, hv_first_suit[i] );
                                suit_hash = sh.append(suit_hash, hv_second_suit[i] );


                                //ranked[i] = 0; continue; // XXX

                                ranked[i] = ev.rank(b.board(), suit_hash, rank_hash, hv_first[i], hv_second[i]);
                        }
                        detail::dispatch_ranked_vector{}(*sub, ranked, n);
                }
                return compute_single_result_t{sub, instr.get_matrix()};
        }
private:
        hash_computer_detail::hash_eval ev;
        holdem_board_decl w;
        rank_hasher rh;
        suit_hasher sh;
};

} // end namespace ps

#endif // PS_EVAL_COMPUTER_HASH_H
