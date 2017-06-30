#include "ps/heads_up.h"
#include "ps/detail/print.h"

#include <numeric>
#include <boost/timer/timer.hpp>

using namespace ps;
using namespace ps::frontend;

/*
                
                                        <>
                                 ______/  \______
                                /                \
                             <SB Push>         <SB Fold>
                         ____/     \_____
                        /                \
                    <BB Call>          <BB Fold>


                               +-----------+
                               |Total Value|
                               +-----------+
                                
                        <SB fold>           = -sb
                        <SB push | BB fold> = bb
                        <SB push | BB call> = 2 S Eq - S  
                                            = S( 2 Eq - 1 )

 */


auto solve_hu_push_fold_bb_maximal_exploitable(ps::class_equity_cacher& cec,
                                               hu_strategy const& sb_push_strat,
                                               double eff_stack, double bb, double sb)
{
        hu_strategy counter_strat{0.0};

        for(holdem_class_id x{0}; x != 169;++x){
                hu_fresult_t result;
                // weight average again villians range
                for(holdem_class_id y{0}; y != 169;++y){
                        auto p{sb_push_strat[y]};

                        hu_fresult_t sim{cec.visit_boards(std::vector<ps::holdem_class_id>{ x,y })};
                        sim *= p;
                        result.append(sim);
                }

                double ev_call{ eff_stack * ( 2 * result.sigma() - 1 ) };
                double ev_fold{ -bb };
                //double ev_fold{ 0.0 };
                auto hero{ holdem_class_decl::get(x) };
                //PRINT_SEQ((hero)(sigma)(equity)(ev_call));

                if( ev_call >= ev_fold ){
                        counter_strat[x] = 1.0;
                }

        }
        return std::move(counter_strat);
} 
auto solve_hu_push_fold_sb_maximal_exploitable(ps::class_equity_cacher& cec,
                                               hu_strategy const& bb_call_strat,
                                               double eff_stack, double bb, double sb)
{
        hu_strategy counter_strat{0.0};
        for(holdem_class_id x{0}; x != 169;++x){
                /*
                 *  We call iff
                 *
                 *    EV<push x| villian call with S> >= 0
                 */


                hu_fresult_t result;

                double win_preflop{0};
                
                // weight average again villians range
                for(holdem_class_id y{0}; y != 169;++y){
                        auto p{bb_call_strat[y]};

                        hu_fresult_t sim{cec.visit_boards(std::vector<ps::holdem_class_id>{ x,y })};
                        win_preflop += sim.sigma() * ( 1 - p );
                        sim *= p;
                        result.append(sim);

                }

                auto total{ result.sigma() + win_preflop };
                double call_p{result.sigma() / total };
                double fold_p{win_preflop    / total};

                /*
                        Ev<Push> = Ev<Villan calls | Push> + Ev<Villian fold | Push>
                */
                double ev_push_call{ eff_stack * ( 2 * result.equity() - 1.0) };

                double ev_push{ call_p * ev_push_call + fold_p * bb };
                double ev_fold{ -sb };
                
                //auto hero{ holdem_class_decl::get(x) };

                //PRINT_SEQ((hero)(sigma)(call_equity)(call_p)(fold_p)(call_p+fold_p)(ev_push_call)(ev_push));

                if( ev_push >= ev_fold ){
                        counter_strat[x] = 1.0;
                }

        }
        return std::move(counter_strat);
} 



double hu_calc_sb_push_value( class_equity_cacher& cec,
                           hu_strategy const& sb_push_strat,
                           hu_strategy const& bb_call_strat,
                           double eff_stack, double bb, double sb)
{
        double sigma{0.0};
        for(holdem_class_id x{0}; x != 169;++x){

                hu_fresult_t win_call;
                double comb_sigma{0.0};


                // weight average again villians range
                for(holdem_class_id y{0}; y != 169;++y){

                        auto p{bb_call_strat[y]};

                        hu_fresult_t sim{cec.visit_boards(std::vector<ps::holdem_class_id>{ x,y })};
                        comb_sigma += sim.sigma();
                        sim *= p;
                        win_call.append(sim);

                }

                // EV = sigma_of_pot * p_win - cost_of_bet
                
                double value_push_call{ eff_stack * ( 2 * win_call.equity() - 1) };
                if( win_call.sigma() == 0 )
                        value_push_call = 0;
                
                double call_p{win_call.sigma() / comb_sigma};
                double fold_p{ 1 - call_p};

                double value{ call_p * value_push_call + fold_p * bb }; 
                
                double weighted_value{ sb_push_strat[x] * value - ( 1- sb_push_strat[x] ) * sb };

                sigma += weighted_value * holdem_class_decl::get(x).prob();

                //PRINT_SEQ((comb_sigma)(value_push_call)(call_p)(fold_p)(value)(weighted_value));

        }
        return sigma;
}


int main(){
        using namespace ps;
        using namespace ps::frontend;

        equity_cacher ec;
        ec.load("cache.bin");
        class_equity_cacher cec(ec);
        cec.load("hc_cache.bin");

        double eff_stack{10.0};
        double bb{1.0};
        double sb{0.5};

        hu_strategy sb_strat{1.0};
        hu_strategy bb_strat{0.0};

        #if 0
        PRINT_SEQ((hu_calc_sb_push_value(cec, bb_strat, sb_strat, eff_stack, bb, sb)));
        PRINT_SEQ((hu_calc_sb_push_value(cec, sb_strat, bb_strat, eff_stack, bb, sb)));
        bb_strat[0] = 1.0;
        PRINT_SEQ((hu_calc_sb_push_value(cec, bb_strat, sb_strat, eff_stack, bb, sb)));
        #endif




#if 1
        for(;;eff_stack += 1.0){
                double alpha{0.5};
                hu_strategy sb_strat{1.0};
                hu_strategy bb_strat{1.0};
                double bb{1.0};
                double sb{0.5};
                for(size_t count=0;;++count){
                        double ev{hu_calc_sb_push_value(cec, bb_strat, sb_strat, eff_stack, bb, sb)};

                        auto bb_me{solve_hu_push_fold_bb_maximal_exploitable(cec,
                                                                             sb_strat,
                                                                             eff_stack,
                                                                             bb,
                                                                             sb)};
                        auto bb_next{ bb_strat * ( 1 - alpha) + bb_me * alpha };
                        auto bb_norm{ (bb_next - bb_strat).norm() };
                        bb_strat = std::move(bb_next);

                        auto sb_me{solve_hu_push_fold_sb_maximal_exploitable(cec,
                                                                             bb_strat,
                                                                             eff_stack,
                                                                             bb,
                                                                             sb)};
                        auto sb_next{ sb_strat * ( 1 - alpha) + sb_me * alpha };
                        auto sb_norm{ (sb_next - sb_strat).norm() };
                        sb_strat = std::move(sb_next);

                        #if 0
                        PRINT(bb_strat);
                        PRINT(sb_strat);
                        #endif

                        #if 0
                        for(size_t i{0};i!=169;++i){
                                if( sb_strat[i] < 1e-3 )
                                        sb_strat[i] = 0.0;
                                else if( std::fabs(1.0 - sb_strat[i]) > 1e-3 )
                                        sb_strat[i] = 1.0;
                        }
                        #endif
                        //PRINT_SEQ((::ps::detail::to_string(sb_strat)));
                        //
                        PRINT_SEQ((sb_norm)(bb_norm)(ev));
                        if( ( sb_norm + bb_norm ) < 1e-5 ){
                                std::cout << "_break_\n";
                                break;
                        }

                }
                std::cout << "-------- BB " << eff_stack << "--------\n";
                for(size_t i{0};i!=169;++i){
                        if( sb_strat[i] < 1e-3 )
                                continue;
                        if( std::fabs(1.0 - sb_strat[i]) > 1e-3 )
                                std::cout << sb_strat[i] << " ";
                        std::cout << holdem_class_decl::get(i) << " ";
                }
                std::cout << "\n";
        }
        #if 0
        for(size_t i{0};i!=3;++i){
                boost::timer::auto_cpu_timer at;
                for(holdem_class_id x{0}; x != 169;++x){
                        for(holdem_class_id y{0}; y != 169;++y){
                                auto hero{ holdem_class_decl::get(x) };
                                auto villian{ holdem_class_decl::get(y) };
                                cec.visit_boards( std::vector<ps::holdem_class_id>{ x,y } );
                                #if 0
                                PRINT_SEQ((hero)(villian)( cec.visit_boards(
                                                std::vector<ps::holdem_class_id>{
                                                        x,y } ) ));
                                                        #endif
                        }
                }
        }
        #endif
#endif

}
