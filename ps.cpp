#include "ps/symbolic.h"
#include "ps/numeric.h"
#include "ps/transforms.h"

#include <type_traits>

#include <boost/timer/timer.hpp>
#include <boost/lexical_cast.hpp>


namespace{
        void run_driver(std::vector<ps::frontend::range> const& players)
        {

                using namespace ps;
                using namespace ps::frontend;

                symbolic_computation::handle star = std::make_shared<symbolic_range>( players );

                star->print();

                ps::numeric::work_scheduler work{players.size()};

                symbolic_computation::transform_schedular sch;

                sch.decl( symbolic_computation::transform_schedular::TransformKind_BottomUp,
                          transforms::to_lowest_permutation() );
                sch.decl( symbolic_computation::transform_schedular::TransformKind_BottomUp,
                          transforms::remove_suit_perms() );
                sch.decl( symbolic_computation::transform_schedular::TransformKind_BottomUp,
                          transforms::work_generator(work) );
                sch.execute(star);
                
                star->print();

                calculation_cache cache;
                auto ret{ star->calculate(cache) };

                auto fmtc = [](auto c){
                        static_assert( std::is_integral< std::decay_t<decltype(c)> >::value, "");
                        std::string mem = boost::lexical_cast<std::string>(c);
                        std::string ret;

                        for(size_t i=0;i!=mem.size();++i){
                                if( i % 3 == 0 && i != 0 )
                                        ret += ',';
                                ret += mem[mem.size()-i-1];
                        }
                        return std::string{ret.rbegin(), ret.rend()};
                };

                const char* title_fmt{ "%2s %12s %12s %10s %20s %10s\n" };
                const char* fmt{       "%2s %12s %12s %10s %20s %10.4f\n" };
                std::cout << boost::format(title_fmt)
                        % "n" % "wins" % "draws" % "sigma" % "equity" % "equity %";
                for(size_t i{0};i!=ret.size1();++i){
                        std::cout << boost::format(fmt)
                                % i % fmtc(ret(i,0)) % fmtc(ret(i,1)) % fmtc(ret(i,2)) % fmtc(ret(i,4)) % ( static_cast<double>(ret(i,4) ) / equity_fixed_prec / ret(i,2) * 100 );
                }

                #if 0
                ps::numeric::result_type ret{players.size()};

                {
                        boost::timer::auto_cpu_timer at;
                        ret = work.compute();
                }

                PRINT(ret);
                #endif
        }
} // anon
 
int main(){

        using namespace ps;
        using namespace ps::frontend;

        range p0;
        range p1;
        range p2;
        range p3;

        range villian;


        #if 1
        p1 += _KQs;
        p0 += _QJs;
        p2 += _AKs;
        #else
        p0 += _AKs;
        p1 += _KQs;
        p2 += _QJs;
        #endif
        
        boost::timer::auto_cpu_timer at;
        //run_driver(std::vector<frontend::range>{p0, p1});
        run_driver(std::vector<frontend::range>{p0, p1, p2});
        
}
