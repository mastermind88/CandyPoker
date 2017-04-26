#ifndef PS_TRANSFORM_CALC_PRIMITIVE_H
#define PS_TRANSFORM_CALC_PRIMITIVE_H

#include "ps/transforms/transform.h"
#include "ps/detail/work_schedular.h"

namespace ps{
namespace transforms{

        struct calc_primitive : symbolic_transform{
                explicit calc_primitive(calculation_context& ctx):symbolic_transform{"calc_primitive"},ctx_{&ctx}{}
                bool apply(symbolic_computation::handle& ptr)override{
                        if( ptr->get_kind() != symbolic_computation::Kind_Symbolic_Primitive )
                                return false;

                        prims_.insert( reinterpret_cast<symbolic_primitive*>(ptr.get()) );
                        return true;
                }
                void end()override{
                        detail::work_scheduler sch;
                        for( auto ptr : prims_){
                                auto j = [this,ptr](){
                                        ptr->calculate( *ctx_ );
                                };
                                sch.decl( std::move(j) );
                        }
                        sch.run();

                        #if 0
                        // pretty print
                        std::set<std::string> mem;
                        for( auto ptr : prims_){
                                #if 0
                                auto ret{ ptr->calculate(*ctx_)  };
                                std::cout << ptr->to_string() << " | ";
                                for(size_t i=0; i != ptr->get_hands().size(); ++i){
                                        auto eq{
                                                static_cast<double>(ret(i,10) ) / computation_equity_fixed_prec / ret(i,9) * 100 };
                                        std::cout << boost::format("%8.4f | ") % eq;
                                }
                                std::cout << "\n";
                                #endif
                                #if 1
                                if( mem.count( ptr->get_hash() ) )
                                        continue;
                                mem.insert( ptr->get_hash() );
                                        
                                auto ret{ ptr->calculate(*ctx_)  };
                                std::cout 
                                        << ptr->get_hash() 
                                        << "," << ret(0,0) 
                                        << "," << ptr->to_string();
                                for(size_t i=0; i != ptr->get_hands().size(); ++i){
                                        auto eq{
                                                static_cast<double>(ret(i,10) ) / computation_equity_fixed_prec / ret(i,9) * 100 };
                                        std::cout << boost::format(",%d,%d,%d,%.4f") % ret(i,0) % ret(i,1) % ret(i,2) % eq;
                                }
                                std::cout << "\n";
                                #endif
                        }
                        #endif
                }
        private:
                calculation_context* ctx_;
                std::set< symbolic_primitive* > prims_;
        };

} // transform
} // ps

#endif // PS_TRANSFORM_CALC_PRIMITIVE_H
