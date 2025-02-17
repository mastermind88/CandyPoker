/*

CandyPoker
https://github.com/sweeterthancandy/CandyPoker

MIT License

Copyright (c) 2019 Gerry Candy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
#include <regex>

#include "ps/base/cards.h"
#include "ps/support/command.h"
#include "ps/detail/print.h"
#include "ps/eval/class_cache.h"
#include "ps/eval/holdem_class_vector_cache.h"
#include "app/pretty_printer.h"
#include "app/serialization_util.h"
#include "ps/detail/graph.h"

#include "ps/sim/computer.h"
#include "ps/sim/game_tree.h"
#include "ps/sim/computer_factory.h"
#include "ps/sim/_extra.h"
#include "ps/sim/solver.h"



namespace ps{
namespace sim{

        struct PipelineArguments : serialization_base_s<PipelineArguments>{
                // we only really need to save the most current one, but
                // we store (S0,S1,...), so that we can continue back at S1 etc
                holdem_binary_strategy_ledger_s ledger;
                size_t index{0};
                std::vector<std::pair<std::string, std::string> > ticker;
                
                friend std::ostream& operator<<(std::ostream& ostr,         PipelineArguments const& self){
                        ostr << "ledger.size() = " << self.ledger.size();
                        ostr << ", index = " << self.index;
                        typedef std::vector<std::pair<std::string, std::string> >::const_iterator CI0;
                        const char* comma = "";
                        ostr << "ticker" << " = {";
                        for(CI0 iter= self.ticker.begin(), end=self.ticker.end();iter!=end;++iter){
                                ostr << comma << iter->first << ":" << iter->second;
                                comma = ", ";
                        }
                        ostr << "}\n";
                        return ostr;
                }

                void Next(std::string const& solver, std::string const& extra = std::string{}){
                        ticker.emplace_back(solver, extra);
                }
        private:
                friend class boost::serialization::access;
                template<class Archive>
                void serialize(Archive & ar, const unsigned int version){
                        ar & ledger;
                        ar & index;
                        ar & ticker;
                }
        };

        struct Pipeline : Solver{
                Pipeline(std::shared_ptr<GameTree> gt_, GraphColouring<AggregateComputer> AG_,
                         PipelineArguments const& args)
                        : gt(gt_), AG(AG_), args_{args}
                {}
                virtual boost::optional<StateType> Execute(SolverContext& ctx)override
                {
                        for(;args_.index!=args_.ticker.size();){
                                auto const& p = args_.ticker[args_.index];
                                PS_LOG(trace) << "running solver " << p.first << " (" << p.second << " ) in pipeline";
                                auto S = args_.ledger.at(args_.index).to_eigen_vv();
                                auto solver = SolverDecl::MakeSolver(p.first, gt, AG, S, p.second);
                                auto result = solver->Execute(ctx);
                                if( ! result )
                                        return {};
                                ++args_.index;
                                args_.ledger.push_back(result.get());
                                args_.save_();
                                
                                DisplayStrategy(*result,6);
                        }
                        return args_.ledger.back().to_eigen_vv();
                }
        private:
                std::shared_ptr<GameTree> gt;
                GraphColouring<AggregateComputer> AG;
                PipelineArguments args_;
        };

        struct PipelineDecl : SolverDecl{
                virtual void Accept(ArgumentVisitor& V)const override{
                        V.DeclArgument("continue-at" , "no"          , "used for development, use --continue-at $ to continue at the end");
                        V.DeclArgument("file"        , ".ps.pipeline", "file to use for state");
                }
                virtual std::shared_ptr<Solver> Make( std::shared_ptr<GameTree> gt,
                                                      GraphColouring<AggregateComputer> AG,
                                                      StateType const& inital_state,
                                                      bpt::ptree const& args)const override
                {

                        PS_LOG(trace) << "Making pipline";

                        auto continue_at = args.get<std::string>("continue-at");
                        auto file        = args.get<std::string>("file");

                        static std::regex rgx{"\\d+"};
                        assert(     std::regex_match("989", rgx) );
                        assert( not std::regex_match("989a", rgx) );
                        assert( not std::regex_match(" 989", rgx) );

                        
                        
                        boost::optional<size_t> continue_idx;
                        bool continue_at_end = false;
                        if( std::regex_match(continue_at, rgx) ){
                                continue_idx = boost::lexical_cast<size_t>(continue_at);
                        } else if( continue_at == "$" ){
                                continue_at_end = true;
                        }

                        PipelineArguments pargs;

                        if( continue_idx || continue_at_end ){
                                PS_LOG(trace) << "Doing continue";
                                if( pargs.try_load_or_default(file) ){
                                        assert( pargs.ledger.size() == pargs.index +1 );
                                        
                                        PS_LOG(trace) << "pargs.ledger.size() => " << pargs.ledger.size();
                                        if( continue_idx ){
                                                if( continue_idx.get() > pargs.index ){
                                                        BOOST_THROW_EXCEPTION(std::domain_error("index to large"));
                                                }
                                                size_t new_size = continue_idx.get();
                                                pargs.ledger.resize(new_size+1);
                                                pargs.index = new_size;
                                                PS_LOG(trace) << "Contining at " << new_size;
                                        } else {
                                                PS_LOG(trace) << "Contining at end";
                                        }
                                } else {
                                        BOOST_THROW_EXCEPTION(std::domain_error("failed to continue"));
                                }
                        } else {
                                PS_LOG(trace) << "Making...";

                                bpt::write_json(std::cout, args);
                                pargs.ledger.push_back(inital_state);
                                for(auto const& p : args.get_child("pipeline") ){
                                        bpt::ptree const& t = p.second;
                                        auto solver = t.get_optional<std::string>("solver");
                                        if( ! solver )
                                                BOOST_THROW_EXCEPTION(std::domain_error("need solver"));
                                        std::string args;
                                        auto opt = p.second.get_child_optional("args");
                                        if( opt ){
                                                std::stringstream sstr;
                                                bpt::write_json(sstr, opt.get(),false);
                                                args  = sstr.str();
                                        }
                                        std::cout << "solver.get() => " << solver.get() << "\n"; // __CandyPrint__(cxx-print-scalar,solver.get())
                                        std::cout << "args => " << args << "\n"; // __CandyPrint__(cxx-print-scalar,args)
                                        pargs.Next(solver.get(), args);
                                }

                                #if 0
                                // else we create the args
                                pargs.Next("trail-solution"  , R"({"clamp-epsilon":1e-4, "level": 10, "factor":0.1})");
                                pargs.Next("trail-solution"  , R"({"clamp-epsilon":1e-4, "level":  5, "factor":0.05})");
                                pargs.Next("numeric-sequence", R"({"clamp-epsilon":1e-4, "factor":0.01, "sequence-type":"level-sequence", "ttl":100})");
                                //pargs.Next("permutation"     , R"({"grid-size":4, "max-evaluations":1000, "max-popcount":1})");
                                #endif
                                pargs.save_as(file);
                        } 
                        PS_LOG(trace) << "Arguments are " << pargs;
                        auto ptr = std::make_shared<Pipeline>(gt, AG, pargs);
                        return ptr;

                }
        };

        static SolverRegister<PipelineDecl> PipelineReg("pipeline");

} // end namespace sim
} // end namespace ps
