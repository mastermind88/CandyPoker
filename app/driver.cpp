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
#include <thread>
#include <numeric>
#include <atomic>
#include <boost/format.hpp>
#include "ps/support/config.h"
#include "ps/base/frontend.h"
#include "ps/base/cards.h"
#include "ps/base/tree.h"
#include "ps/base/board_combination_iterator.h"
#include "ps/support/index_sequence.h"
#include "ps/eval/evaluator_5_card_map.h"
#include "ps/eval/evaluator_6_card_map.h"
#include "app/pretty_printer.h"
#include "ps/base/algorithm.h"
#include "ps/eval/instruction.h"
#include "ps/base/rank_hasher.h"
#include "ps/base/suit_hasher.h"
#include "ps/base/holdem_board_decl.h"
#include "ps/eval/class_cache.h"
#include "ps/eval/pass.h"
#include "ps/eval/pass_eval_hand_instr_vec.h"
#include "ps/base/rank_board_combination_iterator.h"

#include <boost/timer/timer.hpp>

#include <boost/log/trivial.hpp>

#include <Eigen/Dense>
#include <fstream>

#include "ps/support/command.h"
#include "ps/support/persistent.h"
#include "ps/eval/holdem_class_vector_cache.h"

#include <boost/program_options.hpp>
namespace bpo = boost::program_options;

using namespace ps;

#if 0
namespace{

struct PrintMemory : Command{
        explicit
        PrintMemory(std::vector<std::string> const& args):players_s_{args}{}
        virtual int Execute()override{
                using namespace support;
                for(auto iter=persistent_memory_base::begin_decl(), end = persistent_memory_base::end_decl();iter!=end;++iter){
                        std::cout << iter->name() << "\n";
                        iter->display(std::cout);
                }
                return EXIT_SUCCESS;
        }
private:
        std::vector<std::string> const& players_s_;
};
static TrivialCommandDecl<PrintMemory> PrintMemoryDecl{"print-memory"};

struct PrintTree : Command{
        explicit
        PrintTree(std::vector<std::string> const& args):players_s_{args}{}
        virtual int Execute()override{
                std::vector<frontend::range> players;
                for(auto const& s : players_s_ ){
                        players.push_back( frontend::parse(s) );
                }
                tree_range root( players );
                root.display();

                return EXIT_SUCCESS;
        }
private:
        std::vector<std::string> const& players_s_;
};
static TrivialCommandDecl<PrintTree> PrintTreeDecl{"print-tree"};

struct StandardForm : Command{
        explicit
        StandardForm(std::vector<std::string> const& args):args_{args}{}
        virtual int Execute()override{

                holdem_class_vector cv;
                for(auto const& s : args_ ){
                        cv.push_back(s);
                }
                for( auto hvt : cv.to_standard_form_hands()){

                        std::cout << hvt << "\n";
                }

                return EXIT_SUCCESS;
        }
private:
        std::vector<std::string> const& args_;
};
static TrivialCommandDecl<StandardForm> StandardFormDecl{"standard-form"};

struct HandVectors : Command{
        explicit
        HandVectors(std::vector<std::string> const& args):args_{args}{}
        virtual int Execute()override{

                holdem_class_vector cv;
                for(auto const& s : args_ ){
                        cv.push_back(s);
                }
                for( auto hv : cv.get_hand_vectors()){
                        std::cout << "  " << hv << "\n";
                }

                return EXIT_SUCCESS;
        }
private:
        std::vector<std::string> const& args_;
};
static TrivialCommandDecl<HandVectors> HandVectorsDecl{"hand-vectors"};



#if 0
struct SimpleCardEval : Command{
        explicit
        SimpleCardEval(std::vector<std::string> const& args):args_{args}{}
        virtual int Execute()override{

                std::vector<frontend::range> players;
                for(auto const& s : args_ ){
                        players.push_back( frontend::parse(s) );
                }

                auto instr_list = frontend_to_instruction_list(players);

                auto comp = std::make_shared<eval_computer>();

                computation_context comp_ctx{players.size()};
                auto result = comp->compute(comp_ctx, instr_list);

                pretty_print_equity_breakdown(std::cout, *result, args_);

                return EXIT_SUCCESS;
        }
private:
        std::vector<std::string> const& args_;
};
static TrivialCommandDecl<SimpleCardEval> SimpleCardEvalDecl{"eval"};
#endif



struct FrontendDbg : Command{
        explicit
        FrontendDbg(std::vector<std::string> const& args):players_s_{args}{}
        virtual int Execute()override{
                
                std::vector<std::string> title;
                title.push_back("literal");
                title.push_back("range");
                title.push_back("expanded");

                using namespace Pretty;
                std::vector< LineItem > lines;
                lines.push_back(title);
                lines.emplace_back(LineBreak);
                
                for(auto const& s : players_s_ ){
                        auto rng = frontend::parse(s);
                        auto expanded = expand(rng);
                        auto prim_rng = expanded.to_primitive_range();
                        std::vector<std::string> line;
                        line.push_back(s);
                        line.push_back(boost::lexical_cast<std::string>(rng));
                        line.push_back(boost::lexical_cast<std::string>(expanded));

                        try{
                                auto cv = expanded.to_class_vector();
                                line.push_back(boost::lexical_cast<std::string>(cv));
                        }catch(...){
                                line.push_back("error");
                        }
                        #if 0
                        try{
                                auto hv = expanded.to_holdem_vector();
                                line.push_back(boost::lexical_cast<std::string>(hv));
                        }catch(...){
                                line.push_back("error");
                        }
                        #endif

                        lines.push_back(line);
                }
                
                RenderTablePretty(std::cout, lines);
                
                return EXIT_SUCCESS;
        }
private:
        std::vector<std::string> const& players_s_;
};
static TrivialCommandDecl<FrontendDbg> FrontendDbgDecl{"frontend-dbg"};











struct IteratorDbg : Command{
        explicit
        IteratorDbg(std::vector<std::string> const& args):players_s_{args}{}
        virtual int Execute()override{
                enum{ MaxIter = 200 };
                size_t n = 0;
                
                std::cout << std::fixed;
                n = 0;
                std::cout << "rank_board_combination_iterator\n";
                for(rank_board_combination_iterator iter(7),end;iter!=end && n < MaxIter;++iter,++n){
                        std::cout << "    " <<  *iter << "\n";
                }
                return 0;


                std::cout << "holdem_hand_iterator\n";
                n = 0;
                for(holdem_hand_iterator iter(5),end;iter!=end && n < MaxIter;++iter,++n){
                        std::cout << "    " <<  *iter << "\n";
                }

                std::cout << "holdem_class_iterator\n";
                n = 0;
                for(holdem_class_iterator iter(3),end;iter!=end && n < MaxIter;++iter,++n){
                        std::cout << "    " <<  *iter << "\n";
                }

                std::cout << "board_combination_iterator\n";
                n = 0;
                for(board_combination_iterator iter(3),end;iter!=end && n < MaxIter;++iter,++n){
                        std::cout << "    " <<  *iter << "\n";
                }
                
                std::cout << "holdem_class_perm_iterator<2>\n";
                n = 0;
                for(holdem_class_perm_iterator iter(2),end;iter!=end && n < MaxIter;++iter,++n){
                        auto p = iter->prob();
                        std::cout << "    " <<  *iter << " - " << p << "\n";
                }
                
                std::cout << "holdem_class_perm_iterator<3>\n";
                n = 0;
                for(holdem_class_perm_iterator iter(3),end;iter!=end && n < MaxIter;++iter,++n){
                        auto p = iter->prob();
                        std::cout << "    " <<  *iter << " - " << p << "\n";
                }

                double sigma = 0.0;
                for(holdem_class_perm_iterator iter(2),end;iter!=end;++iter){
                        sigma += iter->prob();
                }
                std::cout << "sigma<2> => " << sigma << "\n"; // __CandyPrint__(cxx-print-scalar,sigma)





                return EXIT_SUCCESS;
        }
private:
        std::vector<std::string> const& players_s_;
};
static TrivialCommandDecl<IteratorDbg> IteratorDbgDecl{"iterator-dbg"};


struct PrintRanks : Command{
        explicit
        PrintRanks(std::vector<std::string> const& args):args_{args}{}
        virtual int Execute()override{
                rank_world rankdev;
                for(auto const& rd : rankdev ){
                        std::cout << rd << "\n";
                }

                return EXIT_SUCCESS;
        }
private:
        std::vector<std::string> const& args_;
};
static TrivialCommandDecl<PrintRanks> PrintRanksDecl{"print-ranks"};

struct PrintBoard : Command{
        explicit
        PrintBoard(std::vector<std::string> const& args):args_{args}{}
        virtual int Execute()override{

                holdem_board_decl b;
                for(auto const& _ : b ){
                        std::cout << _.board() << ","
                                  << _.flush_possible() << ","
                                  << +_.flush_suit() << ","
                                  << std::bitset<13>(_.flush_mask()).to_string() << "\n";
                }

                return EXIT_SUCCESS;
        }
private:
        std::vector<std::string> const& args_;
};
static TrivialCommandDecl<PrintBoard> PrintBoardDecl{"print-board"};

} // end namespace anon
#endif



int main(int argc, char** argv){
#if 0
#define USE_PROXY 1
    static char* argv_proxy[] = {
        "dummy",
        "eval",
        "--debug",
        "1",
        "--engine",
        "generic",
        "AhAd",
        "AcAs",
        "KsKd",
        nullptr };
#else
    static char* argv_proxy[] = {
        "dummy",
        "solver",
        nullptr };
#endif
    int argc_proxy = sizeof(argv_proxy) / sizeof(void*) - 1;

        try{
#ifdef USE_PROXY
            CommandDecl::Driver(argc_proxy, argv_proxy);
#else
                CommandDecl::Driver(argc, argv);
#endif
        } catch(std::exception const& e){
                std::cerr << "Caught exception: " << e.what() << "\n";
        }
        return EXIT_SUCCESS;
}

