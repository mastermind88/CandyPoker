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
#ifndef PS_BASE_COMPUTER_H
#define PS_BASE_COMPUTER_H

#include <unordered_map>
#include "ps/eval/instruction.h"

namespace ps{

/*
 * Just want a simple linear extensible pass manager, with the ultimate
 * idea of being able to handle transparently class vs class cache but 
 * with loose coupling
 */


instruction_list frontend_to_instruction_list(std::string const& group, std::vector<frontend::range> const& players, const bool use_perm_cache = true);


struct computation_context;
struct computation_result;

struct computation_pass{
        virtual ~computation_pass()=default;
        virtual void transform(computation_context* ctx, instruction_list* instr_list, computation_result* result = nullptr)=0;
};

struct instruction_map_pass : computation_pass{
        virtual boost::optional<instruction_list> try_map_instruction(computation_context* ctx, instruction* instr)=0;
        virtual void transform(computation_context* ctx, instruction_list* instr_list, computation_result* result)override;
};

        
struct computation_context{
        explicit
        computation_context(size_t num_players, int verbose=0)
                :num_players_(num_players)
                ,verbose_{verbose}
        {}
        size_t NumPlayers()const{
                return num_players_;
        }
        int Verboseicity()const{ return verbose_; }
        friend std::ostream& operator<<(std::ostream& ostr, computation_context const& self){
                ostr << "num_players_ = " << self.num_players_;
                return ostr;
        }
private:
        size_t num_players_;
        int verbose_;
};

/*
        The idea here, is that we have
 */
struct computation_result
        : std::unordered_map<std::string, matrix_t>
{
        explicit computation_result(computation_context const& ctx)
        {
                proto_.resize(ctx.NumPlayers(), ctx.NumPlayers());
                proto_.fill(0);
        }
        matrix_t& allocate_tag(std::string const& key){
                auto iter = find(key);
                if( iter == end()){
                        emplace(key, proto_);
                        return allocate_tag(key);
                }
                return iter->second;
        }

        // error handling
        operator bool()const{ return error_.empty(); }
        void set_error(std::string const& msg){
                // take first
                if( error_.empty()){
                        error_ = msg;
                }
        }
private:
        matrix_t proto_;
        std::string error_;
};


struct computation_pass_manager : std::vector<std::shared_ptr<computation_pass> >{
        template<class PassType, class... Args>
        void add_pass(Args&&... args){
                this->push_back(std::make_shared<PassType>(std::forward<Args>(args)...));
        }

        void execute_(computation_context* ctx, instruction_list* instr_list, computation_result* result){
                for(size_t idx=0;idx!=size();++idx){
                        at(idx)->transform(ctx, instr_list, result);
                }
        }
};


struct pass_permutate : computation_pass{
        virtual void transform(computation_context* ctx, instruction_list* instr_list, computation_result* result)override;
};
struct pass_sort_type : computation_pass{
        virtual void transform(computation_context* ctx, instruction_list* instr_list, computation_result* result)override;
};
template<class EvalInstruction>
struct basic_pass_collect : computation_pass{
        using vector_ty = typename EvalInstruction::vector_type;
        virtual void transform(computation_context* ctx, instruction_list* instr_list, computation_result* result)override{
                using iter_type = decltype(instr_list->begin());

                std::map<
                    vector_ty,
                    std::vector<iter_type>
                > vec_device;

                
                for(iter_type iter(instr_list->begin()),end(instr_list->end());iter!=end;++iter){
                    if ((*iter)->get_type() == EvalInstruction::instruction_type)
                    {
                        const auto ptr = reinterpret_cast<EvalInstruction*>(&**iter);
                        vec_device[ptr->get_vector()].push_back(iter);
                    }
                }

                for (auto const& p : vec_device)
                {
                    if (p.second.size() < 2)
                    {
                        continue;
                    }
                    std::vector<result_description> agg_result_desc;
                    for (auto const& iter : p.second)
                    {
                        const auto ptr = reinterpret_cast<EvalInstruction*>(&**iter);
                        auto const& step_result_desc = ptr->result_desc();
                        std::copy(
                            std::cbegin(step_result_desc),
                            std::cend(step_result_desc),
                            std::back_inserter(agg_result_desc));
                        instr_list->erase(iter);
                    }
                    agg_result_desc = result_description::aggregate(agg_result_desc);
                    auto agg_ptr = std::make_shared< EvalInstruction>(agg_result_desc, p.first);
                    instr_list->push_back(agg_ptr);
                }
        }
};

using pass_collect= basic_pass_collect<card_eval_instruction>;
using pass_collect_class = basic_pass_collect<class_eval_instruction>;

struct pass_print : computation_pass{
        virtual void transform(computation_context* ctx, instruction_list* instr_list, computation_result* result)override;
};
struct pass_permutate_class : computation_pass{
        virtual void transform(computation_context* ctx, instruction_list* instr_list, computation_result* result)override;
};

struct pass_class2cards : instruction_map_pass{
        virtual boost::optional<instruction_list> try_map_instruction(computation_context* ctx, instruction* instrr)override;
};
// the same as pass_class2cards, but will use an expensive cache to map class vectors directly to normalized card vectors
struct pass_class2normalisedcards : computation_pass{
        virtual void transform(computation_context* ctx, instruction_list* instr_list, computation_result* result)override;
};


struct pass_write : computation_pass{
        virtual void transform(computation_context* ctx, instruction_list* instr_list, computation_result* result)override;
};

#if 0
struct pass_check_matrix_duplicates : computation_pass{
        virtual void transform(computation_context* ctx, instruction_list* instr_list, computation_result* result)override{
                for(auto instr : *instr_list){
                        if( instr->get_type() != instruction::T_Matrix )
                                continue;
                        auto ptr = reinterpret_cast<matrix_instruction*>(instr.get());
                        auto const& matrix = ptr->get_matrix();

                }
        }
};
#endif

} // end namespace ps

#endif // PS_BASE_COMPUTER_H
