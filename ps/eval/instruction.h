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
#ifndef PS_BASE_INSTRUCTION_H
#define PS_BASE_INSTRUCTION_H


#include <Eigen/Dense>
#include <map>
#include <cfloat>
#include <memory>
#include <string>
#include <list>
#include <tuple>

#include "ps/base/cards.h"
#include "ps/base/algorithm.h"
#include "ps/base/frontend.h"
#include "ps/base/tree.h"

namespace ps{
        

struct instruction{
        enum type{
                T_ClassEval          =1 << 1,
                T_Matrix             =1 << 2,
                T_ClassVec           =1 << 4,
                // here we segregate the evals
                T_CardNoFlushEval    =1 << 8,
                T_CardMaybeFlushEval =1 << 16,
                T_CardEval           =T_CardNoFlushEval | T_CardMaybeFlushEval,
        };
        explicit instruction(std::string const& grp, int t)
                :group_{grp}, type_{(type)t}
        {}
        type get_type()const{ return type_; }

        virtual std::string to_string()const=0;
        virtual std::shared_ptr<instruction> clone()const=0;
        std::string const& group()const{ return group_; }
private:
        std::string group_;
        type type_;
};




struct class_vec_instruction : instruction{
        explicit class_vec_instruction(std::string const& grp, holdem_class_vector vec)
                :instruction{grp, T_ClassVec}
                ,vec_{std::move(vec)}
        {}
        virtual std::string to_string()const override{
                std::stringstream sstr;
                sstr << "ClassVec{" << vec_.to_string() << "}";
                return sstr.str();
        }
        virtual std::shared_ptr<instruction> clone()const override{
                return std::make_shared<class_vec_instruction>(group(), vec_);
        }
        holdem_class_vector const& get_vector()const{ return vec_; }
private:
        holdem_class_vector vec_;
};

struct matrix_instruction : instruction{
        explicit matrix_instruction(std::string const& grp, matrix_t mat,
                                    std::string const& dbg_msg = std::string{})
                :instruction{grp, T_Matrix}
                ,mat_{std::move(mat)}
                ,dbg_msg_{dbg_msg}
        {}
        virtual std::string to_string()const override{
                std::stringstream sstr;
                sstr << "Matrix{" << matrix_to_string(mat_);
                if( dbg_msg_.size() ){
                        sstr << ", " << dbg_msg_;
                }
                sstr << "}";
                return sstr.str();
        }
        virtual std::shared_ptr<instruction> clone()const override{
                return std::make_shared<matrix_instruction>(group(), mat_);
        }
        matrix_t const& get_matrix()const{ return mat_; }
        std::string const& debug_message()const{ return dbg_msg_; }
private:
        matrix_t mat_;
        std::string dbg_msg_;
};

template<class VectorType>
struct any_eval_vector_instruction : instruction{

        using vector_type = VectorType;

        any_eval_vector_instruction(int type, std::string const& grp, vector_type const& vec)
                : instruction{grp, type }
                , vec_{vec}
                , matrix_{matrix_t::Identity(vec.size(), vec.size())}
        {
        }
        any_eval_vector_instruction(int type, std::string const& grp, vector_type const& vec, matrix_t const& matrix )
                : instruction{grp, type }
                , vec_{vec}
                , matrix_{matrix}
        {
        }
        holdem_hand_vector get_vector()const{
                return vec_;
        }
        void set_vector(holdem_hand_vector const& vec){
                vec_ = vec;
        }
        matrix_t const& get_matrix()const{
                return matrix_;
        }
        void set_matrix(matrix_t const& matrix){
                matrix_ = matrix;
        }
        
        


private:
        vector_type vec_;
        matrix_t matrix_;
};

template<class Traits>
struct basic_eval_instruction : any_eval_vector_instruction<typename Traits::vector_type>{

        using self_type = basic_eval_instruction;
        using base_type = any_eval_vector_instruction<typename Traits::vector_type>;

        template<class... Args>
        basic_eval_instruction(Args&&... args)
                : base_type{ Traits::instruction_type, std::forward<Args>(args)...}
        {}
        virtual std::string to_string()const override{
                return Traits::to_string(*this);
        }
        
        virtual std::shared_ptr<instruction> clone()const override{
                return std::make_shared<self_type>(base_type::group(), base_type::get_vector(), base_type::get_matrix());
        }

        
        friend std::ostream& operator<<(std::ostream& ostr, basic_eval_instruction const& self){
                return ostr << self.to_string();
        }
};

struct class_eval_traits{
        enum{ instruction_type = instruction::T_ClassEval };
        using vector_type = holdem_class_vector;
        template<class Self>
        static std::string to_string(Self const& self){
                std::stringstream sstr;
                sstr << "ClassEval{" << self.get_vector() << ", " << matrix_to_string(self.get_matrix()) << "}";
                return sstr.str();
        }
        using type = basic_eval_instruction<class_eval_traits>;
};

struct card_eval_traits{
        enum{ instruction_type = instruction::T_CardEval };
        using vector_type = holdem_hand_vector;
        template<class Self>
        static std::string to_string(Self const& self){
                std::stringstream sstr;
                sstr << "CardEval{" << self.get_vector() << ", " << matrix_to_string(self.get_matrix()) << "}";
                return sstr.str();
        }
        using type = basic_eval_instruction<card_eval_traits>;
};


struct card_no_flush_traits{
        enum{ instruction_type = instruction::T_CardNoFlushEval };
        using vector_type = holdem_hand_vector;
        template<class Self>
        static std::string to_string(Self const& self){
                std::stringstream sstr;
                sstr << "CardNoFlushEval{" << self.get_vector() << ", " << matrix_to_string(self.get_matrix()) << "}";
                return sstr.str();
        }
        using type = basic_eval_instruction<card_no_flush_traits>;
};

struct card_maybe_flush_traits{
        enum{ instruction_type = instruction::T_CardMaybeFlushEval };
        using vector_type = holdem_hand_vector;
        template<class Self>
        static std::string to_string(Self const& self){
                std::stringstream sstr;
                sstr << "CardMaybeFlushEval{" << self.get_vector() << ", " << matrix_to_string(self.get_matrix()) << "}";
                return sstr.str();
        }

        using type = basic_eval_instruction<card_maybe_flush_traits>;
};


using class_eval_instruction             = basic_eval_instruction<class_eval_traits>;
using card_eval_instruction              = basic_eval_instruction<card_eval_traits>;
using card_no_flush_eval_instruction     = basic_eval_instruction<card_no_flush_traits>;
using card_maybe_flush_eval_instruction  = basic_eval_instruction<card_maybe_flush_traits>;

using any_card_eval_vector_instruction = any_eval_vector_instruction<holdem_hand_vector>;

using instruction_list = std::list<std::shared_ptr<instruction> >;



inline
instruction_list frontend_to_instruction_list(std::string const& group, std::vector<frontend::range> const& players){
        instruction_list instr_list;
        tree_range root( players );

        for( auto const& c : root.children ){

                #if 0
                // this means it's a class vs class evaulation
                if( c.opt_cplayers.size() != 0 ){
                        holdem_class_vector aux{c.opt_cplayers};
                        //agg.append(*class_eval.evaluate(aux));
                        instr_list.push_back(std::make_shared<class_eval_instruction>(aux));
                } else
                #endif
                {
                        for( auto const& d : c.children ){
                                holdem_hand_vector aux{d.players};
                                //agg.append(*eval.evaluate(aux));

                                instr_list.push_back(std::make_shared<card_eval_instruction>(group, aux));
                        }
                }
        }
        return instr_list;
}



} // end namespace ps

#endif // PS_BASE_INSTRUCTION_H
