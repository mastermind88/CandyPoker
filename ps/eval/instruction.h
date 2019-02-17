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
                T_CardEval,
                T_ClassEval,
                T_Matrix,
                T_ClassVec,
                // here we segregate the evals
                T_NoFlushCardEval,
                T_MaybeFlushCardEval,
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

template<class Traits>
struct basic_eval_instruction : instruction{

        using vector_type = typename Traits::vector_type;
        using self_type = basic_eval_instruction;

        basic_eval_instruction(std::string const& grp, vector_type const& vec)
                : instruction{grp, Traits::type }
                , vec_{vec}
                , matrix_{matrix_t::Identity(vec.size(), vec.size())}
        {
        }
        basic_eval_instruction(std::string const& grp, vector_type const& vec, matrix_t const& matrix)
                : instruction{grp, Traits::type}
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
        virtual std::string to_string()const override{
                return Traits::to_string(*this);
                #if 0
                std::stringstream sstr;
                sstr << (Type == T_CardEval ? "CardEval" : "ClassEval" ) << "{" << vec_ << ", " << matrix_to_string(matrix_) << "}";
                return sstr.str();
                #endif
        }
        
        virtual std::shared_ptr<instruction> clone()const override{
                return std::make_shared<self_type>(group(), vec_, matrix_);
        }
        
        friend std::ostream& operator<<(std::ostream& ostr, basic_eval_instruction const& self){
                return ostr << self.to_string();
        }


private:
        vector_type vec_;
        matrix_t matrix_;
};

struct card_eval_traits{
        enum{ type = instruction::T_CardEval };
        using vector_type = holdem_hand_vector;
        template<class Self>
        static std::string to_string(Self const& self){
                std::stringstream sstr;
                sstr << "CardEval{" << self.get_vector() << ", " << matrix_to_string(self.get_matrix()) << "}";
                return sstr.str();
        }
};

struct class_eval_traits{
        enum{ type = instruction::T_ClassEval };
        using vector_type = holdem_class_vector;
        template<class Self>
        static std::string to_string(Self const& self){
                std::stringstream sstr;
                sstr << "ClassEval{" << self.get_vector() << ", " << matrix_to_string(self.get_matrix()) << "}";
                return sstr.str();
        }
};

using card_eval_instruction  = basic_eval_instruction<card_eval_traits>;
using class_eval_instruction = basic_eval_instruction<class_eval_traits>;


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
