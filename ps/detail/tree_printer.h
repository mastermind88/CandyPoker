#ifndef PS_DETAIL_TREE_PRINTER_H
#define PS_DETAIL_TREE_PRINTER_H

#include <boost/mpl/char.hpp>
#include <iostream>

namespace ps{
namespace detail{

        struct tree_printer_detail{

                void non_terminal(const std::string& line){
                        terminal(line);
                        stack_.emplace_back(line.size());
                }
                void terminal(const std::string& line){
                        if( stack_.size() ){
                                std::for_each( stack_.begin(), std::prev(stack_.end()),
                                        [this](format_data& fd){
                                                fd.print(std::cout);
                                });
                                stack_.back().print_alt(std::cout);
                        }
                        std::cout << line << "\n";
                }
                void begin_children_size(size_t n){
                        stack_.back().finalize(n);
                }
                void end_children(){
                        stack_.pop_back();
                }
                void next_child(){
                        stack_.back().next();
                }
        private:

                struct angle_c_traits{
                        using hor_c = boost::mpl::char_<'_'>;
                        using vert_c = boost::mpl::char_<'|'>;
                        using br_c = boost::mpl::char_<'\\'>;
                };
                struct straight_c_traits{
                        using hor_c = boost::mpl::char_<'_'>;
                        using vert_c = boost::mpl::char_<'|'>;
                        using br_c = boost::mpl::char_<'|'>;
                };
                
                using c_traits = angle_c_traits;

                struct format_data{
                        std::string make_(char c, char a){
                                std::string tmp;
                                tmp += std::string((indent_-1)/2,' ');
                                tmp += c;
                                tmp += std::string(indent_-tmp.size(),a);
                                return tmp;
                        }
                        format_data(size_t indent):indent_(indent){
                                auto sz = indent_;
                        }
                        void finalize(size_t num_children){
                                num_children_ = num_children;

                        }
                        void print_alt(std::ostream& ostr){
                                if( indent_ )
                                        ostr << make_(c_traits::br_c(), c_traits::hor_c());
                        }
                        void print(std::ostream& ostr){
                                if( indent_ ){
                                        if( idx_ != num_children_ -1 ){
                                                ostr << make_(c_traits::vert_c(), ' ');
                                        } else{
                                                ostr << std::string(indent_,' ');
                                        }
                                }
                        }
                        void next(){
                                ++idx_;
                                assert( idx_ <= num_children_ );
                        }
                private:
                        size_t idx_{0}; // child index
                        size_t num_children_; // number of children
                        const size_t indent_; // lengh of indent
                };
                std::list<format_data> stack_;
        };
} // detail
} // ps


#endif // PS_DETAIL_TREE_PRINTER_H
