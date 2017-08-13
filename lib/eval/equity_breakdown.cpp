#include "ps/eval/equity_breakdown.h"

#include <boost/lexical_cast.hpp>

namespace ps{
        std::ostream& operator<<(std::ostream& ostr, equity_breakdown const& self){
                std::vector<std::vector<std::string> > line_buffer;
                std::vector<size_t> widths(self.n(), 0);
                ostr << self.sigma() << "\n";
                for(size_t i=0;i!=self.n();++i){
                        line_buffer.emplace_back();
                        for(size_t j=0;j!=self.n();++j){
                                line_buffer.back().emplace_back(
                                        boost::lexical_cast<std::string>(
                                                self.player(i).nwin(j)));
                                widths[j] = std::max(widths[j], line_buffer.back().back().size());
                        }
                }
                for(size_t i=0;i!=self.n();++i){
                        for(size_t j=0;j!=self.n();++j){
                                auto const& tok(line_buffer[i][j]);
                                size_t padding{widths[j]-tok.size()};
                                size_t left_pad{padding/2};
                                size_t right_pad{padding - left_pad};
                                if( j != 0 ){
                                        ostr << " | ";
                                }
                                if( left_pad )
                                        ostr << std::string(left_pad,' ');
                                ostr << tok;
                                if( right_pad )
                                        ostr << std::string(right_pad,' ');

                        }
                        ostr << "\n";
                }
                return ostr;
        }
} // ps
