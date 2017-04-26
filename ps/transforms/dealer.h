#ifndef PS_TRANSFORM_DEALER_H
#define PS_TRANSFORM_DEALER_H

namespace ps{
namespace transforms{
        
        struct dealer : symbolic_transform{
                dealer():symbolic_transform{"dealer"}{}
                bool apply(symbolic_computation::handle& ptr)override{
                        if( ptr->get_kind() != symbolic_computation::Kind_Symbolic_Primitive )
                                return false;
                        auto sc{ reinterpret_cast<symbolic_primitive*>(ptr.get()) };
                        auto hands{ sc->get_hands() };
                        auto board{ sc->get_board() };

                        if( board.size() == 5 )
                                return false;

                        auto head = std::make_shared<symbolic_non_terminal>();

                        std::set<id_type> known;
                        for( auto h : hands){
                                auto d{ holdem_hand_decl::get(h.get())};
                                known.insert(d.first().id());
                                known.insert(d.second().id());
                        }
                        for( auto b : board){
                                known.insert(b);
                        }

                        auto start = 52;
                        if( board.size() ){
                                start = *boost::min_element(board);
                        }
                        
                        // don't want to deal 2c because there is no 
                        // other cards to deal, ie can't deal less than
                        //
                        //      43210
                        
                        //for(id_type c = 0; c != start; ++c){
                        for(id_type c = start; c != 4 - board.size() ;){
                                --c;
                                if( known.count(c) == 1 ){
                                        continue;
                                }
                                auto new_board{board};
                                new_board.emplace_back(c);
                                head->push_child( 
                                        std::make_shared<symbolic_primitive>(
                                                hands,
                                                new_board));
                        }
                        ptr = head;
                        return true;

                }
        };

} // transform
} // ps
#endif // PS_TRANSFORM_DEALER_H
