#ifndef PS_CARDS_FWD_H
#define PS_CARDS_FWD_H

#include <string>
#include <cstdint>

namespace ps{

        #if 0
        using id_type         =  std::uint16_t;

        using suit_id         = std::uint8_t;
        using rank_id         = std::uint8_t; 
        using card_id         = std::uint8_t; 
        using holdem_id       = std::uint16_t;
        using holdem_class_id = std::uint8_t; 
        #else

        using id_type =  unsigned;

        using suit_id   = std::uint_fast8_t;
        using rank_id   = std::uint_fast8_t;
        using card_id   = std::uint_fast8_t;
        using holdem_id = unsigned;
        using holdem_class_id = unsigned;

        #endif


        enum class suit_category{
                any_suit,
                suited,
                offsuit
        };
        
        enum class holdem_class_type{
                pocket_pair,
                suited,
                offsuit
        };


        struct suit_decl;
        struct rank_decl;
        struct holdem_hand_decl;
        struct holdem_class_decl;
        
        
        inline card_id card_suit_from_id(card_id id){
                return id & 0x3;
        }
        inline card_id card_rank_from_id(card_id id){
                return id >> 2;
        }
        
} // ps

#endif // PS_CARDS_FWD_H
